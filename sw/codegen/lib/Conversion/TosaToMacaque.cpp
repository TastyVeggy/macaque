#include "macaque/Conversion/TosaToMacaque.h"

#include <optional>

#include "macaque/Dialect/MacaqueOps.h"
#include "macaque/common/isa.hpp"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Block.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::macaque;
namespace isa = ::macaque::common::isa;

namespace {


uint32_t byteSizeOf(RankedTensorType type) {
  return static_cast<uint32_t>(type.getNumElements() *
                               (type.getElementTypeBitWidth() / 8));
}

// Extract the single value of a 1-element tosa.const.
std::optional<int64_t> getScalarConstValue(Value v) {
  auto constOp = v.getDefiningOp<tosa::ConstOp>();
  if (!constOp) return std::nullopt;
  auto type = cast<RankedTensorType>(constOp.getType());
  if (type.getNumElements() != 1) return std::nullopt;
  auto elements = cast<DenseElementsAttr>(constOp.getValues());
  return elements.getSplatValue<APInt>().getSExtValue();
}

// Only handles both matmul operands being tosa.const for now.
// TODO: Handle a real (non-constant) activation input.
struct MatmulOperands {
  tosa::ConstOp aConst;
  tosa::ConstOp bConst;
  RankedTensorType aType;
  RankedTensorType bType;
  int64_t rows;
};

FailureOr<MatmulOperands> matchMatmulOperands(tosa::MatMulOp matmul,
                                              PatternRewriter& rewriter) {
  auto aConst = matmul.getA().getDefiningOp<tosa::ConstOp>();
  auto bConst = matmul.getB().getDefiningOp<tosa::ConstOp>();
  if (!aConst || !bConst)
    return rewriter.notifyMatchFailure(
        matmul, "only tosa.const operands are supported for now");

  std::optional<int64_t> aZp = getScalarConstValue(matmul.getAZp());
  std::optional<int64_t> bZp = getScalarConstValue(matmul.getBZp());
  if (!aZp || *aZp != 0 || !bZp || *bZp != 0)
    return rewriter.notifyMatchFailure(
        matmul,
        "only zero a_zp/b_zp are supported for now. Zero-point folding "
        "into bias isn't implemented yet");

  auto aType = cast<RankedTensorType>(aConst.getType());
  auto bType = cast<RankedTensorType>(bConst.getType());
  // [batch, rows, k] for a, [batch, k, cols] for b (TOSA's rank-3 matmul).
  // macaque has no batch concept.
  if (aType.getShape()[0] != 1 || bType.getShape()[0] != 1)
    return rewriter.notifyMatchFailure(
        matmul, "only batch=1 matmuls are supported for now");

  return MatmulOperands{aConst, bConst, aType, bType, aType.getShape()[1]};
}

struct MatmulChain {
  tosa::MatMulOp matmul;
  tosa::AddOp biasAdd;      // null if there's no bias
  tosa::ConstOp biasConst;  // null if there's no bias
};

std::optional<MatmulChain> matchMatmulChain(Value rescaleInput) {
  if (auto matmul = rescaleInput.getDefiningOp<tosa::MatMulOp>())
    return MatmulChain{matmul, nullptr, nullptr};

  auto addOp = rescaleInput.getDefiningOp<tosa::AddOp>();
  if (!addOp) return std::nullopt;

  auto matmul = addOp.getInput1().getDefiningOp<tosa::MatMulOp>();
  auto biasConst = addOp.getInput2().getDefiningOp<tosa::ConstOp>();
  if (!matmul) {
    matmul = addOp.getInput2().getDefiningOp<tosa::MatMulOp>();
    biasConst = addOp.getInput1().getDefiningOp<tosa::ConstOp>();
  }
  if (!matmul || !biasConst) return std::nullopt;
  return MatmulChain{matmul, addOp, biasConst};
}

// Pre-pass: sum each region's total bytes across the whole block before
// any conversion runs, so we can dynamically get the offsets
struct DdrRegionTotals {
  uint32_t weightsBytes = 0;
  uint32_t biasesBytes = 0;
  uint32_t inputBytes = 0;
  uint32_t outputBytes = 0;
};

DdrRegionTotals sizeRegions(Block& block) {
  DdrRegionTotals totals;
  for (Operation& op : block) {
    if (auto matmul = dyn_cast<tosa::MatMulOp>(op)) {
      if (auto aConst = matmul.getA().getDefiningOp<tosa::ConstOp>())
        totals.inputBytes +=
            byteSizeOf(cast<RankedTensorType>(aConst.getType()));
      if (auto bConst = matmul.getB().getDefiningOp<tosa::ConstOp>())
        totals.weightsBytes +=
            byteSizeOf(cast<RankedTensorType>(bConst.getType()));
    } else if (auto rescale = dyn_cast<tosa::RescaleOp>(op)) {
      totals.outputBytes +=
          byteSizeOf(cast<RankedTensorType>(rescale.getOutput().getType()));
      if (std::optional<MatmulChain> chain =
              matchMatmulChain(rescale.getInput());
          chain && chain->biasConst) {
        totals.biasesBytes +=
            byteSizeOf(cast<RankedTensorType>(chain->biasConst.getType()));
      }
    }
  }
  return totals;
}

// Layout is in accordance to sw/docs/MEMORY_LAYOUT.md
class DdrLayout {
 public:
  explicit DdrLayout(const DdrRegionTotals& totals) {
    weight_next_ = kWeightBase;
    bias_next_ = alignUp(weight_next_ + totals.weightsBytes);
    input_next_ = alignUp(bias_next_ + totals.biasesBytes);
    output_next_ = alignUp(input_next_ + totals.inputBytes);
  }

  uint32_t allocateWeight(uint32_t bytes) { return bump(weight_next_, bytes); }
  uint32_t allocateBias(uint32_t bytes) { return bump(bias_next_, bytes); }
  uint32_t allocateInput(uint32_t bytes) { return bump(input_next_, bytes); }
  uint32_t allocateOutput(uint32_t bytes) { return bump(output_next_, bytes); }

 private:
  static constexpr uint32_t kWeightBase = 0x0000'1000;

  static uint32_t alignUp(uint32_t x) { return (x + 7u) & ~7u; }

  static uint32_t bump(uint32_t& cursor, uint32_t bytes) {
    uint32_t addr = cursor;
    cursor = alignUp(cursor + bytes);
    return addr;
  }

  uint32_t weight_next_;
  uint32_t bias_next_;
  uint32_t input_next_;
  uint32_t output_next_;
};

void emitLoadWeightAndInput(const MatmulOperands& operands, DdrLayout& layout,
                            Location loc, ConversionPatternRewriter& rewriter) {
  const uint32_t inputBytes = byteSizeOf(operands.aType);
  const uint32_t weightBytes = byteSizeOf(operands.bType);
  const uint32_t inputAddr = layout.allocateInput(inputBytes);
  const uint32_t weightAddr = layout.allocateWeight(weightBytes);

  LoadWeightOp::create(rewriter, loc, TypeRange{}, weightAddr,
                       static_cast<uint16_t>(weightBytes));
  LoadInputOp::create(rewriter, loc, TypeRange{}, inputAddr,
                      static_cast<uint16_t>(inputBytes));
}

// Pattern 1: bare tosa.matmul -> load_weight, load_input, matmul(acc_mode=0)
//
// Only applies to a matmul with no downstream tosa.rescale (feedsRescale returns false)

bool feedsRescale(tosa::MatMulOp matmul) {
  for (Operation* user : matmul->getUsers()) {
    if (isa<tosa::RescaleOp>(user)) return true;
    if (isa<tosa::AddOp>(user))
      for (Operation* addUser : user->getUsers())
        if (isa<tosa::RescaleOp>(addUser)) return true;
  }
  return false;
}

struct MatmulToMacaque : public OpConversionPattern<tosa::MatMulOp> {
  MatmulToMacaque(MLIRContext* ctx, DdrLayout& layout)
      : OpConversionPattern(ctx), layout(layout) {}

  LogicalResult matchAndRewrite(
      tosa::MatMulOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const override {

    FailureOr<MatmulOperands> operands = matchMatmulOperands(op, rewriter);
    if (failed(operands)) return failure();

    Location loc = op.getLoc();
    emitLoadWeightAndInput(*operands, layout, loc, rewriter);
    MatmulOp::create(rewriter, loc, TypeRange{}, /*acc_mode=*/false,
                     static_cast<uint16_t>(operands->rows));

    rewriter.eraseOp(op);
    return success();
  }

 private:
  DdrLayout& layout;
};

// Pattern 2: tosa.rescale(+ bias tosa.add) -> load_weight, [load_bias],
// load_input, matmul(acc_mode=0), activate, store

LogicalResult checkRescaleIsSupported(tosa::RescaleOp op,
                                      ConversionPatternRewriter& rewriter) {
  if (op.getPerChannel())
    return rewriter.notifyMatchFailure(
        op, "per-channel rescale is not supported yet");
  if (op.getInputUnsigned() || op.getOutputUnsigned())
    return rewriter.notifyMatchFailure(op,
                                       "unsigned rescale I/O is not supported");
  if (op.getRoundingMode() != tosa::RoundingMode::SINGLE_ROUND)
    return rewriter.notifyMatchFailure(
        op, "only SINGLE_ROUND matches ACTIVATE's fixed rounding");
  return success();
}

struct RescaleToMacaque : public OpConversionPattern<tosa::RescaleOp> {
  RescaleToMacaque(MLIRContext* ctx, DdrLayout& layout)
      : OpConversionPattern(ctx), layout(layout) {}

  LogicalResult matchAndRewrite(
      tosa::RescaleOp op, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const override {
    if (failed(checkRescaleIsSupported(op, rewriter))) return failure();

    std::optional<int64_t> inputZp = getScalarConstValue(op.getInputZp());
    std::optional<int64_t> outputZp = getScalarConstValue(op.getOutputZp());
    if (!inputZp || *inputZp != 0 || !outputZp || *outputZp != 0)
      return rewriter.notifyMatchFailure(
          op,
          "only zero input/output zero-points are supported for now - "
          "zero-point folding into bias/M/shift isn't implemented yet");

    std::optional<int64_t> multiplier = getScalarConstValue(op.getMultiplier());
    std::optional<int64_t> shift = getScalarConstValue(op.getShift());
    if (!multiplier || !shift)
      return rewriter.notifyMatchFailure(
          op,
          "multiplier/shift must each be a single-element tosa.const "
          "(per-tensor, not per-channel)");

    std::optional<MatmulChain> chain = matchMatmulChain(op.getInput());
    if (!chain)
      return rewriter.notifyMatchFailure(
          op,
          "rescale input must be a matmul, or a matmul plus a constant "
          "bias add");

    FailureOr<MatmulOperands> operands =
        matchMatmulOperands(chain->matmul, rewriter);
    if (failed(operands)) return failure();

    Location loc = op.getLoc();
    emitLoadWeightAndInput(*operands, layout, loc, rewriter);

    if (chain->biasConst) {
      auto biasType = cast<RankedTensorType>(chain->biasConst.getType());
      const uint32_t biasBytes = byteSizeOf(biasType);
      const uint32_t biasAddr = layout.allocateBias(biasBytes);
      LoadBiasOp::create(rewriter, loc, TypeRange{}, biasAddr,
                         static_cast<uint16_t>(biasBytes));
    }

    MatmulOp::create(rewriter, loc, TypeRange{}, /*acc_mode=*/false,
                     static_cast<uint16_t>(operands->rows));
    ActivateOp::create(
        rewriter, loc, TypeRange{},
        /*act_func=*/static_cast<uint8_t>(isa::ActFunc::Passthrough),
        static_cast<uint32_t>(*multiplier), static_cast<uint8_t>(*shift),
        static_cast<uint8_t>(operands->rows));

    // Write the requantized INT8 result tile back to DDR3's Output region which is
    // treated as final output, not intermediate/scratch because
    // matmul currently only accepts const.
    auto outType = cast<RankedTensorType>(op.getOutput().getType());
    const uint32_t outputBytes = byteSizeOf(outType);
    const uint32_t outputAddr = layout.allocateOutput(outputBytes);
    StoreOp::create(rewriter, loc, TypeRange{}, outputAddr,
                    static_cast<uint16_t>(outputBytes));

    rewriter.eraseOp(op);
    if (chain->biasAdd) rewriter.eraseOp(chain->biasAdd);
    rewriter.eraseOp(chain->matmul);
    return success();
  }

 private:
  DdrLayout& layout;
};

}  // namespace

namespace macaque::codegen::conversion {

LogicalResult lowerTosaToMacaque(Block& block) {
  MLIRContext* ctx = block.getParentOp() ? block.getParentOp()->getContext()
                                         : block.front().getContext();
  ConversionTarget target(*ctx);
  target.addLegalDialect<MacaqueDialect>();
  target.addDynamicallyLegalOp<tosa::MatMulOp>(
      [](tosa::MatMulOp op) { return feedsRescale(op); });
  target.addIllegalOp<tosa::RescaleOp>();

  DdrLayout layout(sizeRegions(block));
  RewritePatternSet patterns(ctx);
  patterns.add<RescaleToMacaque>(ctx, layout);
  patterns.add<MatmulToMacaque>(ctx, layout);

  SmallVector<Operation*> ops;
  for (Operation& op : block) ops.push_back(&op);

  return applyPartialConversion(ops, target, std::move(patterns));
}

}  // namespace macaque::codegen::conversion
