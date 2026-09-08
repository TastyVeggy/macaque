#include "macaque/Conversion/TosaToMacaque.hpp"

#include <algorithm>
#include <optional>

#include "macaque/Conversion/DdrLayout.hpp"
#include "macaque/Conversion/TilingCommon.hpp"
#include "macaque/Dialect/MacaqueOps.hpp"
#include "macaque/common/isa.hpp"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Block.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::macaque;
using namespace ::macaque::codegen::conversion;
using namespace ::macaque::codegen::conversion::detail;
namespace macaque_isa = ::macaque::common::isa;

namespace {
// Extract the single value of a 1-element tosa.const.
std::optional<int64_t> getScalarConstValue(Value v) {
  auto constOp = v.getDefiningOp<tosa::ConstOp>();
  if (!constOp)
    return std::nullopt;
  auto type = cast<RankedTensorType>(constOp.getType());
  if (type.getNumElements() != 1)
    return std::nullopt;
  auto elements = cast<DenseElementsAttr>(constOp.getValues());
  return elements.getSplatValue<APInt>().getSExtValue();
}

uint32_t activationTileBytes(RankedTensorType aType, int64_t rows) {
  return static_cast<uint32_t>(rows * kTileWidth) *
         (aType.getElementTypeBitWidth() / 8);
}

struct InputLoadSizing {
  uint32_t byteCount;       // LoadInputOp's byte_count operand
  uint8_t validBytesPerRow; // 0 disables padding
  uint8_t inputRows;        // only meaningful when validBytesPerRow != 0
};

InputLoadSizing computeInputLoadSizing(RankedTensorType aType, int64_t fullK,
                                       int64_t kOffset, int64_t chunkRows,
                                       bool isRuntimeInput) {
  const int64_t elemBytes = aType.getElementTypeBitWidth() / 8;
  const int64_t validCols = std::min(kTileWidth, fullK - kOffset);
  if (!isRuntimeInput || validCols >= kTileWidth)
    return {activationTileBytes(aType, chunkRows), /*validBytesPerRow=*/0,
            /*inputRows=*/0};
  const auto validBytesPerRow = static_cast<uint8_t>(validCols * elemBytes);
  return {static_cast<uint32_t>(chunkRows) * validBytesPerRow, validBytesPerRow,
          static_cast<uint8_t>(chunkRows)};
}

// Materializes every element of a rank-3 [1, dim0, dim1] integer
// tosa.const into a flat, row-major value grid, for `extractTileBytes`
// to index into
SmallVector<int64_t> flattenConstValues(tosa::ConstOp constOp) {
  auto elements = cast<DenseElementsAttr>(constOp.getValues());
  SmallVector<int64_t> values;
  values.reserve(elements.getNumElements());
  for (const APInt &v : elements.getValues<APInt>())
    values.push_back(v.getSExtValue());
  return values;
}

// Extracts one (rowTileSize x colTileSize) tile
// from a flat, row-major (fullRows x fullCols) value grid, as raw
// little-endian bytes. It also handles the zero-padding
SmallVector<uint8_t> extractTileBytes(ArrayRef<int64_t> values,
                                      int64_t fullRows, int64_t fullCols,
                                      int64_t rowBase, int64_t colBase,
                                      int64_t rowTileSize, int64_t colTileSize,
                                      int64_t elemBytes) {
  SmallVector<uint8_t> bytes;
  bytes.reserve(static_cast<size_t>(rowTileSize * colTileSize * elemBytes));
  for (int64_t r = 0; r < rowTileSize; r++) {
    for (int64_t c = 0; c < colTileSize; c++) {
      const int64_t row = rowBase + r, col = colBase + c;
      const uint64_t v =
          (row < fullRows && col < fullCols)
              ? static_cast<uint64_t>(values[row * fullCols + col])
              : 0;
      for (int64_t b = 0; b < elemBytes; b++)
        bytes.push_back(static_cast<uint8_t>((v >> (8 * b)) & 0xFF));
    }
  }
  return bytes;
}

struct MatmulOperands {
  Value a;
  tosa::ConstOp bConst;
  RankedTensorType aType;
  RankedTensorType bType;
  int64_t rows;
  int64_t numKTiles;
  int64_t numNTiles;
  int64_t numMChunks;
};

FailureOr<MatmulOperands> matchMatmulOperands(tosa::MatMulOp matmul,
                                              PatternRewriter &rewriter) {
  Value a = matmul.getA();
  if (!a.getDefiningOp<tosa::ConstOp>() && !isa<BlockArgument>(a) &&
      !a.getDefiningOp<tosa::RescaleOp>() && !a.getDefiningOp<tosa::ClampOp>())
    return rewriter.notifyMatchFailure(
        matmul,
        "activation input must be a tosa.const, a block argument, a prior "
        "tosa.rescale's result, or a fused-ReLU tosa.clamp's result (see "
        "matchReluClamp)");

  auto bConst = matmul.getB().getDefiningOp<tosa::ConstOp>();
  if (!bConst)
    return rewriter.notifyMatchFailure(matmul, "weights must be a tosa.const");

  std::optional<int64_t> aZp = getScalarConstValue(matmul.getAZp());
  std::optional<int64_t> bZp = getScalarConstValue(matmul.getBZp());
  if (!aZp || *aZp != 0 || !bZp || *bZp != 0)
    return rewriter.notifyMatchFailure(
        matmul, "only zero a_zp/b_zp are supported for now. Zero-point folding "
                "into bias isn't implemented yet");

  auto aType = cast<RankedTensorType>(a.getType());
  auto bType = cast<RankedTensorType>(bConst.getType());
  // [batch, rows, k] for a, [batch, k, cols] for b (TOSA's rank-3 matmul).
  // macaque has no batch concept.
  if (aType.getShape()[0] != 1 || bType.getShape()[0] != 1)
    return rewriter.notifyMatchFailure(
        matmul, "only batch=1 matmuls are supported for now");

  const int64_t numKTiles = numTilesFor(bType.getShape()[1]);
  const int64_t numNTiles = numTilesFor(bType.getShape()[2]);
  const int64_t numMChunks = numMChunksFor(aType.getShape()[1]);

  return MatmulOperands{
      a,         bConst,    aType,      bType, aType.getShape()[1],
      numKTiles, numNTiles, numMChunks};
}

// Input tile addresses, addrs[mChunk][kTile]. They are shared across every
// N-tile, since the activation doesn't depend on which output-channel tile is
// being computed. For a chained activation, `intermediateAddr` holds the
// producer's own addrs[nTile][mChunk], which is transposed: the producer's
// N-tile k is this layer's K-tile k
SmallVector<SmallVector<uint32_t>> allocateInputAddrs(
    const MatmulOperands &operands, DdrLayout &layout,
    const DenseMap<Value, SmallVector<SmallVector<uint32_t>>> &intermediateAddr,
    bool &isRuntimeInput) {
  isRuntimeInput = false;
  auto chained = intermediateAddr.find(operands.a);
  if (chained != intermediateAddr.end()) {
    const auto &producerAddrs = chained->second; // [nTile][mChunk]
    SmallVector<SmallVector<uint32_t>> addrs(operands.numMChunks);
    for (int64_t m = 0; m < operands.numMChunks; m++) {
      addrs[m].reserve(operands.numKTiles);
      for (int64_t k = 0; k < operands.numKTiles; k++)
        addrs[m].push_back(producerAddrs[k][m]);
    }
    return addrs;
  }

  // aConst is pretty much just for testing. Activation input
  // should almost always be supplied by the runtime or result from a previous
  // layer
  auto aConst = operands.a.getDefiningOp<tosa::ConstOp>();
  isRuntimeInput = !aConst;
  const SmallVector<int64_t> actValues =
      aConst ? flattenConstValues(aConst) : SmallVector<int64_t>{};
  const int64_t fullK = operands.aType.getShape()[2];
  const int64_t actElemBytes = operands.aType.getElementTypeBitWidth() / 8;

  SmallVector<SmallVector<uint32_t>> addrs;
  addrs.reserve(operands.numMChunks);
  for (int64_t m = 0; m < operands.numMChunks; ++m) {
    const int64_t chunkRows = mChunkRows(operands.rows, m);
    const uint32_t tileBytes = activationTileBytes(operands.aType, chunkRows);
    SmallVector<uint32_t> kAddrs;
    kAddrs.reserve(operands.numKTiles);
    for (int64_t k = 0; k < operands.numKTiles; k++) {
      const uint32_t addr = layout.allocateInput(tileBytes);
      if (aConst) {
        layout.recordData(
            addr, extractTileBytes(actValues, operands.rows, fullK,
                                   /*rowBase=*/m * kMaxMChunkRows,
                                   /*colBase=*/k * kTileWidth, chunkRows,
                                   kTileWidth, actElemBytes));
      } else {
        layout.recordInputTile(addr, tileBytes);
        layout.setInputValidBytes(fullK * actElemBytes);
      }
      kAddrs.push_back(addr);
    }
    addrs.push_back(std::move(kAddrs));
  }
  return addrs;
}

uint32_t allocateBiasAddr(tosa::ConstOp biasConst, int64_t nTile,
                          int64_t mChunk, int64_t chunkRows,
                          DdrLayout &layout) {
  const uint32_t tileBytes =
      static_cast<uint32_t>(chunkRows) * kBiasTileBytes;
  if (!biasConst)
    return layout.zeroBiasAddr();
  const uint32_t addr = layout.allocateBias(tileBytes);
  const SmallVector<int64_t> biasValues = flattenConstValues(biasConst);
  auto biasType = cast<RankedTensorType>(biasConst.getType());
  layout.recordData(
      addr, extractTileBytes(biasValues, /*fullRows=*/biasType.getShape()[1],
                             /*fullCols=*/biasType.getShape()[2],
                             /*rowBase=*/mChunk * kMaxMChunkRows,
                             /*colBase=*/nTile * kTileWidth, chunkRows,
                             kTileWidth, /*elemBytes=*/sizeof(int32_t)));
  return addr;
}

void emitBiasLoad(uint32_t addr, int64_t chunkRows, Location loc,
                  ConversionPatternRewriter &rewriter) {
  LoadBiasOp::create(
      rewriter, loc, TypeRange{}, addr,
      static_cast<uint16_t>(static_cast<uint32_t>(chunkRows) * kBiasTileBytes));
}

// One DDR3 weight tile per K-tile for this N-tile. The tiles are the same for
// every M-chunk, so they are allocated once here and only reloaded from
SmallVector<uint32_t> allocateKTileWeightAddrs(const MatmulOperands &operands,
                                               int64_t nTile,
                                               DdrLayout &layout) {
  const uint32_t weightBytes = weightTileBytes(operands.bType);
  const SmallVector<int64_t> weightValues = flattenConstValues(operands.bConst);
  const int64_t fullK = operands.bType.getShape()[1];
  const int64_t fullN = operands.bType.getShape()[2];
  const int64_t weightElemBytes = operands.bType.getElementTypeBitWidth() / 8;
  SmallVector<uint32_t> addrs;
  addrs.reserve(operands.numKTiles);
  for (int64_t k = 0; k < operands.numKTiles; k++) {
    const uint32_t addr = layout.allocateWeight(weightBytes);
    layout.recordData(addr, extractTileBytes(weightValues, fullK, fullN,
                                             /*rowBase=*/k * kTileWidth,
                                             /*colBase=*/nTile * kTileWidth,
                                             kTileWidth, kTileWidth,
                                             weightElemBytes));
    addrs.push_back(addr);
  }
  return addrs;
}

// Pattern looking out for
//  tosa.rescale(+ bias tosa.add) -> load_weight, [load_bias], load_input,
// matmul(acc_mode=0), activate, store.

LogicalResult checkRescaleIsSupported(tosa::RescaleOp op,
                                      ConversionPatternRewriter &rewriter) {
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
  RescaleToMacaque(
      MLIRContext *ctx, DdrLayout &layout,
      DenseMap<Value, SmallVector<SmallVector<uint32_t>>> &intermediateAddr)
      : OpConversionPattern(ctx), layout(layout),
        intermediateAddr(intermediateAddr) {}

  LogicalResult
  matchAndRewrite(tosa::RescaleOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(checkRescaleIsSupported(op, rewriter)))
      return failure();

    std::optional<int64_t> inputZp = getScalarConstValue(op.getInputZp());
    std::optional<int64_t> outputZp = getScalarConstValue(op.getOutputZp());
    if (!inputZp || *inputZp != 0 || !outputZp || *outputZp != 0)
      return rewriter.notifyMatchFailure(
          op, "only zero input/output zero-points are supported for now - "
              "zero-point folding into bias/M/shift isn't implemented yet");

    std::optional<int64_t> multiplier = getScalarConstValue(op.getMultiplier());
    std::optional<int64_t> shift = getScalarConstValue(op.getShift());
    if (!multiplier || !shift)
      return rewriter.notifyMatchFailure(
          op, "multiplier/shift must each be a single-element tosa.const "
              "(per-tensor, not per-channel)");
    if (*multiplier < 0 || *multiplier > 0x1FFFF)
      return rewriter.notifyMatchFailure(
          op, "rescale multiplier does not fit ACTIVATE's 17-bit scale_m "
              "field - requantize with sw/tools/train_mnist.py's "
              "quantize_multiplier (mult_bits=17) instead of a wider value");

    std::optional<MatmulChain> chain = matchMatmulChain(op.getInput());
    if (!chain)
      return rewriter.notifyMatchFailure(
          op, "rescale input must be a matmul, or a matmul plus a constant "
              "bias add");

    FailureOr<MatmulOperands> operands =
        matchMatmulOperands(chain->matmul, rewriter);
    if (failed(operands))
      return failure();

    // Bias must be shaped like matmul output. Eventually when batching supported
    // then can think about how to broadcast bias in a loop or something but for
    // now just take it as it is, then if the dimensions don't match throw error
    if (chain->biasConst) {
      auto biasType = cast<RankedTensorType>(chain->biasConst.getType());
      if (biasType.getRank() != 3 || biasType.getShape()[0] != 1 ||
          biasType.getShape()[1] != operands->rows ||
          biasType.getShape()[2] != operands->bType.getShape()[2])
        return rewriter.notifyMatchFailure(
            op, "bias must have the matmul output's shape [1, M, N] - "
                "broadcasting a bias over rows isn't supported");
    }

    std::optional<tosa::ClampOp> reluClamp = matchReluClamp(op);
    // TODO: Leaky relu support
    const macaque_isa::ActFunc actFunc =
        reluClamp ? macaque_isa::ActFunc::Relu
                  : macaque_isa::ActFunc::Passthrough;
    Value logicalOutput = logicalRescaleOutput(op);

    const bool isIntermediate = feedsMatmul(op);

    Location loc = op.getLoc();
    auto outType = cast<RankedTensorType>(op.getOutput().getType());
    const uint32_t elemBytes = outType.getElementTypeBitWidth() / 8;

    // Input tiles are shared across every N-tile, so allocate them once
    bool isRuntimeInput = false;
    SmallVector<SmallVector<uint32_t>> inputAddrs =
        allocateInputAddrs(*operands, layout, intermediateAddr, isRuntimeInput);

    // One Scratch slot per (N-tile, M-chunk)
    SmallVector<SmallVector<uint32_t>> scratchAddrs;
    if (isIntermediate)
      scratchAddrs =
          layout.allocateScratch(operands->numNTiles, operands->rows, elemBytes);

    const int64_t fullK = operands->aType.getShape()[2];
    const uint32_t weightBytes = weightTileBytes(operands->bType);

    // N-tile, then M-chunk, then K-tile innermost: an M-chunk's rows stream
    // through the array once per K-tile, accumulating in place in out_buffer
    // (acc_mode), and the chunk is drained (ACTIVATE + STORE) once after its
    // last K-tile
    for (int64_t n = 0; n < operands->numNTiles; n++) {
      SmallVector<uint32_t> weightAddrs =
          allocateKTileWeightAddrs(*operands, n, layout);

      for (int64_t m = 0; m < operands->numMChunks; m++) {
        const int64_t chunkRows = mChunkRows(operands->rows, m);
        const uint32_t biasAddr =
            allocateBiasAddr(chain->biasConst, n, m, chunkRows, layout);
        // With a single K-tile, the weight the first M-chunk loaded is still
        // in the buffer and still the right one, so later chunks hold it
        // (weight-stationary). Holding also keeps the bias bank, so it only
        // applies when every chunk's bias rows are the same zeros. With
        // several K-tiles the buffer ends on the last K-tile, so every chunk
        // reloads from K-tile 0
        const bool held =
            operands->numKTiles == 1 && m > 0 && !chain->biasConst;

        for (int64_t k = 0; k < operands->numKTiles; k++) {
          if (!held) {
            LoadWeightOp::create(rewriter, loc, TypeRange{}, weightAddrs[k],
                                 static_cast<uint16_t>(weightBytes));
            // The bias seeds the accumulator, so only K-tile 0 needs it
            if (k == 0)
              emitBiasLoad(biasAddr, chunkRows, loc, rewriter);
          }
          const InputLoadSizing sizing =
              computeInputLoadSizing(operands->aType, fullK, k * kTileWidth,
                                     chunkRows, isRuntimeInput);
          LoadInputOp::create(rewriter, loc, TypeRange{}, inputAddrs[m][k],
                              static_cast<uint16_t>(sizing.byteCount),
                              sizing.validBytesPerRow, sizing.inputRows);
          MatmulOp::create(rewriter, loc, TypeRange{}, /*acc_mode=*/k > 0,
                           static_cast<uint16_t>(chunkRows),
                           /*weight_hold=*/held);
        }

        ActivateOp::create(rewriter, loc, TypeRange{},
                           /*act_func=*/static_cast<uint8_t>(actFunc),
                           static_cast<uint32_t>(*multiplier),
                           static_cast<uint8_t>(*shift),
                           static_cast<uint8_t>(chunkRows));

        // Write this chunk's requantized INT8 result back to DDR3: Scratch
        // A/B if a downstream matmul consumes it (intermediate, layer-to-
        // layer), Output region if final.
        const uint32_t outputTileBytes =
            static_cast<uint32_t>(chunkRows * kTileWidth) * elemBytes;
        uint32_t outputAddr;
        if (isIntermediate) {
          outputAddr = scratchAddrs[n][m];
        } else {
          outputAddr = layout.allocateOutput(outputTileBytes);
          layout.recordOutputTile(outputAddr, outputTileBytes);
          layout.setOutputValidBytes(outType.getShape()[2] * elemBytes);
        }
        StoreOp::create(rewriter, loc, TypeRange{}, outputAddr,
                        static_cast<uint16_t>(outputTileBytes));
      }
    }
    if (isIntermediate) {
      intermediateAddr[logicalOutput] = std::move(scratchAddrs);
      SyncOp::create(rewriter, loc, TypeRange{});
    }

    if (reluClamp)
      rewriter.eraseOp(*reluClamp);
    rewriter.eraseOp(op);
    if (chain->biasAdd)
      rewriter.eraseOp(chain->biasAdd);
    rewriter.eraseOp(chain->matmul);
    return success();
  }

private:
  DdrLayout &layout;
  DenseMap<Value, SmallVector<SmallVector<uint32_t>>> &intermediateAddr;
};

} // namespace

namespace macaque::codegen::conversion {

LogicalResult lowerTosaToMacaque(Block &block, CompiledProgramInfo *info) {
  MLIRContext *ctx = block.getParentOp() ? block.getParentOp()->getContext()
                                         : block.front().getContext();
  ConversionTarget target(*ctx);
  target.addLegalDialect<MacaqueDialect>();
  target.addDynamicallyLegalOp<tosa::MatMulOp>(
      [](tosa::MatMulOp op) { return feedsRescale(op); });
  target.addIllegalOp<tosa::RescaleOp>();

  DdrLayout layout(sizeRegions(block));
  DenseMap<Value, SmallVector<SmallVector<uint32_t>>> intermediateAddr;
  RewritePatternSet patterns(ctx);
  patterns.add<RescaleToMacaque>(ctx, layout, intermediateAddr);

  SmallVector<Operation *> ops;
  for (Operation &op : block)
    if (!op.hasTrait<OpTrait::IsTerminator>())
      ops.push_back(&op);

  // Prevent double-erase crashing. RescaleToMacaque run its own cleanup
  // erasing matmul, add and rescale but if the add is zero (bias is all zero),
  // then thefolding mode will also erase it.
  ConversionConfig config;
  config.foldingMode = DialectConversionFoldingMode::Never;

  if (failed(applyPartialConversion(ops, target, std::move(patterns), config)))
    return failure();

  // clean up the unused consts
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (isa<tosa::ConstOp>(op) && op.use_empty())
      op.erase();
  }

  if (info) {
    for (const auto &[addr, bytes] : layout.data())
      info->data.emplace_back(addr, bytes);
    for (const auto &[addr, bytes] : layout.inputTiles())
      info->inputTiles.emplace_back(addr, bytes);
    for (const auto &[addr, bytes] : layout.outputTiles())
      info->outputTiles.emplace_back(addr, bytes);
    info->inputValidBytes = layout.inputValidBytes();
    info->outputValidBytes = layout.outputValidBytes();
  }
  return success();
}

} // namespace macaque::codegen::conversion
