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
  if (!constOp) return std::nullopt;
  auto type = cast<RankedTensorType>(constOp.getType());
  if (type.getNumElements() != 1) return std::nullopt;
  auto elements = cast<DenseElementsAttr>(constOp.getValues());
  return elements.getSplatValue<APInt>().getSExtValue();
}

uint32_t activationTileBytes(RankedTensorType aType, int64_t rows) {
  return static_cast<uint32_t>(rows * kTileWidth) *
         (aType.getElementTypeBitWidth() / 8);
}

// Materializes every element of a rank-3 [1, dim0, dim1] integer
// tosa.const into a flat, row-major value grid, for `extractTileBytes`
// to index into
SmallVector<int64_t> flattenConstValues(tosa::ConstOp constOp) {
  auto elements = cast<DenseElementsAttr>(constOp.getValues());
  SmallVector<int64_t> values;
  values.reserve(elements.getNumElements());
  for (const APInt& v : elements.getValues<APInt>())
    values.push_back(v.getSExtValue());
  return values;
}

// Extracts one (rowTileSize x colTileSize) tile 
// from a flat, row-major (fullRows x fullCols) value grid, as raw
// little-endian bytes. It also handles the zero-padding
SmallVector<uint8_t> extractTileBytes(ArrayRef<int64_t> values, int64_t fullRows,
                                      int64_t fullCols, int64_t rowBase,
                                      int64_t colBase, int64_t rowTileSize,
                                      int64_t colTileSize, int64_t elemBytes) {
  SmallVector<uint8_t> bytes;
  bytes.reserve(static_cast<size_t>(rowTileSize * colTileSize * elemBytes));
  for (int64_t r = 0; r < rowTileSize; r++) {
    for (int64_t c = 0; c < colTileSize; c++) {
      const int64_t row = rowBase + r, col = colBase + c;
      const uint64_t v = (row < fullRows && col < fullCols)
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
  int64_t numFlatChunks;
};

FailureOr<MatmulOperands> matchMatmulOperands(tosa::MatMulOp matmul,
                                              PatternRewriter& rewriter) {
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
        matmul,
        "only zero a_zp/b_zp are supported for now. Zero-point folding "
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
  const int64_t numFlatChunks = numFlatChunksFor(aType.getShape()[1]);

  return MatmulOperands{
      a,         bConst,    aType,        bType, aType.getShape()[1],
      numKTiles, numNTiles, numFlatChunks};
}

// K-tile input addresses are shared across every N-tile but each
// M-chunk gets its own set (different row data), so this returns
// addrs[mChunk][kTile]. For a chained activation, `intermediateAddr` holds
// the producer's own addrs[nTile][mChunk] which is then to be consumed by
// next consumer but transposed
SmallVector<SmallVector<uint32_t>> allocateFlatChunkInputAddrs(
    const MatmulOperands& operands, DdrLayout& layout,
    const DenseMap<Value, SmallVector<SmallVector<uint32_t>>>&
        intermediateAddr) {
  auto chained = intermediateAddr.find(operands.a);
  if (chained != intermediateAddr.end()) {
    const auto& producerAddrs = chained->second;  // [nTile][mChunk]
    SmallVector<SmallVector<uint32_t>> addrs(operands.numFlatChunks);
    // transpose
    for (int64_t m = 0; m < operands.numFlatChunks; m++) {
      addrs[m].reserve(operands.numKTiles);
      for (int64_t k = 0; k < operands.numKTiles; k++)
        addrs[m].push_back(producerAddrs[k][m]);
    }
    return addrs;
  }

  // aConst is pretty much just for testing. Activation input
  // should almost always be supplied by the runtime or result from a previous layer
  auto aConst = operands.a.getDefiningOp<tosa::ConstOp>();
  const SmallVector<int64_t> actValues =
      aConst ? flattenConstValues(aConst) : SmallVector<int64_t>{};
  const int64_t fullK = operands.aType.getShape()[2];
  const int64_t actElemBytes = operands.aType.getElementTypeBitWidth() / 8;

  SmallVector<SmallVector<uint32_t>> addrs;
  addrs.reserve(operands.numFlatChunks);
  for (int64_t m = 0; m < operands.numFlatChunks; ++m) {
    const int64_t chunkRows = flatChunkRows(operands.rows, m);
    const uint32_t tileBytes = activationTileBytes(operands.aType, chunkRows);
    SmallVector<uint32_t> kAddrs;
    kAddrs.reserve(operands.numKTiles);
    for (int64_t k = 0; k < operands.numKTiles; k++) {
      const uint32_t addr = layout.allocateInput(tileBytes);
      if (aConst) {
        layout.recordData(
            addr, extractTileBytes(actValues, operands.rows, fullK,
                                   /*rowBase=*/m * kMaxFlatChunkRows,
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

// Emits the first M-chunk's load_weight+load_input+matmul for a single
// N-tile which is the only chunk in the group that actually loads weight.
void emitFlatFreshChunkMatmul(const MatmulOperands& operands, int64_t nTile,
                              int64_t chunkRows, uint32_t inputAddr,
                              DdrLayout& layout, Location loc,
                              ConversionPatternRewriter& rewriter) {
  const uint32_t weightBytes = weightTileBytes(operands.bType);
  const uint32_t weightAddr = layout.allocateWeight(weightBytes);
  LoadWeightOp::create(rewriter, loc, TypeRange{}, weightAddr,
                       static_cast<uint16_t>(weightBytes));

  // numKTiles == 1 here so there's a single K-tile spanning
  // all of K, padded to 14 rows.
  const SmallVector<int64_t> weightValues = flattenConstValues(operands.bConst);
  const int64_t fullK = operands.bType.getShape()[1];
  const int64_t fullN = operands.bType.getShape()[2];
  const int64_t weightElemBytes = operands.bType.getElementTypeBitWidth() / 8;
  layout.recordData(weightAddr,
                    extractTileBytes(weightValues, fullK, fullN, /*rowBase=*/0,
                                     /*colBase=*/nTile * kTileWidth, kTileWidth,
                                     kTileWidth, weightElemBytes));

  const uint32_t inputBytes = activationTileBytes(operands.aType, chunkRows);
  LoadInputOp::create(rewriter, loc, TypeRange{}, inputAddr,
                      static_cast<uint16_t>(inputBytes));

  MatmulOp::create(rewriter, loc, TypeRange{}, /*acc_mode=*/false,
                   static_cast<uint16_t>(chunkRows));
}

// Emits every M-chunk after the group's first: no load_weight/load_bias.
void emitFlatHeldChunkMatmul(const MatmulOperands& operands, int64_t chunkRows,
                             uint32_t inputAddr, Location loc,
                             ConversionPatternRewriter& rewriter) {
  const uint32_t inputBytes = activationTileBytes(operands.aType, chunkRows);
  LoadInputOp::create(rewriter, loc, TypeRange{}, inputAddr,
                      static_cast<uint16_t>(inputBytes));
  MatmulOp::create(rewriter, loc, TypeRange{}, /*acc_mode=*/false,
                   static_cast<uint16_t>(chunkRows), /*weight_hold=*/true);
}

uint32_t allocateBiasAddr(tosa::ConstOp biasConst, int64_t nTile,
                          DdrLayout& layout) {
  if (!biasConst) {
    const uint32_t addr = layout.zeroBiasAddr();
    layout.recordData(addr, SmallVector<uint8_t>(kBiasTileBytes, 0));
    return addr;
  }
  const uint32_t addr = layout.allocateBias(kBiasTileBytes);
  const SmallVector<int64_t> biasValues = flattenConstValues(biasConst);
  const int64_t fullN = cast<RankedTensorType>(biasConst.getType()).getShape()[2];
  layout.recordData(
      addr, extractTileBytes(biasValues, /*fullRows=*/1, fullN, /*rowBase=*/0,
                             /*colBase=*/nTile * kTileWidth, /*rowTileSize=*/1,
                             kTileWidth, /*elemBytes=*/sizeof(int32_t)));
  return addr;
}

void emitBiasLoad(uint32_t addr, Location loc,
                  ConversionPatternRewriter& rewriter) {
  LoadBiasOp::create(rewriter, loc, TypeRange{}, addr,
                     static_cast<uint16_t>(kBiasTileBytes));
}

void emitBias(tosa::ConstOp biasConst, int64_t nTile, DdrLayout& layout,
              Location loc, ConversionPatternRewriter& rewriter) {
  emitBiasLoad(allocateBiasAddr(biasConst, nTile, layout), loc, rewriter);
}

// Held-batch activation addresses for weight-hold combined with K-tiling:
// addrs[globalChunkIdx][kTile], where chunks are numbered across *all*
// hold-batches. For a chained activation, `intermediateAddr` holds the
// producer's own addrs[nTile][mChunk] and since the producer's flat M-chunks
// and this consumer's hold-batches share identical boundaries, producer
// chunk `b` is this consumer's hold-batch `b`. Each hold-chunk `c` within
// that batch is then just a 14-row byte-offset slice into that one shared
// address
SmallVector<SmallVector<uint32_t>> allocateBatchChunkInputAddrs(
    const MatmulOperands& operands, DdrLayout& layout,
    const DenseMap<Value, SmallVector<SmallVector<uint32_t>>>&
        intermediateAddr) {
  auto chained = intermediateAddr.find(operands.a);
  const int64_t elemBytes = operands.aType.getElementTypeBitWidth() / 8;

  // aConst is pretty much just for testing. Activation input
  // should almost always be supplied by the runtime or result from a previous layer
  auto aConst = operands.a.getDefiningOp<tosa::ConstOp>();
  const SmallVector<int64_t> actValues =
      aConst ? flattenConstValues(aConst) : SmallVector<int64_t>{};
  const int64_t fullK = operands.aType.getShape()[2];

  SmallVector<SmallVector<uint32_t>> addrs;
  const int64_t numBatches = numBatchesFor(operands.rows);
  for (int64_t b = 0; b < numBatches; b++) {
    const int64_t batchRows = rowsPerBatch(operands.rows, b);
    const int64_t numChunks = numTilesFor(batchRows);
    for (int64_t c = 0; c < numChunks; c++) {
      SmallVector<uint32_t> kAddrs;
      kAddrs.reserve(operands.numKTiles);
      if (chained != intermediateAddr.end()) {
        const uint32_t chunkOffsetBytes =
            static_cast<uint32_t>(c * kBatchChunkRows * kTileWidth * elemBytes);
        for (int64_t k = 0; k < operands.numKTiles; k++)
          kAddrs.push_back(chained->second[k][b] + chunkOffsetBytes);
      } else {
        const int64_t chunkRows = batchChunkRows(batchRows, c);
        const uint32_t tileBytes = activationTileBytes(operands.aType, chunkRows);
        for (int64_t k = 0; k < operands.numKTiles; k++) {
          const uint32_t addr = layout.allocateInput(tileBytes);
          if (aConst) {
            layout.recordData(
                addr,
                extractTileBytes(actValues, operands.rows, fullK,
                                 /*rowBase=*/b * kMaxBatchRows + c * kBatchChunkRows,
                                 /*colBase=*/k * kTileWidth, chunkRows,
                                 kTileWidth, elemBytes));
          } else {
            layout.recordInputTile(addr, tileBytes);
            layout.setInputValidBytes(fullK * elemBytes);
          }
          kAddrs.push_back(addr);
        }
      }
      addrs.push_back(std::move(kAddrs));
    }
  }
  return addrs;
}

// One hold-batch's row_base/row-count, for the caller to drain (ACTIVATE +
// STORE) after emitBatchMatmuls returns.
struct BatchChunk {
  int64_t rowBase;  // local to this batch/bank: 0, 14, 28, ...
  int64_t rows;
};

// Emits one hold-batch's matmul groups for a single N-tile: K-tile outer, chunk
// inner
SmallVector<BatchChunk> emitBatchMatmuls(
    const MatmulOperands& operands, int64_t batchRows,
    ArrayRef<uint32_t> weightAddrs, uint32_t biasAddr,
    ArrayRef<SmallVector<uint32_t>> batchChunkAddrs, Location loc,
    ConversionPatternRewriter& rewriter) {
  const int64_t numChunks = numTilesFor(batchRows);
  const uint32_t weightBytes = weightTileBytes(operands.bType);

  SmallVector<BatchChunk> chunks;
  chunks.reserve(numChunks);
  for (int64_t c = 0; c < numChunks; c++)
    chunks.push_back({c * kBatchChunkRows, batchChunkRows(batchRows, c)});

  for (int64_t k = 0; k < operands.numKTiles; k++) {
    for (int64_t c = 0; c < numChunks; c++) {
      const bool held = c > 0;
      if (!held) {
        LoadWeightOp::create(rewriter, loc, TypeRange{}, weightAddrs[k],
                             static_cast<uint16_t>(weightBytes));
        if (k == 0) emitBiasLoad(biasAddr, loc, rewriter);
      }
      const uint32_t inputBytes =
          activationTileBytes(operands.aType, chunks[c].rows);
      LoadInputOp::create(rewriter, loc, TypeRange{}, batchChunkAddrs[c][k],
                          static_cast<uint16_t>(inputBytes));
      MatmulOp::create(rewriter, loc, TypeRange{}, /*acc_mode=*/k > 0,
                       static_cast<uint16_t>(chunks[c].rows),
                       /*weight_hold=*/held,
                       static_cast<uint8_t>(chunks[c].rowBase));
    }
  }
  return chunks;
}

// Weight is identical across every hold-batch (same K-tile, same N-tile
// data regardless of which M rows are being processed) so allocate once per
// K-tile here. Due to the design limitations of the dep tracker in the npu, a
// new batch MUST reload the weight again. But do it via emitBatchMatmuls rather
// than here
SmallVector<uint32_t> allocateBatchKTileWeightAddrs(
    const MatmulOperands& operands, int64_t nTile, DdrLayout& layout) {
  const uint32_t weightBytes = weightTileBytes(operands.bType);
  const SmallVector<int64_t> weightValues = flattenConstValues(operands.bConst);
  const int64_t fullK = operands.bType.getShape()[1];
  const int64_t fullN = operands.bType.getShape()[2];
  const int64_t weightElemBytes = operands.bType.getElementTypeBitWidth() / 8;
  SmallVector<uint32_t> addrs;
  addrs.reserve(operands.numKTiles);
  for (int64_t k = 0; k < operands.numKTiles; k++) {
    const uint32_t addr = layout.allocateWeight(weightBytes);
    layout.recordData(
        addr, extractTileBytes(weightValues, fullK, fullN,
                               /*rowBase=*/k * kTileWidth,
                               /*colBase=*/nTile * kTileWidth, kTileWidth,
                               kTileWidth, weightElemBytes));
    addrs.push_back(addr);
  }
  return addrs;
}

// Pattern looking out for
//  tosa.rescale(+ bias tosa.add) -> load_weight, [load_bias], load_input,
// matmul(acc_mode=0), activate, store.

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
  RescaleToMacaque(
      MLIRContext* ctx, DdrLayout& layout,
      DenseMap<Value, SmallVector<SmallVector<uint32_t>>>& intermediateAddr)
      : OpConversionPattern(ctx),
        layout(layout),
        intermediateAddr(intermediateAddr) {}

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
    if (*multiplier < 0 || *multiplier > 0x1FFFF)
      return rewriter.notifyMatchFailure(
          op, "rescale multiplier does not fit ACTIVATE's 17-bit scale_m "
              "field - requantize with sw/tools/train_mnist.py's "
              "quantize_multiplier (mult_bits=17) instead of a wider value");

    std::optional<MatmulChain> chain = matchMatmulChain(op.getInput());
    if (!chain)
      return rewriter.notifyMatchFailure(
          op,
          "rescale input must be a matmul, or a matmul plus a constant "
          "bias add");

    FailureOr<MatmulOperands> operands =
        matchMatmulOperands(chain->matmul, rewriter);
    if (failed(operands)) return failure();

    std::optional<tosa::ClampOp> reluClamp = matchReluClamp(op);
    // TODO: Leaky relu support
    const macaque_isa::ActFunc actFunc =
        reluClamp ? macaque_isa::ActFunc::Relu : macaque_isa::ActFunc::Passthrough;
    Value logicalOutput = logicalRescaleOutput(op);

    const bool isIntermediate = feedsMatmul(op);

    Location loc = op.getLoc();
    auto outType = cast<RankedTensorType>(op.getOutput().getType());
    const uint32_t elemBytes = outType.getElementTypeBitWidth() / 8;

    if (operands->numKTiles > 1) {
      // Input is shared across every N-tile (activation doesn't depend on
      // which output-channel tile is being computed), so it's allocated
      // once, outside the N-tile loop
      SmallVector<SmallVector<uint32_t>> chunkAddrs =
          allocateBatchChunkInputAddrs(*operands, layout, intermediateAddr);

      // One Scratch slot per (N-tile, hold-batch)
      SmallVector<SmallVector<uint32_t>> scratchAddrs;
      if (isIntermediate)
        scratchAddrs =
            layout.allocateScratch(operands->numNTiles, operands->rows,
                                   numFlatChunksFor(operands->rows), elemBytes);

      for (int64_t n = 0; n < operands->numNTiles; n++) {
        // Weight-hold combined with K-tiling: each hold-batch's chunks are
        // drained (ACTIVATE+STORE) right after that batch's matmul sweep -
        // bank_hold skips out_bank_sel's toggle on every chunk but the
        // batch's last, so all of a batch's chunks read back from the same
        // bank.
        SmallVector<uint32_t> weightAddrs =
            allocateBatchKTileWeightAddrs(*operands, n, layout);
        const uint32_t biasAddr = allocateBiasAddr(chain->biasConst, n, layout);
        const int64_t numBatches = numBatchesFor(operands->rows);
        int64_t chunkOffset = 0;
        for (int64_t b = 0; b < numBatches; b++) {
          const int64_t batchRows = rowsPerBatch(operands->rows, b);
          const int64_t numChunks = numTilesFor(batchRows);
          SmallVector<BatchChunk> chunks = emitBatchMatmuls(
              *operands, batchRows, weightAddrs, biasAddr,
              ArrayRef(chunkAddrs).slice(chunkOffset, numChunks), loc,
              rewriter);
          chunkOffset += numChunks;

          for (int64_t c = 0; c < numChunks; c++) {
            const bool last = c == numChunks - 1;
            ActivateOp::create(
                rewriter, loc, TypeRange{},
                /*act_func=*/static_cast<uint8_t>(actFunc),
                static_cast<uint32_t>(*multiplier),
                static_cast<uint8_t>(*shift),
                static_cast<uint8_t>(chunks[c].rows),
                /*act_row_base=*/static_cast<uint8_t>(chunks[c].rowBase),
                /*act_bank_hold=*/!last);

            const uint32_t outputTileBytes =
                static_cast<uint32_t>(chunks[c].rows * kTileWidth) * elemBytes;
            uint32_t outputAddr;
            if (isIntermediate) {
              outputAddr = scratchAddrs[n][b] +
                          static_cast<uint32_t>(chunks[c].rowBase * kTileWidth) * elemBytes;
            } else {
              outputAddr = layout.allocateOutput(outputTileBytes);
              layout.recordOutputTile(outputAddr, outputTileBytes);
              layout.setOutputValidBytes(outType.getShape()[2] * elemBytes);
            }
            StoreOp::create(rewriter, loc, TypeRange{}, outputAddr,
                            static_cast<uint16_t>(outputTileBytes));
          }
        }
      }
      if (isIntermediate)
        intermediateAddr[logicalOutput] = std::move(scratchAddrs);

      if (reluClamp) rewriter.eraseOp(*reluClamp);
      rewriter.eraseOp(op);
      if (chain->biasAdd) rewriter.eraseOp(chain->biasAdd);
      rewriter.eraseOp(chain->matmul);
      return success();
    }

    // numKTiles == 1 case

    SmallVector<SmallVector<uint32_t>> kTileInputAddrs =
        allocateFlatChunkInputAddrs(*operands, layout, intermediateAddr);

    // Scratch A/B's addresses for this producer, allocated up front
    SmallVector<SmallVector<uint32_t>> scratchAddrs;
    if (isIntermediate)
      scratchAddrs = layout.allocateScratch(operands->numNTiles, operands->rows,
                                            operands->numFlatChunks, elemBytes);

    for (int64_t n = 0; n < operands->numNTiles; n++) {
      for (int64_t m = 0; m < operands->numFlatChunks; m++) {
        const int64_t chunkRows = flatChunkRows(operands->rows, m);
        // Weight-hold means every M-chunk but the first reuses the
        // weight/bias bank the first one loaded
        if (m == 0) {
          // Bias must be loaded before each (N-tile, M-chunk)'s first
          // matmul runs.
          emitBias(chain->biasConst, n, layout, loc, rewriter);
          emitFlatFreshChunkMatmul(*operands, n, chunkRows, kTileInputAddrs[m][0],
                                   layout, loc, rewriter);
        } else {
          emitFlatHeldChunkMatmul(*operands, chunkRows, kTileInputAddrs[m][0],
                                  loc, rewriter);
        }

        ActivateOp::create(
            rewriter, loc, TypeRange{},
            /*act_func=*/static_cast<uint8_t>(actFunc),
            static_cast<uint32_t>(*multiplier), static_cast<uint8_t>(*shift),
            static_cast<uint8_t>(chunkRows));

        // One full 14-wide output tile per (N-tile, M-chunk)
        const uint32_t outputTileBytes =
            static_cast<uint32_t>(chunkRows * kTileWidth) * elemBytes;

        // Write this chunk's requantized INT8 result back to DDR3: Scratch
        // A/B if a downstream matmul consumes it (intermediate, layer-to-
        // layer), Output region if final.
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
    if (isIntermediate)
      intermediateAddr[logicalOutput] = std::move(scratchAddrs);

    if (reluClamp) rewriter.eraseOp(*reluClamp);
    rewriter.eraseOp(op);
    if (chain->biasAdd) rewriter.eraseOp(chain->biasAdd);
    rewriter.eraseOp(chain->matmul);
    return success();
  }

 private:
  DdrLayout& layout;
  DenseMap<Value, SmallVector<SmallVector<uint32_t>>>& intermediateAddr;
};

}  // namespace

namespace macaque::codegen::conversion {

LogicalResult lowerTosaToMacaque(Block& block, CompiledProgramInfo* info) {
  MLIRContext* ctx = block.getParentOp() ? block.getParentOp()->getContext()
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

  SmallVector<Operation*> ops;
  for (Operation& op : block)
    if (!op.hasTrait<OpTrait::IsTerminator>()) ops.push_back(&op);

  // Prevent double-erase crashing. RescaleToMacaque run its own cleanup
  // erasing matmul, add and rescale but if the add is zero (bias is all zero), then 
  // thefolding mode will also erase it.
  ConversionConfig config;
  config.foldingMode = DialectConversionFoldingMode::Never;

  if (failed(applyPartialConversion(ops, target, std::move(patterns), config)))
    return failure();

  // clean up the unused consts
  for (Operation& op : llvm::make_early_inc_range(block)) {
    if (isa<tosa::ConstOp>(op) && op.use_empty()) op.erase();
  }

  if (info) {
    for (const auto& [addr, bytes] : layout.data()) info->data.emplace_back(addr, bytes);
    for (const auto& [addr, bytes] : layout.inputTiles())
      info->inputTiles.emplace_back(addr, bytes);
    for (const auto& [addr, bytes] : layout.outputTiles())
      info->outputTiles.emplace_back(addr, bytes);
    info->inputValidBytes = layout.inputValidBytes();
    info->outputValidBytes = layout.outputValidBytes();
  }
  return success();
}

}  // namespace macaque::codegen::conversion
