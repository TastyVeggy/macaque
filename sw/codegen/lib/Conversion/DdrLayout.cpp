#include "macaque/Conversion/DdrLayout.hpp"

#include <algorithm>
#include <optional>

#include "mlir/IR/Block.h"

using namespace mlir;
using namespace macaque::codegen::conversion::detail;

namespace macaque::codegen::conversion {

namespace {

int64_t numNTilesFor(tosa::MatMulOp matmul) {
  auto bConst = matmul.getB().getDefiningOp<tosa::ConstOp>();
  if (!bConst)
    return 1;
  auto bType = cast<RankedTensorType>(bConst.getType());
  if (bType.getRank() != 3)
    return 1;
  return numTilesFor(bType.getShape()[2]);
}

int64_t numKTilesFor(tosa::MatMulOp matmul) {
  auto bConst = matmul.getB().getDefiningOp<tosa::ConstOp>();
  if (!bConst)
    return 1;
  auto bType = cast<RankedTensorType>(bConst.getType());
  if (bType.getRank() != 3)
    return 1;
  return numTilesFor(bType.getShape()[1]);
}

// Total DDR3 bytes needed for `numChunks` M-chunks (each up to
// kMaxFlatChunkRows rows, the last one whatever remains) of a `fullRows`-row,
// 14-column, `elemBytes`-per-element tensor. This is the sum of each
// individual chunk's own aligned size, not one aligned block for the whole
// tensor - every chunk lands at its own independently 8-byte-aligned DDR3
// address (same "every tile gets its own slot" rule as K/N-tiling - see
// MEMORY_LAYOUT.md), so the total can come out slightly larger than
// alignUp(fullRows * 14 * elemBytes) in one shot.
uint32_t flatChunkedRowBytes(int64_t fullRows, int64_t numChunks,
                             int64_t elemBytes) {
  uint32_t total = 0;
  for (int64_t m = 0; m < numChunks; m++) {
    const int64_t rows = numChunks > 1 ? flatChunkRows(fullRows, m) : fullRows;
    total += alignUp(static_cast<uint32_t>(rows * kTileWidth) *
                     static_cast<uint32_t>(elemBytes));
  }
  return total;
}

// Same as flatChunkedRowBytes, but for weight-hold+K-tiling's 14-row
// hold-batch chunks (kBatchChunkRows, batched by kMaxChunksPerBatch).
uint32_t batchChunkedRowBytes(int64_t fullRows, int64_t elemBytes) {
  uint32_t total = 0;
  const int64_t numBatches = numBatchesFor(fullRows);
  for (int64_t b = 0; b < numBatches; b++) {
    const int64_t batchRows = rowsPerBatch(fullRows, b);
    const int64_t numChunks = numTilesFor(batchRows);
    for (int64_t c = 0; c < numChunks; c++) {
      const int64_t rows = batchChunkRows(batchRows, c);
      total += alignUp(static_cast<uint32_t>(rows * kTileWidth) *
                       static_cast<uint32_t>(elemBytes));
    }
  }
  return total;
}

} // namespace

DdrRegionTotals sizeRegions(Block &block) {
  DdrRegionTotals totals;
  for (Operation &op : block) {
    if (auto matmul = dyn_cast<tosa::MatMulOp>(op)) {
      if (auto bConst = matmul.getB().getDefiningOp<tosa::ConstOp>()) {
        auto bType = cast<RankedTensorType>(bConst.getType());
        if (bType.getRank() == 3) {
          const int64_t numK = numTilesFor(bType.getShape()[1]);
          const int64_t numN = numTilesFor(bType.getShape()[2]);
          totals.weightsBytes += static_cast<uint32_t>(numK * numN) *
                                 alignUp(weightTileBytes(bType));

          if (auto aType = dyn_cast<RankedTensorType>(matmul.getA().getType());
              !matmul.getA().getDefiningOp<tosa::RescaleOp>() && aType &&
              aType.getRank() == 3) {
            const int64_t rows = aType.getShape()[1];
            const int64_t elemBytes = aType.getElementTypeBitWidth() / 8;
            const uint32_t perKTileBytes =
                numK == 1 ? flatChunkedRowBytes(rows, numFlatChunksFor(rows),
                                                elemBytes)
                          : batchChunkedRowBytes(rows, elemBytes);
            totals.inputBytes += static_cast<uint32_t>(numK) * perKTileBytes;
          }
          // A bare (non-rescaled) matmul never has a bias so it always
          // needs the zero-bias slot
          if (!feedsRescale(matmul))
            totals.maxZeroBiasBytes =
                std::max(totals.maxZeroBiasBytes, kBiasTileBytes);
        }
      }
    } else if (auto rescale = dyn_cast<tosa::RescaleOp>(op)) {
      auto outType = cast<RankedTensorType>(rescale.getOutput().getType());

      std::optional<MatmulChain> chain = matchMatmulChain(rescale.getInput());
      const int64_t numN = chain ? numNTilesFor(chain->matmul) : 1;
      const int64_t numK = chain ? numKTilesFor(chain->matmul) : 1;

      if (feedsMatmul(rescale)) {
        // Intermediate (chained) outputs get one Scratch slot per (N-tile,
        // M-chunk) and always the flat mChunkedRowBytes shape, regardless of
        // numK: a held-batch (numK>1) producer's Scratch output is one
        // address per hold-batch now. Scratch A/B is sized to
        // the *largest single* producer's full tiled shape
        uint32_t producerBytes;
        if (outType.getRank() == 3) {
          const int64_t rows = outType.getShape()[1];
          const int64_t elemBytes = outType.getElementTypeBitWidth() / 8;
          producerBytes =
              static_cast<uint32_t>(numN) *
              flatChunkedRowBytes(rows, numFlatChunksFor(rows), elemBytes);
        } else {
          producerBytes = alignUp(byteSizeOf(outType));
        }
        totals.maxScratchBytes =
            std::max(totals.maxScratchBytes, producerBytes);
      } else {
        // Each (N-tile, M-chunk) pair gets its own store.
        uint32_t perNTileBytes;
        if (outType.getRank() == 3) {
          const int64_t rows = outType.getShape()[1];
          const int64_t elemBytes = outType.getElementTypeBitWidth() / 8;
          perNTileBytes =
              numK == 1
                  ? flatChunkedRowBytes(rows, numFlatChunksFor(rows), elemBytes)
                  : batchChunkedRowBytes(rows, elemBytes);
        } else {
          perNTileBytes = alignUp(byteSizeOf(outType));
        }
        totals.outputBytes += static_cast<uint32_t>(numN) * perNTileBytes;
      }

      if (chain) {
        if (chain->biasConst) {
          totals.biasesBytes +=
              static_cast<uint32_t>(numN) * alignUp(kBiasTileBytes);
        } else if (chain->matmul.getB().getDefiningOp<tosa::ConstOp>()) {
          totals.maxZeroBiasBytes =
              std::max(totals.maxZeroBiasBytes, kBiasTileBytes);
        }
      }
    }
  }
  return totals;
}

SmallVector<SmallVector<uint32_t>>
DdrLayout::allocateScratch(int64_t numNTiles, int64_t rows,
                           int64_t numFlatChunks, int64_t elemBytes) {
  uint32_t cursor = next_is_a_ ? scratch_a_base_ : scratch_b_base_;
  next_is_a_ = !next_is_a_;
  SmallVector<SmallVector<uint32_t>> addrs(numNTiles);
  for (int64_t n = 0; n < numNTiles; ++n) {
    addrs[n].reserve(numFlatChunks);
    for (int64_t m = 0; m < numFlatChunks; ++m) {
      addrs[n].push_back(cursor);
      const int64_t chunkRows =
          numFlatChunks > 1 ? flatChunkRows(rows, m) : rows;
      cursor = alignUp(cursor + static_cast<uint32_t>(chunkRows * kTileWidth) *
                                    static_cast<uint32_t>(elemBytes));
    }
  }
  return addrs;
}

} // namespace macaque::codegen::conversion
