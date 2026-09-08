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

// Total DDR3 bytes for one 14-column slice of a `fullRows`-row,
// `elemBytes`-per-element tensor, split into M-chunks (each up to
// kMaxMChunkRows rows). This is the sum of each chunk's own aligned size, not
// one aligned block for the whole tensor, because each chunk is its own tile.
uint32_t mChunkedRowBytes(int64_t fullRows, int64_t elemBytes) {
  uint32_t total = 0;
  for (int64_t m = 0; m < numMChunksFor(fullRows); m++)
    total += alignUp(static_cast<uint32_t>(mChunkRows(fullRows, m) * kTileWidth) *
                     static_cast<uint32_t>(elemBytes));
  return total;
}

// Bytes of the shared zero-bias tile a bias-less matmul with `rows` rows
// needs: one 14-lane INT32 row per row of its largest M-chunk.
uint32_t zeroBiasTileBytes(int64_t rows) {
  return alignUp(static_cast<uint32_t>(std::min(rows, kMaxMChunkRows)) *
                 kBiasTileBytes);
}

int64_t rowsOf(RankedTensorType type) {
  return type.getRank() == 3 ? type.getShape()[1] : 1;
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
            totals.inputBytes += static_cast<uint32_t>(numK) *
                                 mChunkedRowBytes(rows, elemBytes);
          }
          // A bare (non-rescaled) matmul never has a bias so it always
          // needs the zero-bias slot
          if (!feedsRescale(matmul))
            totals.maxZeroBiasBytes = std::max(
                totals.maxZeroBiasBytes,
                zeroBiasTileBytes(
                    rowsOf(cast<RankedTensorType>(matmul.getType()))));
        }
      }
    } else if (auto rescale = dyn_cast<tosa::RescaleOp>(op)) {
      auto outType = cast<RankedTensorType>(rescale.getOutput().getType());

      std::optional<MatmulChain> chain = matchMatmulChain(rescale.getInput());
      const int64_t numN = chain ? numNTilesFor(chain->matmul) : 1;

      if (feedsMatmul(rescale)) {
        // Intermediate (chained) outputs get one Scratch slot per (N-tile,
        // M-chunk). Scratch A/B is sized to the *largest single* producer's
        // full tiled shape
        uint32_t producerBytes;
        if (outType.getRank() == 3) {
          const int64_t rows = outType.getShape()[1];
          const int64_t elemBytes = outType.getElementTypeBitWidth() / 8;
          producerBytes =
              static_cast<uint32_t>(numN) * mChunkedRowBytes(rows, elemBytes);
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
          perNTileBytes = mChunkedRowBytes(rows, elemBytes);
        } else {
          perNTileBytes = alignUp(byteSizeOf(outType));
        }
        totals.outputBytes += static_cast<uint32_t>(numN) * perNTileBytes;
      }

      // One bias tile per (N-tile, M-chunk), one row per output row
      if (chain) {
        const int64_t rows = rowsOf(outType);
        if (chain->biasConst) {
          totals.biasesBytes +=
              static_cast<uint32_t>(numN) *
              mChunkedRowBytes(rows, static_cast<int64_t>(sizeof(int32_t)));
        } else if (chain->matmul.getB().getDefiningOp<tosa::ConstOp>()) {
          totals.maxZeroBiasBytes =
              std::max(totals.maxZeroBiasBytes, zeroBiasTileBytes(rows));
        }
      }
    }
  }
  return totals;
}

SmallVector<SmallVector<uint32_t>>
DdrLayout::allocateScratch(int64_t numNTiles, int64_t rows,
                           int64_t elemBytes) {
  uint32_t cursor = next_is_a_ ? scratch_a_base_ : scratch_b_base_;
  next_is_a_ = !next_is_a_;
  const int64_t numMChunks = numMChunksFor(rows);
  SmallVector<SmallVector<uint32_t>> addrs(numNTiles);
  for (int64_t n = 0; n < numNTiles; ++n) {
    addrs[n].reserve(numMChunks);
    for (int64_t m = 0; m < numMChunks; ++m) {
      addrs[n].push_back(cursor);
      cursor = alignUp(cursor + static_cast<uint32_t>(mChunkRows(rows, m) *
                                                      kTileWidth) *
                                    static_cast<uint32_t>(elemBytes));
    }
  }
  return addrs;
}

} // namespace macaque::codegen::conversion
