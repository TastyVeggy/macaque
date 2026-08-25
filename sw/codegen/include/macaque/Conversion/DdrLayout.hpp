#pragma once

#include <cstdint>

#include "macaque/Conversion/TilingCommon.hpp"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Block;
}

namespace macaque::codegen::conversion {

// Pre-pass: sum each region's total bytes across the whole block before any
// conversion runs, so DdrLayout can compute every region's offset up front.
struct DdrRegionTotals {
  uint32_t weightsBytes = 0;
  uint32_t biasesBytes = 0;
  uint32_t inputBytes = 0;
  uint32_t outputBytes = 0;
  // Scratch A/B are ping-ponged (reused across layers), so they're sized to
  // the largest single intermediate activation.
  uint32_t maxScratchBytes = 0;
  uint32_t maxZeroBiasBytes = 0;
};

// Scans `block` for every macaque-bound tosa.matmul/tosa.rescale, computing
// how many bytes each DDR3 region needs in total
DdrRegionTotals sizeRegions(mlir::Block& block);

// DDR3 address allocator, in accordance with sw/docs/MEMORY_LAYOUT.md's
// region layout 
class DdrLayout {
 public:
  explicit DdrLayout(const DdrRegionTotals& totals) {
    weight_next_ = kWeightBase;
    bias_next_ = detail::alignUp(weight_next_ + totals.weightsBytes);
    zero_bias_base_ = detail::alignUp(bias_next_ + totals.biasesBytes);
    input_next_ = detail::alignUp(zero_bias_base_ + totals.maxZeroBiasBytes);
    scratch_a_base_ = detail::alignUp(input_next_ + totals.inputBytes);
    scratch_b_base_ = detail::alignUp(scratch_a_base_ + totals.maxScratchBytes);
    output_next_ = detail::alignUp(scratch_b_base_ + totals.maxScratchBytes);
  }

  uint32_t allocateWeight(uint32_t bytes) { return bump(weight_next_, bytes); }
  uint32_t allocateBias(uint32_t bytes) { return bump(bias_next_, bytes); }
  uint32_t allocateInput(uint32_t bytes) { return bump(input_next_, bytes); }
  uint32_t allocateOutput(uint32_t bytes) { return bump(output_next_, bytes); }

  // Fixed, shared and not written by the running program itself.
  // The runtime is responsible for staging it as zero, same as any other region.
  uint32_t zeroBiasAddr() const { return zero_bias_base_; }

  // Returns addrs[nTile][mChunk], densely packed within one ping-pong bank
  llvm::SmallVector<llvm::SmallVector<uint32_t>> allocateScratch(
      int64_t numNTiles, int64_t rows, int64_t numFlatChunks, int64_t elemBytes);

 private:
  static constexpr uint32_t kWeightBase = 0x0000'1000;

  static uint32_t bump(uint32_t& cursor, uint32_t bytes) {
    uint32_t addr = cursor;
    cursor = detail::alignUp(cursor + bytes);
    return addr;
  }

  uint32_t weight_next_;
  uint32_t bias_next_;
  uint32_t zero_bias_base_;
  uint32_t input_next_;
  uint32_t scratch_a_base_;
  uint32_t scratch_b_base_;
  uint32_t output_next_;
  bool next_is_a_ = true;
};

}  // namespace macaque::codegen::conversion
