#pragma once

#include <cstdint>
#include <utility>

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Block;
}

namespace macaque::codegen::conversion {

// One compile-time-known DDR3 region's contents: for zero-padding to make
// multiple of 14 Covers weights, bias, and any const activation, which the
// caller/runtime is responsible for staging itself.
using DataSegment =
    llvm::SmallVector<std::pair<uint32_t, llvm::SmallVector<uint8_t>>>;

// One DDR3 tile a caller must fill (a runtime input) or may read back (the
// final output). Currently, it is super slow because only
// can stream one image at a time
// TODO: allow multi-row IO
using IoSegment = llvm::SmallVector<std::pair<uint32_t, uint32_t>>;

struct CompiledProgramInfo {
  DataSegment data;
  IoSegment inputTiles;
  IoSegment outputTiles;
  int64_t inputValidBytes = 0;
  int64_t outputValidBytes = 0;
};

mlir::LogicalResult lowerTosaToMacaque(mlir::Block &block,
                                       CompiledProgramInfo *info = nullptr);

} // namespace macaque::codegen::conversion
