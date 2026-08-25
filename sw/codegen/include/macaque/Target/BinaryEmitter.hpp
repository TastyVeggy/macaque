#pragma once

#include <cstdint>
#include <vector>

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Block;
}

namespace macaque::codegen::target {

// Serializes a block's macaque ops, in order, into their 64-bit instruction
// words per the ISA encoding (macaque::common::isa::Instruction::encode()).
// Fails (emitting a diagnostic on the offending op) if the block contains an
// op outside the macaque dialect's seven opcodes.
mlir::FailureOr<std::vector<uint64_t>> emitBinary(mlir::Block &block);

}  // namespace macaque::codegen::target
