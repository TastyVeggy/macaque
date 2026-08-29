#pragma once

#include <cstdint>
#include <vector>

#include "macaque/Conversion/TosaToMacaque.hpp"
#include "llvm/ADT/ArrayRef.h"

namespace macaque::codegen::target {

// Serializes a full compiled program into one .macq binary document for the
// runtime to consume
std::vector<uint8_t> emitProgramBinary(llvm::ArrayRef<uint64_t> instructions,
                                       const conversion::CompiledProgramInfo &info);

} // namespace macaque::codegen::target
