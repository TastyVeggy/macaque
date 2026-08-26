#pragma once

#include <cstdint>
#include <string>

#include "macaque/Conversion/TosaToMacaque.hpp"
#include "llvm/ADT/ArrayRef.h"

namespace macaque::codegen::target {

// Serializes a full compiled program into one JSON document
// for the runtime to consume
std::string emitProgramJson(llvm::ArrayRef<uint64_t> instructions,
                            const conversion::CompiledProgramInfo &info);

} // namespace macaque::codegen::target
