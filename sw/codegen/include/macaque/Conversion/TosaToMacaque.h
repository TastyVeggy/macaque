#pragma once

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Block;
}

namespace macaque::codegen::conversion {

mlir::LogicalResult lowerTosaToMacaque(mlir::Block& block);

}  // namespace macaque::codegen::conversion
