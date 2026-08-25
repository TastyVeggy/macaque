#include "macaque/Dialect/MacaqueDialect.hpp"

#include "macaque/Dialect/MacaqueOps.hpp"
#include "macaque/Dialect/MacaqueOpsDialect.cpp.inc"

void mlir::macaque::MacaqueDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "macaque/Dialect/MacaqueOps.cpp.inc"
      >();
}
