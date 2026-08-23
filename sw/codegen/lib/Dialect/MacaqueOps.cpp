#include "macaque/Dialect/MacaqueOps.h"

#define GET_OP_CLASSES
#include "macaque/Dialect/MacaqueOps.cpp.inc"

using namespace mlir;
using namespace macaque;

LogicalResult LoadWeightOp::verify() {
  if (getDdr3Addr() >= (1u << 28))
    return emitOpError("ddr3_addr must fit in 28 bits [55:28], got ")
           << getDdr3Addr();
  return success();
}
