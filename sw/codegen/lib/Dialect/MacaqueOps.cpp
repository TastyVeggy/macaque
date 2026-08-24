#include "macaque/Dialect/MacaqueOps.h"

#define GET_OP_CLASSES
#include "macaque/Dialect/MacaqueOps.cpp.inc"

using namespace mlir;
using namespace mlir::macaque;

LogicalResult LoadWeightOp::verify() {
  if (getDdr3Addr() >= (1u << 28))
    return emitOpError("ddr3_addr must fit in 28 bits [55:28], got ")
           << getDdr3Addr();
  return success();
}

LogicalResult LoadBiasOp::verify() {
  if (getDdr3Addr() >= (1u << 28))
    return emitOpError("ddr3_addr must fit in 28 bits [55:28], got ")
           << getDdr3Addr();
  return success();
}

LogicalResult LoadInputOp::verify() {
  if (getDdr3Addr() >= (1u << 28))
    return emitOpError("ddr3_addr must fit in 28 bits [55:28], got ")
           << getDdr3Addr();
  return success();
}

LogicalResult MatmulOp::verify() {
  if (getTileParams() >= (1u << 12))
    return emitOpError("tile_params must fit in 12 bits [11:0], got ")
           << getTileParams();
  return success();
}

LogicalResult ActivateOp::verify() {
  if (getActFunc() > 2)
    return emitOpError(
               "act_func must be 0 (ReLU), 1 (leaky-ReLU), or 2 "
               "(passthrough), got ")
           << static_cast<unsigned>(getActFunc());
  if (getActScaleM() >= (1u << 28))
    return emitOpError("act_scale_m must fit in 28 bits, got ")
           << getActScaleM();
  if (getActScaleShift() >= 32)
    return emitOpError("act_scale_shift must fit in 5 bits [0, 32), got ")
           << static_cast<unsigned>(getActScaleShift());
  return success();
}

LogicalResult StoreOp::verify() {
  if (getDdr3Addr() >= (1u << 28))
    return emitOpError("ddr3_addr must fit in 28 bits [55:28], got ")
           << getDdr3Addr();
  return success();
}
