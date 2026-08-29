#include "macaque/Dialect/MacaqueOps.hpp"

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
  if (getValidBytesPerRow() >= 16)
    return emitOpError(
               "valid_bytes_per_row must fit reserved's 4 bits [11:8], got ")
           << getValidBytesPerRow();
  if (getValidBytesPerRow() != 0 && getInputRows() == 0)
    return emitOpError(
        "input_rows must be nonzero when valid_bytes_per_row is set");
  return success();
}

LogicalResult MatmulOp::verify() {
  if (getTileParams() >= (1u << 8))
    return emitOpError("tile_params (row count) must fit in 8 bits [7:0], got ")
           << getTileParams();
  if (getMatRowBase() + getTileParams() > 256)
    return emitOpError(
               "mat_row_base + tile_params must fit the 256-row out_buffer, "
               "got mat_row_base=")
           << getMatRowBase() << " tile_params=" << getTileParams();
  return success();
}

LogicalResult ActivateOp::verify() {
  if (getActFunc() > 2)
    return emitOpError("act_func must be 0 (ReLU), 1 (leaky-ReLU), or 2 "
                       "(passthrough), got ")
           << static_cast<unsigned>(getActFunc());
  if (getActScaleM() >= (1u << 17))
    return emitOpError("act_scale_m must fit in 17 bits as it only occupies "
                       "the top 17 of "
                       "ddr3_addr's 28 bits")
           << getActScaleM();
  if (getActScaleShift() >= 32)
    return emitOpError("act_scale_shift must fit in 5 bits [0, 32), got ")
           << static_cast<unsigned>(getActScaleShift());
  if (getActRowBase() + getActNumRows() > 256)
    return emitOpError("act_row_base + act_num_rows must fit the 256-row "
                       "out_buffer, got act_row_base=")
           << static_cast<unsigned>(getActRowBase())
           << " act_num_rows=" << static_cast<unsigned>(getActNumRows());
  return success();
}

LogicalResult StoreOp::verify() {
  if (getDdr3Addr() >= (1u << 28))
    return emitOpError("ddr3_addr must fit in 28 bits [55:28], got ")
           << getDdr3Addr();
  return success();
}
