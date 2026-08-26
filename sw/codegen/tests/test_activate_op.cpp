#include <gtest/gtest.h>

#include "macaque/Dialect/MacaqueOps.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;
using namespace mlir::macaque;

TEST(ActivateOp, MaxValidFieldsVerify) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  auto op = ActivateOp::create(builder, loc, TypeRange{}, /*act_func=*/2,
                               /*act_scale_m=*/(1u << 17) - 1,
                               /*act_scale_shift=*/31, /*act_num_rows=*/255);
  EXPECT_TRUE(succeeded(verify(op)));
}

TEST(ActivateOp, ActFuncAboveTwoFailsVerification) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  auto op = ActivateOp::create(builder, loc, TypeRange{}, /*act_func=*/3,
                               /*act_scale_m=*/0, /*act_scale_shift=*/0,
                               /*act_num_rows=*/0);
  EXPECT_TRUE(failed(verify(op)));
}

TEST(ActivateOp, ActScaleMOneOverMaxFailsVerification) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  auto op = ActivateOp::create(builder, loc, TypeRange{}, /*act_func=*/0,
                               /*act_scale_m=*/(1u << 17),
                               /*act_scale_shift=*/0, /*act_num_rows=*/0);
  EXPECT_TRUE(failed(verify(op)));
}

TEST(ActivateOp, ActScaleShiftOneOverMaxFailsVerification) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  auto op = ActivateOp::create(builder, loc, TypeRange{}, /*act_func=*/0,
                               /*act_scale_m=*/0, /*act_scale_shift=*/32,
                               /*act_num_rows=*/0);
  EXPECT_TRUE(failed(verify(op)));
}
