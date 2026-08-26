#include <gtest/gtest.h>

#include "macaque/Dialect/MacaqueOps.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;
using namespace mlir::macaque;

TEST(MatmulOp, MaxValidTileParamsVerifies) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  auto op = MatmulOp::create(builder, loc, TypeRange{}, /*acc_mode=*/true,
                             /*tile_params=*/(1u << 8) - 1);
  EXPECT_TRUE(succeeded(verify(op)));
}

TEST(MatmulOp, OneOverMaxTileParamsFailsVerification) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  auto op = MatmulOp::create(builder, loc, TypeRange{}, /*acc_mode=*/false,
                             /*tile_params=*/(1u << 8));
  EXPECT_TRUE(failed(verify(op)));
}

TEST(MatmulOp, WeightHoldDefaultsToFalse) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  // weight_hold is DefaultValuedAttr<UI1Attr, "false"> - existing callers
  // that predate the attribute (and every non-M-streaming matmul) must keep
  // building without setting it.
  auto op = MatmulOp::create(builder, loc, TypeRange{}, /*acc_mode=*/false,
                             /*tile_params=*/14);
  EXPECT_TRUE(succeeded(verify(op)));
  EXPECT_EQ(op.getWeightHold(), false);
}

TEST(MatmulOp, WeightHoldRoundTripsTrue) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  auto op = MatmulOp::create(builder, loc, TypeRange{}, /*acc_mode=*/false,
                             /*tile_params=*/14, /*weight_hold=*/true);
  EXPECT_TRUE(succeeded(verify(op)));
  EXPECT_EQ(op.getWeightHold(), true);
}
