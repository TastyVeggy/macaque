#include <gtest/gtest.h>

#include "macaque/Dialect/MacaqueOps.h"
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
                              /*tile_params=*/(1u << 12) - 1);
  EXPECT_TRUE(succeeded(verify(op)));
}

TEST(MatmulOp, OneOverMaxTileParamsFailsVerification) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  auto op = MatmulOp::create(builder, loc, TypeRange{}, /*acc_mode=*/false,
                              /*tile_params=*/(1u << 12));
  EXPECT_TRUE(failed(verify(op)));
}
