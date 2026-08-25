#include <gtest/gtest.h>

#include "macaque/Dialect/MacaqueOps.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;
using namespace mlir::macaque;

TEST(StoreOp, MaxValidAddressVerifies) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  auto op = StoreOp::create(builder, loc, TypeRange{}, (1u << 28) - 1, 200);
  EXPECT_TRUE(succeeded(verify(op)));
}

TEST(StoreOp, OneOverMaxAddressFailsVerification) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  auto op = StoreOp::create(builder, loc, TypeRange{}, (1u << 28), 200);
  EXPECT_TRUE(failed(verify(op)));
}
