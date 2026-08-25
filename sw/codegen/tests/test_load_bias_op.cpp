#include <gtest/gtest.h>

#include "macaque/Dialect/MacaqueOps.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;
using namespace mlir::macaque;

TEST(LoadBiasOp, MaxValidAddressVerifies) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  auto op =
      LoadBiasOp::create(builder, loc, TypeRange{}, (1u << 28) - 1, 200);
  EXPECT_TRUE(succeeded(verify(op)));
}

TEST(LoadBiasOp, OneOverMaxAddressFailsVerification) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  auto op = LoadBiasOp::create(builder, loc, TypeRange{}, (1u << 28), 200);
  EXPECT_TRUE(failed(verify(op)));
}
