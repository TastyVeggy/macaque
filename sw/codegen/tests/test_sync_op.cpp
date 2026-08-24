#include <gtest/gtest.h>

#include "macaque/Dialect/MacaqueOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;
using namespace macaque;

TEST(SyncOp, Verifies) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  auto op = SyncOp::create(builder, loc, TypeRange{});
  EXPECT_TRUE(succeeded(verify(op)));
}
