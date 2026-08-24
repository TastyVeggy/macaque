#include <gtest/gtest.h>

#include "macaque/Dialect/MacaqueOps.h"
#include "macaque/Target/BinaryEmitter.h"
#include "macaque/common/isa.hpp"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace macaque;
namespace target = ::macaque::codegen::target;
namespace isa = ::macaque::common::isa;

namespace {

struct BinaryEmitterTest : public ::testing::Test {
  MLIRContext context;
  std::unique_ptr<OpBuilder> builder;
  Location loc = UnknownLoc::get(&context);
  Block block;

  void SetUp() override {
    context.getOrLoadDialect<MacaqueDialect>();
    builder = std::make_unique<OpBuilder>(&context);
    builder->setInsertionPointToStart(&block);
  }
};

}  // namespace

TEST_F(BinaryEmitterTest, LoadWeightMatchesHandBuiltInstruction) {
  LoadWeightOp::create(*builder, loc, TypeRange{}, 0x123'4567u, 200);

  auto words = target::emitBinary(block);
  ASSERT_TRUE(succeeded(words));
  ASSERT_EQ(words->size(), 1u);

  isa::Instruction expected{isa::Opcode::LoadWeight, false, 0, 0x123'4567u,
                             200, 0};
  EXPECT_EQ((*words)[0], expected.encode());
}

TEST_F(BinaryEmitterTest, MatmulMatchesHandBuiltInstruction) {
  MatmulOp::create(*builder, loc, TypeRange{}, /*acc_mode=*/true,
                    /*tile_params=*/14);

  auto words = target::emitBinary(block);
  ASSERT_TRUE(succeeded(words));
  ASSERT_EQ(words->size(), 1u);

  isa::Instruction expected{isa::Opcode::Matmul, true, 0, 0, 0, 14};
  EXPECT_EQ((*words)[0], expected.encode());
}

TEST_F(BinaryEmitterTest, ActivateMatchesHandBuiltInstruction) {
  ActivateOp::create(*builder, loc, TypeRange{}, /*act_func=*/1,
                      /*act_scale_m=*/0xABCDEFu, /*act_scale_shift=*/9,
                      /*act_num_rows=*/64);

  auto words = target::emitBinary(block);
  ASSERT_TRUE(succeeded(words));
  ASSERT_EQ(words->size(), 1u);

  isa::Instruction expected{isa::Opcode::Activate, false, 1, 0xABCDEFu, 9,
                             64};
  EXPECT_EQ((*words)[0], expected.encode());
}

TEST_F(BinaryEmitterTest, SyncMatchesHandBuiltInstruction) {
  SyncOp::create(*builder, loc, TypeRange{});

  auto words = target::emitBinary(block);
  ASSERT_TRUE(succeeded(words));
  ASSERT_EQ(words->size(), 1u);

  isa::Instruction expected{isa::Opcode::Sync, false, 0, 0, 0, 0};
  EXPECT_EQ((*words)[0], expected.encode());
}

TEST_F(BinaryEmitterTest, EmitsWordsInProgramOrder) {
  LoadWeightOp::create(*builder, loc, TypeRange{}, 0x10u, 4);
  LoadInputOp::create(*builder, loc, TypeRange{}, 0x20u, 8);
  MatmulOp::create(*builder, loc, TypeRange{}, /*acc_mode=*/false,
                    /*tile_params=*/4);
  StoreOp::create(*builder, loc, TypeRange{}, 0x30u, 4);

  auto words = target::emitBinary(block);
  ASSERT_TRUE(succeeded(words));
  ASSERT_EQ(words->size(), 4u);

  EXPECT_EQ((isa::Instruction::decode((*words)[0])).opcode,
            isa::Opcode::LoadWeight);
  EXPECT_EQ((isa::Instruction::decode((*words)[1])).opcode,
            isa::Opcode::LoadInput);
  EXPECT_EQ((isa::Instruction::decode((*words)[2])).opcode,
            isa::Opcode::Matmul);
  EXPECT_EQ((isa::Instruction::decode((*words)[3])).opcode,
            isa::Opcode::Store);
}

TEST_F(BinaryEmitterTest, UnsupportedOpFailsEmission) {
  UnrealizedConversionCastOp::create(*builder, loc, TypeRange{},
                                      ValueRange{});

  auto words = target::emitBinary(block);
  EXPECT_TRUE(failed(words));
}
