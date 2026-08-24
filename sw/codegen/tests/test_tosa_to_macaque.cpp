#include <gtest/gtest.h>

#include "macaque/Conversion/TosaToMacaque.h"
#include "macaque/Dialect/MacaqueOps.h"
#include "macaque/common/isa.hpp"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace mlir::macaque;
using namespace ::macaque::codegen::conversion;
namespace isa = ::macaque::common::isa;

namespace {

// Builds a single tosa.matmul(a, b) with both operands tosa.const:
//   a: [batch, rows, 14]  (activation)   b: [batch, 14, 14]  (weight)
// into the given block, and returns the matmul op.
tosa::MatMulOp buildConstMatmul(OpBuilder& builder, Location loc, int rows,
                                int batch = 1, int64_t aZpValue = 0) {
  auto i8Ty = builder.getIntegerType(8);
  auto i32Ty = builder.getIntegerType(32);
  auto zpTy = RankedTensorType::get({1}, i8Ty);
  auto aTy = RankedTensorType::get({batch, rows, 14}, i8Ty);
  auto bTy = RankedTensorType::get({batch, 14, 14}, i8Ty);
  auto outTy = RankedTensorType::get({batch, rows, 14}, i32Ty);

  auto a = tosa::ConstOp::create(
      builder, loc, aTy, DenseElementsAttr::get(aTy, static_cast<int8_t>(1)));
  auto b = tosa::ConstOp::create(
      builder, loc, bTy, DenseElementsAttr::get(bTy, static_cast<int8_t>(2)));
  auto aZp = tosa::ConstOp::create(
      builder, loc, zpTy,
      DenseElementsAttr::get(zpTy, static_cast<int8_t>(aZpValue)));
  auto bZp = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));

  return tosa::MatMulOp::create(builder, loc, outTy, a, b, aZp, bZp);
}

tosa::ConstOp buildScalarConst(OpBuilder& builder, Location loc, Type elemTy,
                               int64_t value) {
  auto ty = RankedTensorType::get({1}, elemTy);
  return tosa::ConstOp::create(
      builder, loc, ty,
      DenseElementsAttr::get(ty, builder.getIntegerAttr(elemTy, value)));
}

tosa::RescaleOp buildRescale(OpBuilder& builder, Location loc, Value input,
                             Type outElemTy, int64_t multiplier, int64_t shift,
                             bool perChannel = false, bool inputUnsigned = false,
                             bool outputUnsigned = false) {
  auto i8Ty = builder.getIntegerType(8);
  auto i32Ty = builder.getIntegerType(32);
  auto inShape = cast<RankedTensorType>(input.getType()).getShape();
  auto outTy = RankedTensorType::get(inShape, outElemTy);

  auto multiplierConst = buildScalarConst(builder, loc, i32Ty, multiplier);
  auto shiftConst = buildScalarConst(builder, loc, i8Ty, shift);
  auto inputZp = buildScalarConst(builder, loc, i8Ty, 0);
  auto outputZp = buildScalarConst(builder, loc, i8Ty, 0);

  return tosa::RescaleOp::create(
      builder, loc, outTy, input, multiplierConst, shiftConst, inputZp,
      outputZp, /*scale32=*/true, tosa::RoundingMode::SINGLE_ROUND,
      perChannel, inputUnsigned, outputUnsigned);
}

// Builds a tosa.matmul(activation, weight) against a fresh 14x14 tosa.const
// weight tile, where `activation` may be a tosa.const, a block argument, or
// (for chain tests) a prior rescale's result. Output keeps activation's own
// shape, matching buildConstMatmul's K=N=14 convention.
tosa::MatMulOp buildChainedMatmul(OpBuilder& builder, Location loc,
                                  Value activation, int8_t weightValue) {
  auto i8Ty = builder.getIntegerType(8);
  auto i32Ty = builder.getIntegerType(32);
  auto activationShape = cast<RankedTensorType>(activation.getType()).getShape();
  auto bTy = RankedTensorType::get({1, 14, 14}, i8Ty);
  auto b = tosa::ConstOp::create(
      builder, loc, bTy, DenseElementsAttr::get(bTy, weightValue));
  auto zpTy = RankedTensorType::get({1}, i8Ty);
  auto aZp = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto bZp = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto outTy = RankedTensorType::get(activationShape, i32Ty);
  return tosa::MatMulOp::create(builder, loc, outTy, activation, b, aZp, bZp);
}

}  // namespace

TEST(TosaToMacaque, LowersConstMatmulToLoadLoadMatmul) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  buildConstMatmul(builder, loc, /*rows=*/2);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  // expected: tosa.matmul gone, exactly one load_weight, one load_input
  // and one matmul remains
  LoadWeightOp loadWeight;
  LoadInputOp loadInput;
  MatmulOp matmul;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeight = o;
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInput = o;
    if (auto o = dyn_cast<MatmulOp>(op)) matmul = o;
    EXPECT_FALSE(isa<tosa::MatMulOp>(op));
  }

  ASSERT_TRUE(loadWeight);
  ASSERT_TRUE(loadInput);
  ASSERT_TRUE(matmul);

  EXPECT_EQ(loadWeight.getByteCount(), 14u * 14u);
  EXPECT_EQ(loadInput.getByteCount(), 2u * 14u);
  EXPECT_EQ(matmul.getAccMode(), false);
  EXPECT_EQ(matmul.getTileParams(), 2u);

  // ensure the ddr3 memory allocation don't overlap
  EXPECT_NE(loadWeight.getDdr3Addr(), loadInput.getDdr3Addr());
}

TEST(TosaToMacaque, LowersBlockArgumentActivationToLoadInput) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  auto i8Ty = builder.getIntegerType(8);
  auto aTy = RankedTensorType::get({1, 2, 14}, i8Ty);
  BlockArgument a = block.addArgument(aTy, loc);

  auto bTy = RankedTensorType::get({1, 14, 14}, i8Ty);
  auto b = tosa::ConstOp::create(
      builder, loc, bTy, DenseElementsAttr::get(bTy, static_cast<int8_t>(2)));
  auto zpTy = RankedTensorType::get({1}, i8Ty);
  auto aZp = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto bZp = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto outTy = RankedTensorType::get({1, 2, 14}, builder.getIntegerType(32));
  tosa::MatMulOp::create(builder, loc, outTy, a, b, aZp, bZp);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  LoadWeightOp loadWeight;
  LoadInputOp loadInput;
  MatmulOp matmulOp;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeight = o;
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInput = o;
    if (auto o = dyn_cast<MatmulOp>(op)) matmulOp = o;
    EXPECT_FALSE(isa<tosa::MatMulOp>(op));
  }

  ASSERT_TRUE(loadWeight);
  ASSERT_TRUE(loadInput);
  ASSERT_TRUE(matmulOp);
  EXPECT_EQ(loadInput.getByteCount(), 2u * 14u);
  EXPECT_NE(loadWeight.getDdr3Addr(), loadInput.getDdr3Addr());
}

TEST(TosaToMacaque, UnhandledActivationProducerFailsToConvert) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  auto i8Ty = builder.getIntegerType(8);
  auto aTy = RankedTensorType::get({1, 2, 14}, i8Ty);
  auto x = tosa::ConstOp::create(
      builder, loc, aTy, DenseElementsAttr::get(aTy, static_cast<int8_t>(1)));
  auto y = tosa::ConstOp::create(
      builder, loc, aTy, DenseElementsAttr::get(aTy, static_cast<int8_t>(1)));
  auto a = tosa::AddOp::create(builder, loc, aTy, x.getResult(), y.getResult());

  auto bTy = RankedTensorType::get({1, 14, 14}, i8Ty);
  auto b = tosa::ConstOp::create(
      builder, loc, bTy, DenseElementsAttr::get(bTy, static_cast<int8_t>(2)));
  auto zpTy = RankedTensorType::get({1}, i8Ty);
  auto aZp = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto bZp = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto outTy = RankedTensorType::get({1, 2, 14}, builder.getIntegerType(32));
  tosa::MatMulOp::create(builder, loc, outTy, a.getResult(), b, aZp, bZp);

  EXPECT_TRUE(failed(lowerTosaToMacaque(block)));
}

TEST(TosaToMacaque, BatchNotOneFailsToConvert) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  buildConstMatmul(builder, loc, /*rows=*/2, /*batch=*/2);

  EXPECT_TRUE(failed(lowerTosaToMacaque(block)));
}

TEST(TosaToMacaque, LowersMatmulRescaleToActivateWithoutBias) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  tosa::MatMulOp matmul = buildConstMatmul(builder, loc, /*rows=*/2);
  buildRescale(builder, loc, matmul.getResult(), builder.getIntegerType(8),
              /*multiplier=*/12345, /*shift=*/9);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  LoadWeightOp loadWeight;
  LoadInputOp loadInput;
  LoadBiasOp loadBias;
  MatmulOp matmulOp;
  ActivateOp activate;
  StoreOp store;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeight = o;
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInput = o;
    if (auto o = dyn_cast<LoadBiasOp>(op)) loadBias = o;
    if (auto o = dyn_cast<MatmulOp>(op)) matmulOp = o;
    if (auto o = dyn_cast<ActivateOp>(op)) activate = o;
    if (auto o = dyn_cast<StoreOp>(op)) store = o;
    EXPECT_FALSE(isa<tosa::MatMulOp>(op));
    EXPECT_FALSE(isa<tosa::RescaleOp>(op));
  }

  ASSERT_TRUE(loadWeight);
  ASSERT_TRUE(loadInput);
  ASSERT_TRUE(matmulOp);
  ASSERT_TRUE(activate);
  ASSERT_TRUE(store);
  EXPECT_FALSE(loadBias);

  EXPECT_EQ(activate.getActScaleM(), 12345u);
  EXPECT_EQ(activate.getActScaleShift(), 9u);
  EXPECT_EQ(activate.getActNumRows(), 2u);
  EXPECT_EQ(static_cast<isa::ActFunc>(activate.getActFunc()),
            isa::ActFunc::Passthrough);

  // Output tile: rows=2, 14 channels, INT8 -> 28 bytes.
  EXPECT_EQ(store.getByteCount(), 2u * 14u);
  EXPECT_NE(store.getDdr3Addr(), loadWeight.getDdr3Addr());
  EXPECT_NE(store.getDdr3Addr(), loadInput.getDdr3Addr());
}

TEST(TosaToMacaque, LowersMatmulAddRescaleToActivateWithBias) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  tosa::MatMulOp matmul = buildConstMatmul(builder, loc, /*rows=*/2);
  // Bias: one INT32 value per output channel, broadcast across rows -
  // [1, 1, 14], matching load_bias's one-row convention.
  auto biasTy = RankedTensorType::get({1, 1, 14}, builder.getIntegerType(32));
  auto bias = tosa::ConstOp::create(
      builder, loc, biasTy, DenseElementsAttr::get(biasTy, 7));
  auto add = tosa::AddOp::create(builder, loc, matmul.getType(),
                                 matmul.getResult(), bias.getResult());
  buildRescale(builder, loc, add.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  LoadBiasOp loadBias;
  MatmulOp matmulOp;
  ActivateOp activate;
  StoreOp store;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadBiasOp>(op)) loadBias = o;
    if (auto o = dyn_cast<MatmulOp>(op)) matmulOp = o;
    if (auto o = dyn_cast<ActivateOp>(op)) activate = o;
    if (auto o = dyn_cast<StoreOp>(op)) store = o;
    EXPECT_FALSE(isa<tosa::MatMulOp>(op));
    EXPECT_FALSE(isa<tosa::AddOp>(op));
    EXPECT_FALSE(isa<tosa::RescaleOp>(op));
  }

  ASSERT_TRUE(loadBias);
  ASSERT_TRUE(matmulOp);
  ASSERT_TRUE(activate);
  ASSERT_TRUE(store);
  EXPECT_EQ(loadBias.getByteCount(), 14u * sizeof(int32_t));
  EXPECT_EQ(store.getByteCount(), 2u * 14u);
}

// Follows layout in accordance to conventions set at sw/docs/MEMORY_LAYOUT.md
TEST(TosaToMacaque, DdrRegionsFollowIntendedLayout) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  tosa::MatMulOp matmul = buildConstMatmul(builder, loc, /*rows=*/2);
  auto biasTy = RankedTensorType::get({1, 1, 14}, builder.getIntegerType(32));
  auto bias = tosa::ConstOp::create(
      builder, loc, biasTy, DenseElementsAttr::get(biasTy, 7));
  auto add = tosa::AddOp::create(builder, loc, matmul.getType(),
                                 matmul.getResult(), bias.getResult());
  buildRescale(builder, loc, add.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  LoadWeightOp loadWeight;
  LoadBiasOp loadBias;
  LoadInputOp loadInput;
  StoreOp store;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeight = o;
    if (auto o = dyn_cast<LoadBiasOp>(op)) loadBias = o;
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInput = o;
    if (auto o = dyn_cast<StoreOp>(op)) store = o;
  }
  ASSERT_TRUE(loadWeight);
  ASSERT_TRUE(loadBias);
  ASSERT_TRUE(loadInput);
  ASSERT_TRUE(store);

  EXPECT_EQ(loadWeight.getDdr3Addr(), 0x1000u);
  EXPECT_EQ(loadBias.getDdr3Addr(), 4296u);
  EXPECT_EQ(loadInput.getDdr3Addr(), 4352u);
  EXPECT_EQ(store.getDdr3Addr(), 4384u);
}

TEST(TosaToMacaque, ChainsRescaleOutputIntoNextMatmulViaScratch) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  // Layer 1: const matmul -> rescale. Its result feeds layer 2 below, so
  // it's intermediate (Scratch A/B), not final output.
  tosa::MatMulOp matmul1 = buildConstMatmul(builder, loc, /*rows=*/2);
  tosa::RescaleOp rescale1 =
      buildRescale(builder, loc, matmul1.getResult(), builder.getIntegerType(8),
                  /*multiplier=*/1, /*shift=*/0);

  // Layer 2: matmul whose activation is layer 1's rescale result, not a
  // tosa.const or a block argument.
  auto i8Ty = builder.getIntegerType(8);
  auto i32Ty = builder.getIntegerType(32);
  auto bTy = RankedTensorType::get({1, 14, 14}, i8Ty);
  auto b2 = tosa::ConstOp::create(
      builder, loc, bTy, DenseElementsAttr::get(bTy, static_cast<int8_t>(3)));
  auto zpTy = RankedTensorType::get({1}, i8Ty);
  auto aZp2 = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto bZp2 = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto outTy2 = RankedTensorType::get({1, 2, 14}, i32Ty);
  tosa::MatMulOp matmul2 = tosa::MatMulOp::create(
      builder, loc, outTy2, rescale1.getResult(), b2, aZp2, bZp2);

  // Layer 2's rescale - nothing downstream, so it's the final output.
  buildRescale(builder, loc, matmul2.getResult(), i8Ty, /*multiplier=*/1,
              /*shift=*/0);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  SmallVector<LoadInputOp> loadInputs;
  SmallVector<StoreOp> stores;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
    if (auto o = dyn_cast<StoreOp>(op)) stores.push_back(o);
    EXPECT_FALSE(isa<tosa::MatMulOp>(op));
    EXPECT_FALSE(isa<tosa::RescaleOp>(op));
  }

  ASSERT_EQ(loadInputs.size(), 2u);
  ASSERT_EQ(stores.size(), 2u);

  // Exactly one load_input's address matches exactly one store's address -
  // that's the chained pair: layer 2 reads back layer 1's intermediate
  // store instead of getting its own fresh Input-region slot.
  int matches = 0;
  for (auto& li : loadInputs)
    for (auto& s : stores)
      if (li.getDdr3Addr() == s.getDdr3Addr()) matches++;
  EXPECT_EQ(matches, 1);

  // The two stores land in different regions (layer 1: scratch, layer 2:
  // output) and the two load_inputs come from different regions (layer 1:
  // fresh Input slot, layer 2: chained scratch read) - neither pair may
  // collide.
  EXPECT_NE(stores[0].getDdr3Addr(), stores[1].getDdr3Addr());
  EXPECT_NE(loadInputs[0].getDdr3Addr(), loadInputs[1].getDdr3Addr());

  // Exact region math: weight(196)x2 -> bias(0) -> input(28, layer 1 only)
  // -> scratch A(28) -> scratch B(28) -> output(28).
  EXPECT_EQ(loadInputs[0].getDdr3Addr(), 4488u);   // layer 1: fresh Input slot
  EXPECT_EQ(stores[0].getDdr3Addr(), 4520u);       // layer 1: Scratch A
  EXPECT_EQ(loadInputs[1].getDdr3Addr(), 4520u);   // layer 2: chained from Scratch A
  EXPECT_EQ(stores[1].getDdr3Addr(), 4584u);       // layer 2: Output
}

TEST(TosaToMacaque, FourLayerChainAlternatesScratchAB) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);
  auto i8Ty = builder.getIntegerType(8);

  // Layer 1 (intermediate -> Scratch A) -> layer 2 (intermediate ->
  // Scratch B) -> layer 3 (intermediate -> Scratch A again, reusing layer
  // 1's slot) -> layer 4 (final -> Output).
  tosa::MatMulOp matmul1 = buildConstMatmul(builder, loc, /*rows=*/2);
  tosa::RescaleOp rescale1 =
      buildRescale(builder, loc, matmul1.getResult(), i8Ty, 1, 0);

  tosa::MatMulOp matmul2 =
      buildChainedMatmul(builder, loc, rescale1.getResult(), 3);
  tosa::RescaleOp rescale2 =
      buildRescale(builder, loc, matmul2.getResult(), i8Ty, 1, 0);

  tosa::MatMulOp matmul3 =
      buildChainedMatmul(builder, loc, rescale2.getResult(), 5);
  tosa::RescaleOp rescale3 =
      buildRescale(builder, loc, matmul3.getResult(), i8Ty, 1, 0);

  tosa::MatMulOp matmul4 =
      buildChainedMatmul(builder, loc, rescale3.getResult(), 7);
  buildRescale(builder, loc, matmul4.getResult(), i8Ty, 1, 0);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  SmallVector<LoadInputOp> loadInputs;
  SmallVector<StoreOp> stores;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
    if (auto o = dyn_cast<StoreOp>(op)) stores.push_back(o);
    EXPECT_FALSE(isa<tosa::MatMulOp>(op));
    EXPECT_FALSE(isa<tosa::RescaleOp>(op));
  }

  ASSERT_EQ(loadInputs.size(), 4u);
  ASSERT_EQ(stores.size(), 4u);

  // Program order is preserved by conversion, so stores/loadInputs appear
  // in layer order: [layer1, layer2, layer3, layer4].
  const uint32_t scratchA = stores[0].getDdr3Addr();
  const uint32_t scratchB = stores[1].getDdr3Addr();
  EXPECT_NE(scratchA, scratchB);

  // Ping-pong: layer 1 -> A, layer 2 -> B, layer 3 -> A again (reusing
  // layer 1's slot - safe, since layer 2 already consumed it by then).
  EXPECT_EQ(stores[2].getDdr3Addr(), scratchA);
  // Layer 4 is final output, not scratch - must not collide with either.
  EXPECT_NE(stores[3].getDdr3Addr(), scratchA);
  EXPECT_NE(stores[3].getDdr3Addr(), scratchB);

  // Each layer's load_input reads back exactly the previous layer's store
  // (chained), except layer 1, which gets its own fresh Input-region slot.
  EXPECT_NE(loadInputs[0].getDdr3Addr(), scratchA);
  EXPECT_NE(loadInputs[0].getDdr3Addr(), scratchB);
  EXPECT_EQ(loadInputs[1].getDdr3Addr(), scratchA);  // layer 2 <- layer 1's store
  EXPECT_EQ(loadInputs[2].getDdr3Addr(), scratchB);  // layer 3 <- layer 2's store
  EXPECT_EQ(loadInputs[3].getDdr3Addr(), scratchA);  // layer 4 <- layer 3's store (A again)
}

TEST(TosaToMacaque, PerChannelRescaleFailsToConvert) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  tosa::MatMulOp matmul = buildConstMatmul(builder, loc, /*rows=*/2);
  buildRescale(builder, loc, matmul.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0, /*perChannel=*/true);

  EXPECT_TRUE(failed(lowerTosaToMacaque(block)));
}

TEST(TosaToMacaque, NonzeroZeroPointFailsToConvert) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  tosa::MatMulOp matmul = buildConstMatmul(builder, loc, /*rows=*/2);
  auto i8Ty = builder.getIntegerType(8);
  auto i32Ty = builder.getIntegerType(32);
  auto outTy = RankedTensorType::get({1, 2, 14}, i8Ty);
  auto multiplierConst = buildScalarConst(builder, loc, i32Ty, 1);
  auto shiftConst = buildScalarConst(builder, loc, i8Ty, 0);
  auto inputZp = buildScalarConst(builder, loc, i8Ty, /*value=*/3);  // nonzero
  auto outputZp = buildScalarConst(builder, loc, i8Ty, 0);
  tosa::RescaleOp::create(builder, loc, outTy, matmul.getResult(),
                          multiplierConst, shiftConst, inputZp, outputZp,
                          /*scale32=*/true, tosa::RoundingMode::SINGLE_ROUND,
                          /*perChannel=*/false, /*inputUnsigned=*/false,
                          /*outputUnsigned=*/false);

  EXPECT_TRUE(failed(lowerTosaToMacaque(block)));
}

TEST(TosaToMacaque, NonzeroAZpFailsToConvert) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  buildConstMatmul(builder, loc, /*rows=*/2, /*batch=*/1, /*aZpValue=*/3);

  EXPECT_TRUE(failed(lowerTosaToMacaque(block)));
}
