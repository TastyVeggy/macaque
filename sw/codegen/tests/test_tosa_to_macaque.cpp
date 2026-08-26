#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "macaque/Conversion/TosaToMacaque.hpp"
#include "macaque/Dialect/MacaqueOps.hpp"
#include "macaque/common/isa.hpp"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace mlir::macaque;
using namespace ::macaque::codegen::conversion;
namespace macaque_isa = ::macaque::common::isa;

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

// Same shape as buildConstMatmul, but with an arbitrary K (contraction)
// dimension instead of always 14 - for K-tiling tests.
tosa::MatMulOp buildConstMatmulK(OpBuilder& builder, Location loc, int rows,
                                 int64_t k) {
  auto i8Ty = builder.getIntegerType(8);
  auto i32Ty = builder.getIntegerType(32);
  auto zpTy = RankedTensorType::get({1}, i8Ty);
  auto aTy = RankedTensorType::get({1, rows, k}, i8Ty);
  auto bTy = RankedTensorType::get({1, k, 14}, i8Ty);
  auto outTy = RankedTensorType::get({1, rows, 14}, i32Ty);

  auto a = tosa::ConstOp::create(
      builder, loc, aTy, DenseElementsAttr::get(aTy, static_cast<int8_t>(1)));
  auto b = tosa::ConstOp::create(
      builder, loc, bTy, DenseElementsAttr::get(bTy, static_cast<int8_t>(2)));
  auto aZp = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto bZp = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));

  return tosa::MatMulOp::create(builder, loc, outTy, a, b, aZp, bZp);
}

// Same as buildConstMatmulK, but with an arbitrary N (output channel)
// dimension too - for N-tiling tests.
tosa::MatMulOp buildConstMatmulKN(OpBuilder& builder, Location loc, int rows,
                                  int64_t k, int64_t n) {
  auto i8Ty = builder.getIntegerType(8);
  auto i32Ty = builder.getIntegerType(32);
  auto zpTy = RankedTensorType::get({1}, i8Ty);
  auto aTy = RankedTensorType::get({1, rows, k}, i8Ty);
  auto bTy = RankedTensorType::get({1, k, n}, i8Ty);
  auto outTy = RankedTensorType::get({1, rows, n}, i32Ty);

  auto a = tosa::ConstOp::create(
      builder, loc, aTy, DenseElementsAttr::get(aTy, static_cast<int8_t>(1)));
  auto b = tosa::ConstOp::create(
      builder, loc, bTy, DenseElementsAttr::get(bTy, static_cast<int8_t>(2)));
  auto aZp = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
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

// Like buildChainedMatmul, but the weight's K is `activation`'s own last
// dimension (so a chained consumer's K-tiling correctly matches its
// producer's real N) and N is chosen independently - for tests chaining an
// N-tiled producer into a K-tiled consumer.
tosa::MatMulOp buildChainedMatmulN(OpBuilder& builder, Location loc,
                                   Value activation, int64_t n,
                                   int8_t weightValue) {
  auto i8Ty = builder.getIntegerType(8);
  auto i32Ty = builder.getIntegerType(32);
  auto activationShape = cast<RankedTensorType>(activation.getType()).getShape();
  const int64_t rows = activationShape[1];
  const int64_t k = activationShape[2];
  auto bTy = RankedTensorType::get({1, k, n}, i8Ty);
  auto b = tosa::ConstOp::create(builder, loc, bTy,
                                 DenseElementsAttr::get(bTy, weightValue));
  auto zpTy = RankedTensorType::get({1}, i8Ty);
  auto aZp = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto bZp = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto outTy = RankedTensorType::get({1, rows, n}, i32Ty);
  return tosa::MatMulOp::create(builder, loc, outTy, activation, b, aZp, bZp);
}

}  // namespace

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
  auto matmul = tosa::MatMulOp::create(builder, loc, outTy, a, b, aZp, bZp);
  buildRescale(builder, loc, matmul.getResult(), i8Ty, /*multiplier=*/1, /*shift=*/0);

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
  auto matmul = tosa::MatMulOp::create(builder, loc, outTy, a.getResult(), b, aZp, bZp);
  buildRescale(builder, loc, matmul.getResult(), i8Ty, /*multiplier=*/1, /*shift=*/0);

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

  tosa::MatMulOp matmul = buildConstMatmul(builder, loc, /*rows=*/2, /*batch=*/2);
  buildRescale(builder, loc, matmul.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  EXPECT_TRUE(failed(lowerTosaToMacaque(block)));
}

TEST(TosaToMacaque, KTilesMatmulIntoTwoAccumulatingGroups) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  tosa::MatMulOp matmul = buildConstMatmulK(builder, loc, /*rows=*/2, /*k=*/28);
  buildRescale(builder, loc, matmul.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<LoadInputOp> loadInputs;
  SmallVector<MatmulOp> matmuls;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
    if (auto o = dyn_cast<MatmulOp>(op)) matmuls.push_back(o);
    EXPECT_FALSE(isa<tosa::MatMulOp>(op));
  }

  // K=28 / 14 = 2 tiles: one load_weight/load_input/matmul group per tile.
  ASSERT_EQ(loadWeights.size(), 2u);
  ASSERT_EQ(loadInputs.size(), 2u);
  ASSERT_EQ(matmuls.size(), 2u);

  // First tile seeds the accumulator from the bias buffer, the rest
  // accumulate into it - the ISA's existing K-tiling support.
  EXPECT_EQ(matmuls[0].getAccMode(), false);
  EXPECT_EQ(matmuls[1].getAccMode(), true);

  // Each tile is a 14-row weight slice (196 bytes) and a 2-row, 14-wide
  // activation slice (28 bytes) - half of the full K=28 tensor each.
  EXPECT_EQ(loadWeights[0].getByteCount(), 14u * 14u);
  EXPECT_EQ(loadWeights[1].getByteCount(), 14u * 14u);
  EXPECT_EQ(loadInputs[0].getByteCount(), 2u * 14u);
  EXPECT_EQ(loadInputs[1].getByteCount(), 2u * 14u);

  // No two tiles' addresses collide.
  EXPECT_NE(loadWeights[0].getDdr3Addr(), loadWeights[1].getDdr3Addr());
  EXPECT_NE(loadInputs[0].getDdr3Addr(), loadInputs[1].getDdr3Addr());
  EXPECT_NE(loadWeights[0].getDdr3Addr(), loadInputs[0].getDdr3Addr());
}

TEST(TosaToMacaque, PartialKTilePadsToTwoFullTiles) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  // K=20 = one full 14-wide tile plus a 6-wide boundary tile. The array has
  // no masking, so the boundary tile is still staged/read as a full 14-wide
  // tile - see sw/docs/MEMORY_LAYOUT.md's zero-padding convention.
  tosa::MatMulOp matmul = buildConstMatmulK(builder, loc, /*rows=*/2, /*k=*/20);
  buildRescale(builder, loc, matmul.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<LoadInputOp> loadInputs;
  SmallVector<MatmulOp> matmuls;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
    if (auto o = dyn_cast<MatmulOp>(op)) matmuls.push_back(o);
  }

  // ceil(20/14) = 2 tiles, same as an exact K=28 would give - the boundary
  // tile doesn't collapse into a smaller instruction, it's padded instead.
  ASSERT_EQ(loadWeights.size(), 2u);
  ASSERT_EQ(loadInputs.size(), 2u);
  ASSERT_EQ(matmuls.size(), 2u);
  EXPECT_EQ(matmuls[0].getAccMode(), false);
  EXPECT_EQ(matmuls[1].getAccMode(), true);

  // Every tile - including the boundary one - is a full 14-wide byte count,
  // not a smaller 6-wide one.
  EXPECT_EQ(loadWeights[0].getByteCount(), 14u * 14u);
  EXPECT_EQ(loadWeights[1].getByteCount(), 14u * 14u);
  EXPECT_EQ(loadInputs[0].getByteCount(), 2u * 14u);
  EXPECT_EQ(loadInputs[1].getByteCount(), 2u * 14u);
}

TEST(TosaToMacaque, PartialNTileDataIsZeroPadded) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  auto i8Ty = builder.getIntegerType(8);
  auto i32Ty = builder.getIntegerType(32);
  constexpr int64_t kRows = 2, kK = 14, kN = 20;
  auto weightVal = [](int64_t k, int64_t n) -> int8_t {
    return static_cast<int8_t>((k + n) % 7 - 3);
  };
  auto biasVal = [](int64_t n) -> int32_t { return static_cast<int32_t>(n * 10 - 5); };

  auto aTy = RankedTensorType::get({1, kRows, kK}, i8Ty);
  auto bTy = RankedTensorType::get({1, kK, kN}, i8Ty);
  auto zpTy = RankedTensorType::get({1}, i8Ty);
  auto biasTy = RankedTensorType::get({1, 1, kN}, i32Ty);

  auto a = tosa::ConstOp::create(builder, loc, aTy,
                                 DenseElementsAttr::get(aTy, static_cast<int8_t>(1)));

  std::vector<int8_t> bData;
  for (int64_t k = 0; k < kK; ++k)
    for (int64_t n = 0; n < kN; ++n) bData.push_back(weightVal(k, n));
  auto b = tosa::ConstOp::create(builder, loc, bTy,
                                 DenseElementsAttr::get(bTy, ArrayRef<int8_t>(bData)));

  auto aZp = tosa::ConstOp::create(builder, loc, zpTy,
                                   DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto bZp = tosa::ConstOp::create(builder, loc, zpTy,
                                   DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto matmulOutTy = RankedTensorType::get({1, kRows, kN}, i32Ty);
  auto matmul = tosa::MatMulOp::create(builder, loc, matmulOutTy, a.getResult(),
                                       b.getResult(), aZp.getResult(), bZp.getResult());

  std::vector<int32_t> biasData;
  for (int64_t n = 0; n < kN; ++n) biasData.push_back(biasVal(n));
  auto bias = tosa::ConstOp::create(builder, loc, biasTy,
                                    DenseElementsAttr::get(biasTy, ArrayRef<int32_t>(biasData)));
  auto add = tosa::AddOp::create(builder, loc, matmulOutTy, matmul.getResult(),
                                 bias.getResult());
  buildRescale(builder, loc, add.getResult(), i8Ty, /*multiplier=*/1, /*shift=*/0);

  CompiledProgramInfo info;
  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block, &info)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<LoadBiasOp> loadBiases;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<LoadBiasOp>(op)) loadBiases.push_back(o);
  }
  ASSERT_EQ(loadWeights.size(), 2u);  // ceil(20/14) = 2 N-tiles
  ASSERT_EQ(loadBiases.size(), 2u);

  auto findData = [&](uint32_t addr) -> const SmallVector<uint8_t>* {
    for (auto& [recordedAddr, bytes] : info.data)
      if (recordedAddr == addr) return &bytes;
    return nullptr;
  };

  // N-tile 0: fully real (global columns 0-13).
  const SmallVector<uint8_t>* w0 = findData(loadWeights[0].getDdr3Addr());
  ASSERT_NE(w0, nullptr);
  ASSERT_EQ(w0->size(), 14u * 14u);
  for (int64_t k = 0; k < 14; ++k)
    for (int64_t n = 0; n < 14; ++n)
      EXPECT_EQ(static_cast<int8_t>((*w0)[k * 14 + n]), weightVal(k, n))
          << "k=" << k << " n=" << n;

  // N-tile 1 (the boundary tile): local cols 0-5 (global 14-19) are real,
  // local cols 6-13 are the zero-padding convention in action.
  const SmallVector<uint8_t>* w1 = findData(loadWeights[1].getDdr3Addr());
  ASSERT_NE(w1, nullptr);
  ASSERT_EQ(w1->size(), 14u * 14u);
  for (int64_t k = 0; k < 14; ++k) {
    for (int64_t localN = 0; localN < 14; ++localN) {
      const int64_t n = 14 + localN;
      const int8_t expected = n < kN ? weightVal(k, n) : 0;
      EXPECT_EQ(static_cast<int8_t>((*w1)[k * 14 + localN]), expected)
          << "k=" << k << " localN=" << localN;
    }
  }

  // Bias tile 1: same boundary, but int32 little-endian.
  const SmallVector<uint8_t>* bias1 = findData(loadBiases[1].getDdr3Addr());
  ASSERT_NE(bias1, nullptr);
  ASSERT_EQ(bias1->size(), 14u * 4u);
  for (int64_t localN = 0; localN < 14; ++localN) {
    const int64_t n = 14 + localN;
    const int32_t expected = n < kN ? biasVal(n) : 0;
    int32_t got = 0;
    for (int byteIdx = 0; byteIdx < 4; ++byteIdx)
      got |= static_cast<int32_t>((*bias1)[localN * 4 + byteIdx]) << (8 * byteIdx);
    EXPECT_EQ(got, expected) << "localN=" << localN;
  }
}

TEST(TosaToMacaque, NTilesRescaleIntoTwoIndependentGroups) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  // N=28 = two 14-wide output-channel tiles, each a fully independent
  // load_weight/[load_bias]/load_input/matmul/activate/store group - unlike
  // K-tiling, N-tiles don't accumulate into each other.
  tosa::MatMulOp matmul = buildConstMatmulKN(builder, loc, /*rows=*/2, /*k=*/14, /*n=*/28);
  auto biasTy = RankedTensorType::get({1, 1, 28}, builder.getIntegerType(32));
  auto bias = tosa::ConstOp::create(builder, loc, biasTy, DenseElementsAttr::get(biasTy, 7));
  auto add = tosa::AddOp::create(builder, loc, matmul.getType(), matmul.getResult(),
                                 bias.getResult());
  buildRescale(builder, loc, add.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<LoadBiasOp> loadBiases;
  SmallVector<LoadInputOp> loadInputs;
  SmallVector<MatmulOp> matmuls;
  SmallVector<ActivateOp> activates;
  SmallVector<StoreOp> stores;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<LoadBiasOp>(op)) loadBiases.push_back(o);
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
    if (auto o = dyn_cast<MatmulOp>(op)) matmuls.push_back(o);
    if (auto o = dyn_cast<ActivateOp>(op)) activates.push_back(o);
    if (auto o = dyn_cast<StoreOp>(op)) stores.push_back(o);
  }

  // K=14 -> 1 K-tile; N=28 -> 2 N-tiles. One weight/bias/matmul/activate/
  // store per N-tile; the (single) K-tile's input is reloaded once per
  // N-tile rather than shared across the two independent groups.
  ASSERT_EQ(loadWeights.size(), 2u);
  ASSERT_EQ(loadBiases.size(), 2u);
  ASSERT_EQ(loadInputs.size(), 2u);
  ASSERT_EQ(matmuls.size(), 2u);
  ASSERT_EQ(activates.size(), 2u);
  ASSERT_EQ(stores.size(), 2u);

  // Both N-tiles seed fresh (acc_mode=false) - N-tiles don't accumulate.
  EXPECT_EQ(matmuls[0].getAccMode(), false);
  EXPECT_EQ(matmuls[1].getAccMode(), false);

  // Each N-tile writes to its own, non-colliding output address.
  EXPECT_NE(stores[0].getDdr3Addr(), stores[1].getDdr3Addr());
  EXPECT_NE(loadWeights[0].getDdr3Addr(), loadWeights[1].getDdr3Addr());
  EXPECT_NE(loadBiases[0].getDdr3Addr(), loadBiases[1].getDdr3Addr());
}

TEST(TosaToMacaque, NTiledIntermediateFeedsKTiledConsumer) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  // Layer 1: N=28 (2 N-tiles) rescale feeding layer 2 - Scratch A/B now
  // holds one slot per (N-tile, M-chunk), so an N-tiled intermediate
  // producer is supported. Layer 2's weight K must match layer 1's real N
  // (28), so its consumer sees 2 K-tiles too - one per layer 1 N-tile.
  tosa::MatMulOp matmul1 = buildConstMatmulKN(builder, loc, /*rows=*/2, /*k=*/14, /*n=*/28);
  tosa::RescaleOp rescale1 = buildRescale(builder, loc, matmul1.getResult(),
                                          builder.getIntegerType(8),
                                          /*multiplier=*/1, /*shift=*/0);
  tosa::MatMulOp matmul2 = buildChainedMatmulN(builder, loc, rescale1.getResult(),
                                               /*n=*/14, /*weightValue=*/3);
  buildRescale(builder, loc, matmul2.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  // Sentinel: an unrelated, unchained matmul placed *after* layer 2.
  // Address-matching alone can't distinguish "layer 2 correctly read layer
  // 1's Scratch addresses" from "layer 2 wrongly allocated fresh Input
  // bytes that happen to land on the same addresses" - Input always ends
  // exactly where Scratch begins by construction, so a first, unaccounted
  // phantom read coincidentally lines up regardless of which one happened.
  // If layer 2 wrongly consumed real Input-cursor bytes for its chained
  // read (2 tiles' worth, unbudgeted by sizeRegions since a chained
  // activation is meant to need none), this sentinel's own real,
  // correctly-budgeted Input allocation would be pushed forward into where
  // Scratch A/B actually live, corrupting it - observable as its address
  // landing inside the stores' region instead of before it.
  tosa::MatMulOp sentinel = buildConstMatmul(builder, loc, /*rows=*/3);
  buildRescale(builder, loc, sentinel.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  SmallVector<StoreOp> stores;
  SmallVector<LoadInputOp> loadInputs;
  SmallVector<MatmulOp> matmuls;
  for (Operation& op : block) {
    if (auto o = dyn_cast<StoreOp>(op)) stores.push_back(o);
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
    if (auto o = dyn_cast<MatmulOp>(op)) matmuls.push_back(o);
  }

  // Layer 1: 2 independent N-tile groups (K=14 single-tile), each its own
  // load_input (re-issued per N-tile group, same shared address, since
  // input doesn't depend on N but the load instruction still is) and store
  // (into Scratch, one per N-tile). Layer 2: 2 K-tiles, each reading one of
  // layer 1's stores back, plus its own fresh weight - one accumulating
  // matmul per K-tile, plus its own real output store. Sentinel: 1
  // load_input, 1 matmul, 1 real output store.
  ASSERT_EQ(stores.size(), 4u);
  ASSERT_EQ(loadInputs.size(), 5u);
  ASSERT_EQ(matmuls.size(), 5u);  // 2 layer 1 + 2 layer 2 + 1 sentinel

  // Exactly layer 2's 2 load_inputs match layer 1's 2 stores - the chain
  // link, generalized to 2 tiles instead of 1 (layer 1's and the sentinel's
  // own load_inputs read from the Input region, not Scratch, so they never
  // match a store).
  int matches = 0;
  for (auto& li : loadInputs)
    for (auto& s : stores)
      if (li.getDdr3Addr() == s.getDdr3Addr()) matches++;
  EXPECT_EQ(matches, 2);
  EXPECT_NE(stores[0].getDdr3Addr(), stores[1].getDdr3Addr());

  // The sentinel's own load_input - the last one emitted - must sit in the
  // Input region, strictly before both of layer 1's Scratch stores. 
  uint32_t sentinelAddr = loadInputs.back().getDdr3Addr();
  EXPECT_LT(sentinelAddr, stores[0].getDdr3Addr());
  EXPECT_LT(sentinelAddr, stores[1].getDdr3Addr());

  // Layer 2's matmuls (indices 2-3, after layer 1's 2 and before the
  // sentinel's 1): acc_mode false for the first K-tile (seeds from bias),
  // true for the second (accumulates) - the same K-tiling pattern as a
  // fresh (non-chained) K-tiled matmul.
  MatmulOp k0 = matmuls[2];
  MatmulOp k1 = matmuls[3];
  EXPECT_FALSE(k0.getAccMode());
  EXPECT_TRUE(k1.getAccMode());
}

TEST(TosaToMacaque, CombinedNAndKTilingProducesFourMatmuls) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  // K=28 (2 K-tiles) x N=28 (2 N-tiles): each N-tile independently
  // accumulates across both K-tiles - 2x2 = 4 matmuls total.
  tosa::MatMulOp matmul = buildConstMatmulKN(builder, loc, /*rows=*/2, /*k=*/28, /*n=*/28);
  buildRescale(builder, loc, matmul.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<LoadInputOp> loadInputs;
  SmallVector<MatmulOp> matmuls;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
    if (auto o = dyn_cast<MatmulOp>(op)) matmuls.push_back(o);
  }

  ASSERT_EQ(loadWeights.size(), 4u);  // 2 N-tiles x 2 K-tiles
  ASSERT_EQ(loadInputs.size(), 4u);   // reloaded per N-tile, 2 K-tiles each
  ASSERT_EQ(matmuls.size(), 4u);

  // acc_mode alternates false/true within each N-tile's own 2 K-tiles.
  EXPECT_EQ(matmuls[0].getAccMode(), false);
  EXPECT_EQ(matmuls[1].getAccMode(), true);
  EXPECT_EQ(matmuls[2].getAccMode(), false);
  EXPECT_EQ(matmuls[3].getAccMode(), true);

  // No two weight tiles collide - 4 genuinely distinct (K-tile, N-tile) slots.
  for (int i = 0; i < 4; ++i)
    for (int j = i + 1; j < 4; ++j)
      EXPECT_NE(loadWeights[i].getDdr3Addr(), loadWeights[j].getDdr3Addr());
}

TEST(TosaToMacaque, MChunksMatmulIntoTwoHeldGroups) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  // rows=300 = one 252-row chunk plus a 48-row boundary chunk. 252 (not
  // tile_params' full 255-row encoding range) is the real per-instruction
  // cap
  tosa::MatMulOp matmul = buildConstMatmulK(builder, loc, /*rows=*/300, /*k=*/14);
  buildRescale(builder, loc, matmul.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<LoadBiasOp> loadBiases;
  SmallVector<LoadInputOp> loadInputs;
  SmallVector<MatmulOp> matmuls;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<LoadBiasOp>(op)) loadBiases.push_back(o);
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
    if (auto o = dyn_cast<MatmulOp>(op)) matmuls.push_back(o);
  }

  // ceil(300/252) = 2 M-chunks, but only the first gets a real load_weight/
  // load_bias - the second's matmul sets weight_hold instead.
  ASSERT_EQ(loadWeights.size(), 1u);
  ASSERT_EQ(loadBiases.size(), 1u);
  ASSERT_EQ(loadInputs.size(), 2u);
  ASSERT_EQ(matmuls.size(), 2u);

  EXPECT_EQ(matmuls[0].getTileParams(), 252u);
  EXPECT_EQ(matmuls[1].getTileParams(), 48u);
  EXPECT_EQ(matmuls[0].getWeightHold(), false);
  EXPECT_EQ(matmuls[1].getWeightHold(), true);
  EXPECT_EQ(loadInputs[0].getByteCount(), 252u * 14u);
  EXPECT_EQ(loadInputs[1].getByteCount(), 48u * 14u);
  EXPECT_NE(loadInputs[0].getDdr3Addr(), loadInputs[1].getDdr3Addr());
}

TEST(TosaToMacaque, WeightHoldCombinedWithKTilingIntoHeldBatch) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  // rows=30 (3 14-row hold-batch chunks: 14+14+2) x K=28 (2 K-tiles): weight
  // is reloaded once per K-tile (not per chunk) and held across every chunk
  // after the first in that K-tile's pass - each chunk's partial sum stays
  // resident in out_buffer at its own mat_row_base across the whole K-tile
  // sweep.
  tosa::MatMulOp matmul = buildConstMatmulK(builder, loc, /*rows=*/30, /*k=*/28);
  buildRescale(builder, loc, matmul.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<MatmulOp> matmuls;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<MatmulOp>(op)) matmuls.push_back(o);
  }

  // One hold-batch (30 rows fits well within the 252-row batch capacity),
  // so weight is loaded exactly once per K-tile - not once per chunk.
  ASSERT_EQ(loadWeights.size(), 2u);
  // numKTiles(2) x numChunks(3) = 6 matmuls: 3 chunks per K-tile pass.
  ASSERT_EQ(matmuls.size(), 6u);

  // K-tile 0's pass: chunk 0 real (unheld), chunks 1-2 held.
  EXPECT_EQ(matmuls[0].getWeightHold(), false);
  EXPECT_EQ(matmuls[1].getWeightHold(), true);
  EXPECT_EQ(matmuls[2].getWeightHold(), true);
  // K-tile 1's pass: same pattern - its own fresh load, then held.
  EXPECT_EQ(matmuls[3].getWeightHold(), false);
  EXPECT_EQ(matmuls[4].getWeightHold(), true);
  EXPECT_EQ(matmuls[5].getWeightHold(), true);

  // Each chunk's mat_row_base is local to the batch: 0, 14, 28 - the same
  // for both K-tile passes (chunk 1 always lands at row 14 in out_buffer,
  // regardless of which K-tile is currently accumulating into it).
  EXPECT_EQ(matmuls[0].getMatRowBase(), 0u);
  EXPECT_EQ(matmuls[1].getMatRowBase(), 14u);
  EXPECT_EQ(matmuls[2].getMatRowBase(), 28u);
  EXPECT_EQ(matmuls[3].getMatRowBase(), 0u);
  EXPECT_EQ(matmuls[4].getMatRowBase(), 14u);
  EXPECT_EQ(matmuls[5].getMatRowBase(), 28u);

  // Row counts: 14, 14, 2 (30 = 14+14+2) per K-tile pass.
  EXPECT_EQ(matmuls[0].getTileParams(), 14u);
  EXPECT_EQ(matmuls[1].getTileParams(), 14u);
  EXPECT_EQ(matmuls[2].getTileParams(), 2u);
}

TEST(TosaToMacaque, WeightHoldWithKTilingSpansMultipleHoldBatches) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  // rows=260: one full hold-batch (252 rows = 18 chunks of 14) plus an
  // 8-row second batch (1 chunk) - the on-chip out_buffer (256 rows) can't
  // hold more than 252 rows' worth of 14-row chunks at once, so weight
  // reloads once more at the second batch's start.
  tosa::MatMulOp matmul = buildConstMatmulK(builder, loc, /*rows=*/260, /*k=*/28);
  buildRescale(builder, loc, matmul.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<MatmulOp> matmuls;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<MatmulOp>(op)) matmuls.push_back(o);
  }

  // 2 hold-batches x 2 K-tiles = 4 weight loads (vs. 2 without batching, or
  // 19 x 2 = 38 without holding at all).
  ASSERT_EQ(loadWeights.size(), 4u);
  // batch0: 18 chunks x 2 K-tiles = 36. batch1: 1 chunk x 2 K-tiles = 2.
  ASSERT_EQ(matmuls.size(), 38u);

  // First chunk of each batch's each K-tile pass is unheld (fresh load);
  // batch0 has 18 chunks per K-tile pass, batch1 has 1.
  EXPECT_EQ(matmuls[0].getWeightHold(), false);   // batch0, K-tile0, chunk0
  EXPECT_EQ(matmuls[17].getWeightHold(), true);   // batch0, K-tile0, chunk17
  EXPECT_EQ(matmuls[18].getWeightHold(), false);  // batch0, K-tile1, chunk0
  EXPECT_EQ(matmuls[36].getWeightHold(), false);  // batch1, K-tile0, chunk0
  EXPECT_EQ(matmuls[36].getMatRowBase(), 0u);     // batch1 restarts its own row_base at 0
  EXPECT_EQ(matmuls[36].getTileParams(), 8u);     // batch1's only chunk: 260-252=8 rows
  EXPECT_EQ(matmuls[37].getWeightHold(), false);  // batch1, K-tile1, chunk0
}

TEST(TosaToMacaque, HeldBatchKTilingLoopsOverNTilesThroughRescale) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  // K=28 (2 K-tiles, so the held-batch branch fires) x N=28 (2 N-tiles),
  // rows=2 (single M-hold-chunk).
  tosa::MatMulOp matmul = buildConstMatmulKN(builder, loc, /*rows=*/2, /*k=*/28, /*n=*/28);
  buildRescale(builder, loc, matmul.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<MatmulOp> matmuls;
  SmallVector<StoreOp> stores;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<MatmulOp>(op)) matmuls.push_back(o);
    if (auto o = dyn_cast<StoreOp>(op)) stores.push_back(o);
  }

  // 2 N-tiles x 2 K-tiles = 4 independent weight loads and matmuls, 2
  // independent stores (one per N-tile)
  ASSERT_EQ(loadWeights.size(), 4u);
  ASSERT_EQ(matmuls.size(), 4u);
  ASSERT_EQ(stores.size(), 2u);

  // All 4 weight addresses are pairwise distinct - each N-tile's 2 K-tiles
  // get their own real weight data, not a shared/reused address.
  for (size_t i = 0; i < loadWeights.size(); ++i)
    for (size_t j = i + 1; j < loadWeights.size(); ++j)
      EXPECT_NE(loadWeights[i].getDdr3Addr(), loadWeights[j].getDdr3Addr())
          << "weight " << i << " vs " << j;

  // Each N-tile's own K-tile pass: acc_mode false then true (seed, then
  // accumulate) - the pattern repeats per N-tile, not just once overall.
  EXPECT_FALSE(matmuls[0].getAccMode());
  EXPECT_TRUE(matmuls[1].getAccMode());
  EXPECT_FALSE(matmuls[2].getAccMode());
  EXPECT_TRUE(matmuls[3].getAccMode());

  EXPECT_NE(stores[0].getDdr3Addr(), stores[1].getDdr3Addr());
}

TEST(TosaToMacaque, MChunksRescaleIntoTwoHeldGroupsWithSeparateStores) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  // rows=300, K=14 (single K-tile, hold-eligible), N=14 (single N-tile) -
  // each M-chunk still needs its own activate+store (out_buffer holds one
  // M-chunk at a time), even while weight/bias stay resident.
  tosa::MatMulOp matmul = buildConstMatmulK(builder, loc, /*rows=*/300, /*k=*/14);
  buildRescale(builder, loc, matmul.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<LoadBiasOp> loadBiases;
  SmallVector<MatmulOp> matmuls;
  SmallVector<ActivateOp> activates;
  SmallVector<StoreOp> stores;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<LoadBiasOp>(op)) loadBiases.push_back(o);
    if (auto o = dyn_cast<MatmulOp>(op)) matmuls.push_back(o);
    if (auto o = dyn_cast<ActivateOp>(op)) activates.push_back(o);
    if (auto o = dyn_cast<StoreOp>(op)) stores.push_back(o);
  }

  ASSERT_EQ(loadWeights.size(), 1u);
  ASSERT_EQ(loadBiases.size(), 1u);
  ASSERT_EQ(matmuls.size(), 2u);
  ASSERT_EQ(activates.size(), 2u);
  ASSERT_EQ(stores.size(), 2u);

  EXPECT_EQ(matmuls[0].getWeightHold(), false);
  EXPECT_EQ(matmuls[1].getWeightHold(), true);
  EXPECT_EQ(activates[0].getActNumRows(), 252u);
  EXPECT_EQ(activates[1].getActNumRows(), 48u);
  EXPECT_EQ(stores[0].getByteCount(), 252u * 14u);
  EXPECT_EQ(stores[1].getByteCount(), 48u * 14u);
  EXPECT_NE(stores[0].getDdr3Addr(), stores[1].getDdr3Addr());
}

TEST(TosaToMacaque, MChunkedIntermediateFeedsMChunkedConsumer) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  // Layer 1: rows=300 (2 M-chunks, K=14 single-tile so weight-hold applies)
  // feeding layer 2 - Scratch A/B now holds one slot per M-chunk. Layer 2
  // shares the same 300 rows (M doesn't change between layers), so it
  // M-chunks the same way.
  tosa::MatMulOp matmul1 = buildConstMatmulK(builder, loc, /*rows=*/300, /*k=*/14);
  tosa::RescaleOp rescale1 = buildRescale(builder, loc, matmul1.getResult(),
                                          builder.getIntegerType(8),
                                          /*multiplier=*/1, /*shift=*/0);
  tosa::MatMulOp matmul2 =
      buildChainedMatmul(builder, loc, rescale1.getResult(), /*weightValue=*/3);
  buildRescale(builder, loc, matmul2.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  SmallVector<StoreOp> stores;
  SmallVector<LoadInputOp> loadInputs;
  for (Operation& op : block) {
    if (auto o = dyn_cast<StoreOp>(op)) stores.push_back(o);
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
  }

  // Layer 1: 2 M-chunks, each its own load_input (fresh) and store (into
  // Scratch). Layer 2: 2 M-chunks, each its own load_input reading one of
  // layer 1's 2 stores, plus its own 2 real output stores - 4 load_inputs,
  // 4 stores total.
  ASSERT_EQ(stores.size(), 4u);
  ASSERT_EQ(loadInputs.size(), 4u);

  int matches = 0;
  for (auto& li : loadInputs)
    for (auto& s : stores)
      if (li.getDdr3Addr() == s.getDdr3Addr()) matches++;
  EXPECT_EQ(matches, 2);
  EXPECT_NE(stores[0].getDdr3Addr(), stores[1].getDdr3Addr());
}

TEST(TosaToMacaque, HeldBatchProducerSharesOneScratchAddressPerBatch) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  // Layer 1: K=28 (2 K-tiles, own K needs the held-batch scheme), N=14
  // (single tile), rows=30 (3 hold-chunks: 14+14+2) - feeds layer 2 (K=14,
  // matching layer 1's N, so a plain flat consumer).
  tosa::MatMulOp matmul1 = buildConstMatmulK(builder, loc, /*rows=*/30, /*k=*/28);
  tosa::RescaleOp rescale1 = buildRescale(builder, loc, matmul1.getResult(),
                                          builder.getIntegerType(8),
                                          /*multiplier=*/1, /*shift=*/0);
  tosa::MatMulOp matmul2 =
      buildChainedMatmul(builder, loc, rescale1.getResult(), /*weightValue=*/3);
  buildRescale(builder, loc, matmul2.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  SmallVector<LoadInputOp> loadInputs;
  SmallVector<StoreOp> stores;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
    if (auto o = dyn_cast<StoreOp>(op)) stores.push_back(o);
  }
  // Layer 1: 3 hold-chunks x 2 K-tiles = 6 own load_inputs, 3 stores (one
  // per chunk, sharing one batch's base address at computed offsets).
  // Layer 2: 1 load_input (single K-tile, chained, reading that same base),
  // plus its own real output store.
  ASSERT_EQ(loadInputs.size(), 7u);
  ASSERT_EQ(stores.size(), 4u);

  // The 3 chunks' stores land at offsets 0, 14*14, 28*14 bytes (row_base
  // 0/14/28, 14 columns, 1 byte/element) from one shared base address
  const uint32_t base = stores[0].getDdr3Addr();
  EXPECT_EQ(stores[1].getDdr3Addr(), base + 14u * 14u);
  EXPECT_EQ(stores[2].getDdr3Addr(), base + 28u * 14u);

  // Layer 2's chained read (the last load_input) reads that exact base
  // address
  EXPECT_EQ(loadInputs.back().getDdr3Addr(), base);
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
  //  No TOSA-level bias, but load_bias must still be emitted, pointing at the shared zero-bias slot.
  ASSERT_TRUE(loadBias);
  EXPECT_EQ(loadBias.getByteCount(), 14u * sizeof(int32_t));

  EXPECT_EQ(activate.getActScaleM(), 12345u);
  EXPECT_EQ(activate.getActScaleShift(), 9u);
  EXPECT_EQ(activate.getActNumRows(), 2u);
  EXPECT_EQ(static_cast<macaque_isa::ActFunc>(activate.getActFunc()),
            macaque_isa::ActFunc::Passthrough);

  // Output tile: rows=2, 14 channels, INT8 -> 28 bytes.
  EXPECT_EQ(store.getByteCount(), 2u * 14u);
  EXPECT_NE(store.getDdr3Addr(), loadWeight.getDdr3Addr());
  EXPECT_NE(store.getDdr3Addr(), loadInput.getDdr3Addr());
}

TEST(TosaToMacaque, MultiplierAtSeventeenBitMaxConverts) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  tosa::MatMulOp matmul = buildConstMatmul(builder, loc, /*rows=*/2);
  buildRescale(builder, loc, matmul.getResult(), builder.getIntegerType(8),
              /*multiplier=*/0x1FFFF, /*shift=*/9);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  ActivateOp activate;
  for (Operation& op : block)
    if (auto o = dyn_cast<ActivateOp>(op)) activate = o;
  ASSERT_TRUE(activate);
  EXPECT_EQ(activate.getActScaleM(), 0x1FFFFu);
}

TEST(TosaToMacaque, MultiplierAboveSeventeenBitMaxFailsToConvert) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  tosa::MatMulOp matmul = buildConstMatmul(builder, loc, /*rows=*/2);
  buildRescale(builder, loc, matmul.getResult(), builder.getIntegerType(8),
              /*multiplier=*/0x20000, /*shift=*/9);

  EXPECT_TRUE(failed(lowerTosaToMacaque(block)));
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

  // Exact region math: weight(196, 8-byte-aligned to 200)x2 -> bias(0, no
  // real bias) -> zero-bias(56, shared by both no-bias layers) -> input(28,
  // aligned to 32, layer 1 only) -> scratch A(28) -> scratch B(28) ->
  // output(28).
  EXPECT_EQ(loadInputs[0].getDdr3Addr(), 4552u);   // layer 1: fresh Input slot
  EXPECT_EQ(stores[0].getDdr3Addr(), 4584u);       // layer 1: Scratch A
  EXPECT_EQ(loadInputs[1].getDdr3Addr(), 4584u);   // layer 2: chained from Scratch A
  EXPECT_EQ(stores[1].getDdr3Addr(), 4648u);       // layer 2: Output
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

  tosa::MatMulOp matmul =
      buildConstMatmul(builder, loc, /*rows=*/2, /*batch=*/1, /*aZpValue=*/3);
  buildRescale(builder, loc, matmul.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  EXPECT_TRUE(failed(lowerTosaToMacaque(block)));
}

TEST(TosaToMacaque, ZeroBiasIsNotFoldedAway) {
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
      builder, loc, biasTy,
      DenseElementsAttr::get(biasTy, ArrayRef<int32_t>(std::vector<int32_t>(14, 0))));
  auto add = tosa::AddOp::create(builder, loc, matmul.getType(), matmul.getResult(),
                                 bias.getResult());
  buildRescale(builder, loc, add.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  CompiledProgramInfo info;
  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block, &info)));

  SmallVector<LoadBiasOp> loadBiases;
  for (Operation& op : block)
    if (auto o = dyn_cast<LoadBiasOp>(op)) loadBiases.push_back(o);
  ASSERT_EQ(loadBiases.size(), 1u);

  const SmallVector<uint8_t>* biasBytes = nullptr;
  for (auto& [addr, bytes] : info.data)
    if (addr == loadBiases[0].getDdr3Addr()) biasBytes = &bytes;
  ASSERT_NE(biasBytes, nullptr);
  ASSERT_EQ(biasBytes->size(), 14u * 4u);
  for (uint8_t b : *biasBytes) EXPECT_EQ(b, 0u);
}

// A chained rescale->clamp(0, 127) should select ActFunc::Relu for that
// layer, erase the clamp entirely
TEST(TosaToMacaque, ReluClampFusesIntoActivateAndErases) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  tosa::MatMulOp matmul1 = buildConstMatmul(builder, loc, /*rows=*/2);
  tosa::RescaleOp rescale1 = buildRescale(builder, loc, matmul1.getResult(),
                                         builder.getIntegerType(8),
                                         /*multiplier=*/1, /*shift=*/0);
  auto relu = tosa::ClampOp::create(
      builder, loc, rescale1.getType(), rescale1.getResult(),
      builder.getI8IntegerAttr(0), builder.getI8IntegerAttr(127),
      tosa::NanPropagationMode::PROPAGATE);

  tosa::MatMulOp matmul2 =
      buildChainedMatmul(builder, loc, relu.getResult(), /*weightValue=*/3);
  buildRescale(builder, loc, matmul2.getResult(), builder.getIntegerType(8),
              /*multiplier=*/1, /*shift=*/0);

  ASSERT_TRUE(succeeded(lowerTosaToMacaque(block)));

  // The clamp must not survive as its own op - it has no macaque
  // equivalent, it's fused into layer 1's ActivateOp.
  for (Operation& op : block) EXPECT_FALSE(isa<tosa::ClampOp>(op));

  SmallVector<ActivateOp> activates;
  SmallVector<StoreOp> stores;
  SmallVector<LoadInputOp> loadInputs;
  for (Operation& op : block) {
    if (auto o = dyn_cast<ActivateOp>(op)) activates.push_back(o);
    if (auto o = dyn_cast<StoreOp>(op)) stores.push_back(o);
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
  }
  ASSERT_EQ(activates.size(), 2u);
  EXPECT_EQ(activates[0].getActFunc(), static_cast<uint8_t>(macaque_isa::ActFunc::Relu));
  EXPECT_EQ(activates[1].getActFunc(),
           static_cast<uint8_t>(macaque_isa::ActFunc::Passthrough));

  ASSERT_EQ(stores.size(), 2u);
  ASSERT_GE(loadInputs.size(), 2u);
  EXPECT_EQ(loadInputs[1].getDdr3Addr(), stores[0].getDdr3Addr());
}
