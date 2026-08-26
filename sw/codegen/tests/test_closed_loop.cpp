#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "macaque/Conversion/TosaToMacaque.hpp"
#include "macaque/Dialect/MacaqueOps.hpp"
#include "macaque/Target/BinaryEmitter.hpp"
#include "macaque/common/isa.hpp"
#include "macaque/sim/simulator.hpp"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

// Two kinds of "closed loop" test live in this file, both checked against
// the simulator rather than just the shape of the emitted IR:
//   1. Hand-built macaque IR -> emitBinary -> simulator. Skips the
//      TOSA -> macaque conversion pass entirely; proves the binary
//      emitter + simulator agree on the ISA encoding.
//   2. TOSA IR -> lowerTosaToMacaque -> emitBinary -> simulator. Exercises
//      the *whole* compiler pipeline, including chained multi-layer
//      addressing (Scratch A/B) - the only place that does.
// Both are grouped under the ClosedLoop test suite; the section banners
// below mark which is which.

using namespace mlir;
using namespace mlir::macaque;
namespace sim = ::macaque::sim;
namespace target = ::macaque::codegen::target;
namespace isa = ::macaque::common::isa;
namespace conversion = ::macaque::codegen::conversion;

//===----------------------------------------------------------------------===//
// 1. Hand-built macaque IR (no TOSA, no conversion pass)
//===----------------------------------------------------------------------===//

namespace {

constexpr int kRows = 2;

int8_t weightA(int k, int c) { return static_cast<int8_t>(k - c); }
int8_t weightB(int k, int c) { return static_cast<int8_t>(c - k); }
int8_t actA(int r, int k) { return static_cast<int8_t>(r + k + 1); }
int8_t actB(int r, int k) { return static_cast<int8_t>(2 * r + k); }
int32_t bias(int c) { return 2 * c; }

constexpr uint32_t kWeightAAddr = 0;
constexpr uint32_t kWeightBAddr = 256;
constexpr uint32_t kBiasAddr = 512;
constexpr uint32_t kActAAddr = 640;
constexpr uint32_t kActBAddr = 704;
constexpr uint32_t kOutputAddr = 768;

std::vector<uint8_t> serializeWeightTile(int8_t (*w)(int, int)) {
  std::vector<uint8_t> bytes;
  for (int k = 0; k < sim::kArraySize; ++k)
    for (int c = 0; c < sim::kArraySize; ++c)
      bytes.push_back(static_cast<uint8_t>(w(k, c)));
  return bytes;
}

std::vector<uint8_t> serializeActTile(int8_t (*a)(int, int)) {
  std::vector<uint8_t> bytes;
  for (int r = 0; r < kRows; ++r)
    for (int k = 0; k < sim::kArraySize; ++k)
      bytes.push_back(static_cast<uint8_t>(a(r, k)));
  return bytes;
}

std::vector<uint8_t> serializeBias() {
  std::vector<uint8_t> bytes;
  for (int c = 0; c < sim::kArraySize; ++c) {
    int32_t v = bias(c);
    for (int b = 0; b < 4; ++b)
      bytes.push_back(static_cast<uint8_t>((v >> (8 * b)) & 0xFF));
  }
  return bytes;
}

uint8_t expected(int r, int c) {
  int64_t acc = bias(c);
  for (int k = 0; k < sim::kArraySize; ++k)
    acc += static_cast<int64_t>(actA(r, k)) * static_cast<int64_t>(weightA(k, c));
  for (int k = 0; k < sim::kArraySize; ++k)
    acc += static_cast<int64_t>(actB(r, k)) * static_cast<int64_t>(weightB(k, c));
  int64_t relu = acc < 0 ? 0 : acc;           // ACT_RELU
  int64_t clamped = relu > 127 ? 127 : relu;  // signed INT8 clamp
  return static_cast<uint8_t>(clamped);
}

}  // namespace

TEST(ClosedLoop, TwoTileKTilingWithReluIsBitExact) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  LoadWeightOp::create(builder, loc, TypeRange{}, kWeightAAddr,
                        sim::kArraySize * sim::kArraySize);
  LoadBiasOp::create(builder, loc, TypeRange{}, kBiasAddr,
                      sim::kArraySize * sizeof(int32_t));
  LoadInputOp::create(builder, loc, TypeRange{}, kActAAddr,
                       kRows * sim::kArraySize);
  MatmulOp::create(builder, loc, TypeRange{}, /*acc_mode=*/false,
                    /*tile_params=*/kRows);

  LoadWeightOp::create(builder, loc, TypeRange{}, kWeightBAddr,
                        sim::kArraySize * sim::kArraySize);
  LoadInputOp::create(builder, loc, TypeRange{}, kActBAddr,
                       kRows * sim::kArraySize);
  MatmulOp::create(builder, loc, TypeRange{}, /*acc_mode=*/true,
                    /*tile_params=*/kRows);

  ActivateOp::create(builder, loc, TypeRange{},
                      /*act_func=*/static_cast<uint8_t>(isa::ActFunc::Relu),
                      /*act_scale_m=*/1, /*act_scale_shift=*/0,
                      /*act_num_rows=*/kRows);
  StoreOp::create(builder, loc, TypeRange{}, kOutputAddr,
                   kRows * sim::kArraySize);

  auto words = target::emitBinary(block);
  ASSERT_TRUE(succeeded(words));
  ASSERT_EQ(words->size(), 9u);

  sim::Simulator simulator(/*ddr_bytes=*/2048);
  auto wa = serializeWeightTile(weightA);
  auto wb = serializeWeightTile(weightB);
  auto b = serializeBias();
  auto aa = serializeActTile(actA);
  auto ab = serializeActTile(actB);
  simulator.write(kWeightAAddr, wa.data(), wa.size());
  simulator.write(kWeightBAddr, wb.data(), wb.size());
  simulator.write(kBiasAddr, b.data(), b.size());
  simulator.write(kActAAddr, aa.data(), aa.size());
  simulator.write(kActBAddr, ab.data(), ab.size());

  simulator.run(*words);

  std::vector<uint8_t> result =
      simulator.read(kOutputAddr, kRows * sim::kArraySize);
  ASSERT_EQ(result.size(), static_cast<size_t>(kRows * sim::kArraySize));
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < sim::kArraySize; ++c) {
      EXPECT_EQ(result[r * sim::kArraySize + c], expected(r, c))
          << "row " << r << " channel " << c;
    }
  }
}

// 2. TOSA-sourced tests
namespace {

constexpr int kChainRows = 2;
constexpr int kChainK = 14;  // == sim::kArraySize; fixed until real tiling lands

int8_t chainAct0(int r, int k) { return static_cast<int8_t>((r * 3 + k) % 9 - 4); }
int8_t chainWeight1(int k, int c) { return static_cast<int8_t>((k - c) % 5); }
int32_t chainBias1(int c) { return c - 4; }
int8_t chainWeight2(int k, int c) { return static_cast<int8_t>((k + c) % 6 - 3); }
int8_t chainWeight3(int k, int c) { return static_cast<int8_t>((c - k) % 4); }
int32_t chainBias3(int c) { return 2 * c - 6; }

constexpr uint32_t kL1Mult = 1, kL1Shift = 0;
constexpr uint32_t kL2Mult = 1, kL2Shift = 0;
constexpr uint32_t kL3Mult = 3, kL3Shift = 1;

// Independent quantisation oracle so this test isn't
// just checking the simulator agrees with itself.
uint8_t chainRequantizePassthrough(int64_t acc, uint32_t m, uint32_t shift) {
  int64_t scaled = acc * static_cast<int64_t>(m);
  if (shift > 0) scaled += static_cast<int64_t>(1) << (shift - 1);  // round-half-up
  int64_t requant = scaled >> shift;
  return static_cast<uint8_t>(std::clamp<int64_t>(requant, -128, 127));
}

uint8_t chainLayer1Out(int r, int c) {
  int64_t acc = chainBias1(c);
  for (int k = 0; k < kChainK; ++k)
    acc += static_cast<int64_t>(chainAct0(r, k)) *
           static_cast<int64_t>(chainWeight1(k, c));
  return chainRequantizePassthrough(acc, kL1Mult, kL1Shift);
}
int8_t chainLayer1OutSigned(int r, int c) {
  return static_cast<int8_t>(chainLayer1Out(r, c));
}

uint8_t chainLayer2Out(int r, int c) {
  int64_t acc = 0;
  for (int k = 0; k < kChainK; ++k)
    acc += static_cast<int64_t>(chainLayer1OutSigned(r, k)) *
           static_cast<int64_t>(chainWeight2(k, c));
  return chainRequantizePassthrough(acc, kL2Mult, kL2Shift);
}
int8_t chainLayer2OutSigned(int r, int c) {
  return static_cast<int8_t>(chainLayer2Out(r, c));
}

uint8_t chainLayer3Out(int r, int c) {
  int64_t acc = chainBias3(c);
  for (int k = 0; k < kChainK; ++k)
    acc += static_cast<int64_t>(chainLayer2OutSigned(r, k)) *
           static_cast<int64_t>(chainWeight3(k, c));
  return chainRequantizePassthrough(acc, kL3Mult, kL3Shift);
}

tosa::ConstOp buildChainI8Tensor(OpBuilder& builder, Location loc, int rows,
                                 int cols, int8_t (*gen)(int, int)) {
  auto ty = RankedTensorType::get({1, rows, cols}, builder.getIntegerType(8));
  std::vector<int8_t> data;
  data.reserve(static_cast<size_t>(rows) * cols);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) data.push_back(gen(r, c));
  return tosa::ConstOp::create(builder, loc, ty,
                               DenseElementsAttr::get(ty, ArrayRef<int8_t>(data)));
}

tosa::ConstOp buildChainI32Bias(OpBuilder& builder, Location loc, int cols,
                                int32_t (*gen)(int)) {
  auto ty = RankedTensorType::get({1, 1, cols}, builder.getIntegerType(32));
  std::vector<int32_t> data;
  data.reserve(static_cast<size_t>(cols));
  for (int c = 0; c < cols; ++c) data.push_back(gen(c));
  return tosa::ConstOp::create(builder, loc, ty,
                               DenseElementsAttr::get(ty, ArrayRef<int32_t>(data)));
}

tosa::ConstOp buildChainScalarI8(OpBuilder& builder, Location loc, int8_t v) {
  auto ty = RankedTensorType::get({1}, builder.getIntegerType(8));
  return tosa::ConstOp::create(builder, loc, ty, DenseElementsAttr::get(ty, v));
}

tosa::ConstOp buildChainScalarI32(OpBuilder& builder, Location loc, int32_t v) {
  auto ty = RankedTensorType::get({1}, builder.getIntegerType(32));
  return tosa::ConstOp::create(builder, loc, ty, DenseElementsAttr::get(ty, v));
}

tosa::RescaleOp buildChainLayer(OpBuilder& builder, Location loc,
                                Value activation, tosa::ConstOp weight,
                                tosa::ConstOp bias, int32_t multiplier,
                                int8_t shift) {
  auto aZp = buildChainScalarI8(builder, loc, 0);
  auto bZp = buildChainScalarI8(builder, loc, 0);
  auto activationShape = cast<RankedTensorType>(activation.getType()).getShape();
  auto matmulOutTy =
      RankedTensorType::get(activationShape, builder.getIntegerType(32));
  auto matmul = tosa::MatMulOp::create(builder, loc, matmulOutTy, activation,
                                       weight.getResult(), aZp.getResult(),
                                       bZp.getResult());

  Value rescaleInput = matmul.getResult();
  if (bias) {
    auto add = tosa::AddOp::create(builder, loc, matmulOutTy, matmul.getResult(),
                                   bias.getResult());
    rescaleInput = add.getResult();
  }

  auto multiplierConst = buildChainScalarI32(builder, loc, multiplier);
  auto shiftConst = buildChainScalarI8(builder, loc, shift);
  auto inputZp = buildChainScalarI8(builder, loc, 0);
  auto outputZp = buildChainScalarI8(builder, loc, 0);
  auto outTy = RankedTensorType::get(activationShape, builder.getIntegerType(8));
  return tosa::RescaleOp::create(
      builder, loc, outTy, rescaleInput, multiplierConst.getResult(),
      shiftConst.getResult(), inputZp.getResult(), outputZp.getResult(),
      /*scale32=*/true, tosa::RoundingMode::SINGLE_ROUND,
      /*perChannel=*/false, /*inputUnsigned=*/false, /*outputUnsigned=*/false);
}

// Like buildChainLayer, but the output's channel count comes from `weight`'s
// own N (weight's shape[2]) instead of reusing activation's shape - for
// chaining a layer whose N differs from its own K (e.g. an N-tiled producer
// feeding a K-tiled consumer).
tosa::RescaleOp buildChainLayerN(OpBuilder& builder, Location loc,
                                 Value activation, tosa::ConstOp weight,
                                 tosa::ConstOp bias, int32_t multiplier,
                                 int8_t shift) {
  auto aZp = buildChainScalarI8(builder, loc, 0);
  auto bZp = buildChainScalarI8(builder, loc, 0);
  auto rows = cast<RankedTensorType>(activation.getType()).getShape()[1];
  auto cols = cast<RankedTensorType>(weight.getType()).getShape()[2];
  auto matmulOutTy =
      RankedTensorType::get({1, rows, cols}, builder.getIntegerType(32));
  auto matmul = tosa::MatMulOp::create(builder, loc, matmulOutTy, activation,
                                       weight.getResult(), aZp.getResult(),
                                       bZp.getResult());

  Value rescaleInput = matmul.getResult();
  if (bias) {
    auto add = tosa::AddOp::create(builder, loc, matmulOutTy, matmul.getResult(),
                                   bias.getResult());
    rescaleInput = add.getResult();
  }

  auto multiplierConst = buildChainScalarI32(builder, loc, multiplier);
  auto shiftConst = buildChainScalarI8(builder, loc, shift);
  auto inputZp = buildChainScalarI8(builder, loc, 0);
  auto outputZp = buildChainScalarI8(builder, loc, 0);
  auto outTy = RankedTensorType::get({1, rows, cols}, builder.getIntegerType(8));
  return tosa::RescaleOp::create(
      builder, loc, outTy, rescaleInput, multiplierConst.getResult(),
      shiftConst.getResult(), inputZp.getResult(), outputZp.getResult(),
      /*scale32=*/true, tosa::RoundingMode::SINGLE_ROUND,
      /*perChannel=*/false, /*inputUnsigned=*/false, /*outputUnsigned=*/false);
}

}  // namespace

TEST(ClosedLoop, ThreeLayerChainFromTosaIsBitExact) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  auto act0Const = buildChainI8Tensor(builder, loc, kChainRows, kChainK, chainAct0);
  auto weight1Const = buildChainI8Tensor(builder, loc, kChainK, kChainK, chainWeight1);
  auto bias1Const = buildChainI32Bias(builder, loc, kChainK, chainBias1);
  auto weight2Const = buildChainI8Tensor(builder, loc, kChainK, kChainK, chainWeight2);
  auto weight3Const = buildChainI8Tensor(builder, loc, kChainK, kChainK, chainWeight3);
  auto bias3Const = buildChainI32Bias(builder, loc, kChainK, chainBias3);

  auto rescale1 = buildChainLayer(builder, loc, act0Const.getResult(), weight1Const,
                                  bias1Const, static_cast<int32_t>(kL1Mult),
                                  static_cast<int8_t>(kL1Shift));
  auto rescale2 = buildChainLayer(builder, loc, rescale1.getResult(), weight2Const,
                                  /*bias=*/nullptr, static_cast<int32_t>(kL2Mult),
                                  static_cast<int8_t>(kL2Shift));
  buildChainLayer(builder, loc, rescale2.getResult(), weight3Const, bias3Const,
                  static_cast<int32_t>(kL3Mult), static_cast<int8_t>(kL3Shift));

  ASSERT_TRUE(succeeded(conversion::lowerTosaToMacaque(block)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<LoadBiasOp> loadBiases;
  SmallVector<LoadInputOp> loadInputs;
  SmallVector<StoreOp> stores;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<LoadBiasOp>(op)) loadBiases.push_back(o);
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
    if (auto o = dyn_cast<StoreOp>(op)) stores.push_back(o);
  }
  ASSERT_EQ(loadWeights.size(), 3u);
  // layer 1 and layer 3 have real biases; layer 2 has none, but still gets
  // a load_bias pointing at the shared zero-bias slot (see emitBias in
  // TosaToMacaque.cpp) - acc_mode=0 always seeds from the bias buffer, so
  // skipping the load entirely would silently reuse whatever a *previous*
  // layer last loaded there.
  ASSERT_EQ(loadBiases.size(), 3u);
  ASSERT_EQ(loadInputs.size(), 3u);
  ASSERT_EQ(stores.size(), 3u);

  auto words = target::emitBinary(block);
  ASSERT_TRUE(succeeded(words));

  sim::Simulator simulator(/*ddr_bytes=*/8192);

  auto writeI8Tile = [&](uint32_t addr, int rows, int cols, int8_t (*gen)(int, int)) {
    std::vector<uint8_t> bytes;
    bytes.reserve(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c) bytes.push_back(static_cast<uint8_t>(gen(r, c)));
    simulator.write(addr, bytes.data(), bytes.size());
  };
  auto writeI32Bias = [&](uint32_t addr, int cols, int32_t (*gen)(int)) {
    std::vector<uint8_t> bytes;
    bytes.reserve(static_cast<size_t>(cols) * 4);
    for (int c = 0; c < cols; ++c) {
      int32_t v = gen(c);
      for (int b = 0; b < 4; ++b) bytes.push_back(static_cast<uint8_t>((v >> (8 * b)) & 0xFF));
    }
    simulator.write(addr, bytes.data(), bytes.size());
  };

  // Only genuinely external data needs staging: layer 1's activation, every
  // layer's weights, and layer 1/3's real biases. Layer 2's zero-bias slot
  // is deliberately left untouched - the simulator's DDR3 model is
  // zero-initialized, matching what a real runtime staging that region
  // once (not per-run) would guarantee. Layers 2 and 3's activations are
  // chained - their load_input reads back whatever the previous layer's
  // own store wrote, which the simulator produces itself while executing
  // those instructions, exactly like real hardware would - nothing to
  // inject for those either.
  writeI8Tile(loadInputs[0].getDdr3Addr(), kChainRows, kChainK, chainAct0);
  writeI8Tile(loadWeights[0].getDdr3Addr(), kChainK, kChainK, chainWeight1);
  writeI32Bias(loadBiases[0].getDdr3Addr(), kChainK, chainBias1);
  writeI8Tile(loadWeights[1].getDdr3Addr(), kChainK, kChainK, chainWeight2);
  writeI8Tile(loadWeights[2].getDdr3Addr(), kChainK, kChainK, chainWeight3);
  writeI32Bias(loadBiases[2].getDdr3Addr(), kChainK, chainBias3);

  simulator.run(*words);

  std::vector<uint8_t> l1 = simulator.read(stores[0].getDdr3Addr(), kChainRows * kChainK);
  for (int r = 0; r < kChainRows; ++r)
    for (int c = 0; c < kChainK; ++c)
      EXPECT_EQ(l1[r * kChainK + c], chainLayer1Out(r, c)) << "L1 row " << r << " c " << c;

  std::vector<uint8_t> l2 = simulator.read(stores[1].getDdr3Addr(), kChainRows * kChainK);
  for (int r = 0; r < kChainRows; ++r)
    for (int c = 0; c < kChainK; ++c)
      EXPECT_EQ(l2[r * kChainK + c], chainLayer2Out(r, c)) << "L2 row " << r << " c " << c;

  std::vector<uint8_t> result =
      simulator.read(stores[2].getDdr3Addr(), kChainRows * kChainK);
  ASSERT_EQ(result.size(), static_cast<size_t>(kChainRows * kChainK));
  for (int r = 0; r < kChainRows; ++r) {
    for (int c = 0; c < kChainK; ++c) {
      EXPECT_EQ(result[r * kChainK + c], chainLayer3Out(r, c))
          << "row " << r << " channel " << c;
    }
  }
}

// N-tiled producer feeding a K-tiled consumer - the real-world case this
// generalization exists for (a hidden layer wider than 14 channels chaining
// into the next layer). Layer 1: K=14, N=28 (2 N-tiles). Layer 2: K=28
// (matches layer 1's real N, so 2 K-tiles), N=14, final output.
namespace {

constexpr int kChainNKRows = 2;
constexpr int64_t kChainNKN1 = 28;  // layer 1's N (2 N-tiles)
constexpr int64_t kChainNKK2 = 28;  // layer 2's K (== layer 1's N)
constexpr int64_t kChainNKN2 = 14;  // layer 2's N (single tile)

int8_t chainNKAct0(int r, int k) { return static_cast<int8_t>((r * 7 + k) % 9 - 4); }
int8_t chainNKWeight1(int k, int c) { return static_cast<int8_t>((k + 2 * c) % 7 - 3); }
int32_t chainNKBias1(int c) { return c % 5 - 2; }
int8_t chainNKWeight2(int k, int c) { return static_cast<int8_t>((k - c) % 6); }
int32_t chainNKBias2(int c) { return 2 * c - 3; }

constexpr uint32_t kNK1Mult = 1, kNK1Shift = 0;
constexpr uint32_t kNK2Mult = 1, kNK2Shift = 0;

uint8_t chainNKRequantizePassthrough(int64_t acc, uint32_t m, uint32_t shift) {
  int64_t scaled = acc * static_cast<int64_t>(m);
  if (shift > 0) scaled += static_cast<int64_t>(1) << (shift - 1);
  int64_t requant = scaled >> shift;
  return static_cast<uint8_t>(std::clamp<int64_t>(requant, -128, 127));
}

uint8_t chainNKLayer1Out(int r, int c) {  // c in [0, 28)
  int64_t acc = chainNKBias1(c);
  for (int k = 0; k < 14; ++k)
    acc += static_cast<int64_t>(chainNKAct0(r, k)) *
           static_cast<int64_t>(chainNKWeight1(k, c));
  return chainNKRequantizePassthrough(acc, kNK1Mult, kNK1Shift);
}
int8_t chainNKLayer1OutSigned(int r, int c) {
  return static_cast<int8_t>(chainNKLayer1Out(r, c));
}

uint8_t chainNKLayer2Out(int r, int c) {  // c in [0, 14)
  int64_t acc = chainNKBias2(c);
  for (int k = 0; k < 28; ++k)
    acc += static_cast<int64_t>(chainNKLayer1OutSigned(r, k)) *
           static_cast<int64_t>(chainNKWeight2(k, c));
  return chainNKRequantizePassthrough(acc, kNK2Mult, kNK2Shift);
}

}  // namespace

TEST(ClosedLoop, NTiledChainIntoKTiledConsumerIsBitExact) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  auto act0Const = buildChainI8Tensor(builder, loc, kChainNKRows, 14, chainNKAct0);
  auto weight1Const = buildChainI8Tensor(builder, loc, 14, kChainNKN1, chainNKWeight1);
  auto bias1Const = buildChainI32Bias(builder, loc, kChainNKN1, chainNKBias1);
  auto weight2Const =
      buildChainI8Tensor(builder, loc, kChainNKK2, kChainNKN2, chainNKWeight2);
  auto bias2Const = buildChainI32Bias(builder, loc, kChainNKN2, chainNKBias2);

  auto rescale1 = buildChainLayerN(builder, loc, act0Const.getResult(), weight1Const,
                                   bias1Const, static_cast<int32_t>(kNK1Mult),
                                   static_cast<int8_t>(kNK1Shift));
  buildChainLayerN(builder, loc, rescale1.getResult(), weight2Const, bias2Const,
                   static_cast<int32_t>(kNK2Mult), static_cast<int8_t>(kNK2Shift));

  ASSERT_TRUE(succeeded(conversion::lowerTosaToMacaque(block)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<LoadBiasOp> loadBiases;
  SmallVector<LoadInputOp> loadInputs;
  SmallVector<StoreOp> stores;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<LoadBiasOp>(op)) loadBiases.push_back(o);
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
    if (auto o = dyn_cast<StoreOp>(op)) stores.push_back(o);
  }
  // Layer 1: 2 N-tiles (K=14 single-tile) -> 2 of each. Layer 2: 2 K-tiles
  // (N=14 single-tile) -> 2 weight/matmul, 1 bias (loaded once, at the
  // first K-tile), 2 chained load_inputs, 1 store (final output).
  ASSERT_EQ(loadWeights.size(), 4u);
  ASSERT_EQ(loadBiases.size(), 3u);
  ASSERT_EQ(loadInputs.size(), 4u);  // 2 layer 1 (fresh) + 2 layer 2 (chained)
  ASSERT_EQ(stores.size(), 3u);      // 2 layer 1 (scratch) + 1 layer 2 (output)

  auto words = target::emitBinary(block);
  ASSERT_TRUE(succeeded(words));

  sim::Simulator simulator(/*ddr_bytes=*/8192);

  auto writeI8Tile = [&](uint32_t addr, int rows, int cols, int8_t (*gen)(int, int)) {
    std::vector<uint8_t> bytes;
    bytes.reserve(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c) bytes.push_back(static_cast<uint8_t>(gen(r, c)));
    simulator.write(addr, bytes.data(), bytes.size());
  };
  auto writeI32Bias = [&](uint32_t addr, int cols, int32_t (*gen)(int)) {
    std::vector<uint8_t> bytes;
    bytes.reserve(static_cast<size_t>(cols) * 4);
    for (int c = 0; c < cols; ++c) {
      int32_t v = gen(c);
      for (int b = 0; b < 4; ++b) bytes.push_back(static_cast<uint8_t>((v >> (8 * b)) & 0xFF));
    }
    simulator.write(addr, bytes.data(), bytes.size());
  };

  // Layer 1's activation is fresh (one shared address, reloaded per N-tile -
  // both loadInputs[0]/[1] point at it). Layer 1's weight is a 14x28 tensor
  // split into 2 N-tiles' worth of 14x14 slices - each load_weight only
  // covers 14 columns, so stage each N-tile's real slice at its own
  // address, offset into layer 1's logical 14x28 weight/28-channel bias by
  // the N-tile's own column range.
  writeI8Tile(loadInputs[0].getDdr3Addr(), kChainNKRows, 14, chainNKAct0);
  for (int n = 0; n < 2; ++n) {
    std::vector<uint8_t> wBytes;
    for (int k = 0; k < 14; ++k)
      for (int c = 0; c < 14; ++c)
        wBytes.push_back(static_cast<uint8_t>(chainNKWeight1(k, n * 14 + c)));
    simulator.write(loadWeights[n].getDdr3Addr(), wBytes.data(), wBytes.size());

    std::vector<uint8_t> bBytes;
    for (int c = 0; c < 14; ++c) {
      int32_t v = chainNKBias1(n * 14 + c);
      for (int b = 0; b < 4; ++b) bBytes.push_back(static_cast<uint8_t>((v >> (8 * b)) & 0xFF));
    }
    simulator.write(loadBiases[n].getDdr3Addr(), bBytes.data(), bBytes.size());
  }
  // Layer 2's weight (28x14, split into 2 K-tiles of 14x14) and real bias
  // (14 channels, loaded once at the first K-tile).
  for (int k = 0; k < 2; ++k) {
    std::vector<uint8_t> wBytes;
    for (int kk = 0; kk < 14; ++kk)
      for (int c = 0; c < 14; ++c)
        wBytes.push_back(static_cast<uint8_t>(chainNKWeight2(k * 14 + kk, c)));
    simulator.write(loadWeights[2 + k].getDdr3Addr(), wBytes.data(), wBytes.size());
  }
  writeI32Bias(loadBiases[2].getDdr3Addr(), kChainNKN2, chainNKBias2);

  simulator.run(*words);

  // Layer 1's 2 N-tile stores, reassembled into the true 28-channel result.
  std::vector<uint8_t> l1n0 = simulator.read(stores[0].getDdr3Addr(), kChainNKRows * 14);
  std::vector<uint8_t> l1n1 = simulator.read(stores[1].getDdr3Addr(), kChainNKRows * 14);
  for (int r = 0; r < kChainNKRows; ++r) {
    for (int c = 0; c < 14; ++c) {
      EXPECT_EQ(l1n0[r * 14 + c], chainNKLayer1Out(r, c)) << "L1 N-tile0 row " << r << " c " << c;
      EXPECT_EQ(l1n1[r * 14 + c], chainNKLayer1Out(r, c + 14))
          << "L1 N-tile1 row " << r << " c " << c;
    }
  }

  std::vector<uint8_t> result = simulator.read(stores[2].getDdr3Addr(), kChainNKRows * kChainNKN2);
  ASSERT_EQ(result.size(), static_cast<size_t>(kChainNKRows * kChainNKN2));
  for (int r = 0; r < kChainNKRows; ++r) {
    for (int c = 0; c < kChainNKN2; ++c) {
      EXPECT_EQ(result[r * kChainNKN2 + c], chainNKLayer2Out(r, c))
          << "row " << r << " channel " << c;
    }
  }
}

// 3. K-tiling, whole-pipeline closed loop
//
// K=28 (2 tiles of 14) sourced from real TOSA IR, checked bit-exact - the
// actual correctness oracle for Milestone 3's K-tiling increment. Unlike
// test_tosa_to_macaque.cpp's KTilesMatmulIntoTwoAccumulatingGroups (which
// only checks the shape/addresses of the emitted IR), this writes real
// weight/activation/bias bytes into the simulator's DDR3 and checks the
// numeric result against an independent oracle - proving the two tiles
// actually accumulate to the right answer, not just that two tiles exist.

namespace {

constexpr int kKTileRows = 3;
constexpr int64_t kKTileK = 28;  // 2 tiles of 14

int8_t kTileAct(int r, int k) { return static_cast<int8_t>((r * 5 + k) % 11 - 5); }
int8_t kTileWeight(int k, int c) { return static_cast<int8_t>((k - c) % 7); }
int32_t kTileBias(int c) { return 3 * c - 10; }
constexpr uint32_t kKTileMult = 2, kKTileShift = 1;

uint8_t kTileRequantizePassthrough(int64_t acc, uint32_t m, uint32_t shift) {
  int64_t scaled = acc * static_cast<int64_t>(m);
  if (shift > 0) scaled += static_cast<int64_t>(1) << (shift - 1);  // round-half-up
  int64_t requant = scaled >> shift;
  return static_cast<uint8_t>(std::clamp<int64_t>(requant, -128, 127));
}

uint8_t kTileExpected(int r, int c) {
  int64_t acc = kTileBias(c);
  for (int64_t k = 0; k < kKTileK; ++k)
    acc += static_cast<int64_t>(kTileAct(r, static_cast<int>(k))) *
           static_cast<int64_t>(kTileWeight(static_cast<int>(k), c));
  return kTileRequantizePassthrough(acc, kKTileMult, kKTileShift);
}

}  // namespace

TEST(ClosedLoop, KTiledMatmulFromTosaIsBitExact) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  auto i8Ty = builder.getIntegerType(8);
  auto i32Ty = builder.getIntegerType(32);
  auto aTy = RankedTensorType::get({1, kKTileRows, kKTileK}, i8Ty);
  auto bTy = RankedTensorType::get({1, kKTileK, 14}, i8Ty);
  auto zpTy = RankedTensorType::get({1}, i8Ty);
  auto biasTy = RankedTensorType::get({1, 1, 14}, i32Ty);

  std::vector<int8_t> aData;
  for (int r = 0; r < kKTileRows; ++r)
    for (int64_t k = 0; k < kKTileK; ++k) aData.push_back(kTileAct(r, static_cast<int>(k)));
  auto a = tosa::ConstOp::create(builder, loc, aTy,
                                 DenseElementsAttr::get(aTy, ArrayRef<int8_t>(aData)));

  std::vector<int8_t> bData;
  for (int64_t k = 0; k < kKTileK; ++k)
    for (int c = 0; c < 14; ++c) bData.push_back(kTileWeight(static_cast<int>(k), c));
  auto b = tosa::ConstOp::create(builder, loc, bTy,
                                 DenseElementsAttr::get(bTy, ArrayRef<int8_t>(bData)));

  auto aZp = tosa::ConstOp::create(builder, loc, zpTy,
                                   DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto bZp = tosa::ConstOp::create(builder, loc, zpTy,
                                   DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto matmulOutTy = RankedTensorType::get({1, kKTileRows, 14}, i32Ty);
  auto matmul = tosa::MatMulOp::create(builder, loc, matmulOutTy, a.getResult(),
                                       b.getResult(), aZp.getResult(), bZp.getResult());

  std::vector<int32_t> biasData;
  for (int c = 0; c < 14; ++c) biasData.push_back(kTileBias(c));
  auto bias = tosa::ConstOp::create(builder, loc, biasTy,
                                    DenseElementsAttr::get(biasTy, ArrayRef<int32_t>(biasData)));
  auto add = tosa::AddOp::create(builder, loc, matmulOutTy, matmul.getResult(),
                                 bias.getResult());

  auto multiplierConst = tosa::ConstOp::create(
      builder, loc, zpTy.clone(i32Ty),
      DenseElementsAttr::get(RankedTensorType::get({1}, i32Ty),
                             static_cast<int32_t>(kKTileMult)));
  auto shiftConst = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(kKTileShift)));
  auto rescaleOutTy = RankedTensorType::get({1, kKTileRows, 14}, i8Ty);
  tosa::RescaleOp::create(builder, loc, rescaleOutTy, add.getResult(), multiplierConst.getResult(),
                          shiftConst.getResult(), aZp.getResult(), aZp.getResult(),
                          /*scale32=*/true, tosa::RoundingMode::SINGLE_ROUND,
                          /*perChannel=*/false, /*inputUnsigned=*/false,
                          /*outputUnsigned=*/false);

  ASSERT_TRUE(succeeded(conversion::lowerTosaToMacaque(block)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<LoadBiasOp> loadBiases;
  SmallVector<LoadInputOp> loadInputs;
  SmallVector<StoreOp> stores;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<LoadBiasOp>(op)) loadBiases.push_back(o);
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
    if (auto o = dyn_cast<StoreOp>(op)) stores.push_back(o);
  }
  ASSERT_EQ(loadWeights.size(), 2u);  // one per K-tile
  ASSERT_EQ(loadBiases.size(), 1u);   // bias loads once, before the first tile
  ASSERT_EQ(loadInputs.size(), 2u);   // one per K-tile
  ASSERT_EQ(stores.size(), 1u);       // one final result, not per-tile

  auto words = target::emitBinary(block);
  ASSERT_TRUE(succeeded(words));

  sim::Simulator simulator(/*ddr_bytes=*/8192);

  // Each K-tile's weight/activation is staged as its own contiguous
  // [rows,14] (or [14,14]) block at its own address - not as one
  // contiguous [rows,28]/[28,14] blob. This is the real-world implication
  // of K-tiling: the runtime must pre-split tiled tensors into per-tile
  // chunks before staging, the same kind of DDR3-layout convention as the
  // zero-bias slot.
  for (int tile = 0; tile < 2; ++tile) {
    std::vector<uint8_t> wBytes;
    for (int k = 0; k < 14; ++k)
      for (int c = 0; c < 14; ++c) wBytes.push_back(static_cast<uint8_t>(kTileWeight(tile * 14 + k, c)));
    simulator.write(loadWeights[tile].getDdr3Addr(), wBytes.data(), wBytes.size());

    std::vector<uint8_t> aBytes;
    for (int r = 0; r < kKTileRows; ++r)
      for (int k = 0; k < 14; ++k) aBytes.push_back(static_cast<uint8_t>(kTileAct(r, tile * 14 + k)));
    simulator.write(loadInputs[tile].getDdr3Addr(), aBytes.data(), aBytes.size());
  }

  std::vector<uint8_t> biasBytes;
  for (int c = 0; c < 14; ++c) {
    int32_t v = kTileBias(c);
    for (int bByte = 0; bByte < 4; ++bByte)
      biasBytes.push_back(static_cast<uint8_t>((v >> (8 * bByte)) & 0xFF));
  }
  simulator.write(loadBiases[0].getDdr3Addr(), biasBytes.data(), biasBytes.size());

  simulator.run(*words);

  std::vector<uint8_t> result = simulator.read(stores[0].getDdr3Addr(), kKTileRows * 14);
  ASSERT_EQ(result.size(), static_cast<size_t>(kKTileRows * 14));
  for (int r = 0; r < kKTileRows; ++r) {
    for (int c = 0; c < 14; ++c) {
      EXPECT_EQ(result[r * 14 + c], kTileExpected(r, c)) << "row " << r << " channel " << c;
    }
  }
}

//===----------------------------------------------------------------------===//
// 4. Partial K-tile zero-padding, whole-pipeline closed loop
//
// K=20 - one full 14-wide tile plus a 6-wide boundary tile - checked
// bit-exact. This is the actual correctness oracle for the zero-padding
// convention documented in sw/docs/MEMORY_LAYOUT.md: the boundary tile is
// staged as a full 14-wide block with only the first 6 rows real and the
// rest zero, and the oracle sums over the true K=20 (not 28) to prove that
// convention actually produces the right answer, not just the right shape.
//===----------------------------------------------------------------------===//

namespace {

constexpr int kPartialRows = 2;
constexpr int64_t kPartialK = 20;

int8_t partialAct(int r, int k) { return static_cast<int8_t>((r * 3 + k) % 9 - 4); }
int8_t partialWeight(int k, int c) { return static_cast<int8_t>((k - c) % 5); }
int32_t partialBias(int c) { return c - 3; }
constexpr uint32_t kPartialMult = 3, kPartialShift = 2;

uint8_t partialRequantizePassthrough(int64_t acc, uint32_t m, uint32_t shift) {
  int64_t scaled = acc * static_cast<int64_t>(m);
  if (shift > 0) scaled += static_cast<int64_t>(1) << (shift - 1);  // round-half-up
  int64_t requant = scaled >> shift;
  return static_cast<uint8_t>(std::clamp<int64_t>(requant, -128, 127));
}

uint8_t partialExpected(int r, int c) {
  int64_t acc = partialBias(c);
  for (int64_t k = 0; k < kPartialK; ++k)  // sums over the *real* K=20 only
    acc += static_cast<int64_t>(partialAct(r, static_cast<int>(k))) *
           static_cast<int64_t>(partialWeight(static_cast<int>(k), c));
  return partialRequantizePassthrough(acc, kPartialMult, kPartialShift);
}

}  // namespace

TEST(ClosedLoop, PartialKTileZeroPadsCorrectly) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  auto i8Ty = builder.getIntegerType(8);
  auto i32Ty = builder.getIntegerType(32);
  auto aTy = RankedTensorType::get({1, kPartialRows, kPartialK}, i8Ty);
  auto bTy = RankedTensorType::get({1, kPartialK, 14}, i8Ty);
  auto zpTy = RankedTensorType::get({1}, i8Ty);
  auto biasTy = RankedTensorType::get({1, 1, 14}, i32Ty);

  std::vector<int8_t> aData;
  for (int r = 0; r < kPartialRows; ++r)
    for (int64_t k = 0; k < kPartialK; ++k) aData.push_back(partialAct(r, static_cast<int>(k)));
  auto a = tosa::ConstOp::create(builder, loc, aTy,
                                 DenseElementsAttr::get(aTy, ArrayRef<int8_t>(aData)));

  std::vector<int8_t> bData;
  for (int64_t k = 0; k < kPartialK; ++k)
    for (int c = 0; c < 14; ++c) bData.push_back(partialWeight(static_cast<int>(k), c));
  auto b = tosa::ConstOp::create(builder, loc, bTy,
                                 DenseElementsAttr::get(bTy, ArrayRef<int8_t>(bData)));

  auto aZp = tosa::ConstOp::create(builder, loc, zpTy,
                                   DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto bZp = tosa::ConstOp::create(builder, loc, zpTy,
                                   DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto matmulOutTy = RankedTensorType::get({1, kPartialRows, 14}, i32Ty);
  auto matmul = tosa::MatMulOp::create(builder, loc, matmulOutTy, a.getResult(), b.getResult(),
                                       aZp.getResult(), bZp.getResult());

  std::vector<int32_t> biasData;
  for (int c = 0; c < 14; ++c) biasData.push_back(partialBias(c));
  auto bias = tosa::ConstOp::create(builder, loc, biasTy,
                                    DenseElementsAttr::get(biasTy, ArrayRef<int32_t>(biasData)));
  auto add = tosa::AddOp::create(builder, loc, matmulOutTy, matmul.getResult(), bias.getResult());

  auto multiplierConst = tosa::ConstOp::create(
      builder, loc, RankedTensorType::get({1}, i32Ty),
      DenseElementsAttr::get(RankedTensorType::get({1}, i32Ty),
                             static_cast<int32_t>(kPartialMult)));
  auto shiftConst = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(kPartialShift)));
  auto rescaleOutTy = RankedTensorType::get({1, kPartialRows, 14}, i8Ty);
  tosa::RescaleOp::create(builder, loc, rescaleOutTy, add.getResult(), multiplierConst.getResult(),
                          shiftConst.getResult(), aZp.getResult(), aZp.getResult(),
                          /*scale32=*/true, tosa::RoundingMode::SINGLE_ROUND,
                          /*perChannel=*/false, /*inputUnsigned=*/false,
                          /*outputUnsigned=*/false);

  ASSERT_TRUE(succeeded(conversion::lowerTosaToMacaque(block)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<LoadBiasOp> loadBiases;
  SmallVector<LoadInputOp> loadInputs;
  SmallVector<StoreOp> stores;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<LoadBiasOp>(op)) loadBiases.push_back(o);
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
    if (auto o = dyn_cast<StoreOp>(op)) stores.push_back(o);
  }
  ASSERT_EQ(loadWeights.size(), 2u);
  ASSERT_EQ(loadBiases.size(), 1u);
  ASSERT_EQ(loadInputs.size(), 2u);
  ASSERT_EQ(stores.size(), 1u);

  auto words = target::emitBinary(block);
  ASSERT_TRUE(succeeded(words));

  sim::Simulator simulator(/*ddr_bytes=*/8192);

  // Tile 0: a full, real 14-wide K-slice (global K 0..13).
  std::vector<uint8_t> w0;
  for (int k = 0; k < 14; ++k)
    for (int c = 0; c < 14; ++c) w0.push_back(static_cast<uint8_t>(partialWeight(k, c)));
  simulator.write(loadWeights[0].getDdr3Addr(), w0.data(), w0.size());

  std::vector<uint8_t> a0;
  for (int r = 0; r < kPartialRows; ++r)
    for (int k = 0; k < 14; ++k) a0.push_back(static_cast<uint8_t>(partialAct(r, k)));
  simulator.write(loadInputs[0].getDdr3Addr(), a0.data(), a0.size());

  // Tile 1: the boundary tile - local rows 0..5 are real (global K 14..19),
  // local rows 6..13 are the zero-padding convention in action. Both the
  // weight's padding rows AND the activation's padding columns are zeroed
  // (either alone would suffice numerically, since the product is zero if
  // either factor is, but zeroing both avoids relying on that subtlety).
  std::vector<uint8_t> w1;
  for (int localK = 0; localK < 14; ++localK) {
    const int globalK = 14 + localK;
    for (int c = 0; c < 14; ++c) {
      const int8_t v = globalK < kPartialK ? partialWeight(globalK, c) : 0;
      w1.push_back(static_cast<uint8_t>(v));
    }
  }
  simulator.write(loadWeights[1].getDdr3Addr(), w1.data(), w1.size());

  std::vector<uint8_t> a1;
  for (int r = 0; r < kPartialRows; ++r) {
    for (int localK = 0; localK < 14; ++localK) {
      const int globalK = 14 + localK;
      const int8_t v = globalK < kPartialK ? partialAct(r, globalK) : 0;
      a1.push_back(static_cast<uint8_t>(v));
    }
  }
  simulator.write(loadInputs[1].getDdr3Addr(), a1.data(), a1.size());

  std::vector<uint8_t> biasBytes;
  for (int c = 0; c < 14; ++c) {
    int32_t v = partialBias(c);
    for (int byteIdx = 0; byteIdx < 4; ++byteIdx)
      biasBytes.push_back(static_cast<uint8_t>((v >> (8 * byteIdx)) & 0xFF));
  }
  simulator.write(loadBiases[0].getDdr3Addr(), biasBytes.data(), biasBytes.size());

  simulator.run(*words);

  std::vector<uint8_t> result = simulator.read(stores[0].getDdr3Addr(), kPartialRows * 14);
  ASSERT_EQ(result.size(), static_cast<size_t>(kPartialRows * 14));
  for (int r = 0; r < kPartialRows; ++r) {
    for (int c = 0; c < 14; ++c) {
      EXPECT_EQ(result[r * 14 + c], partialExpected(r, c)) << "row " << r << " channel " << c;
    }
  }
}

//===----------------------------------------------------------------------===//
// 4b. Partial N-tile, whole-pipeline closed loop, driven by the compiler's
// own DataSegment output instead of hand-constructed bytes.
//
// Test does not hand write the zero-padded bytes, but check if
// staged into DDR3 as part of the DataSegment output
//===----------------------------------------------------------------------===//

namespace {

constexpr int kPartialNRows = 2;
constexpr int64_t kPartialN = 20;

int8_t partialNAct(int r, int k) { return static_cast<int8_t>((r * 7 + k) % 9 - 4); }
int8_t partialNWeight(int k, int c) { return static_cast<int8_t>((k + 2 * c) % 5 - 2); }
int32_t partialNBias(int c) { return c - 5; }
constexpr uint32_t kPartialNMult = 2, kPartialNShift = 1;

uint8_t partialNExpected(int r, int c) {  // c ranges over the true N=20
  int64_t acc = partialNBias(c);
  for (int k = 0; k < 14; ++k)
    acc += static_cast<int64_t>(partialNAct(r, k)) * static_cast<int64_t>(partialNWeight(k, c));
  int64_t scaled = acc * static_cast<int64_t>(kPartialNMult);
  scaled += static_cast<int64_t>(1) << (kPartialNShift - 1);  // round-half-up
  int64_t requant = scaled >> kPartialNShift;
  return static_cast<uint8_t>(std::clamp<int64_t>(requant, -128, 127));
}

}  // namespace

TEST(ClosedLoop, PartialNTileZeroPadsCorrectly) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  auto i8Ty = builder.getIntegerType(8);
  auto i32Ty = builder.getIntegerType(32);
  auto aTy = RankedTensorType::get({1, kPartialNRows, 14}, i8Ty);
  auto bTy = RankedTensorType::get({1, 14, kPartialN}, i8Ty);
  auto zpTy = RankedTensorType::get({1}, i8Ty);
  auto biasTy = RankedTensorType::get({1, 1, kPartialN}, i32Ty);

  std::vector<int8_t> aData;
  for (int r = 0; r < kPartialNRows; ++r)
    for (int k = 0; k < 14; ++k) aData.push_back(partialNAct(r, k));
  auto a = tosa::ConstOp::create(builder, loc, aTy,
                                 DenseElementsAttr::get(aTy, ArrayRef<int8_t>(aData)));

  std::vector<int8_t> bData;
  for (int k = 0; k < 14; ++k)
    for (int64_t c = 0; c < kPartialN; ++c) bData.push_back(partialNWeight(k, static_cast<int>(c)));
  auto b = tosa::ConstOp::create(builder, loc, bTy,
                                 DenseElementsAttr::get(bTy, ArrayRef<int8_t>(bData)));

  auto aZp = tosa::ConstOp::create(builder, loc, zpTy,
                                   DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto bZp = tosa::ConstOp::create(builder, loc, zpTy,
                                   DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto matmulOutTy = RankedTensorType::get({1, kPartialNRows, kPartialN}, i32Ty);
  auto matmul = tosa::MatMulOp::create(builder, loc, matmulOutTy, a.getResult(), b.getResult(),
                                       aZp.getResult(), bZp.getResult());

  std::vector<int32_t> biasData;
  for (int64_t c = 0; c < kPartialN; ++c) biasData.push_back(partialNBias(static_cast<int>(c)));
  auto bias = tosa::ConstOp::create(builder, loc, biasTy,
                                    DenseElementsAttr::get(biasTy, ArrayRef<int32_t>(biasData)));
  auto add = tosa::AddOp::create(builder, loc, matmulOutTy, matmul.getResult(), bias.getResult());

  auto multiplierConst = tosa::ConstOp::create(
      builder, loc, RankedTensorType::get({1}, i32Ty),
      DenseElementsAttr::get(RankedTensorType::get({1}, i32Ty),
                             static_cast<int32_t>(kPartialNMult)));
  auto shiftConst = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(kPartialNShift)));
  auto rescaleOutTy = RankedTensorType::get({1, kPartialNRows, kPartialN}, i8Ty);
  tosa::RescaleOp::create(builder, loc, rescaleOutTy, add.getResult(), multiplierConst.getResult(),
                          shiftConst.getResult(), aZp.getResult(), aZp.getResult(),
                          /*scale32=*/true, tosa::RoundingMode::SINGLE_ROUND,
                          /*perChannel=*/false, /*inputUnsigned=*/false,
                          /*outputUnsigned=*/false);

  conversion::CompiledProgramInfo info;
  ASSERT_TRUE(succeeded(conversion::lowerTosaToMacaque(block, &info)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<LoadBiasOp> loadBiases;
  SmallVector<LoadInputOp> loadInputs;
  SmallVector<StoreOp> stores;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<LoadBiasOp>(op)) loadBiases.push_back(o);
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
    if (auto o = dyn_cast<StoreOp>(op)) stores.push_back(o);
  }
  ASSERT_EQ(loadWeights.size(), 2u);  // ceil(20/14) = 2 N-tiles
  ASSERT_EQ(loadBiases.size(), 2u);
  ASSERT_EQ(loadInputs.size(), 2u);   // reloaded once per N-tile
  ASSERT_EQ(stores.size(), 2u);

  auto words = target::emitBinary(block);
  ASSERT_TRUE(succeeded(words));

  sim::Simulator simulator(/*ddr_bytes=*/8192);

  // No hand-constructed bytes here - every weight/bias/activation address is
  // staged straight from the compiler's own DataSegment output.
  for (const auto& [addr, bytes] : info.data)
    simulator.write(addr, bytes.data(), bytes.size());

  simulator.run(*words);

  // N-tile 0: channels 0-13, fully real.
  std::vector<uint8_t> result0 =
      simulator.read(stores[0].getDdr3Addr(), kPartialNRows * 14);
  ASSERT_EQ(result0.size(), static_cast<size_t>(kPartialNRows * 14));
  for (int r = 0; r < kPartialNRows; ++r)
    for (int c = 0; c < 14; ++c)
      EXPECT_EQ(result0[r * 14 + c], partialNExpected(r, c)) << "row " << r << " channel " << c;

  // N-tile 1 (the boundary tile): local channels 0-5 (global 14-19) are
  // real; the compiler pads the rest of this N-tile's weight/bias with
  // zeros, but the *output* itself is never padded - the store here is
  // still only 14 columns wide per the instruction shape (ACTIVATE/STORE
  // don't know the true N=20 bound), so only the real local channels are
  // checked against a reference.
  std::vector<uint8_t> result1 =
      simulator.read(stores[1].getDdr3Addr(), kPartialNRows * 14);
  ASSERT_EQ(result1.size(), static_cast<size_t>(kPartialNRows * 14));
  for (int r = 0; r < kPartialNRows; ++r) {
    for (int localC = 0; localC < 6; ++localC) {
      const int c = 14 + localC;
      EXPECT_EQ(result1[r * 14 + localC], partialNExpected(r, c))
          << "row " << r << " channel " << c;
    }
  }
}

//===----------------------------------------------------------------------===//
// 5. N-tiling, whole-pipeline closed loop
//
// N=28 - two fully independent 14-wide output-channel tiles - checked
// bit-exact. Unlike K-tiling, N-tiles don't accumulate: this is the oracle
// proving the two tiles' stores correctly reassemble into the true N=28
// result, each channel computed from its own weight/bias slice and the
// *same*, shared activation data (reloaded once per N-tile, not re-derived
// or split).
//===----------------------------------------------------------------------===//

namespace {

constexpr int kNTileRows = 2;
constexpr int64_t kNTileN = 28;

int8_t nTileAct(int r, int k) { return static_cast<int8_t>((r * 5 + k) % 9 - 4); }
int8_t nTileWeight(int k, int c) { return static_cast<int8_t>((k + c) % 7 - 3); }
int32_t nTileBias(int c) { return 2 * c - 15; }
constexpr uint32_t kNTileMult = 1, kNTileShift = 0;

uint8_t nTileRequantizePassthrough(int64_t acc, uint32_t m, uint32_t shift) {
  int64_t scaled = acc * static_cast<int64_t>(m);
  if (shift > 0) scaled += static_cast<int64_t>(1) << (shift - 1);
  int64_t requant = scaled >> shift;
  return static_cast<uint8_t>(std::clamp<int64_t>(requant, -128, 127));
}

uint8_t nTileExpected(int r, int c) {  // c ranges over the true N=28
  int64_t acc = nTileBias(c);
  for (int k = 0; k < 14; ++k)
    acc += static_cast<int64_t>(nTileAct(r, k)) * static_cast<int64_t>(nTileWeight(k, c));
  return nTileRequantizePassthrough(acc, kNTileMult, kNTileShift);
}

}  // namespace

TEST(ClosedLoop, NTiledMatmulFromTosaIsBitExact) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  auto i8Ty = builder.getIntegerType(8);
  auto i32Ty = builder.getIntegerType(32);
  auto aTy = RankedTensorType::get({1, kNTileRows, 14}, i8Ty);
  auto bTy = RankedTensorType::get({1, 14, kNTileN}, i8Ty);
  auto zpTy = RankedTensorType::get({1}, i8Ty);
  auto biasTy = RankedTensorType::get({1, 1, kNTileN}, i32Ty);

  std::vector<int8_t> aData;
  for (int r = 0; r < kNTileRows; ++r)
    for (int k = 0; k < 14; ++k) aData.push_back(nTileAct(r, k));
  auto a = tosa::ConstOp::create(builder, loc, aTy,
                                 DenseElementsAttr::get(aTy, ArrayRef<int8_t>(aData)));

  std::vector<int8_t> bData;
  for (int k = 0; k < 14; ++k)
    for (int64_t c = 0; c < kNTileN; ++c) bData.push_back(nTileWeight(k, static_cast<int>(c)));
  auto b = tosa::ConstOp::create(builder, loc, bTy,
                                 DenseElementsAttr::get(bTy, ArrayRef<int8_t>(bData)));

  auto aZp = tosa::ConstOp::create(builder, loc, zpTy,
                                   DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto bZp = tosa::ConstOp::create(builder, loc, zpTy,
                                   DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto matmulOutTy = RankedTensorType::get({1, kNTileRows, kNTileN}, i32Ty);
  auto matmul = tosa::MatMulOp::create(builder, loc, matmulOutTy, a.getResult(), b.getResult(),
                                       aZp.getResult(), bZp.getResult());

  std::vector<int32_t> biasData;
  for (int64_t c = 0; c < kNTileN; ++c) biasData.push_back(nTileBias(static_cast<int>(c)));
  auto bias = tosa::ConstOp::create(builder, loc, biasTy,
                                    DenseElementsAttr::get(biasTy, ArrayRef<int32_t>(biasData)));
  auto add = tosa::AddOp::create(builder, loc, matmulOutTy, matmul.getResult(), bias.getResult());

  auto multiplierConst = tosa::ConstOp::create(
      builder, loc, RankedTensorType::get({1}, i32Ty),
      DenseElementsAttr::get(RankedTensorType::get({1}, i32Ty),
                             static_cast<int32_t>(kNTileMult)));
  auto shiftConst = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(kNTileShift)));
  auto rescaleOutTy = RankedTensorType::get({1, kNTileRows, kNTileN}, i8Ty);
  tosa::RescaleOp::create(builder, loc, rescaleOutTy, add.getResult(), multiplierConst.getResult(),
                          shiftConst.getResult(), aZp.getResult(), aZp.getResult(),
                          /*scale32=*/true, tosa::RoundingMode::SINGLE_ROUND,
                          /*perChannel=*/false, /*inputUnsigned=*/false,
                          /*outputUnsigned=*/false);

  ASSERT_TRUE(succeeded(conversion::lowerTosaToMacaque(block)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<LoadBiasOp> loadBiases;
  SmallVector<LoadInputOp> loadInputs;
  SmallVector<StoreOp> stores;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<LoadBiasOp>(op)) loadBiases.push_back(o);
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
    if (auto o = dyn_cast<StoreOp>(op)) stores.push_back(o);
  }
  ASSERT_EQ(loadWeights.size(), 2u);  // one per N-tile
  ASSERT_EQ(loadBiases.size(), 2u);   // one per N-tile
  ASSERT_EQ(loadInputs.size(), 2u);   // reloaded once per N-tile
  ASSERT_EQ(stores.size(), 2u);       // one per N-tile

  auto words = target::emitBinary(block);
  ASSERT_TRUE(succeeded(words));

  sim::Simulator simulator(/*ddr_bytes=*/8192);

  // Same activation data staged at both reload addresses - N-tiling doesn't
  // change which activation rows are needed, only which output columns.
  std::vector<uint8_t> actBytes;
  for (int r = 0; r < kNTileRows; ++r)
    for (int k = 0; k < 14; ++k) actBytes.push_back(static_cast<uint8_t>(nTileAct(r, k)));
  simulator.write(loadInputs[0].getDdr3Addr(), actBytes.data(), actBytes.size());
  simulator.write(loadInputs[1].getDdr3Addr(), actBytes.data(), actBytes.size());

  // N-tile 0: weight/bias for output channels 0..13. N-tile 1: channels
  // 14..27. Each tile's weight columns are local 0..13, mapping to global
  // channel `tile*14 + localC`.
  for (int tile = 0; tile < 2; ++tile) {
    std::vector<uint8_t> wBytes;
    for (int k = 0; k < 14; ++k)
      for (int localC = 0; localC < 14; ++localC)
        wBytes.push_back(static_cast<uint8_t>(nTileWeight(k, tile * 14 + localC)));
    simulator.write(loadWeights[tile].getDdr3Addr(), wBytes.data(), wBytes.size());

    std::vector<uint8_t> biasBytes;
    for (int localC = 0; localC < 14; ++localC) {
      int32_t v = nTileBias(tile * 14 + localC);
      for (int byteIdx = 0; byteIdx < 4; ++byteIdx)
        biasBytes.push_back(static_cast<uint8_t>((v >> (8 * byteIdx)) & 0xFF));
    }
    simulator.write(loadBiases[tile].getDdr3Addr(), biasBytes.data(), biasBytes.size());
  }

  simulator.run(*words);

  for (int tile = 0; tile < 2; ++tile) {
    std::vector<uint8_t> result = simulator.read(stores[tile].getDdr3Addr(), kNTileRows * 14);
    ASSERT_EQ(result.size(), static_cast<size_t>(kNTileRows * 14));
    for (int r = 0; r < kNTileRows; ++r) {
      for (int localC = 0; localC < 14; ++localC) {
        EXPECT_EQ(result[r * 14 + localC], nTileExpected(r, tile * 14 + localC))
            << "tile " << tile << " row " << r << " local channel " << localC;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// 6. M-chunking + weight-hold, whole-pipeline closed loop
//
// rows=270 (252 + 18) - two M-chunks, K=14 single-tile so weight-hold
// applies: only the first chunk's weight/bias is loaded, the second's
// matmul sets weight_hold and reuses them. This is the numeric oracle for
// weight-stationary M-streaming - proves the held chunk's matmul actually
// produces the right numbers using the *reused* weight/bias, not just that
// it skips the reload instructions (test_tosa_to_macaque.cpp's
// MChunksRescaleIntoTwoHeldGroupsWithSeparateStores only checks shape).
//===----------------------------------------------------------------------===//

namespace {

constexpr int64_t kHoldRows = 270;

int8_t holdAct(int r, int k) { return static_cast<int8_t>((r * 7 + k) % 13 - 6); }
int8_t holdWeight(int k, int c) { return static_cast<int8_t>((k - c) % 5); }
int32_t holdBias(int c) { return c - 4; }
constexpr uint32_t kHoldMult = 3, kHoldShift = 2;

uint8_t holdRequantizePassthrough(int64_t acc, uint32_t m, uint32_t shift) {
  int64_t scaled = acc * static_cast<int64_t>(m);
  if (shift > 0) scaled += static_cast<int64_t>(1) << (shift - 1);  // round-half-up
  int64_t requant = scaled >> shift;
  return static_cast<uint8_t>(std::clamp<int64_t>(requant, -128, 127));
}

uint8_t holdExpected(int r, int c) {
  int64_t acc = holdBias(c);
  for (int k = 0; k < 14; ++k)
    acc += static_cast<int64_t>(holdAct(r, k)) * static_cast<int64_t>(holdWeight(k, c));
  return holdRequantizePassthrough(acc, kHoldMult, kHoldShift);
}

}  // namespace

TEST(ClosedLoop, MChunkedWeightHoldMatmulFromTosaIsBitExact) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  auto i8Ty = builder.getIntegerType(8);
  auto i32Ty = builder.getIntegerType(32);
  auto aTy = RankedTensorType::get({1, kHoldRows, 14}, i8Ty);
  auto bTy = RankedTensorType::get({1, 14, 14}, i8Ty);
  auto zpTy = RankedTensorType::get({1}, i8Ty);
  auto biasTy = RankedTensorType::get({1, 1, 14}, i32Ty);

  std::vector<int8_t> aData;
  for (int64_t r = 0; r < kHoldRows; ++r)
    for (int k = 0; k < 14; ++k) aData.push_back(holdAct(static_cast<int>(r), k));
  auto a = tosa::ConstOp::create(builder, loc, aTy,
                                 DenseElementsAttr::get(aTy, ArrayRef<int8_t>(aData)));

  std::vector<int8_t> bData;
  for (int k = 0; k < 14; ++k)
    for (int c = 0; c < 14; ++c) bData.push_back(holdWeight(k, c));
  auto b = tosa::ConstOp::create(builder, loc, bTy,
                                 DenseElementsAttr::get(bTy, ArrayRef<int8_t>(bData)));

  auto aZp = tosa::ConstOp::create(builder, loc, zpTy,
                                   DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto bZp = tosa::ConstOp::create(builder, loc, zpTy,
                                   DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto matmulOutTy = RankedTensorType::get({1, kHoldRows, 14}, i32Ty);
  auto matmul = tosa::MatMulOp::create(builder, loc, matmulOutTy, a.getResult(), b.getResult(),
                                       aZp.getResult(), bZp.getResult());

  std::vector<int32_t> biasData;
  for (int c = 0; c < 14; ++c) biasData.push_back(holdBias(c));
  auto bias = tosa::ConstOp::create(builder, loc, biasTy,
                                    DenseElementsAttr::get(biasTy, ArrayRef<int32_t>(biasData)));
  auto add = tosa::AddOp::create(builder, loc, matmulOutTy, matmul.getResult(), bias.getResult());

  auto multiplierConst = tosa::ConstOp::create(
      builder, loc, RankedTensorType::get({1}, i32Ty),
      DenseElementsAttr::get(RankedTensorType::get({1}, i32Ty),
                             static_cast<int32_t>(kHoldMult)));
  auto shiftConst = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(kHoldShift)));
  auto rescaleOutTy = RankedTensorType::get({1, kHoldRows, 14}, i8Ty);
  tosa::RescaleOp::create(builder, loc, rescaleOutTy, add.getResult(), multiplierConst.getResult(),
                          shiftConst.getResult(), aZp.getResult(), aZp.getResult(),
                          /*scale32=*/true, tosa::RoundingMode::SINGLE_ROUND,
                          /*perChannel=*/false, /*inputUnsigned=*/false,
                          /*outputUnsigned=*/false);

  ASSERT_TRUE(succeeded(conversion::lowerTosaToMacaque(block)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<LoadBiasOp> loadBiases;
  SmallVector<LoadInputOp> loadInputs;
  SmallVector<MatmulOp> matmuls;
  SmallVector<StoreOp> stores;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<LoadBiasOp>(op)) loadBiases.push_back(o);
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
    if (auto o = dyn_cast<MatmulOp>(op)) matmuls.push_back(o);
    if (auto o = dyn_cast<StoreOp>(op)) stores.push_back(o);
  }
  // ceil(270/252) = 2 M-chunks; K=14 is single-tile so weight-hold applies -
  // only the first chunk reloads weight/bias.
  ASSERT_EQ(loadWeights.size(), 1u);
  ASSERT_EQ(loadBiases.size(), 1u);
  ASSERT_EQ(loadInputs.size(), 2u);
  ASSERT_EQ(matmuls.size(), 2u);
  ASSERT_EQ(stores.size(), 2u);
  EXPECT_EQ(matmuls[0].getWeightHold(), false);
  EXPECT_EQ(matmuls[1].getWeightHold(), true);

  auto words = target::emitBinary(block);
  ASSERT_TRUE(succeeded(words));

  sim::Simulator simulator(/*ddr_bytes=*/65536);

  std::vector<uint8_t> wBytes;
  for (int k = 0; k < 14; ++k)
    for (int c = 0; c < 14; ++c) wBytes.push_back(static_cast<uint8_t>(holdWeight(k, c)));
  simulator.write(loadWeights[0].getDdr3Addr(), wBytes.data(), wBytes.size());

  std::vector<uint8_t> biasBytes;
  for (int c = 0; c < 14; ++c) {
    int32_t v = holdBias(c);
    for (int byteIdx = 0; byteIdx < 4; ++byteIdx)
      biasBytes.push_back(static_cast<uint8_t>((v >> (8 * byteIdx)) & 0xFF));
  }
  simulator.write(loadBiases[0].getDdr3Addr(), biasBytes.data(), biasBytes.size());

  // Chunk 0: rows 0..251 (252 rows). Chunk 1: rows 252..269 (18 rows) - each
  // chunk's own activation data at its own load_input address, staged
  // separately since the row ranges differ. Weight/bias are staged only
  // once above - the held chunk's matmul must reuse that same DDR3 data via
  // the RTL's weight-stationary bank reuse, not a fresh load.
  const int64_t chunkRowCounts[2] = {252, 18};
  int64_t rowOffset = 0;
  for (int chunk = 0; chunk < 2; ++chunk) {
    std::vector<uint8_t> aBytes;
    for (int64_t r = 0; r < chunkRowCounts[chunk]; ++r)
      for (int k = 0; k < 14; ++k)
        aBytes.push_back(static_cast<uint8_t>(holdAct(static_cast<int>(rowOffset + r), k)));
    simulator.write(loadInputs[chunk].getDdr3Addr(), aBytes.data(), aBytes.size());
    rowOffset += chunkRowCounts[chunk];
  }

  simulator.run(*words);

  rowOffset = 0;
  for (int chunk = 0; chunk < 2; ++chunk) {
    std::vector<uint8_t> result =
        simulator.read(stores[chunk].getDdr3Addr(), chunkRowCounts[chunk] * 14);
    ASSERT_EQ(result.size(), static_cast<size_t>(chunkRowCounts[chunk] * 14));
    for (int64_t r = 0; r < chunkRowCounts[chunk]; ++r) {
      for (int c = 0; c < 14; ++c) {
        EXPECT_EQ(result[r * 14 + c], holdExpected(static_cast<int>(rowOffset + r), c))
            << "chunk " << chunk << " row " << r << " channel " << c;
      }
    }
    rowOffset += chunkRowCounts[chunk];
  }
}

//===----------------------------------------------------------------------===//
// 7. Weight-hold combined with K-tiling, whole-pipeline closed loop
//
// rows=30 (3 14-row hold-batch chunks: 14+14+2), K=28 (2 K-tiles) - the
// numeric oracle for emitBatchMatmuls: proves weight reloaded once per
// K-tile (not once per chunk) and held across chunks still produces the
// right numbers for every chunk, at every K-tile, including the correct
// accumulation across K-tiles via each chunk's own mat_row_base.
//===----------------------------------------------------------------------===//

namespace {

constexpr int64_t kHKRows = 30;
constexpr int64_t kHKK = 28;

int8_t hkAct(int r, int k) { return static_cast<int8_t>((r * 3 + k) % 11 - 5); }
int8_t hkWeight(int k, int c) { return static_cast<int8_t>((k - c) % 7); }
int32_t hkBias(int c) { return 2 * c - 5; }
constexpr uint32_t kHKMult = 2, kHKShift = 1;

uint8_t hkRequantizePassthrough(int64_t acc, uint32_t m, uint32_t shift) {
  int64_t scaled = acc * static_cast<int64_t>(m);
  if (shift > 0) scaled += static_cast<int64_t>(1) << (shift - 1);  // round-half-up
  int64_t requant = scaled >> shift;
  return static_cast<uint8_t>(std::clamp<int64_t>(requant, -128, 127));
}

uint8_t hkExpected(int r, int c) {
  int64_t acc = hkBias(c);
  for (int64_t k = 0; k < kHKK; ++k)
    acc += static_cast<int64_t>(hkAct(r, static_cast<int>(k))) *
           static_cast<int64_t>(hkWeight(static_cast<int>(k), c));
  return hkRequantizePassthrough(acc, kHKMult, kHKShift);
}

}  // namespace

TEST(ClosedLoop, WeightHoldWithKTilingFromTosaIsBitExact) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  auto i8Ty = builder.getIntegerType(8);
  auto i32Ty = builder.getIntegerType(32);
  auto aTy = RankedTensorType::get({1, kHKRows, kHKK}, i8Ty);
  auto bTy = RankedTensorType::get({1, kHKK, 14}, i8Ty);
  auto zpTy = RankedTensorType::get({1}, i8Ty);
  auto biasTy = RankedTensorType::get({1, 1, 14}, i32Ty);

  std::vector<int8_t> aData;
  for (int64_t r = 0; r < kHKRows; ++r)
    for (int64_t k = 0; k < kHKK; ++k)
      aData.push_back(hkAct(static_cast<int>(r), static_cast<int>(k)));
  auto a = tosa::ConstOp::create(builder, loc, aTy,
                                 DenseElementsAttr::get(aTy, ArrayRef<int8_t>(aData)));

  std::vector<int8_t> bData;
  for (int64_t k = 0; k < kHKK; ++k)
    for (int c = 0; c < 14; ++c) bData.push_back(hkWeight(static_cast<int>(k), c));
  auto b = tosa::ConstOp::create(builder, loc, bTy,
                                 DenseElementsAttr::get(bTy, ArrayRef<int8_t>(bData)));

  auto aZp = tosa::ConstOp::create(builder, loc, zpTy,
                                   DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto bZp = tosa::ConstOp::create(builder, loc, zpTy,
                                   DenseElementsAttr::get(zpTy, static_cast<int8_t>(0)));
  auto matmulOutTy = RankedTensorType::get({1, kHKRows, 14}, i32Ty);
  auto matmul = tosa::MatMulOp::create(builder, loc, matmulOutTy, a.getResult(), b.getResult(),
                                       aZp.getResult(), bZp.getResult());

  std::vector<int32_t> biasData;
  for (int c = 0; c < 14; ++c) biasData.push_back(hkBias(c));
  auto bias = tosa::ConstOp::create(builder, loc, biasTy,
                                    DenseElementsAttr::get(biasTy, ArrayRef<int32_t>(biasData)));
  auto add = tosa::AddOp::create(builder, loc, matmulOutTy, matmul.getResult(), bias.getResult());

  auto multiplierConst = tosa::ConstOp::create(
      builder, loc, RankedTensorType::get({1}, i32Ty),
      DenseElementsAttr::get(RankedTensorType::get({1}, i32Ty),
                             static_cast<int32_t>(kHKMult)));
  auto shiftConst = tosa::ConstOp::create(
      builder, loc, zpTy, DenseElementsAttr::get(zpTy, static_cast<int8_t>(kHKShift)));
  auto rescaleOutTy = RankedTensorType::get({1, kHKRows, 14}, i8Ty);
  tosa::RescaleOp::create(builder, loc, rescaleOutTy, add.getResult(), multiplierConst.getResult(),
                          shiftConst.getResult(), aZp.getResult(), aZp.getResult(),
                          /*scale32=*/true, tosa::RoundingMode::SINGLE_ROUND,
                          /*perChannel=*/false, /*inputUnsigned=*/false,
                          /*outputUnsigned=*/false);

  ASSERT_TRUE(succeeded(conversion::lowerTosaToMacaque(block)));

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
  // 1 hold-batch (30 rows fits well under the 252-row batch capacity), 2
  // K-tiles: weight loaded exactly once per K-tile, not once per chunk.
  ASSERT_EQ(loadWeights.size(), 2u);
  ASSERT_EQ(loadBiases.size(), 1u);
  ASSERT_EQ(loadInputs.size(), 6u);  // 3 chunks x 2 K-tiles
  ASSERT_EQ(matmuls.size(), 6u);
  ASSERT_EQ(activates.size(), 3u);   // one per chunk, after the full K-tile sweep
  ASSERT_EQ(stores.size(), 3u);

  auto words = target::emitBinary(block);
  ASSERT_TRUE(succeeded(words));

  sim::Simulator simulator(/*ddr_bytes=*/65536);

  // Weight: one tile per K-tile, staged once - reused (via weight_hold) by
  // every chunk in that K-tile's pass, not restaged per chunk.
  for (int k = 0; k < 2; ++k) {
    std::vector<uint8_t> wBytes;
    for (int kk = 0; kk < 14; ++kk)
      for (int c = 0; c < 14; ++c)
        wBytes.push_back(static_cast<uint8_t>(hkWeight(k * 14 + kk, c)));
    simulator.write(loadWeights[k].getDdr3Addr(), wBytes.data(), wBytes.size());
  }

  std::vector<uint8_t> biasBytes;
  for (int c = 0; c < 14; ++c) {
    int32_t v = hkBias(c);
    for (int byteIdx = 0; byteIdx < 4; ++byteIdx)
      biasBytes.push_back(static_cast<uint8_t>((v >> (8 * byteIdx)) & 0xFF));
  }
  simulator.write(loadBiases[0].getDdr3Addr(), biasBytes.data(), biasBytes.size());

  // Activation: K-tile outer, chunk inner - matches emitBatchMatmuls's
  // emission order (each K-tile's pass visits every chunk in turn).
  const int64_t chunkRows[3] = {14, 14, 2};
  int loadInputIdx = 0;
  for (int k = 0; k < 2; ++k) {
    int64_t chunkRowOffset = 0;
    for (int c = 0; c < 3; ++c) {
      std::vector<uint8_t> aBytes;
      for (int64_t r = 0; r < chunkRows[c]; ++r)
        for (int kk = 0; kk < 14; ++kk)
          aBytes.push_back(static_cast<uint8_t>(
              hkAct(static_cast<int>(chunkRowOffset + r), k * 14 + kk)));
      simulator.write(loadInputs[loadInputIdx].getDdr3Addr(), aBytes.data(), aBytes.size());
      chunkRowOffset += chunkRows[c];
      ++loadInputIdx;
    }
  }

  simulator.run(*words);

  int64_t rowOffset = 0;
  for (int c = 0; c < 3; ++c) {
    std::vector<uint8_t> result = simulator.read(stores[c].getDdr3Addr(), chunkRows[c] * 14);
    ASSERT_EQ(result.size(), static_cast<size_t>(chunkRows[c] * 14));
    for (int64_t r = 0; r < chunkRows[c]; ++r) {
      for (int cc = 0; cc < 14; ++cc) {
        EXPECT_EQ(result[r * 14 + cc], hkExpected(static_cast<int>(rowOffset + r), cc))
            << "chunk " << c << " row " << r << " channel " << cc;
      }
    }
    rowOffset += chunkRows[c];
  }
}

//===----------------------------------------------------------------------===//
// 8. Chained held-batch producer and consumer, whole-pipeline closed loop
//
// Layer 1: K=28 (2 K-tiles, held-batch producer), N=28 (2 N-tiles), rows=30
// (3 hold-chunks: 14+14+2) - its own K needs the held-batch scheme *and* it
// feeds layer 2. Layer 2: K=28 (matches layer 1's real N, so its own K also
// needs the held-batch scheme), N=14, same rows=30. Exercises both newly
// -unified code paths at once: RescaleToMacaque's numKTiles>1 branch storing
// into one shared per-(N-tile, hold-batch) Scratch address instead of one
// per chunk, and allocateBatchChunkInputAddrs's chained branch reading that
// same address at computed 14-row offsets. Previously rejected outright
// (rows=30 > 14, so both sides of the link span 3 batch-chunks, not 1).
//===----------------------------------------------------------------------===//

namespace {

constexpr int64_t kHC2Rows = 30;
constexpr int64_t kHC2K1 = 28;  // layer 1's own K (2 K-tiles)
constexpr int64_t kHC2N1 = 28;  // layer 1's N (2 N-tiles) == layer 2's K
constexpr int64_t kHC2N2 = 14;  // layer 2's N (1 N-tile)

int8_t hc2Act0(int r, int k) { return static_cast<int8_t>((r * 3 + k) % 11 - 5); }
int8_t hc2Weight1(int k, int c) { return static_cast<int8_t>((k + c) % 9 - 4); }
int32_t hc2Bias1(int c) { return c % 7 - 3; }
int8_t hc2Weight2(int k, int c) { return static_cast<int8_t>((k - c) % 5); }
int32_t hc2Bias2(int c) { return 2 * c - 5; }
constexpr uint32_t kHC2Mult1 = 1, kHC2Shift1 = 0;
constexpr uint32_t kHC2Mult2 = 1, kHC2Shift2 = 0;

uint8_t hc2Requant(int64_t acc, uint32_t m, uint32_t shift) {
  int64_t scaled = acc * static_cast<int64_t>(m);
  if (shift > 0) scaled += static_cast<int64_t>(1) << (shift - 1);  // round-half-up
  int64_t requant = scaled >> shift;
  return static_cast<uint8_t>(std::clamp<int64_t>(requant, -128, 127));
}

uint8_t hc2Layer1Out(int r, int c) {  // c in [0, 28)
  int64_t acc = hc2Bias1(c);
  for (int64_t k = 0; k < kHC2K1; ++k)
    acc += static_cast<int64_t>(hc2Act0(r, static_cast<int>(k))) *
           static_cast<int64_t>(hc2Weight1(static_cast<int>(k), c));
  return hc2Requant(acc, kHC2Mult1, kHC2Shift1);
}
int8_t hc2Layer1OutSigned(int r, int c) {
  return static_cast<int8_t>(hc2Layer1Out(r, c));
}

uint8_t hc2Layer2Out(int r, int c) {  // c in [0, 14)
  int64_t acc = hc2Bias2(c);
  for (int64_t k = 0; k < kHC2N1; ++k)
    acc += static_cast<int64_t>(hc2Layer1OutSigned(r, static_cast<int>(k))) *
           static_cast<int64_t>(hc2Weight2(static_cast<int>(k), c));
  return hc2Requant(acc, kHC2Mult2, kHC2Shift2);
}

}  // namespace

TEST(ClosedLoop, ChainedHeldBatchProducerAndConsumerIsBitExact) {
  MLIRContext context;
  context.getOrLoadDialect<MacaqueDialect>();
  context.getOrLoadDialect<tosa::TosaDialect>();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  Block block;
  builder.setInsertionPointToStart(&block);

  auto act0Const = buildChainI8Tensor(builder, loc, kHC2Rows, kHC2K1, hc2Act0);
  auto weight1Const = buildChainI8Tensor(builder, loc, kHC2K1, kHC2N1, hc2Weight1);
  auto bias1Const = buildChainI32Bias(builder, loc, kHC2N1, hc2Bias1);
  auto weight2Const = buildChainI8Tensor(builder, loc, kHC2N1, kHC2N2, hc2Weight2);
  auto bias2Const = buildChainI32Bias(builder, loc, kHC2N2, hc2Bias2);

  auto rescale1 = buildChainLayerN(builder, loc, act0Const.getResult(), weight1Const,
                                   bias1Const, static_cast<int32_t>(kHC2Mult1),
                                   static_cast<int8_t>(kHC2Shift1));
  buildChainLayerN(builder, loc, rescale1.getResult(), weight2Const, bias2Const,
                   static_cast<int32_t>(kHC2Mult2), static_cast<int8_t>(kHC2Shift2));

  ASSERT_TRUE(succeeded(conversion::lowerTosaToMacaque(block)));

  SmallVector<LoadWeightOp> loadWeights;
  SmallVector<LoadBiasOp> loadBiases;
  SmallVector<LoadInputOp> loadInputs;
  SmallVector<StoreOp> stores;
  for (Operation& op : block) {
    if (auto o = dyn_cast<LoadWeightOp>(op)) loadWeights.push_back(o);
    if (auto o = dyn_cast<LoadBiasOp>(op)) loadBiases.push_back(o);
    if (auto o = dyn_cast<LoadInputOp>(op)) loadInputs.push_back(o);
    if (auto o = dyn_cast<StoreOp>(op)) stores.push_back(o);
  }
  // Layer 1: 2 K-tiles x 2 N-tiles = 4 weight loads, 2 bias loads (one per
  // N-tile), 6 distinct input addresses (3 chunks x 2 K-tiles) each
  // re-issued once per N-tile = 12 load_inputs, 6 stores (3 chunks x 2
  // N-tiles). Layer 2: 2 K-tiles x 1 N-tile = 2 weight loads, 1 bias load,
  // 6 chained load_inputs (3 chunks x 2 K-tiles), 3 stores (final output).
  ASSERT_EQ(loadWeights.size(), 6u);
  ASSERT_EQ(loadBiases.size(), 3u);
  ASSERT_EQ(loadInputs.size(), 18u);
  ASSERT_EQ(stores.size(), 9u);

  auto words = target::emitBinary(block);
  ASSERT_TRUE(succeeded(words));

  sim::Simulator simulator(/*ddr_bytes=*/65536);

  // Layer 1's weight1: 2 K-tiles x 2 N-tiles = 4 distinct 14x14 slices,
  // emitted K-tile outer within each N-tile's own pass (N-tile outer).
  const int weight1KTileOf[4] = {0, 1, 0, 1};
  const int weight1NTileOf[4] = {0, 0, 1, 1};
  for (int i = 0; i < 4; ++i) {
    int kTile = weight1KTileOf[i], nTile = weight1NTileOf[i];
    std::vector<uint8_t> wBytes;
    for (int kk = 0; kk < 14; ++kk)
      for (int cc = 0; cc < 14; ++cc)
        wBytes.push_back(
            static_cast<uint8_t>(hc2Weight1(kTile * 14 + kk, nTile * 14 + cc)));
    simulator.write(loadWeights[i].getDdr3Addr(), wBytes.data(), wBytes.size());
  }

  // Layer 1's bias1: 2 N-tiles.
  for (int n = 0; n < 2; ++n) {
    std::vector<uint8_t> bBytes;
    for (int cc = 0; cc < 14; ++cc) {
      int32_t v = hc2Bias1(n * 14 + cc);
      for (int byteIdx = 0; byteIdx < 4; ++byteIdx)
        bBytes.push_back(static_cast<uint8_t>((v >> (8 * byteIdx)) & 0xFF));
    }
    simulator.write(loadBiases[n].getDdr3Addr(), bBytes.data(), bBytes.size());
  }

  // Layer 2's weight2: 2 K-tiles, single N-tile (14 cols each).
  for (int k = 0; k < 2; ++k) {
    std::vector<uint8_t> wBytes;
    for (int kk = 0; kk < 14; ++kk)
      for (int cc = 0; cc < 14; ++cc)
        wBytes.push_back(static_cast<uint8_t>(hc2Weight2(k * 14 + kk, cc)));
    simulator.write(loadWeights[4 + k].getDdr3Addr(), wBytes.data(), wBytes.size());
  }

  // Layer 2's bias2: real, single N-tile.
  std::vector<uint8_t> bias2Bytes;
  for (int cc = 0; cc < 14; ++cc) {
    int32_t v = hc2Bias2(cc);
    for (int byteIdx = 0; byteIdx < 4; ++byteIdx)
      bias2Bytes.push_back(static_cast<uint8_t>((v >> (8 * byteIdx)) & 0xFF));
  }
  simulator.write(loadBiases[2].getDdr3Addr(), bias2Bytes.data(), bias2Bytes.size());

  // Layer 1's own activation: 3 held-batch chunks (14+14+2 rows) x 2
  // K-tiles, K-tile outer/chunk inner, matching emitBatchMatmuls's
  // emission order. Only loadInputs[0..5] (N-tile 0's pass) need staging -
  // N-tile 1's pass (loadInputs[6..11]) re-reads the exact same addresses.
  // Layer 2's own reads (loadInputs[12..17]) are chained - computed by the
  // simulator from layer 1's stores during run(), nothing to stage.
  const int64_t chunkRows[3] = {14, 14, 2};
  int loadInputIdx = 0;
  for (int k = 0; k < 2; ++k) {
    int64_t rowOffset = 0;
    for (int c = 0; c < 3; ++c) {
      std::vector<uint8_t> aBytes;
      for (int64_t r = 0; r < chunkRows[c]; ++r)
        for (int kk = 0; kk < 14; ++kk)
          aBytes.push_back(static_cast<uint8_t>(
              hc2Act0(static_cast<int>(rowOffset + r), k * 14 + kk)));
      simulator.write(loadInputs[loadInputIdx].getDdr3Addr(), aBytes.data(), aBytes.size());
      rowOffset += chunkRows[c];
      ++loadInputIdx;
    }
  }

  simulator.run(*words);

  // Layer 1's 2 N-tiles' worth of scratch stores, reassembled into the true
  // 28-channel intermediate result.
  for (int n = 0; n < 2; ++n) {
    int64_t rowOffset = 0;
    for (int c = 0; c < 3; ++c) {
      int storeIdx = n * 3 + c;
      std::vector<uint8_t> result =
          simulator.read(stores[storeIdx].getDdr3Addr(), chunkRows[c] * 14);
      ASSERT_EQ(result.size(), static_cast<size_t>(chunkRows[c] * 14));
      for (int64_t r = 0; r < chunkRows[c]; ++r) {
        for (int cc = 0; cc < 14; ++cc) {
          EXPECT_EQ(result[r * 14 + cc],
                    hc2Layer1Out(static_cast<int>(rowOffset + r), n * 14 + cc))
              << "L1 N-tile" << n << " chunk " << c << " row " << r << " col " << cc;
        }
      }
      rowOffset += chunkRows[c];
    }
  }

  // Layer 2's final 3 chunks.
  {
    int64_t rowOffset = 0;
    for (int c = 0; c < 3; ++c) {
      int storeIdx = 6 + c;
      std::vector<uint8_t> result =
          simulator.read(stores[storeIdx].getDdr3Addr(), chunkRows[c] * 14);
      ASSERT_EQ(result.size(), static_cast<size_t>(chunkRows[c] * 14));
      for (int64_t r = 0; r < chunkRows[c]; ++r) {
        for (int cc = 0; cc < 14; ++cc) {
          EXPECT_EQ(result[r * 14 + cc], hc2Layer2Out(static_cast<int>(rowOffset + r), cc))
              << "L2 chunk " << c << " row " << r << " col " << cc;
        }
      }
      rowOffset += chunkRows[c];
    }
  }
}
