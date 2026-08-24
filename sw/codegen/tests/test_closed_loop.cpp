#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "macaque/Conversion/TosaToMacaque.h"
#include "macaque/Dialect/MacaqueOps.h"
#include "macaque/Target/BinaryEmitter.h"
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

// TOSA-sourced tests
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
