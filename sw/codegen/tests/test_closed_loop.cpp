#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "macaque/Dialect/MacaqueOps.h"
#include "macaque/Target/BinaryEmitter.h"
#include "macaque/common/isa.hpp"
#include "macaque/sim/simulator.hpp"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace mlir::macaque;
namespace sim = ::macaque::sim;
namespace target = ::macaque::codegen::target;
namespace isa = ::macaque::common::isa;

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
