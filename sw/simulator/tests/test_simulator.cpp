#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "macaque/common/isa.hpp"
#include "macaque/sim/requantizeAndActivate.hpp"
#include "macaque/sim/simulator.hpp"

using macaque::common::isa::ActFunc;
using macaque::common::isa::ActivateFields;
using macaque::common::isa::encodeActivate;
using macaque::common::isa::encodeMatmul;
using macaque::common::isa::Instruction;
using macaque::common::isa::MatmulFields;
using macaque::common::isa::Opcode;
using macaque::sim::kArraySize;
using macaque::sim::requantizeAndActivate;
using macaque::sim::Simulator;

TEST(requantizeAndActivate, BitMatchVectors) {
  EXPECT_EQ(requantizeAndActivate(100, 8192, 13, ActFunc::Relu), 100);
  EXPECT_EQ(requantizeAndActivate(-100, 8192, 13, ActFunc::Relu), 0);
  EXPECT_EQ(requantizeAndActivate(-100, 8192, 13, ActFunc::Passthrough), -100);
  EXPECT_EQ(requantizeAndActivate(100, 8192, 13, ActFunc::Passthrough), 100);
  EXPECT_EQ(requantizeAndActivate(-100, 8192, 13, ActFunc::LeakyRelu),
            -7); // -100 >> 4
  EXPECT_EQ(requantizeAndActivate(100, 8192, 13, ActFunc::LeakyRelu),
            100); // positives unchanged
  EXPECT_EQ(requantizeAndActivate(32768, 8192, 13, ActFunc::Relu), 127);
  EXPECT_EQ(requantizeAndActivate(-40000, 8192, 13, ActFunc::Relu), 0);
  EXPECT_EQ(requantizeAndActivate(123, 16384, 13, ActFunc::Relu), 127);
  EXPECT_EQ(requantizeAndActivate(1, 3, 1, ActFunc::Relu), 2); // round-half-up
  EXPECT_EQ(requantizeAndActivate(255, 128, 7, ActFunc::Relu), 127);
}

TEST(requantizeAndActivate, LeakyMatchesArithmeticShift) {
  EXPECT_EQ(requantizeAndActivate(-16, 8192, 13, ActFunc::LeakyRelu), -1);
  EXPECT_EQ(requantizeAndActivate(-1, 8192, 13, ActFunc::LeakyRelu),
            -1); // -1 >> 4 = -1
  EXPECT_EQ(requantizeAndActivate(-128, 8192, 13, ActFunc::LeakyRelu),
            -8); // -128 >> 4
  EXPECT_EQ(requantizeAndActivate(-1600, 8192, 13, ActFunc::LeakyRelu),
            -100); // -1600 >> 4
}

TEST(requantizeAndActivate, BiasMatchesActivate) {
  EXPECT_EQ(requantizeAndActivate(-64, 8192, 13, ActFunc::Relu), 0);
  EXPECT_EQ(requantizeAndActivate(79, 8192, 13, ActFunc::Relu), 79);
}

TEST(Engine, LoadMatmulActivateStore) {
  const uint32_t kWeightAddr = 0x100;
  const uint32_t kBiasAddr = 0x200;
  const uint32_t kActAddr = 0x300;
  const uint32_t kOutAddr = 0x400;
  const uint32_t kByteCount = kArraySize * kArraySize;

  std::vector<uint8_t> weights(kByteCount, 0);
  for (int k = 0; k < kArraySize; ++k) {
    weights[k * kArraySize + k] = 1;
  }

  std::vector<uint8_t> acts(kByteCount, 0);
  std::array<std::array<int32_t, kArraySize>, kArraySize> act_ref{};
  for (int r = 0; r < kArraySize; ++r) {
    for (int c = 0; c < kArraySize; ++c) {
      const int8_t v = static_cast<int8_t>(10 * r + c - 64);
      acts[r * kArraySize + c] = static_cast<uint8_t>(v);
      act_ref[r][c] = v;
    }
  }

  std::vector<uint8_t> biases(14 * 4, 0);

  const uint32_t kM = 8192;
  const uint16_t kShift = 13;

  Simulator sim(0x1000);
  sim.write(kWeightAddr, weights.data(), weights.size());
  sim.write(kBiasAddr, biases.data(), biases.size());
  sim.write(kActAddr, acts.data(), acts.size());

  const Instruction load_w{Opcode::LoadWeight, kWeightAddr,
                           static_cast<uint16_t>(kByteCount)};
  const Instruction load_b{Opcode::LoadBias, kBiasAddr, 14 * 4};
  const Instruction load_a{Opcode::LoadInput, kActAddr,
                           static_cast<uint16_t>(kByteCount)};
  const uint64_t mm =
      encodeMatmul({/*acc_mode=*/false, /*weight_hold=*/false,
                    /*mat_row_base=*/0, static_cast<uint8_t>(kArraySize)});
  const uint64_t act = encodeActivate(
      {ActFunc::Relu, kM, /*act_bank_hold=*/false, /*act_row_base=*/0,
       static_cast<uint8_t>(kShift), static_cast<uint8_t>(kArraySize)});
  const Instruction store{Opcode::Store, kOutAddr,
                          static_cast<uint16_t>(kByteCount)};

  const std::vector<uint64_t> program{
      load_w.encode(), load_b.encode(), load_a.encode(), mm, act,
      store.encode()};

  sim.run(program);

  const std::vector<uint8_t> got = sim.read(kOutAddr, kByteCount);
  for (int r = 0; r < kArraySize; ++r) {
    for (int c = 0; c < kArraySize; ++c) {
      const int8_t expected =
          requantizeAndActivate(act_ref[r][c], kM, kShift, ActFunc::Relu);
      EXPECT_EQ(static_cast<int8_t>(got[r * kArraySize + c]), expected)
          << "row " << r << " col " << c;
    }
  }
}

TEST(Engine, BiasAddedOnceInMatmulNotActivate) {
  const uint32_t kWeightAddr = 0x100;
  const uint32_t kBiasAddr = 0x200;
  const uint32_t kActAddr = 0x300;
  const uint32_t kOutAddr = 0x400;
  const uint32_t kByteCount = kArraySize * kArraySize;

  std::vector<uint8_t> weights(kByteCount, 0);
  for (int k = 0; k < kArraySize; ++k)
    weights[k * kArraySize + k] = 1;

  std::vector<uint8_t> acts(kByteCount, 1);

  std::vector<uint8_t> biases(14 * 4, 0);
  for (int c = 0; c < kArraySize; ++c) {
    const int32_t b = 1000;
    const uint8_t *p = reinterpret_cast<const uint8_t *>(&b); // little-endian
    for (size_t j = 0; j < sizeof(int32_t); ++j)
      biases[c * 4 + j] = p[j];
  }

  const uint32_t kM = 8192;
  const uint16_t kShift = 13;

  Simulator sim(0x1000);
  sim.write(kWeightAddr, weights.data(), weights.size());
  sim.write(kBiasAddr, biases.data(), biases.size());
  sim.write(kActAddr, acts.data(), acts.size());

  const Instruction load_w{Opcode::LoadWeight, kWeightAddr,
                           static_cast<uint16_t>(kByteCount)};
  const Instruction load_b{Opcode::LoadBias, kBiasAddr, 14 * 4};
  const Instruction load_a{Opcode::LoadInput, kActAddr,
                           static_cast<uint16_t>(kByteCount)};
  const uint64_t mm =
      encodeMatmul({/*acc_mode=*/false, /*weight_hold=*/false,
                    /*mat_row_base=*/0, static_cast<uint8_t>(kArraySize)});
  const uint64_t act = encodeActivate(
      {ActFunc::Relu, kM, /*act_bank_hold=*/false, /*act_row_base=*/0,
       static_cast<uint8_t>(kShift), static_cast<uint8_t>(kArraySize)});
  const Instruction store{Opcode::Store, kOutAddr,
                          static_cast<uint16_t>(kByteCount)};

  const std::vector<uint64_t> program{
      load_w.encode(), load_b.encode(), load_a.encode(), mm, act,
      store.encode()};

  sim.run(program);

  const std::vector<uint8_t> got = sim.read(kOutAddr, kByteCount);
  for (int r = 0; r < kArraySize; ++r) {
    for (int c = 0; c < kArraySize; ++c) {
      const int8_t expected =
          requantizeAndActivate(1000 + 1, kM, kShift, ActFunc::Relu);
      EXPECT_EQ(static_cast<int8_t>(got[r * kArraySize + c]), expected)
          << "row " << r << " col " << c;
    }
  }
}
