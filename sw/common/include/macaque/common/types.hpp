#pragma once

#include <cstdint>

namespace macaque::common {
constexpr int kArraySize = 14;
constexpr int kBramDepth = 1024;
constexpr int kClkFreqHz = 50'000'000;
constexpr int kBaudRate = 115'200;
constexpr int kBitPeriod = kClkFreqHz / kBaudRate;

using Weight = int8_t;
using Activation = int8_t;
using Accum = int32_t;

enum class Opcode : uint8_t {
  LoadWeight = 0x0,
  LoadInput = 0x1,
  Matmul = 0x2,
  Activate = 0x3,
  Store = 0x4,
  DmaRead = 0x5,
  DmaWrite = 0x6,
  Sync = 0x7
};

// 64-bit instruction word layout:
//  [63:60] opcode    (4 bits)
//  [59:32] operand_a (28 bits)
//  [31:0]  operand_b (32 bits)
struct Instruction {
  Opcode opcode;
  uint32_t operand_a;
  uint32_t operand_b;

  [[nodiscard]] uint64_t encode() const {
    return ((static_cast<uint64_t>(opcode) & 0xF) << 60 |
            static_cast<uint64_t>(operand_a & 0x0FFF'FFFF) << 32 |
            static_cast<uint64_t>(operand_b));
  }

  [[nodiscard]] static Instruction decode(uint64_t word) {
    return {static_cast<Opcode>(word >> 60 & 0xF),
            static_cast<uint32_t>((word >> 32) & 0x0FFF'FFFF),
            static_cast<uint32_t>(word) & 0xFFFF'FFFF};
  }
};
} // namespace macaque::common
