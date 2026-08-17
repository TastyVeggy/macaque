#pragma once

#include <cstdint>
namespace macaque::common::isa {

// Instruction field layout:
//  [63:60] opcode        (4 bits)
//  [59]    acc_mode      (1 bit, MATMUL only)
//  [58:56] target/flags  (3 bits)
//  [55:28] ddr3_addr     (28 bits)
//  [27:12] byte_count    (16 bits)
//  [11:0]  tile_params   (12 bits)
constexpr int kOpcodeBits = 4;
constexpr int kAccModeBits = 1;
constexpr int kTargetBits = 3;
constexpr int kDdr3AddrBits = 28;
constexpr int kByteCountBits = 16;
constexpr int kTileParamsBits = 12;

enum class Opcode : uint8_t {
  LoadWeight = 0x0,
  LoadBias   = 0x1,
  LoadInput  = 0x2,
  Matmul     = 0x3,
  Activate   = 0x4,
  Store      = 0x5,
  Sync       = 0x6
};

// ACTIVATE activation-function selector (carried in target[58:56]; SPEC_ISA.md §2.1)
enum class ActFunc : uint8_t {
  Relu        = 0,
  LeakyRelu   = 1,
  Passthrough = 2
};

// Leaky-ReLU negative slope (power-of-two): alpha = 2^-kLeakyReluShift = 1/16.
constexpr int kLeakyReluShift = 4;

struct Instruction {
  Opcode   opcode;       // [63:60]
  bool     acc_mode;     // [59]
  uint8_t  target;       // [58:56]
  uint32_t ddr3_addr;    // [55:28]
  uint16_t byte_count;   // [27:12]
  uint16_t tile_params;  // [11:0]

  [[nodiscard]] uint64_t encode() const {
    return (static_cast<uint64_t>(opcode) & 0xF) << 60 |
           (static_cast<uint64_t>(acc_mode) & 1) << 59 |
           (static_cast<uint64_t>(target) & 0x7) << 56 |
           (static_cast<uint64_t>(ddr3_addr) & 0xFFFFFFF) << 28 |
           (static_cast<uint64_t>(byte_count) & 0xFFFF) << 12 |
           (static_cast<uint64_t>(tile_params) & 0xFFF);
  }

  [[nodiscard]] static Instruction decode(uint64_t word) {
    return {static_cast<Opcode>((word >> 60) & 0xF),
            static_cast<bool>((word >> 59) & 1),
            static_cast<uint8_t>((word >> 56) & 0x7),
            static_cast<uint32_t>((word >> 28) & 0xFFFFFFF),
            static_cast<uint16_t>((word >> 12) & 0xFFFF),
            static_cast<uint16_t>(word & 0xFFF)};
  }
};

// ACTIVATE field reinterpretation
//   ddr3_addr[55:28]  → scale_m      (28-bit unsigned fixed-point M)
//   byte_count[4:0]   → scale_shift  (0..31)
//   byte_count[15:5]  → leaky_shift  (per-instruction leaky slope; future)
//   target[58:56]     → act_func
//   tile_params[7:0]  → num_rows     (M dimension)
//   tile_params[11:8] → reserved
[[nodiscard]] inline constexpr uint32_t scale_m(const Instruction& i) noexcept {
  return i.ddr3_addr;
}
[[nodiscard]] inline constexpr uint32_t scale_shift(const Instruction& i) noexcept {
  return i.byte_count & 0x1F;
}
[[nodiscard]] inline constexpr uint32_t leaky_shift(const Instruction& i) noexcept {
  return (i.byte_count >> 5) & 0x7FF;
}
[[nodiscard]] inline constexpr ActFunc act_func(const Instruction& i) noexcept {
  return static_cast<ActFunc>(i.target);
}
[[nodiscard]] inline constexpr uint32_t num_rows(const Instruction& i) noexcept {
  return i.tile_params & 0xFF;
}
}  // namespace macaque::common::isa
