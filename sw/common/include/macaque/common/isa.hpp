#pragma once

#include <cstdint>

namespace macaque::common::isa {

enum class Opcode : uint8_t {
  LoadWeight = 0x0,
  LoadBias = 0x1,
  LoadInput = 0x2,
  Matmul = 0x3,
  Activate = 0x4,
  Store = 0x5,
  Sync = 0x6
};

// ACTIVATE activation-function selector
enum class ActFunc : uint8_t { Relu = 0, LeakyRelu = 1, Passthrough = 2 };

// Leaky-ReLU negative slope (power-of-two): alpha = 2^-kLeakyReluShift = 1/16.
constexpr int kLeakyReluShift = 4;

[[nodiscard]] inline constexpr Opcode decodeOpcode(uint64_t word) noexcept {
  return static_cast<Opcode>((word >> 60) & 0xF);
}
[[nodiscard]] inline constexpr uint64_t encodeOpcode(Opcode op) noexcept {
  return (static_cast<uint64_t>(op) & 0xF) << 60;
}

struct Instruction {
  Opcode opcode;                   // [63:60]
  uint32_t ddr3_addr;              // [55:28]
  uint16_t byte_count;             // [27:12]
  uint8_t valid_bytes_per_row = 0; // [11:8] - LOAD_INPUT only
  uint8_t input_rows = 0;          // [7:0] - LOAD_INPUT only

  [[nodiscard]] uint64_t encode() const {
    return encodeOpcode(opcode) |
           (static_cast<uint64_t>(ddr3_addr) & 0xFFFFFFF) << 28 |
           (static_cast<uint64_t>(byte_count) & 0xFFFF) << 12 |
           (static_cast<uint64_t>(valid_bytes_per_row) & 0xF) << 8 |
           (static_cast<uint64_t>(input_rows) & 0xFF);
  }

  [[nodiscard]] static Instruction decode(uint64_t word) {
    return {decodeOpcode(word), static_cast<uint32_t>((word >> 28) & 0xFFFFFFF),
            static_cast<uint16_t>((word >> 12) & 0xFFFF),
            static_cast<uint8_t>((word >> 8) & 0xF),
            static_cast<uint8_t>(word & 0xFF)};
  }
};

// MATMUL reinterpretation
struct MatmulFields {
  bool acc_mode;        // [59] K-tile accumulate flag
  bool weight_hold;     // [56] - reuse the currently-loaded weight/bias
                        //       bank instead of a fresh load
  uint8_t mat_row_base; // [35:28] - out_buffer row this M-chunk's
                        //          accumulator starts at
  uint8_t tile_params;  // [7:0] - row count (M) to feed
};

[[nodiscard]] inline constexpr uint64_t
encodeMatmul(const MatmulFields &m) noexcept {
  return encodeOpcode(Opcode::Matmul) |
         (static_cast<uint64_t>(m.acc_mode) & 1) << 59 |
         (static_cast<uint64_t>(m.weight_hold) & 1) << 56 |
         (static_cast<uint64_t>(m.mat_row_base) & 0xFF) << 28 |
         (static_cast<uint64_t>(m.tile_params) & 0xFF);
}

[[nodiscard]] inline constexpr MatmulFields
decodeMatmul(uint64_t word) noexcept {
  return {static_cast<bool>((word >> 59) & 1),
          static_cast<bool>((word >> 56) & 1),
          static_cast<uint8_t>((word >> 28) & 0xFF),
          static_cast<uint8_t>(word & 0xFF)};
}

// ACTIVATE reinterpretation
struct ActivateFields {
  ActFunc act_func;        // [58:56]
  uint32_t act_scale_m;    // [55:39] - requantize multiplier (17-bit
                           //           fixed-point)
  bool act_bank_hold;      // [25] - skip the out_bank_sel toggle
  uint8_t act_row_base;    // [24:17] - out_buffer row this M-chunk's
                           //           accumulator starts at
  uint8_t act_scale_shift; // [16:12] - requantize right-shift
  uint8_t act_num_rows;    // [7:0] - rows to requantize
};

[[nodiscard]] inline constexpr uint64_t
encodeActivate(const ActivateFields &a) noexcept {
  return encodeOpcode(Opcode::Activate) |
         (static_cast<uint64_t>(a.act_func) & 0x7) << 56 |
         (static_cast<uint64_t>(a.act_scale_m) & 0x1FFFF) << 39 |
         (static_cast<uint64_t>(a.act_bank_hold) & 1) << 25 |
         (static_cast<uint64_t>(a.act_row_base) & 0xFF) << 17 |
         (static_cast<uint64_t>(a.act_scale_shift) & 0x1F) << 12 |
         (static_cast<uint64_t>(a.act_num_rows) & 0xFF);
}

[[nodiscard]] inline constexpr ActivateFields
decodeActivate(uint64_t word) noexcept {
  return {static_cast<ActFunc>((word >> 56) & 0x7),
          static_cast<uint32_t>((word >> 39) & 0x1FFFF),
          static_cast<bool>((word >> 25) & 1),
          static_cast<uint8_t>((word >> 17) & 0xFF),
          static_cast<uint8_t>((word >> 12) & 0x1F),
          static_cast<uint8_t>(word & 0xFF)};
}

} // namespace macaque::common::isa
