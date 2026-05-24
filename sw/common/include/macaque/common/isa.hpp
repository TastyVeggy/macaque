#pragma once

#include <cstdint>
namespace macaque::common::isa {
enum class Opcode : uint8_t {
  // [63:60]=0x0 [59:32]=DDR3_addr [31:0]=byte_count
  LoadWeight = 0x0,  // ddr3 to weight buffer
  // [63:60]=0x1 [59:32]=DDR3_addr [31:0]=byte_count
  LoadBias = 0x1,  // ddr3 to bias buffer
  // [63:60]=0x2 [59:32]=DDR3_addr [31:0]=byte_count
  LoadInput = 0x2,  // ddr3 to input buffer
  // [63:60]=0x3 [59:32]=N_rows [31:0]=reserved
  Matmul = 0x3,
  // [63:60]=0x4 [59:32]=func [31:0]=reserved
  Activate = 0x4,
  // [63:60]=0x5 [59:32]=DDR3_addr [31:0]=byte_count
  Store = 0x5,  // output buffer to ddr3
  // [63:60]=0x6  [59:32]=unit_mask [31:0]=reserved
  Sync = 0x6
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
}  // namespace macaque::common::isa
