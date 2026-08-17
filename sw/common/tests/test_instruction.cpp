#include <gtest/gtest.h>

#include "macaque/common/isa.hpp"

using namespace macaque::common::isa;

TEST(InstructionEncoding, RoundtripPreservesAllFields) {
  Instruction original{Opcode::Matmul, true, 0x3, 0x00ABCDE, 0x1234, 0x567};

  uint64_t encoded = original.encode();
  Instruction decoded = Instruction::decode(encoded);

  EXPECT_EQ(decoded.opcode, original.opcode);
  EXPECT_EQ(decoded.acc_mode, original.acc_mode);
  EXPECT_EQ(decoded.target, original.target);
  EXPECT_EQ(decoded.ddr3_addr, original.ddr3_addr);
  EXPECT_EQ(decoded.byte_count, original.byte_count);
  EXPECT_EQ(decoded.tile_params, original.tile_params);
}

TEST(InstructionEncoding, OpcodeOccupiesTopFourBits) {
  Instruction instr{static_cast<Opcode>(0xF), false, 0, 0, 0, 0};
  uint64_t encoded = instr.encode();

  EXPECT_EQ(encoded >> 60, 0xFULL);
}

TEST(InstructionEncoding, FieldsDoNotBleedIntoEachOther) {
  Instruction instr{Opcode::Sync, true, 0x7, 0xFFFFFFF, 0xFFFF, 0xFFF};
  uint64_t encoded = instr.encode();
  Instruction decoded = Instruction::decode(encoded);

  EXPECT_EQ(decoded.acc_mode, true);
  EXPECT_EQ(decoded.target, 0x7u);
  EXPECT_EQ(decoded.ddr3_addr, 0xFFFFFFFu);
  EXPECT_EQ(decoded.byte_count, 0xFFFFu);
  EXPECT_EQ(decoded.tile_params, 0xFFFu);
}

TEST(InstructionEncoding, ZeroInstructionEncodesToZero) {
  Instruction instr{static_cast<Opcode>(0x0), false, 0, 0, 0, 0};
  EXPECT_EQ(instr.encode(), 0ULL);
}

TEST(InstructionEncoding, AllOpcodeValuesRoundtrip) {
  for (uint8_t op = 0; op <= 0x7; ++op) {
    Instruction instr{
        static_cast<Opcode>(op), false, 0, 0xABCDE, 0x1234, 0x567};
    Instruction decoded = Instruction::decode(instr.encode());
    EXPECT_EQ(static_cast<uint8_t>(decoded.opcode), op)
        << "Failed for opcode 0x" << std::hex << (int)op;
  }
}
