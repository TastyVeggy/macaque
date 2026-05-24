#include <gtest/gtest.h>

#include "macaque/common/isa.hpp"

using namespace macaque::common::isa;

TEST(InstructionEncoding, RoundtripPreservesAllFields) {
  Instruction original{Opcode::Matmul, 0x00ABCDEF, 0x12345678};

  uint64_t encoded = original.encode();
  Instruction decoded = Instruction::decode(encoded);

  EXPECT_EQ(decoded.opcode, original.opcode);
  EXPECT_EQ(decoded.operand_a, original.operand_a);
  EXPECT_EQ(decoded.operand_b, original.operand_b);
}

TEST(InstructionEncoding, OpcodeOccupiesTopFourBits) {
  Instruction instr{static_cast<Opcode>(0xF), 0, 0};
  uint64_t encoded = instr.encode();

  EXPECT_EQ(encoded >> 60, 0xFULL);
}

TEST(InstructionEncoding, OperandsDoNotBleedIntoEachOther) {
  Instruction instr{Opcode::Sync, 0x0FFFFFFF, 0xFFFFFFFF};
  uint64_t encoded = instr.encode();
  Instruction decoded = Instruction::decode(encoded);

  EXPECT_EQ(decoded.operand_a, 0x0FFFFFFFu);
  EXPECT_EQ(decoded.operand_b, 0xFFFFFFFFu);
}

TEST(InstructionEncoding, ZeroInstructionEncodesToZero) {
  Instruction instr{static_cast<Opcode>(0x0), 0x0, 0x0};
  EXPECT_EQ(instr.encode(), 0ULL);
}

TEST(InstructionEncoding, AllOpcodeValuesRoundtrip) {
  for (uint8_t op = 0; op <= 0x7; ++op) {
    Instruction instr{static_cast<Opcode>(op), 0x1234, 0x5678};
    Instruction decoded = Instruction::decode(instr.encode());
    EXPECT_EQ(static_cast<uint8_t>(decoded.opcode), op)
        << "Failed for opcode 0x" << std::hex << (int)op;
  }
}
