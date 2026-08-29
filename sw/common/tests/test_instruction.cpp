#include <gtest/gtest.h>

#include "macaque/common/isa.hpp"

using namespace macaque::common::isa;

TEST(InstructionEncoding, RoundtripPreservesAllFields) {
  Instruction original{Opcode::LoadInput, 0x00ABCDE, 0x1234, 0x5, 0x67};

  uint64_t encoded = original.encode();
  Instruction decoded = Instruction::decode(encoded);

  EXPECT_EQ(decoded.opcode, original.opcode);
  EXPECT_EQ(decoded.ddr3_addr, original.ddr3_addr);
  EXPECT_EQ(decoded.byte_count, original.byte_count);
  EXPECT_EQ(decoded.valid_bytes_per_row, original.valid_bytes_per_row);
  EXPECT_EQ(decoded.input_rows, original.input_rows);
}

TEST(InstructionEncoding, OpcodeOccupiesTopFourBits) {
  Instruction instr{static_cast<Opcode>(0xF), 0, 0, 0, 0};
  uint64_t encoded = instr.encode();

  EXPECT_EQ(encoded >> 60, 0xFULL);
}

TEST(InstructionEncoding, FieldsDoNotBleedIntoEachOther) {
  Instruction instr{Opcode::Sync, 0xFFFFFFF, 0xFFFF, 0xF, 0xFF};
  uint64_t encoded = instr.encode();
  Instruction decoded = Instruction::decode(encoded);

  EXPECT_EQ(decoded.ddr3_addr, 0xFFFFFFFu);
  EXPECT_EQ(decoded.byte_count, 0xFFFFu);
  EXPECT_EQ(decoded.valid_bytes_per_row, 0xFu);
  EXPECT_EQ(decoded.input_rows, 0xFFu);
}

TEST(InstructionEncoding, ZeroInstructionEncodesToZero) {
  Instruction instr{static_cast<Opcode>(0x0), 0, 0, 0, 0};
  EXPECT_EQ(instr.encode(), 0ULL);
}

TEST(InstructionEncoding, AllOpcodeValuesRoundtrip) {
  for (uint8_t op = 0; op <= 0x7; ++op) {
    Instruction instr{static_cast<Opcode>(op), 0xABCDE, 0x1234, 0x5, 0x67};
    Instruction decoded = Instruction::decode(instr.encode());
    EXPECT_EQ(static_cast<uint8_t>(decoded.opcode), op)
        << "Failed for opcode 0x" << std::hex << (int)op;
  }
}

TEST(InstructionEncoding, MatmulRoundtripPreservesAllFields) {
  MatmulFields original{/*acc_mode=*/true, /*weight_hold=*/true,
                        /*mat_row_base=*/0xAB, /*tile_params=*/0xCD};

  uint64_t encoded = encodeMatmul(original);
  MatmulFields decoded = decodeMatmul(encoded);

  EXPECT_EQ(decoded.acc_mode, original.acc_mode);
  EXPECT_EQ(decoded.weight_hold, original.weight_hold);
  EXPECT_EQ(decoded.mat_row_base, original.mat_row_base);
  EXPECT_EQ(decoded.tile_params, original.tile_params);
  EXPECT_EQ(decodeOpcode(encoded), Opcode::Matmul);
}

TEST(InstructionEncoding, ActivateRoundtripPreservesAllFields) {
  ActivateFields original{ActFunc::LeakyRelu,       /*act_scale_m=*/0x1ABCDu,
                          /*act_bank_hold=*/true,   /*act_row_base=*/0xAB,
                          /*act_scale_shift=*/0x1F, /*act_num_rows=*/0xCD};

  uint64_t encoded = encodeActivate(original);
  ActivateFields decoded = decodeActivate(encoded);

  EXPECT_EQ(decoded.act_func, original.act_func);
  EXPECT_EQ(decoded.act_scale_m, original.act_scale_m);
  EXPECT_EQ(decoded.act_bank_hold, original.act_bank_hold);
  EXPECT_EQ(decoded.act_row_base, original.act_row_base);
  EXPECT_EQ(decoded.act_scale_shift, original.act_scale_shift);
  EXPECT_EQ(decoded.act_num_rows, original.act_num_rows);
  EXPECT_EQ(decodeOpcode(encoded), Opcode::Activate);
}

TEST(InstructionEncoding, SyncEncodesToOpcodeOnly) {
  EXPECT_EQ(encodeOpcode(Opcode::Sync), static_cast<uint64_t>(Opcode::Sync)
                                            << 60);
}
