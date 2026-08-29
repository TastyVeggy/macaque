#include "macaque/sim/simulator.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace macaque::sim {

namespace {
using macaque::common::isa::Instruction;
using macaque::common::isa::Opcode;
} // namespace

Simulator::Simulator(size_t ddr_bytes) : mem_(ddr_bytes, 0) {}

void Simulator::write(uint32_t addr, const uint8_t *data, size_t len) {
  std::copy_n(data, len, mem_.begin() + addr);
}

std::vector<uint8_t> Simulator::read(uint32_t addr, size_t len) const {
  return {mem_.begin() + addr, mem_.begin() + addr + len};
}

void Simulator::run(const std::vector<uint64_t> &program) {
  for (const uint64_t word : program) {
    execute(word);
  }
}

void Simulator::execute(uint64_t word) {
  switch (macaque::common::isa::decodeOpcode(word)) {
  case Opcode::LoadWeight: {
    const Instruction ins = Instruction::decode(word);
    const size_t rows = ins.byte_count / kArraySize;
    if (rows > static_cast<size_t>(kBramDepth)) {
      throw std::invalid_argument(
          "load exceeds the on-chip buffer depth (M not chunked to fit)");
    }
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < static_cast<size_t>(kArraySize); ++c) {
        weight_[r][c] =
            static_cast<int8_t>(mem_[ins.ddr3_addr + r * kArraySize + c]);
      }
    }
    break;
  }
  case Opcode::LoadInput: {
    const Instruction ins = Instruction::decode(word);
    const size_t rows = ins.valid_bytes_per_row != 0
                            ? ins.input_rows
                            : ins.byte_count / kArraySize;
    if (rows > static_cast<size_t>(kBramDepth)) {
      throw std::invalid_argument(
          "load exceeds the on-chip buffer depth (M not chunked to fit)");
    }
    size_t srcOffset = 0;
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < static_cast<size_t>(kArraySize); ++c) {
        const bool real =
            ins.valid_bytes_per_row == 0 || c < ins.valid_bytes_per_row;
        act_[r][c] =
            real ? static_cast<int8_t>(mem_[ins.ddr3_addr + srcOffset++])
                 : static_cast<int8_t>(0);
      }
    }
    break;
  }
  case Opcode::LoadBias: {
    const Instruction ins = Instruction::decode(word);
    const size_t lanes = ins.byte_count / sizeof(int32_t);
    const size_t n = std::min(lanes, static_cast<size_t>(kArraySize));
    for (size_t c = 0; c < n; ++c) {
      int32_t v = 0;
      // INT32 little-endian
      for (size_t b = 0; b < sizeof(int32_t); ++b) {
        v |= static_cast<int32_t>(mem_[ins.ddr3_addr + c * sizeof(int32_t) + b])
             << static_cast<int>(8 * b);
      }
      bias_[0][c] = v;
    }
    break;
  }
  case Opcode::Matmul: {
    const auto ins = macaque::common::isa::decodeMatmul(word);
    const size_t m = ins.tile_params;
    const size_t rowBase = ins.mat_row_base;
    if (rowBase + m > static_cast<size_t>(kBramDepth)) {
      throw std::invalid_argument(
          "mat_row_base + rows exceeds the on-chip buffer depth");
    }
    for (size_t r = 0; r < m; ++r) {
      for (size_t c = 0; c < static_cast<size_t>(kArraySize); ++c) {
        int32_t acc = ins.acc_mode ? out_[rowBase + r][c] : bias_[0][c];
        for (size_t k = 0; k < static_cast<size_t>(kArraySize); ++k) {
          acc += static_cast<int32_t>(act_[r][k]) *
                 static_cast<int32_t>(weight_[k][c]);
        }
        out_[rowBase + r][c] = acc;
      }
    }
    break;
  }
  case Opcode::Activate: {
    const auto ins = macaque::common::isa::decodeActivate(word);
    const size_t m = ins.act_num_rows;
    const size_t rowBase = ins.act_row_base;
    if (rowBase + m > static_cast<size_t>(kBramDepth)) {
      throw std::invalid_argument(
          "act_row_base + rows exceeds the on-chip buffer depth");
    }
    for (size_t r = 0; r < m; ++r) {
      for (size_t c = 0; c < static_cast<size_t>(kArraySize); ++c) {
        act_[r][c] =
            requantizeAndActivate(out_[rowBase + r][c], ins.act_scale_m,
                                  ins.act_scale_shift, ins.act_func);
      }
    }
    break;
  }
  case Opcode::Store: {
    const Instruction ins = Instruction::decode(word);
    const size_t rows = ins.byte_count / kArraySize;
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < static_cast<size_t>(kArraySize); ++c) {
        mem_[ins.ddr3_addr + r * kArraySize + c] =
            static_cast<uint8_t>(act_[r][c]);
      }
    }
    break;
  }
  case Opcode::Sync:
    // Barrier: numeric no-op in the behavioral model.
    break;
  }
}

} // namespace macaque::sim
