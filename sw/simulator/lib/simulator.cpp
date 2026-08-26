#include "macaque/sim/simulator.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace macaque::sim {

namespace {
using macaque::common::isa::Opcode;
}

Simulator::Simulator(size_t ddr_bytes) : mem_(ddr_bytes, 0) {}

void Simulator::write(uint32_t addr, const uint8_t *data, size_t len) {
  std::copy_n(data, len, mem_.begin() + addr);
}

std::vector<uint8_t> Simulator::read(uint32_t addr, size_t len) const {
  return {mem_.begin() + addr, mem_.begin() + addr + len};
}

void Simulator::run(const std::vector<uint64_t> &program) {
  for (const uint64_t word : program) {
    execute(macaque::common::isa::Instruction::decode(word));
  }
}

void Simulator::execute(const macaque::common::isa::Instruction &ins) {
  switch (ins.opcode) {
  case Opcode::LoadWeight:
  case Opcode::LoadInput: {
    const size_t rows = ins.byte_count / kArraySize;
    assert(rows <= static_cast<size_t>(kBramDepth) &&
           "load exceeds the on-chip buffer depth (M not chunked to fit)");
    std::array<std::array<int8_t, kArraySize>, kBramDepth> &buf =
        (ins.opcode == Opcode::LoadWeight) ? weight_ : act_;
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < static_cast<size_t>(kArraySize); ++c) {
        buf[r][c] =
            static_cast<int8_t>(mem_[ins.ddr3_addr + r * kArraySize + c]);
      }
    }
    break;
  }
  case Opcode::LoadBias: {
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
    const size_t m = ins.tile_params;
    const size_t rowBase = macaque::common::isa::mat_row_base(ins);
    assert(rowBase + m <= static_cast<size_t>(kBramDepth) &&
           "mat_row_base + rows exceeds the on-chip buffer depth");
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
    const size_t m = macaque::common::isa::num_rows(ins);
    const uint32_t mm = macaque::common::isa::scale_m(ins);
    const uint32_t shift = macaque::common::isa::scale_shift(ins);
    const auto func = macaque::common::isa::act_func(ins);
    const size_t rowBase = macaque::common::isa::act_row_base(ins);
    assert(rowBase + m <= static_cast<size_t>(kBramDepth) &&
           "act_row_base + rows exceeds the on-chip buffer depth");
    for (size_t r = 0; r < m; ++r) {
      for (size_t c = 0; c < static_cast<size_t>(kArraySize); ++c) {
        act_[r][c] =
            requantizeAndActivate(out_[rowBase + r][c], mm, shift, func);
      }
    }
    break;
  }
  case Opcode::Store: {
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
