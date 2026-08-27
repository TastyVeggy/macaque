#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "macaque/common/defs.hpp"
#include "macaque/common/isa.hpp"
#include "macaque/sim/requantizeAndActivate.hpp"

namespace macaque::sim {

using namespace macaque::common;

// TODO: Make a cycle-accurate simulator. Currently it is behavioral
class Simulator {
public:
  explicit Simulator(size_t ddr_bytes);

  // Stage data into the DDR3 model (weights/acts/biases/instruction bytes).
  void write(uint32_t addr, const uint8_t *data, size_t len);
  [[nodiscard]] std::vector<uint8_t> read(uint32_t addr, size_t len) const;

  // Execute an instruction stream in order.
  void run(const std::vector<uint64_t> &program);

private:
  void execute(const macaque::common::isa::Instruction &ins);

  std::vector<uint8_t> mem_;

  // Simulate buffers
  // Out[r][out_ch] = Sum_k (Act[r][k] Weight[k][out_ch]
  std::array<std::array<int8_t, kArraySize>, kBramDepth>
      weight_{}; // [k][out_ch]
  std::array<std::array<int32_t, kArraySize>, kBramDepth>
      bias_{};                                                    // [0][out_ch]
  std::array<std::array<int8_t, kArraySize>, kBramDepth> act_{};  // [r][k]
  std::array<std::array<int32_t, kArraySize>, kBramDepth> out_{}; // [r][out_ch]
};

} // namespace macaque::sim
