#pragma once

#include <algorithm>
#include <cstdint>

#include "macaque/common/isa.hpp"

namespace macaque::sim {

[[nodiscard]] inline int8_t
requantizeAndActivate(int32_t biased, uint32_t m, uint32_t shift,
                      macaque::common::isa::ActFunc func) {
  int64_t scaled = static_cast<int64_t>(biased) * static_cast<int64_t>(m);
  if (shift > 0) {
    scaled += static_cast<int64_t>(1) << (shift - 1); // round-half-up
  }
  const int64_t requant = scaled >> shift;

  int64_t act = requant;
  switch (func) {
  case macaque::common::isa::ActFunc::Relu:
    act = std::max<int64_t>(0, requant);
    break;
  case macaque::common::isa::ActFunc::LeakyRelu:
    act = (requant >= 0) ? requant
                         : (requant >> macaque::common::isa::kLeakyReluShift);
    break;
  case macaque::common::isa::ActFunc::Passthrough:
  default:
    act = requant;
    break;
  }
  act = std::clamp<int64_t>(act, -128, 127);
  return static_cast<int8_t>(act);
}

} // namespace macaque::sim
