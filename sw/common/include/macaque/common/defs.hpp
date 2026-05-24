#pragma once

#include <cstdint>

namespace macaque::common {
constexpr int kArraySize = 14;
constexpr int kBramDepth = 256;

constexpr int kClkFreqHz = 50'000'000;
constexpr int kBaudRate = 115'200;
constexpr int kBitPeriod = kClkFreqHz / kBaudRate;

using Weight = int8_t;
using Activation = int8_t;
using Accum = int32_t;

}  // namespace macaque::common
