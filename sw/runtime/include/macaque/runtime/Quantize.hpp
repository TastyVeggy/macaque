#pragma once

#include <cstdint>
#include <vector>

#include "macaque/runtime/Image.hpp"

namespace macaque::runtime {

struct QuantParams {
  float scale = 1.0f;
  int32_t zeroPoint = 0; // currently zero point need to be 0
};

// Quantizes every pixel sample in `img` (in row-major, channel-interleaved
// order) per `params`.
[[nodiscard]] std::vector<int8_t> quantize(const RawImage &img,
                                           const QuantParams &params);

} // namespace macaque::runtime
