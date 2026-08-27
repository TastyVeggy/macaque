#include "macaque/runtime/Quantize.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace macaque::runtime {

std::vector<int8_t> quantize(const RawImage &img, const QuantParams &params) {
  if (params.scale <= 0.0f)
    throw std::runtime_error("QuantParams.scale must be positive");

  if (params.zeroPoint != 0)
    throw std::logic_error("Non-symmetric quantisation is not supported yet");

  std::vector<int8_t> out;
  out.reserve(img.pixels.size());
  for (uint8_t pixel : img.pixels) {
    const float scaled = static_cast<float>(pixel) / params.scale;
    const int32_t q =
        static_cast<int32_t>(std::lround(scaled)) + params.zeroPoint;
    out.push_back(static_cast<int8_t>(std::clamp(q, -128, 127)));
  }
  return out;
}

} // namespace macaque::runtime
