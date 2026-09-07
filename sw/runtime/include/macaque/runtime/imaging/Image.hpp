#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace macaque::runtime {

struct RawImage {
  std::vector<uint8_t> pixels;
  int width = 0;
  int height = 0;
  int channels = 0;
};

[[nodiscard]] RawImage loadPng(const std::string &path,
                               int desiredChannels = 0);
[[nodiscard]] RawImage decodePng(std::span<const uint8_t> bytes,
                                 int desiredChannels = 0);

} // namespace macaque::runtime
