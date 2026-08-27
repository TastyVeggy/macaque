#include "macaque/runtime/Image.hpp"

#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_STDIO // we open files ourselves so errors carry the path
#include <stb_image.h>

#include <fstream>

namespace macaque::runtime {

namespace {

RawImage fromStb(const uint8_t *pixels, int width, int height, int channels,
                 const std::string &what) {
  if (!pixels)
    throw std::runtime_error("failed to decode PNG (" + what +
                             "): " + stbi_failure_reason());
  RawImage img;
  img.width = width;
  img.height = height;
  img.channels = channels;
  img.pixels.assign(pixels, pixels + static_cast<size_t>(width) *
                                         static_cast<size_t>(height) *
                                         static_cast<size_t>(channels));
  stbi_image_free(const_cast<uint8_t *>(pixels));
  return img;
}

} // namespace

namespace {
RawImage decodeMemory(std::span<const uint8_t> bytes, int desiredChannels,
                      const std::string &what) {
  int width = 0, height = 0, channels = 0;
  uint8_t *pixels =
      stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()),
                            &width, &height, &channels, desiredChannels);
  return fromStb(pixels, width, height,
                 desiredChannels ? desiredChannels : channels, what);
}
} // namespace

RawImage decodePng(std::span<const uint8_t> bytes, int desiredChannels) {
  return decodeMemory(bytes, desiredChannels, "<memory>");
}

RawImage loadPng(const std::string &path, int desiredChannels) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("failed to open image file: " + path);
  const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                                   std::istreambuf_iterator<char>());
  return decodeMemory(bytes, desiredChannels, path);
}

} // namespace macaque::runtime
