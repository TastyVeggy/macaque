#include <gtest/gtest.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "macaque/runtime/Image.hpp"
#include "macaque/runtime/Quantize.hpp"

namespace rt = macaque::runtime;

namespace {

std::string tempPngPath() {
  const auto stamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  return (std::filesystem::temp_directory_path() /
          ("macaque_test_" + std::to_string(stamp) + ".png"))
      .string();
}

} // namespace

TEST(Image, DecodesGrayscalePngRoundTrip) {
  const std::vector<uint8_t> pixels = {10, 20, 30, 40}; // 2x2 grayscale
  const std::string path = tempPngPath();
  ASSERT_TRUE(stbi_write_png(path.c_str(), /*w=*/2, /*h=*/2, /*channels=*/1,
                             pixels.data(), 2));

  const rt::RawImage img = rt::loadPng(path, /*desiredChannels=*/1);
  std::filesystem::remove(path);

  EXPECT_EQ(img.width, 2);
  EXPECT_EQ(img.height, 2);
  EXPECT_EQ(img.channels, 1);
  EXPECT_EQ(img.pixels, pixels);
}

TEST(Image, DecodePngThrowsOnGarbageBytes) {
  const std::vector<uint8_t> garbage = {0x00, 0x01, 0x02, 0x03};
  EXPECT_THROW((void)rt::decodePng(garbage), std::runtime_error);
}

TEST(Image, LoadPngThrowsOnMissingFile) {
  EXPECT_THROW((void)rt::loadPng("/nonexistent/path/does/not/exist.png"),
               std::runtime_error);
}

TEST(Quantize, SymmetricMatchesExpectedInt8) {
  rt::RawImage img;
  img.width = 4;
  img.height = 1;
  img.channels = 1;
  img.pixels = {0, 64, 128, 255};

  // scale=2.0, zeroPoint=0 (pure symmetric): q = round(pixel / 2.0)
  const std::vector<int8_t> q = rt::quantize(img, rt::QuantParams{2.0f, 0});
  EXPECT_EQ(q,
            (std::vector<int8_t>{
                0, 32, 64, 127})); // 255/2=127.5 rounds to 128, clamps to 127
}

TEST(Quantize, ZeroPointShiftsThrows) {
  rt::RawImage img;
  img.width = 2;
  img.height = 1;
  img.channels = 1;
  img.pixels = {0, 255};

  EXPECT_THROW((void)rt::quantize(img, rt::QuantParams{1.0f, -128}),
               std::logic_error);
}

TEST(Quantize, NonPositiveScaleThrows) {
  rt::RawImage img;
  img.pixels = {1};
  EXPECT_THROW((void)rt::quantize(img, rt::QuantParams{0.0f, 0}),
               std::runtime_error);
  EXPECT_THROW((void)rt::quantize(img, rt::QuantParams{-1.0f, 0}),
               std::runtime_error);
}
