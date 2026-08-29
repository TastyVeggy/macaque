#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "macaque/runtime/Program.hpp"

namespace rt = macaque::runtime;

namespace {

std::string writeTempMacq(const std::vector<uint8_t> &bytes) {
  const auto stamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("macaque_test_" + std::to_string(stamp) + ".macq");
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(bytes.data()),
         static_cast<std::streamsize>(bytes.size()));
  return path.string();
}

void appendU32(std::vector<uint8_t> &out, uint32_t v) {
  for (int i = 0; i < 4; ++i)
    out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void appendU64(std::vector<uint8_t> &out, uint64_t v) {
  for (int i = 0; i < 8; ++i)
    out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void appendI64(std::vector<uint8_t> &out, int64_t v) {
  appendU64(out, static_cast<uint64_t>(v));
}

// Builds a sample .macq matching this shape:
//   instructions: {0x00000100000c4000, 0x4000000000010005}
//   data: [{addr=0x1000, bytes={1,2,3,4,5,6,7,8}}]
//   input_tiles: [{addr=0x9678, bytes=14}]
//   output_tiles: [{addr=0x9a88, bytes=14}]
//   inputValidBytes=10, outputValidBytes=5
std::vector<uint8_t> sampleProgramBytes() {
  std::vector<uint8_t> out{'M', 'A', 'C', 'Q'};
  appendU32(out, /*version=*/1);
  appendU32(out, /*numInstructions=*/2);
  appendU32(out, /*numDataTiles=*/1);
  appendU32(out, /*numInputTiles=*/1);
  appendU32(out, /*numOutputTiles=*/1);
  appendI64(out, /*inputValidBytes=*/10);
  appendI64(out, /*outputValidBytes=*/5);

  appendU64(out, 0x00000100000c4000ull);
  appendU64(out, 0x4000000000010005ull);

  appendU32(out, /*addr=*/0x1000);
  appendU32(out, /*len=*/8);
  for (uint8_t b :
      {uint8_t{1}, uint8_t{2}, uint8_t{3}, uint8_t{4}, uint8_t{5}, uint8_t{6},
       uint8_t{7}, uint8_t{8}})
    out.push_back(b);

  appendU32(out, /*addr=*/0x9678);
  appendU32(out, /*bytes=*/14);

  appendU32(out, /*addr=*/0x9a88);
  appendU32(out, /*bytes=*/14);

  return out;
}

} // namespace

TEST(Program, ParsesAllFields) {
  const std::string path = writeTempMacq(sampleProgramBytes());
  const rt::Program prog = rt::Program::load(path);
  std::filesystem::remove(path);

  ASSERT_EQ(prog.instructions.size(), 2u);
  EXPECT_EQ(prog.instructions[0], 0x00000100000c4000ull);
  EXPECT_EQ(prog.instructions[1], 0x4000000000010005ull);

  ASSERT_EQ(prog.data.size(), 1u);
  EXPECT_EQ(prog.data[0].addr, 0x1000u);
  EXPECT_EQ(prog.data[0].bytes, (std::vector<uint8_t>{0x01, 0x02, 0x03, 0x04,
                                                      0x05, 0x06, 0x07, 0x08}));

  ASSERT_EQ(prog.inputTiles.size(), 1u);
  EXPECT_EQ(prog.inputTiles[0].addr, 0x9678u);
  EXPECT_EQ(prog.inputTiles[0].bytes, 14u);

  ASSERT_EQ(prog.outputTiles.size(), 1u);
  EXPECT_EQ(prog.outputTiles[0].addr, 0x9a88u);
  EXPECT_EQ(prog.outputTiles[0].bytes, 14u);

  EXPECT_EQ(prog.inputValidBytes, 10);
  EXPECT_EQ(prog.outputValidBytes, 5);
}

TEST(Program, MissingFileThrows) {
  EXPECT_THROW((void)rt::Program::load("/nonexistent/path/does/not/exist.macq"),
               std::runtime_error);
}

TEST(Program, TooShortThrows) {
  const std::string path = writeTempMacq({'M', 'A', 'C', 'Q', 0, 0});
  EXPECT_THROW((void)rt::Program::load(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(Program, BadMagicThrows) {
  std::vector<uint8_t> bytes = sampleProgramBytes();
  bytes[0] = 'X';
  const std::string path = writeTempMacq(bytes);
  EXPECT_THROW((void)rt::Program::load(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(Program, WrongVersionThrows) {
  std::vector<uint8_t> bytes = sampleProgramBytes();
  bytes[4] = 2; // version field, little-endian byte 0
  const std::string path = writeTempMacq(bytes);
  EXPECT_THROW((void)rt::Program::load(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(Program, TruncatedMidSectionThrows) {
  std::vector<uint8_t> bytes = sampleProgramBytes();
  bytes.resize(bytes.size() - 4); // cut off the last output tile's bytes field
  const std::string path = writeTempMacq(bytes);
  EXPECT_THROW((void)rt::Program::load(path), std::runtime_error);
  std::filesystem::remove(path);
}
