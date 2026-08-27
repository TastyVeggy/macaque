#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include "macaque/runtime/Program.hpp"

namespace rt = macaque::runtime;

namespace {

std::string writeTempJson(const std::string &contents) {
  const auto stamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("macaque_test_" + std::to_string(stamp) + ".json");
  std::ofstream f(path);
  f << contents;
  return path.string();
}

constexpr const char *kSampleProgram = R"JSON({
  "instructions": ["0x00000100000c4000", "0x4000000000010005"],
  "data": [
    {"addr": "0x00001000", "bytes": "0102030405060708"}
  ],
  "input_tiles": [
    {"addr": "0x00009678", "bytes": 14}
  ],
  "output_tiles": [
    {"addr": "0x00009a88", "bytes": 14}
  ],
  "input_valid_bytes": 10,
  "output_valid_bytes": 5
})JSON";

} // namespace

TEST(Program, ParsesAllFields) {
  const std::string path = writeTempJson(kSampleProgram);
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
  EXPECT_THROW((void)rt::Program::load("/nonexistent/path/does/not/exist.json"),
               std::runtime_error);
}

TEST(Program, MalformedJsonThrows) {
  const std::string path = writeTempJson("{not valid json");
  EXPECT_THROW((void)rt::Program::load(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(Program, MissingFieldThrows) {
  const std::string path = writeTempJson(R"({"instructions": []})");
  EXPECT_THROW((void)rt::Program::load(path), std::runtime_error);
  std::filesystem::remove(path);
}
