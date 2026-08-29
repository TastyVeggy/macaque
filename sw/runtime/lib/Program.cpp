#include "macaque/runtime/Program.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace macaque::runtime {

namespace {

// .macq format: a fixed 40-byte header, then instructions/data/input/output
// tiles packed back-to-back, all little-endian, no text encoding.
constexpr char kMagic[4] = {'M', 'A', 'C', 'Q'};
constexpr uint32_t kVersion = 1;
constexpr size_t kHeaderSize = 40;

uint32_t readU32(const std::vector<uint8_t> &buf, size_t &offset) {
  if (offset + 4 > buf.size())
    throw std::runtime_error("truncated .macq file");
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i)
    v |= static_cast<uint32_t>(buf[offset + static_cast<size_t>(i)]) << (8 * i);
  offset += 4;
  return v;
}

uint64_t readU64(const std::vector<uint8_t> &buf, size_t &offset) {
  if (offset + 8 > buf.size())
    throw std::runtime_error("truncated .macq file");
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i)
    v |= static_cast<uint64_t>(buf[offset + static_cast<size_t>(i)]) << (8 * i);
  offset += 8;
  return v;
}

int64_t readI64(const std::vector<uint8_t> &buf, size_t &offset) {
  return static_cast<int64_t>(readU64(buf, offset));
}

std::vector<uint8_t> readBytes(const std::vector<uint8_t> &buf, size_t &offset,
                               size_t len) {
  if (offset + len > buf.size())
    throw std::runtime_error("truncated .macq file");
  std::vector<uint8_t> out(buf.begin() + static_cast<ptrdiff_t>(offset),
                           buf.begin() + static_cast<ptrdiff_t>(offset + len));
  offset += len;
  return out;
}

} // namespace

Program Program::load(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("failed to open program file: " + path);

  const std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());

  if (buf.size() < kHeaderSize)
    throw std::runtime_error("malformed .macq file (" + path + "): too short");
  if (!std::equal(buf.begin(), buf.begin() + 4, kMagic))
    throw std::runtime_error("malformed .macq file (" + path +
                             "): bad magic - not a .macq program");

  Program prog;
  try {
    size_t offset = 4;
    const uint32_t version = readU32(buf, offset);
    if (version != kVersion)
      throw std::runtime_error("unsupported .macq version " +
                               std::to_string(version) + " (expected " +
                               std::to_string(kVersion) + ")");

    const uint32_t numInstructions = readU32(buf, offset);
    const uint32_t numDataTiles = readU32(buf, offset);
    const uint32_t numInputTiles = readU32(buf, offset);
    const uint32_t numOutputTiles = readU32(buf, offset);
    prog.inputValidBytes = readI64(buf, offset);
    prog.outputValidBytes = readI64(buf, offset);

    prog.instructions.reserve(numInstructions);
    for (uint32_t i = 0; i < numInstructions; ++i)
      prog.instructions.push_back(readU64(buf, offset));

    prog.data.reserve(numDataTiles);
    for (uint32_t i = 0; i < numDataTiles; ++i) {
      const uint32_t addr = readU32(buf, offset);
      const uint32_t len = readU32(buf, offset);
      prog.data.push_back(DataTile{addr, readBytes(buf, offset, len)});
    }

    prog.inputTiles.reserve(numInputTiles);
    for (uint32_t i = 0; i < numInputTiles; ++i) {
      const uint32_t addr = readU32(buf, offset);
      const uint32_t bytes = readU32(buf, offset);
      prog.inputTiles.push_back(IoTile{addr, bytes});
    }

    prog.outputTiles.reserve(numOutputTiles);
    for (uint32_t i = 0; i < numOutputTiles; ++i) {
      const uint32_t addr = readU32(buf, offset);
      const uint32_t bytes = readU32(buf, offset);
      prog.outputTiles.push_back(IoTile{addr, bytes});
    }
  } catch (const std::runtime_error &e) {
    throw std::runtime_error("malformed .macq file (" + path + "): " + e.what());
  }

  return prog;
}

} // namespace macaque::runtime
