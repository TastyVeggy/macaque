#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace macaque::runtime {

// One DDR3 read/write segment: an absolute address plus a byte count.
struct IoTile {
  uint32_t addr = 0;
  uint32_t bytes = 0;
};

// One compile-time-known constant to embed in DDR3 (weights/bias/const
// activations) before running the program.
struct DataTile {
  uint32_t addr = 0;
  std::vector<uint8_t> bytes;
};

struct Program {
  std::vector<uint64_t> instructions;
  std::vector<DataTile> data;
  std::vector<IoTile> inputTiles;
  std::vector<IoTile> outputTiles;
  int64_t inputValidBytes = 0;
  int64_t outputValidBytes = 0;

  [[nodiscard]] static Program load(const std::string &path);
};

} // namespace macaque::runtime
