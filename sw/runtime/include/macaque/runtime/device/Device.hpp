#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "macaque/common/defs.hpp"
#include "macaque/runtime/Image.hpp"
#include "macaque/runtime/Program.hpp"
#include "macaque/runtime/Quantize.hpp"
#include "macaque/runtime/SerialTransport.hpp"

namespace macaque::runtime {

struct PmuCounters {
  uint64_t cycles = 0;
  uint64_t compute = 0;
  uint64_t stall = 0;
  uint64_t dmaBytesRd = 0;
  uint64_t dmaBytesWr = 0;
};

class Device {
public:
  explicit Device(const std::string &serialPort,
                  int baud = macaque::common::kBaudRate)
      : transport_(serialPort, baud) {}

  void stageProgramData(const Program &prog);

  void stageInput(const Program &prog, std::span<const uint8_t> imageBytes);

  void loadInstructions(const Program &prog);

  // Starts the loaded program and polls STATUS until done or error.
  void triggerAndWait(double timeoutSeconds = 10.0);

  // Concatenates every output tile's bytes in order, then trims to
  // `prog.outputValidBytes`.
  [[nodiscard]] std::vector<uint8_t> readOutput(const Program &prog);

  [[nodiscard]] PmuCounters readPmu();

  [[nodiscard]] std::vector<uint8_t>
  infer(const Program &prog, const RawImage &img, const QuantParams &params);

private:
  SerialTransport transport_;
};

} // namespace macaque::runtime
