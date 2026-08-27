#include "macaque/runtime/Device.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>

#include "macaque/common/regs.hpp"
#include "macaque/runtime/Protocol.hpp"

namespace macaque::runtime {

namespace regs = macaque::common::regs;

void Device::stageProgramData(const Program &prog) {
  for (const DataTile &tile : prog.data)
    stageBytes(transport_, tile.addr, tile.bytes);
}

void Device::stageInput(const Program &prog,
                        std::span<const uint8_t> imageBytes) {
  if (static_cast<int64_t>(imageBytes.size()) != prog.inputValidBytes) {
    throw std::invalid_argument(
        "input is " + std::to_string(imageBytes.size()) +
        " bytes, program expects " + std::to_string(prog.inputValidBytes));
  }
  size_t cursor = 0;
  for (const IoTile &tile : prog.inputTiles) {
    const size_t chunkLen =
        std::min<size_t>(tile.bytes, imageBytes.size() - cursor);
    stageBytes(transport_, tile.addr, imageBytes.subspan(cursor, chunkLen));

    cursor += tile.bytes;
  }
}

void Device::loadInstructions(const Program &prog) {
  for (size_t i = 0; i < prog.instructions.size(); ++i)
    write64(transport_, regs::kImemBase + 8 * static_cast<uint32_t>(i),
            prog.instructions[i]);
  write64(transport_, regs::kRegBase + regs::kInstrAddr, 0);
  write64(transport_, regs::kRegBase + regs::kInstrLen,
          prog.instructions.size());
}

void Device::triggerAndWait(double timeoutSeconds) {
  write64(transport_, regs::kRegBase + regs::kPmuCtrl, regs::kPmuCtrlClear);
  write64(transport_, regs::kRegBase + regs::kPmuCtrl, regs::kPmuCtrlEnable);
  write64(transport_, regs::kRegBase + regs::kCtrl, regs::kCtrlStart);

  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(timeoutSeconds));
  while (std::chrono::steady_clock::now() < deadline) {
    const uint64_t status = read64(transport_, regs::kRegBase + regs::kStatus);
    if (status & regs::kStatusError)
      throw std::runtime_error("npu_error asserted (STATUS=0x" +
                               std::to_string(status) + ")");
    if (status & regs::kStatusDone)
      return;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  throw std::runtime_error("timed out waiting for STATUS.done after " +
                           std::to_string(timeoutSeconds) + "s");
}

std::vector<uint8_t> Device::readOutput(const Program &prog) {
  std::vector<uint8_t> raw;
  for (const IoTile &tile : prog.outputTiles) {
    std::vector<uint8_t> chunk = readBytes(transport_, tile.addr, tile.bytes);
    raw.insert(raw.end(), chunk.begin(), chunk.end());
  }
  raw.resize(
      std::min<size_t>(static_cast<size_t>(prog.outputValidBytes), raw.size()));
  return raw;
}

PmuCounters Device::readPmu() {
  return PmuCounters{
      read64(transport_, regs::kRegBase + regs::kPmuCycles),
      read64(transport_, regs::kRegBase + regs::kPmuCompute) & 0xFFFFFFFFu,
      read64(transport_, regs::kRegBase + regs::kPmuStall) & 0xFFFFFFFFu,
      read64(transport_, regs::kRegBase + regs::kPmuDmaBytesRd) & 0xFFFFFFFFu,
      read64(transport_, regs::kRegBase + regs::kPmuDmaBytesWr) & 0xFFFFFFFFu,
  };
}

std::vector<uint8_t> Device::infer(const Program &prog, const RawImage &img,
                                   const QuantParams &params) {
  if (static_cast<int64_t>(img.pixels.size()) != prog.inputValidBytes) {
    throw std::invalid_argument("image has " +
                                std::to_string(img.pixels.size()) +
                                " pixel samples, program expects " +
                                std::to_string(prog.inputValidBytes) +
                                " - infer() does not resize images");
  }
  const std::vector<int8_t> quantized = quantize(img, params);
  std::vector<uint8_t> asBytes(quantized.size());
  for (size_t i = 0; i < quantized.size(); ++i)
    asBytes[i] = static_cast<uint8_t>(quantized[i]);

  stageInput(prog, asBytes);
  triggerAndWait();
  return readOutput(prog);
}

} // namespace macaque::runtime
