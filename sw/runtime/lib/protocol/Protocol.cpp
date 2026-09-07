#include "macaque/runtime/Protocol.hpp"

#include <stdexcept>

namespace macaque::runtime {

namespace {

std::vector<uint8_t> expectExact(SerialTransport &transport, size_t n,
                                 const char *what) {
  std::vector<uint8_t> data = transport.read(n);
  if (data.size() != n) {
    throw std::runtime_error(std::string("short read for ") + what + ": got " +
                             std::to_string(data.size()) + "/" +
                             std::to_string(n) + " bytes");
  }
  return data;
}

void writeU32LE(SerialTransport &transport, uint32_t v) {
  uint8_t bytes[4] = {static_cast<uint8_t>(v), static_cast<uint8_t>(v >> 8),
                      static_cast<uint8_t>(v >> 16),
                      static_cast<uint8_t>(v >> 24)};
  transport.write(bytes, sizeof(bytes));
}

void writeU64LE(SerialTransport &transport, uint64_t v) {
  uint8_t bytes[8];
  for (int i = 0; i < 8; ++i)
    bytes[i] = static_cast<uint8_t>(v >> (8 * i));
  transport.write(bytes, sizeof(bytes));
}

uint64_t readU64LE(const std::vector<uint8_t> &bytes) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i)
    v |= static_cast<uint64_t>(bytes[static_cast<size_t>(i)]) << (8 * i);
  return v;
}

} // namespace

void write64(SerialTransport &transport, uint32_t addr, uint64_t data) {
  const uint8_t opcode = kOpWrite;
  transport.write(&opcode, 1);
  writeU32LE(transport, addr);
  writeU64LE(transport, data);
  transport.flush();

  std::vector<uint8_t> ack = expectExact(transport, 1, "write ack");
  if (ack[0] != 0x00) {
    throw std::runtime_error("write ack != 0x00 (got 0x" +
                             std::to_string(static_cast<unsigned>(ack[0])) +
                             ")");
  }
}

uint64_t read64(SerialTransport &transport, uint32_t addr) {
  const uint8_t opcode = kOpRead;
  transport.write(&opcode, 1);
  writeU32LE(transport, addr);
  transport.flush();

  return readU64LE(expectExact(transport, 8, "read data"));
}

void stageBytes(SerialTransport &transport, uint32_t addr,
                std::span<const uint8_t> bytes) {
  for (size_t i = 0; i < bytes.size(); i += 8) {
    uint64_t val = 0;
    const size_t chunkLen = std::min<size_t>(8, bytes.size() - i);
    for (size_t j = 0; j < chunkLen; ++j)
      val |= static_cast<uint64_t>(bytes[i + j]) << (8 * j);
    write64(transport, addr + static_cast<uint32_t>(i), val);
  }
}

std::vector<uint8_t> readBytes(SerialTransport &transport, uint32_t addr,
                               size_t n) {
  std::vector<uint8_t> out;
  out.reserve(n);
  for (size_t i = 0; i < n; i += 8) {
    const uint64_t word = read64(transport, addr + static_cast<uint32_t>(i));
    const size_t chunkLen = std::min<size_t>(8, n - i);
    for (size_t j = 0; j < chunkLen; ++j)
      out.push_back(static_cast<uint8_t>(word >> (8 * j)));
  }
  return out;
}

} // namespace macaque::runtime
