#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "macaque/runtime/SerialTransport.hpp"

namespace macaque::runtime {

inline constexpr uint8_t kOpWrite = 0x57;
inline constexpr uint8_t kOpRead = 0x52;

// Writes a 64-bit little-endian word to `addr` and confirms the ack byte.
// Throws std::runtime_error on a short read or a non-zero ack.
void write64(SerialTransport &transport, uint32_t addr, uint64_t data);

// Reads a 64-bit little-endian word from `addr`.
[[nodiscard]] uint64_t read64(SerialTransport &transport, uint32_t addr);

// Writes `bytes` to DDR3 starting at `addr`, in 8-byte little-endian chunks,
// zero-padded to the next 8-byte boundary.
void stageBytes(SerialTransport &transport, uint32_t addr,
                std::span<const uint8_t> bytes);

// Reads exactly `n` bytes from DDR3 starting at `addr`.
[[nodiscard]] std::vector<uint8_t> readBytes(SerialTransport &transport,
                                             uint32_t addr, size_t n);

} // namespace macaque::runtime
