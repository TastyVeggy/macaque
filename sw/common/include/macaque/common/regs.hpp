#pragma once

#include <cstdint>

namespace macaque::common::regs {
constexpr uint32_t kRegBase = 0x4000'0000;

constexpr uint32_t kCtrl = 0x0000;
constexpr uint32_t kStatus = 0x0008;
constexpr uint32_t kInstrAddr = 0x0010;
constexpr uint32_t kInstrLen = 0x0018;
constexpr uint32_t kPmuCtrl = 0x0020;
constexpr uint32_t kPmuCycles = 0x0028; // 64-bit
constexpr uint32_t kPmuCompute = 0x0030;
constexpr uint32_t kPmuStall = 0x0038;
constexpr uint32_t kPmuDmaBytesRd = 0x0040;
constexpr uint32_t kPmuDmaBytesWr = 0x0048;
} // namespace macaque::common::regs
