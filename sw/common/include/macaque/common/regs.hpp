#pragma once

#include <cstdint>

namespace macaque::common::regs {
constexpr uint32_t kCtrl = 0x0000;
constexpr uint32_t kStatus = 0x0004;
constexpr uint32_t kInstrAddr = 0x0008;
constexpr uint32_t kInstrLen = 0x000C;
constexpr uint32_t kPmuCtrl = 0x0010;
constexpr uint32_t kPmuCyclesLo = 0x0014;
constexpr uint32_t kPmuCyclesHi = 0x0018;
constexpr uint32_t kPmuCompute = 0x001C;
constexpr uint32_t kPmuStall = 0x0020;
constexpr uint32_t kPmuDmaBytesRd = 0x0024;
constexpr uint32_t kPmuDmaBytesWr = 0x0028;
}  // namespace macaque::common::regs
