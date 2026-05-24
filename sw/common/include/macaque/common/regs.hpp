#pragma once

#include <cstdint>

namespace macaque::common::regs {
constexpr uint32_t kCtrl = 0x0000;
constexpr uint32_t kStatus = 0x0004;
constexpr uint32_t kInstrAddr = 0x0008;
constexpr uint32_t kInstrLen = 0x000C;
constexpr uint32_t kPmuCtrl = 0x0024;
constexpr uint32_t kPmuCyclesLo = 0x0028;
constexpr uint32_t kPmuCyclesHi = 0x002C;
constexpr uint32_t kPmuCompute = 0x0030;
constexpr uint32_t kPmuStall = 0x0034;
constexpr uint32_t kPmuDmaBytesRd = 0x0038;
constexpr uint32_t kPmuDmaBytesWr = 0x003C;
}  // namespace macaque::common::regs
