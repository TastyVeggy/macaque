#pragma once

#include <cstdint>

namespace macaque::common::regs {
constexpr uint32_t kRegBase = 0x4000'0000;
constexpr uint32_t kImemBase = 0x5000'0000;

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

// CTRL register: writing 1 to bit 0 starts the loaded program
constexpr uint32_t kCtrlStart = 0x1;

// STATUS register: response from device
constexpr uint32_t kStatusReady = 0x1;
constexpr uint32_t kStatusBusy = 0x2;
constexpr uint32_t kStatusDone = 0x4;
constexpr uint32_t kStatusError = 0x8;

// PMU_CTRL register: bit 0 enables counting, bit 1 clears (write-1-to-clear)
constexpr uint32_t kPmuCtrlEnable = 0x1;
constexpr uint32_t kPmuCtrlClear = 0x2;
} // namespace macaque::common::regs
