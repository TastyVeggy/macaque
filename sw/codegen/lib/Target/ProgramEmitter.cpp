#include "macaque/Target/ProgramEmitter.hpp"

namespace macaque::codegen::target {

namespace {

// .macq format: a fixed 40-byte header, then instructions/data/input/output tiles
// packed back-to-back, all little-endian, no text encoding.
constexpr char kMagic[4] = {'M', 'A', 'C', 'Q'};
constexpr uint32_t kVersion = 1;

void appendU32(std::vector<uint8_t> &out, uint32_t v) {
  for (int i = 0; i < 4; ++i)
    out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void appendU64(std::vector<uint8_t> &out, uint64_t v) {
  for (int i = 0; i < 8; ++i)
    out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void appendI64(std::vector<uint8_t> &out, int64_t v) {
  appendU64(out, static_cast<uint64_t>(v));
}

} // namespace

std::vector<uint8_t> emitProgramBinary(llvm::ArrayRef<uint64_t> instructions,
                                       const conversion::CompiledProgramInfo &info) {
  std::vector<uint8_t> out;
  out.reserve(40 + instructions.size() * 8 + info.data.size() * 8 +
             (info.inputTiles.size() + info.outputTiles.size()) * 8);

  out.insert(out.end(), kMagic, kMagic + 4);
  appendU32(out, kVersion);
  appendU32(out, static_cast<uint32_t>(instructions.size()));
  appendU32(out, static_cast<uint32_t>(info.data.size()));
  appendU32(out, static_cast<uint32_t>(info.inputTiles.size()));
  appendU32(out, static_cast<uint32_t>(info.outputTiles.size()));
  appendI64(out, info.inputValidBytes);
  appendI64(out, info.outputValidBytes);

  for (uint64_t word : instructions)
    appendU64(out, word);

  for (const auto &[addr, bytes] : info.data) {
    appendU32(out, addr);
    appendU32(out, static_cast<uint32_t>(bytes.size()));
    out.insert(out.end(), bytes.begin(), bytes.end());
  }

  for (const auto &[addr, bytes] : info.inputTiles) {
    appendU32(out, addr);
    appendU32(out, bytes);
  }

  for (const auto &[addr, bytes] : info.outputTiles) {
    appendU32(out, addr);
    appendU32(out, bytes);
  }

  return out;
}

} // namespace macaque::codegen::target
