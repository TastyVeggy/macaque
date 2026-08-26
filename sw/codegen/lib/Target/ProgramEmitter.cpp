#include "macaque/Target/ProgramEmitter.hpp"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace macaque::codegen::target {

namespace {

std::string hexAddr(uint32_t addr) {
  return "0x" + llvm::utohexstr(static_cast<uint64_t>(addr), /*LowerCase=*/true,
                                /*Width=*/8);
}

llvm::json::Array ioTilesToJson(const conversion::IoSegment &tiles) {
  llvm::json::Array arr;
  arr.reserve(tiles.size());
  for (const auto &[addr, bytes] : tiles)
    arr.push_back(
        llvm::json::Object{{"addr", hexAddr(addr)}, {"bytes", bytes}});
  return arr;
}

} // namespace

std::string emitProgramJson(llvm::ArrayRef<uint64_t> instructions,
                            const conversion::CompiledProgramInfo &info) {
  llvm::json::Array instrArray;
  instrArray.reserve(instructions.size());
  for (uint64_t word : instructions)
    instrArray.push_back(
        "0x" + llvm::utohexstr(word, /*LowerCase=*/true, /*Width=*/16));

  llvm::json::Array dataArray;
  dataArray.reserve(info.data.size());
  for (const auto &[addr, bytes] : info.data) {
    dataArray.push_back(llvm::json::Object{
        {"addr", hexAddr(addr)},
        {"bytes",
         llvm::toHex(llvm::ArrayRef<uint8_t>(bytes.data(), bytes.size()),
                     /*LowerCase=*/true)}});
  }

  llvm::json::Object root{
      {"instructions", std::move(instrArray)},
      {"data", std::move(dataArray)},
      {"input_tiles", ioTilesToJson(info.inputTiles)},
      {"output_tiles", ioTilesToJson(info.outputTiles)},
      {"input_valid_bytes", info.inputValidBytes},
      {"output_valid_bytes", info.outputValidBytes},
  };

  std::string out;
  llvm::raw_string_ostream os(out);
  os << llvm::json::Value(std::move(root));
  return out;
}

} // namespace macaque::codegen::target
