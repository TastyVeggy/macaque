#include "macaque/runtime/Program.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace macaque::runtime {

namespace {

using json = nlohmann::json;

uint8_t hexNibble(char c) {
  if (c >= '0' && c <= '9')
    return static_cast<uint8_t>(c - '0');
  if (c >= 'a' && c <= 'f')
    return static_cast<uint8_t>(c - 'a' + 10);
  if (c >= 'A' && c <= 'F')
    return static_cast<uint8_t>(c - 'A' + 10);
  throw std::runtime_error("invalid hex digit in program JSON");
}

// Parses a "0x..." (or plain, no-prefix) hex string into an integer.
uint64_t parseHexInt(const std::string &s) {
  const size_t start =
      (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) ? 2 : 0;
  if (start == s.size())
    throw std::runtime_error("empty hex integer in program JSON");
  uint64_t v = 0;
  for (size_t i = start; i < s.size(); ++i)
    v = (v << 4) | hexNibble(s[i]);
  return v;
}

// Parses a plain (no "0x" prefix) hex string of raw byte content.
std::vector<uint8_t> parseHexBytes(const std::string &s) {
  if (s.size() % 2 != 0)
    throw std::runtime_error("odd-length hex byte string in program JSON");
  std::vector<uint8_t> out;
  out.reserve(s.size() / 2);
  for (size_t i = 0; i < s.size(); i += 2)
    out.push_back(
        static_cast<uint8_t>((hexNibble(s[i]) << 4) | hexNibble(s[i + 1])));
  return out;
}

std::vector<IoTile> parseIoTiles(const json &arr) {
  std::vector<IoTile> tiles;
  tiles.reserve(arr.size());
  for (const json &t : arr)
    tiles.push_back(IoTile{
        static_cast<uint32_t>(parseHexInt(t.at("addr").get<std::string>())),
        t.at("bytes").get<uint32_t>()});
  return tiles;
}

} // namespace

Program Program::load(const std::string &path) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("failed to open program file: " + path);

  json j;
  try {
    f >> j;
  } catch (const json::parse_error &e) {
    throw std::runtime_error("failed to parse program JSON (" + path +
                             "): " + e.what());
  }

  Program prog;
  try {
    prog.instructions.reserve(j.at("instructions").size());
    for (const json &s : j.at("instructions"))
      prog.instructions.push_back(parseHexInt(s.get<std::string>()));

    prog.data.reserve(j.at("data").size());
    for (const json &t : j.at("data"))
      prog.data.push_back(DataTile{
          static_cast<uint32_t>(parseHexInt(t.at("addr").get<std::string>())),
          parseHexBytes(t.at("bytes").get<std::string>())});

    prog.inputTiles = parseIoTiles(j.at("input_tiles"));
    prog.outputTiles = parseIoTiles(j.at("output_tiles"));
    prog.inputValidBytes = j.at("input_valid_bytes").get<int64_t>();
    prog.outputValidBytes = j.at("output_valid_bytes").get<int64_t>();
  } catch (const json::exception &e) {
    throw std::runtime_error("malformed program JSON (" + path +
                             "): " + e.what());
  }

  return prog;
}

} // namespace macaque::runtime
