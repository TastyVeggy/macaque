#include <iostream>
#include <stdexcept>

#include "macaque/runtime/Device.hpp"
#include "macaque/runtime/Image.hpp"
#include "macaque/runtime/Program.hpp"

namespace rt = macaque::runtime;

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0]
              << " <program.json> <digit.png> <serial_port>\n";
    return 1;
  }

  try {
    const rt::Program prog = rt::Program::load(argv[1]);
    rt::Device device(argv[3]);
    device.stageProgramData(prog);
    device.loadInstructions(prog);

    const rt::RawImage img = rt::loadPng(argv[2], /*desiredChannels=*/1);

    constexpr rt::QuantParams kMnistInputQuant{/*scale=*/255.0f / 127.0f,
                                               /*zeroPoint=*/0};

    const std::vector<uint8_t> output =
        device.infer(prog, img, kMnistInputQuant);

    if (output.empty())
      throw std::runtime_error("device.infer() returned no output bytes - program has no output tiles");

    int predicted = 0;
    int8_t best = static_cast<int8_t>(output[0]);
    for (size_t c = 1; c < output.size(); ++c) {
      const auto v = static_cast<int8_t>(output[c]);
      if (v > best) {
        best = v;
        predicted = static_cast<int>(c);
      }
    }
    std::cout << "predicted digit: " << predicted << "\n";
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
