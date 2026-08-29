#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "macaque/common/defs.hpp"
#include "macaque/runtime/Device.hpp"
#include "macaque/runtime/Image.hpp"
#include "macaque/runtime/Program.hpp"

namespace rt = macaque::runtime;

namespace {

struct Args {
  std::vector<std::string> positional;
  std::optional<std::string> port, input, image, output;
  float scale = 0.0f;
  int32_t zeroPoint = 0;
  int baud = macaque::common::kBaudRate;
  double timeoutSeconds = 10.0;
};

Args parseArgs(int argc, char **argv, int startIdx) {
  Args args;
  for (int i = startIdx; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char *flag) -> std::string {
      if (i + 1 >= argc)
        throw std::runtime_error(std::string(flag) + " needs a value");
      return argv[++i];
    };
    if (a == "--port")
      args.port = next("--port");
    else if (a == "--input")
      args.input = next("--input");
    else if (a == "--image")
      args.image = next("--image");
    else if (a == "--output")
      args.output = next("--output");
    else if (a == "--scale")
      args.scale = std::stof(next("--scale"));
    else if (a == "--zero-point")
      args.zeroPoint = std::stoi(next("--zero-point"));
    else if (a == "--baud")
      args.baud = std::stoi(next("--baud"));
    else if (a == "--timeout")
      args.timeoutSeconds = std::stod(next("--timeout"));
    else if (!a.empty() && a[0] == '-')
      throw std::runtime_error("unknown flag: " + a);
    else
      args.positional.push_back(a);
  }
  return args;
}

rt::Device openDevice(const Args &args) {
  if (!args.port)
    throw std::runtime_error("--port is required");
  return rt::Device(*args.port, args.baud);
}

int8_t toSigned(uint8_t b) { return static_cast<int8_t>(b); }

int cmdRun(const Args &args) {
  if (args.positional.empty())
    throw std::runtime_error("run needs a program.macq argument");
  const rt::Program prog = rt::Program::load(args.positional[0]);
  rt::Device device = openDevice(args);

  std::cout << "staging " << prog.data.size() << " data tiles...\n";
  device.stageProgramData(prog);

  std::vector<uint8_t> out;
  const auto t0 = std::chrono::steady_clock::now();

  if (args.image) {
    if (args.scale <= 0.0f)
      throw std::runtime_error("--image requires --scale");
    const rt::RawImage img = rt::loadPng(*args.image, /*desiredChannels=*/0);
    std::cout << "loading " << prog.instructions.size() << " instructions...\n";
    device.loadInstructions(prog);
    out = device.infer(prog, img, rt::QuantParams{args.scale, args.zeroPoint});
  } else {
    if (args.input) {
      std::ifstream f(*args.input, std::ios::binary);
      if (!f)
        throw std::runtime_error("failed to open --input file: " + *args.input);
      const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                                       std::istreambuf_iterator<char>());
      device.stageInput(prog, bytes);
    } else if (!prog.inputTiles.empty()) {
      std::cerr
          << "warning: program expects " << prog.inputValidBytes
          << " bytes of runtime input (--input/--image not given, leaving "
             "DDR3 unwritten there)\n";
    }
    std::cout << "loading " << prog.instructions.size() << " instructions...\n";
    device.loadInstructions(prog);
    device.triggerAndWait(args.timeoutSeconds);
    if (!prog.outputTiles.empty())
      out = device.readOutput(prog);
  }

  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  std::printf("run complete in %.3fs\n", elapsed);

  if (!out.empty()) {
    if (args.output) {
      std::ofstream f(*args.output, std::ios::binary);
      f.write(reinterpret_cast<const char *>(out.data()),
              static_cast<std::streamsize>(out.size()));
      std::cout << "wrote " << out.size() << " output bytes to " << *args.output
                << "\n";
    } else {
      std::cout << "output (signed int8): [";
      for (size_t i = 0; i < out.size(); ++i)
        std::cout << (i ? ", " : "") << static_cast<int>(toSigned(out[i]));
      std::cout << "]\n";
    }
  }

  const rt::PmuCounters pmu = device.readPmu();
  std::cout << "PMU: cycles=" << pmu.cycles << " compute=" << pmu.compute
            << " stall=" << pmu.stall << " dma_rd=" << pmu.dmaBytesRd
            << " dma_wr=" << pmu.dmaBytesWr << "\n";
  return 0;
}

int cmdStatus(const Args &args) {
  rt::Device device = openDevice(args);
  const rt::PmuCounters pmu = device.readPmu();
  std::cout << "PMU: cycles=" << pmu.cycles << " compute=" << pmu.compute
            << " stall=" << pmu.stall << " dma_rd=" << pmu.dmaBytesRd
            << " dma_wr=" << pmu.dmaBytesWr << "\n";
  return 0;
}

void printUsage() {
  std::cerr
      << "usage: macaque <run|status> ...\n"
         "  macaque run <program.macq> --port PORT [--input FILE | --image "
         "FILE.png --scale S [--zero-point Z]] [--output FILE]\n"
         "  macaque status --port PORT\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    printUsage();
    return 1;
  }
  const std::string command = argv[1];
  try {
    const Args args = parseArgs(argc, argv, 2);
    if (command == "run")
      return cmdRun(args);
    if (command == "status")
      return cmdStatus(args);
    printUsage();
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
