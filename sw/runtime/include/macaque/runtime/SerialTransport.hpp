#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace macaque::runtime {

class SerialTransport {
public:
  SerialTransport(const std::string &device, int baud,
                  double timeoutSeconds = 2.0);
  ~SerialTransport();

  SerialTransport(const SerialTransport &) = delete;
  SerialTransport &operator=(const SerialTransport &) = delete;

  void write(const uint8_t *data, size_t len);
  void flush();

  // Blocks until exactly `n` bytes are available and returns them, or
  // throws std::runtime_error on a short read/timeout.
  [[nodiscard]] std::vector<uint8_t> read(size_t n);

private:
  int fd_ = -1;
  double timeoutSeconds_;
};

} // namespace macaque::runtime
