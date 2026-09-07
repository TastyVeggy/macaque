#include "macaque/runtime/SerialTransport.hpp"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace macaque::runtime {

namespace {

speed_t baudToSpeed(int baud) {
  switch (baud) {
  case 9600:
    return B9600;
  case 19200:
    return B19200;
  case 38400:
    return B38400;
  case 57600:
    return B57600;
  case 115200:
    return B115200;
  case 230400:
    return B230400;
  case 460800:
    return B460800;
  case 921600:
    return B921600;
  default:
    throw std::runtime_error(
        "unsupported baud rate " + std::to_string(baud) +
        " (only standard POSIX termios rates are supported)");
  }
}

} // namespace

SerialTransport::SerialTransport(const std::string &device, int baud,
                                 double timeoutSeconds)
    : timeoutSeconds_(timeoutSeconds) {
  fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0)
    throw std::runtime_error("failed to open " + device + ": " +
                             std::strerror(errno));

  termios tty{};
  if (::tcgetattr(fd_, &tty) != 0) {
    ::close(fd_);
    throw std::runtime_error("tcgetattr failed on " + device + ": " +
                             std::strerror(errno));
  }

  ::cfmakeraw(&tty);
  const speed_t speed = baudToSpeed(baud);
  ::cfsetispeed(&tty, speed);
  ::cfsetospeed(&tty, speed);
  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_cflag &= ~static_cast<tcflag_t>(PARENB); // no parity
  tty.c_cflag &= ~static_cast<tcflag_t>(CSTOPB); // 1 stop bit
  tty.c_cflag &= ~static_cast<tcflag_t>(CSIZE);
  tty.c_cflag |= CS8; // 8 data bits
  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 1;

  if (::tcsetattr(fd_, TCSANOW, &tty) != 0) {
    ::close(fd_);
    throw std::runtime_error("tcsetattr failed on " + device + ": " +
                             std::strerror(errno));
  }

  // The port itself should block (via VTIME) rather than the fd; undo
  // O_NONBLOCK now that termios is configured.
  const int flags = ::fcntl(fd_, F_GETFL, 0);
  ::fcntl(fd_, F_SETFL, flags & ~O_NONBLOCK);
}

SerialTransport::~SerialTransport() {
  if (fd_ >= 0)
    ::close(fd_);
}

void SerialTransport::write(const uint8_t *data, size_t len) {
  size_t written = 0;
  while (written < len) {
    const ssize_t n = ::write(fd_, data + written, len - written);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      throw std::runtime_error(std::string("serial write failed: ") +
                               std::strerror(errno));
    }
    written += static_cast<size_t>(n);
  }
}

void SerialTransport::flush() {
  if (::tcdrain(fd_) != 0)
    throw std::runtime_error(std::string("tcdrain failed: ") +
                             std::strerror(errno));
}

std::vector<uint8_t> SerialTransport::read(size_t n) {
  std::vector<uint8_t> out;
  out.reserve(n);
  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(timeoutSeconds_));

  while (out.size() < n) {
    if (std::chrono::steady_clock::now() >= deadline)
      break;
    uint8_t buf[256];
    const ssize_t got = ::read(fd_, buf, std::min(sizeof(buf), n - out.size()));
    if (got < 0) {
      if (errno == EINTR)
        continue;
      throw std::runtime_error(std::string("serial read failed: ") +
                               std::strerror(errno));
    }
    out.insert(out.end(), buf, buf + got);
  }

  if (out.size() != n) {
    throw std::runtime_error("serial read timed out: got " +
                             std::to_string(out.size()) + "/" +
                             std::to_string(n) + " bytes");
  }
  return out;
}

} // namespace macaque::runtime
