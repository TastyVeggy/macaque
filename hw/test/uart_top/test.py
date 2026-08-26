import argparse
import sys
import time

import serial

PAYLOAD = bytes([0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF, 0x5A, 0xA5]) + b"hello macaque"


def main():
    ap = argparse.ArgumentParser(description="uart_top UART echo test")
    ap.add_argument("port", help="serial device, e.g. /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument(
        "--inter-byte",
        type=float,
        default=0.02,
        help="seconds to pause between sent bytes (echo has no flow control)",
    )
    ap.add_argument("--timeout", type=float, default=2.0)
    args = ap.parse_args()

    port = serial.Serial(args.port, args.baud, timeout=args.timeout)
    time.sleep(0.2)  # let the FPGA resettle after programming

    port.reset_input_buffer()

    # Send one byte at a time, reading each echo back before the next.
    failures = 0
    for i, b in enumerate(PAYLOAD):
        port.write(bytes([b]))
        port.flush()
        got = port.read(1)
        if len(got) != 1:
            print(f"[{i:02d}]  tx 0x{b:02x}  rx <TIMEOUT>")
            failures += 1
            continue
        ok = got[0] == b
        failures += not ok
        print(
            f"[{i:02d}]  tx 0x{b:02x}  rx 0x{got[0]:02x}  {'OK' if ok else 'MISMATCH'}"
        )
        time.sleep(args.inter_byte)

    print(f"\nsent {len(PAYLOAD)} bytes, {failures} failure(s)")
    print("PASS" if failures == 0 else "FAIL")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
