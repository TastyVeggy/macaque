import argparse
import sys
import time

import serial

OP_STATUS = 0x53
OP_WRITE = 0x57
OP_READ = 0x52

# Test addresses and a distinctive 64-bit pattern (little-endian on the wire).
TEST_ADDRS = [0x0000, 0x0010, 0x0100, 0x1000, 0x1FFF0]
PATTERN = 0xDEAD_BEEF_CAFE_F00D


def expect_exact(port, n, what):
    data = port.read(n)
    if len(data) != n:
        raise RuntimeError(f"short read for {what}: got {len(data)}/{n} bytes")
    return data


def send_status(port):
    port.write(bytes([OP_STATUS]))
    port.flush()
    s = expect_exact(port, 1, "status")[0]
    return s


def write64(port, addr, data):
    port.write(bytes([OP_WRITE]))
    port.write(addr.to_bytes(4, "little"))
    port.write(data.to_bytes(8, "little"))
    port.flush()
    ack = expect_exact(port, 1, "write ack")[0]
    if ack != 0x00:
        raise RuntimeError(f"write ack != 0x00 (got 0x{ack:02x})")


def read64(port, addr):
    port.write(bytes([OP_READ]))
    port.write(addr.to_bytes(4, "little"))
    port.flush()
    d = expect_exact(port, 8, "read data")
    return int.from_bytes(d, "little")


def main():
    ap = argparse.ArgumentParser(description="DDR3 UART read/write self-test")
    ap.add_argument("port", help="serial device, e.g. /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument(
        "--read",
        type=lambda x: int(x, 0),
        default=None,
        help="single address to read (hex ok) instead of self-test",
    )
    ap.add_argument(
        "--write-only", action="store_true", help="skip read-back verification"
    )
    args = ap.parse_args()

    port = serial.Serial(args.port, args.baud, timeout=2.0)
    time.sleep(0.2)  # let the FPGA settle

    # Status byte: bit0 = mmcm_locked, bit1 = init_calib_complete.
    s = send_status(port)
    locked = bool(s & 0x01)
    calib = bool(s & 0x02)
    print(f"status=0x{s:02x} mmcm_locked={int(locked)} calib={int(calib)}")
    if not locked or not calib:
        print("WARNING: MIG not ready (mmcm_locked/calib low) — DDR3 may not work.")

    if args.read is not None:
        v = read64(port, args.read)
        print(f"read  0x{args.read:08x} -> 0x{v:016x}")
        return

    # Self-test: write a distinct pattern to several addresses, read them back.
    fails = 0
    for i, addr in enumerate(TEST_ADDRS):
        val = PATTERN ^ i
        write64(port, addr, val)
        if args.write_only:
            print(f"write 0x{addr:08x} <- 0x{val:016x}")
            continue
        got = read64(port, addr)
        ok = got == val
        fails += not ok
        print(
            f"0x{addr:08x}: wrote 0x{val:016x}  read 0x{got:016x}  "
            f"{'OK' if ok else 'MISMATCH'}"
        )

    if args.write_only:
        print("write-only done (no verification)")
    else:
        print("PASS" if fails == 0 else f"FAIL ({fails} mismatches)")
        sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
