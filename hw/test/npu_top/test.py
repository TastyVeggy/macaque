import argparse
import sys
import time

import serial

REG_BASE = 0x4000_0000
REG_CTRL = REG_BASE + 0x00
REG_STATUS = REG_BASE + 0x08
REG_INSTR_ADDR = REG_BASE + 0x10
REG_INSTR_LEN = REG_BASE + 0x18
REG_PMU_CTRL = REG_BASE + 0x20
REG_PMU_CYCLES = REG_BASE + 0x28   # 64-bit
REG_PMU_COMPUTE = REG_BASE + 0x30
REG_PMU_STALL = REG_BASE + 0x38
REG_PMU_DMA_RD = REG_BASE + 0x40
REG_PMU_DMA_WR = REG_BASE + 0x48
IMEM_BASE = 0x5000_0000

OP_LOAD_W = 0x0
OP_LOAD_B = 0x1
OP_LOAD_ACT = 0x2
OP_MATMUL = 0x3
OP_ACTIVATE = 0x4
OP_STORE = 0x5

ACT_PASSTHROUGH = 0x2

ARRAY = 14  # npu_pkg::ARRAY_SIZE

M, K, N = 100, 130, 150

OP_STATUS = 0x53
OP_WRITE = 0x57
OP_READ = 0x52


def expect_exact(port, n, what):
    data = port.read(n)
    if len(data) != n:
        raise RuntimeError(f"short read for {what}: got {len(data)}/{n} bytes")
    return data


def write64(port, addr, data):
    port.write(bytes([OP_WRITE]))
    port.write((addr & 0xFFFFFFFF).to_bytes(4, "little"))
    port.write((data & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"))
    port.flush()
    ack = expect_exact(port, 1, "write ack")[0]
    if ack != 0x00:
        raise RuntimeError(f"write ack != 0x00 (got 0x{ack:02x})")


def read64(port, addr):
    port.write(bytes([OP_READ]))
    port.write((addr & 0xFFFFFFFF).to_bytes(4, "little"))
    port.flush()
    d = expect_exact(port, 8, "read data")
    return int.from_bytes(d, "little")


def send_status(port):
    port.write(bytes([OP_STATUS]))
    port.flush()
    return expect_exact(port, 1, "status")[0]


def enc(opcode, acc_mode=0, target=0, ddr3_addr=0, byte_count=0, tile_params=0):
    return ((opcode & 0xF) << 60) | ((acc_mode & 1) << 59) | ((target & 0x7) << 56) \
        | ((ddr3_addr & 0xFFFFFFF) << 28) | ((byte_count & 0xFFFF) << 12) \
        | (tile_params & 0xFFF)


def s8(v):
    return v & 0xFF


def gen_matrices():
    A = [[((m + k) % 3) - 1 for k in range(K)] for m in range(M)]
    W = [[((k + n) % 3) - 1 for n in range(N)] for k in range(K)]
    return A, W


def ref_matmul(A, W):
    O = [[0] * N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            s = 0
            for k in range(K):
                s += A[m][k] * W[k][n]
            O[m][n] = s
    return O


def make_tiles(total, size=ARRAY):
    tiles = []
    off = 0
    while off < total:
        sz = min(size, total - off)
        tiles.append((off, sz))
        off += sz
    return tiles


def align_up(addr, align=0x1000):
    return (addr + align - 1) // align * align


def build(A, W):
    O = ref_matmul(A, W)

    M_tiles = make_tiles(M)
    K_tiles = make_tiles(K)
    N_tiles = make_tiles(N)

    weight_base = 0x001000
    weight_total = len(K_tiles) * len(N_tiles) * 200
    bias_base = align_up(weight_base + weight_total)
    bias_total = len(N_tiles) * 56
    act_base = align_up(bias_base + bias_total)
    act_total = len(M_tiles) * len(K_tiles) * 200
    out_base = align_up(act_base + act_total)

    instrs = []
    ddr = []
    out = []

    # ---- weights ----
    waddr = weight_base
    weight_tile_addr = {}  # (k_idx, n_idx) -> addr
    for ki, (ko, ks) in enumerate(K_tiles):
        for ni, (no, ns) in enumerate(N_tiles):
            rows = []
            for kk in range(ARRAY):          # K row within tile
                row = []
                for nn in range(ARRAY):      # N col within tile
                    k = ko + kk
                    n = no + nn
                    row.append(s8(W[k][n]) if (k < K and n < N) else 0)
                rows.extend(row)
            weight_tile_addr[(ki, ni)] = waddr
            ddr.append((waddr, rows))
            waddr += 200  # 14*14 = 196 -> pad to 200

    # ---- bias ----
    baddr = bias_base
    bias_tile_addr = {}
    for ni, (no, ns) in enumerate(N_tiles):
        vals = [0] * ARRAY  # bias = 0 for this test
        bias_tile_addr[ni] = baddr
        ddr.append((baddr, flatten_i32(vals)))
        baddr += 56

    # ---- activations ----
    aaddr = act_base
    act_tile_addr = {}  # (m_idx, k_idx) -> addr
    for mi, (mo, ms) in enumerate(M_tiles):
        for ki, (ko, ks) in enumerate(K_tiles):
            rows = []
            for mm in range(ms):             # M row within tile
                for kk in range(ARRAY):      # K lane within tile
                    k = ko + kk
                    rows.append(s8(A[mo + mm][k]) if (k < K) else 0)
            act_tile_addr[(mi, ki)] = aaddr
            ddr.append((aaddr, rows))
            aaddr += 200  # pad each tile to a multiple of 8

    # ---- instruction stream + expected output ----
    oaddr = out_base
    for ni, (no, ns) in enumerate(N_tiles):
        for mi, (mo, ms) in enumerate(M_tiles):
            for ki, (ko, ks) in enumerate(K_tiles):
                instrs.append(enc(OP_LOAD_W, ddr3_addr=weight_tile_addr[(ki, ni)],
                                  byte_count=14 * ARRAY))
                if ki == 0:
                    instrs.append(enc(OP_LOAD_B, ddr3_addr=bias_tile_addr[ni],
                                      byte_count=ARRAY * 4))
                instrs.append(enc(OP_LOAD_ACT, ddr3_addr=act_tile_addr[(mi, ki)],
                                  byte_count=ms * ARRAY))
                instrs.append(enc(OP_MATMUL, acc_mode=(0 if ki == 0 else 1),
                                  tile_params=ms))
            # ACTIVATE passthrough: scale_m=1, shift=0, func=PASSTHROUGH
            instrs.append(enc(OP_ACTIVATE, target=ACT_PASSTHROUGH,
                              ddr3_addr=1, byte_count=0, tile_params=ms))
            instrs.append(enc(OP_STORE, ddr3_addr=oaddr, byte_count=ms * ARRAY))

            exp = []
            for mm in range(ms):
                for nn in range(ARRAY):
                    m = mo + mm
                    n = no + nn
                    if n < N:
                        # passthrough (scale_m=1, shift=0): activate_unit still
                        # clamps to signed INT8 before truncating. Unlike
                        # test.py's small K=30 (sums always fit int8 unclamped),
                        # K=130 here can produce |sum| > 127, so the clamp is
                        # required for the golden value to match hardware.
                        exp.append(max(-128, min(127, O[m][n])))
                    else:
                        exp.append(0)
            out.append((oaddr, exp))
            oaddr += 200

    return instrs, ddr, out


def flatten_i32(vals):
    b = []
    for v in vals:
        b.extend([(v >> 0) & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF])
    return b


def stage_bytes(port, addr, byte_list):
    """Write a byte list to DDR3 in 8-byte little-endian chunks (zero padded)."""
    n = len(byte_list)
    for i in range(0, n, 8):
        chunk = byte_list[i:i + 8]
        chunk += [0] * (8 - len(chunk))
        val = 0
        for j, b in enumerate(chunk):
            val |= b << (8 * j)
        write64(port, addr + i, val)


def main():
    ap = argparse.ArgumentParser(
        description=f"NPU {M}x{K} * {K}x{N} matmul stress test (much larger than test.py)")
    ap.add_argument("port", help="serial device, e.g. /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    port = serial.Serial(args.port, args.baud, timeout=2.0)
    time.sleep(0.2)

    s = send_status(port)
    print(f"status=0x{s:02x} mmcm_locked={int(bool(s & 1))} calib={int(bool(s & 2))}")
    if not (s & 0x03):
        print("ERROR: MIG not ready (locked/calib low)")
        sys.exit(1)

    A, W = gen_matrices()
    O = ref_matmul(A, W)
    t0 = time.time()
    instrs, ddr, out = build(A, W)

    print(f"generated {len(instrs)} instructions, "
          f"{sum(len(d) for _, d in ddr)} bytes to stage, "
          f"{sum(len(e) for _, e in out)} output bytes to verify")

    # ---- stage weights / bias / activations into DDR3 ----
    for addr, data in ddr:
        stage_bytes(port, addr, data)
    print(f"staged in {time.time() - t0:.1f}s")

    # ---- load instruction stream into imem ----
    t1 = time.time()
    for i, w in enumerate(instrs):
        write64(port, IMEM_BASE + 8 * i, w)
    print(f"program loaded in {time.time() - t1:.1f}s")

    # ---- PMU: clear then enable ----
    write64(port, REG_PMU_CTRL, 0x2)  # clear (W1C)
    write64(port, REG_PMU_CTRL, 0x1)  # enable

    # ---- program start ----
    write64(port, REG_INSTR_ADDR, 0)
    write64(port, REG_INSTR_LEN, len(instrs))
    write64(port, REG_CTRL, 0x1)  # start

    # ---- poll done ----
    t2 = time.time()
    done = False
    for _ in range(2_000_000):
        st = read64(port, REG_STATUS)
        if st & 0x4:  # done
            done = True
            break
        if bool(st & 0x8):
            print(f"ERROR: npu_error asserted (STATUS=0x{st:08x})")
            sys.exit(1)
        time.sleep(0.001)
    if not done:
        print("ERROR: timeout waiting for STATUS.done")
        sys.exit(1)
    print(f"NPU run complete in {time.time() - t2:.1f}s")

    # ---- verify output ----
    t3 = time.time()
    fails = 0
    for addr, exp in out:
        for i in range(0, len(exp), 8):
            got64 = read64(port, addr + i)
            for j in range(8):
                idx = i + j
                if idx >= len(exp):
                    break
                gb = (got64 >> (8 * j)) & 0xFF
                g = gb - 256 if gb >= 128 else gb
                if g != exp[idx]:
                    fails += 1
                    if fails <= 8:
                        print(f"  OUT@{addr + idx:#x}: got {g}, want {exp[idx]}")
    print(f"verified in {time.time() - t3:.1f}s")

    print("MATMUL", "PASS" if fails == 0 else f"FAIL ({fails} mismatches)")

    # ---- read PMU ----
    pmu_cycles = read64(port, REG_PMU_CYCLES)
    pmu_compute = read64(port, REG_PMU_COMPUTE) & 0xFFFFFFFF
    pmu_stall = read64(port, REG_PMU_STALL) & 0xFFFFFFFF
    pmu_dma_rd = read64(port, REG_PMU_DMA_RD) & 0xFFFFFFFF
    pmu_dma_wr = read64(port, REG_PMU_DMA_WR) & 0xFFFFFFFF
    print("PMU:")
    print(f"  cycles        = {pmu_cycles}")
    print(f"  compute       = {pmu_compute}")
    print(f"  stall         = {pmu_stall}")
    print(f"  dma_bytes_rd  = {pmu_dma_rd}")
    print(f"  dma_bytes_wr  = {pmu_dma_wr}")

    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
