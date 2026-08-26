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
REG_PMU_CYCLES = REG_BASE + 0x28  # 64-bit
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

ACT_RELU = 0x0
ACT_LEAKY_RELU = 0x1
ACT_PASSTHROUGH = 0x2

LEAKY_RELU_SHIFT = 4

MATMUL_WEIGHT_HOLD = 0x1

ARRAY = 14  # npu_pkg::ARRAY_SIZE

M, K, N = 100, 130, 150

HOLD_M, HOLD_K, HOLD_N = 50, 10, 20

HOLD_KTILE_M, HOLD_KTILE_K, HOLD_KTILE_N = 270, 28, 14

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
    return (
        ((opcode & 0xF) << 60)
        | ((acc_mode & 1) << 59)
        | ((target & 0x7) << 56)
        | ((ddr3_addr & 0xFFFFFFF) << 28)
        | ((byte_count & 0xFFFF) << 12)
        | (tile_params & 0xFFF)
    )


def s8(v):
    return v & 0xFF


def gen_matrices(M, K, N, seed=0):
    A = [[((m + k + seed) % 3) - 1 for k in range(K)] for m in range(M)]
    W = [[((k + n + seed) % 3) - 1 for n in range(N)] for k in range(K)]
    return A, W


def ref_matmul(A, W, M, K, N):
    O = [[0] * N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            s = 0
            for k in range(K):
                s += A[m][k] * W[k][n]
            O[m][n] = s
    return O


def requantize_activate(acc, mult, shift, act_func):
    scaled = acc * mult
    if shift > 0:
        scaled += 1 << (shift - 1)
    requant = scaled >> shift
    if act_func == ACT_RELU:
        act = max(0, requant)
    elif act_func == ACT_LEAKY_RELU:
        act = requant if requant >= 0 else (requant >> LEAKY_RELU_SHIFT)
    else:  # ACT_PASSTHROUGH
        act = requant
    return max(-128, min(127, act))


def act_byte_count(shift, row_base=0, bank_hold=0):
    """ACTIVATE's byte_count packs three fields: [4:0] scale_shift, [12:5]
    row_base (out_buffer row this M-chunk's accumulator starts at - nonzero
    only for weight-hold combined with K-tiling), [13] bank_hold (skip the
    out_bank_sel toggle - more chunks from the same held-batch bank still
    need draining). See README.md's ACTIVATE field reinterpretation table.
    """
    return (shift & 0x1F) | ((row_base & 0xFF) << 5) | ((bank_hold & 1) << 13)


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


def build(
    A, W, M, K, N, use_weight_hold=False, act_func=ACT_PASSTHROUGH, mult=1, shift=0
):
    O = ref_matmul(A, W, M, K, N)

    M_tiles = make_tiles(M)
    K_tiles = make_tiles(K)
    N_tiles = make_tiles(N)
    hold = use_weight_hold
    max_tiles_per_hold_batch = 256 // ARRAY

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
            for kk in range(ARRAY):  # K row within tile
                row = []
                for nn in range(ARRAY):  # N col within tile
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
            for mm in range(ms):  # M row within tile
                for kk in range(ARRAY):  # K lane within tile
                    k = ko + kk
                    rows.append(s8(A[mo + mm][k]) if (k < K) else 0)
            act_tile_addr[(mi, ki)] = aaddr
            ddr.append((aaddr, rows))
            aaddr += 200  # pad each tile to a multiple of 8

    # ---- instruction stream + expected output ----
    oaddr = out_base

    def emit_chunk_output(mi, mo, ms, no):
        nonlocal oaddr
        instrs.append(enc(OP_STORE, ddr3_addr=oaddr, byte_count=ms * ARRAY))
        exp = []
        for mm in range(ms):
            for nn in range(ARRAY):
                m, n = mo + mm, no + nn
                exp.append(
                    requantize_activate(O[m][n], mult, shift, act_func) if n < N else 0
                )
        out.append((oaddr, exp))
        oaddr += 200

    if hold and len(K_tiles) > 1:
        for ni, (no, ns) in enumerate(N_tiles):
            for batch_start in range(0, len(M_tiles), max_tiles_per_hold_batch):
                batch = M_tiles[batch_start : batch_start + max_tiles_per_hold_batch]
                for ki, (ko, ks) in enumerate(K_tiles):
                    for ci, (mo, ms) in enumerate(batch):
                        mi = batch_start + ci
                        held = ci > 0
                        row_base = ci * ARRAY
                        if not held:
                            instrs.append(
                                enc(
                                    OP_LOAD_W,
                                    ddr3_addr=weight_tile_addr[(ki, ni)],
                                    byte_count=14 * ARRAY,
                                )
                            )
                            if ki == 0:
                                instrs.append(
                                    enc(
                                        OP_LOAD_B,
                                        ddr3_addr=bias_tile_addr[ni],
                                        byte_count=ARRAY * 4,
                                    )
                                )
                        instrs.append(
                            enc(
                                OP_LOAD_ACT,
                                ddr3_addr=act_tile_addr[(mi, ki)],
                                byte_count=ms * ARRAY,
                            )
                        )
                        instrs.append(
                            enc(
                                OP_MATMUL,
                                acc_mode=(0 if ki == 0 else 1),
                                target=(MATMUL_WEIGHT_HOLD if held else 0),
                                ddr3_addr=row_base,
                                tile_params=ms,
                            )
                        )
                # Drain the whole batch: one ACTIVATE+STORE per M-tile,
                # bank_hold=1 on every chunk but the batch's last (which
                # toggles out_bank_sel, same as an ordinary activate).
                for ci, (mo, ms) in enumerate(batch):
                    mi = batch_start + ci
                    last = ci == len(batch) - 1
                    instrs.append(
                        enc(
                            OP_ACTIVATE,
                            target=act_func,
                            ddr3_addr=mult,
                            byte_count=act_byte_count(
                                shift, ci * ARRAY, bank_hold=0 if last else 1
                            ),
                            tile_params=ms,
                        )
                    )
                    emit_chunk_output(mi, mo, ms, no)
        return instrs, ddr, out

    for ni, (no, ns) in enumerate(N_tiles):
        for mi, (mo, ms) in enumerate(M_tiles):
            for ki, (ko, ks) in enumerate(K_tiles):
                # hold implies exactly one K-tile, so ki==0 always here; mi>0
                # is what makes this a held (reuse, no reload) M-chunk.
                held = hold and mi > 0
                if not held:
                    instrs.append(
                        enc(
                            OP_LOAD_W,
                            ddr3_addr=weight_tile_addr[(ki, ni)],
                            byte_count=14 * ARRAY,
                        )
                    )
                    if ki == 0:
                        instrs.append(
                            enc(
                                OP_LOAD_B,
                                ddr3_addr=bias_tile_addr[ni],
                                byte_count=ARRAY * 4,
                            )
                        )
                instrs.append(
                    enc(
                        OP_LOAD_ACT,
                        ddr3_addr=act_tile_addr[(mi, ki)],
                        byte_count=ms * ARRAY,
                    )
                )
                instrs.append(
                    enc(
                        OP_MATMUL,
                        acc_mode=(0 if ki == 0 else 1),
                        target=(MATMUL_WEIGHT_HOLD if held else 0),
                        tile_params=ms,
                    )
                )
            instrs.append(
                enc(
                    OP_ACTIVATE,
                    target=act_func,
                    ddr3_addr=mult,
                    byte_count=act_byte_count(shift),
                    tile_params=ms,
                )
            )
            emit_chunk_output(mi, mo, ms, no)

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
        chunk = byte_list[i : i + 8]
        chunk += [0] * (8 - len(chunk))
        val = 0
        for j, b in enumerate(chunk):
            val |= b << (8 * j)
        write64(port, addr + i, val)


SCENARIO_RESULTS = []


def run_program(port, name, instrs, ddr, verify):
    """Stage `ddr` into DDR3, load `instrs` into imem, run to completion, and
    check every (addr, expected_bytes) entry in `verify` - shared by every
    scenario below, single-layer or chained, regardless of how the program
    was built.
    """
    print(f"\n=== scenario: {name} ===")
    print(
        f"{len(instrs)} instructions, {sum(len(d) for _, d in ddr)} bytes to "
        f"stage, {sum(len(e) for _, e in verify)} output bytes to verify"
    )

    t0 = time.time()
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
    for addr, exp in verify:
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

    print(f"{name}:", "PASS" if fails == 0 else f"FAIL ({fails} mismatches)")
    SCENARIO_RESULTS.append((name, fails == 0, fails))

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

    return fails == 0


ACT_FUNC_NAMES = {
    ACT_RELU: "relu",
    ACT_LEAKY_RELU: "leaky_relu",
    ACT_PASSTHROUGH: "passthrough",
}


def run_scenario(
    port, name, M, K, N, use_weight_hold=False, seed=0, act_func=ACT_PASSTHROUGH
):
    tags = []
    if use_weight_hold:
        tags.append("weight_hold")
    if act_func != ACT_PASSTHROUGH:
        tags.append(f"act={ACT_FUNC_NAMES[act_func]}")
    tag_str = f", {', '.join(tags)}" if tags else ""
    label = f"{name} ({M}x{K} * {K}x{N}{tag_str})"
    A, W = gen_matrices(M, K, N, seed=seed)
    instrs, ddr, out = build(
        A, W, M, K, N, use_weight_hold=use_weight_hold, act_func=act_func
    )
    return run_program(port, label, instrs, ddr, out)


def build_chain(M, layers):
    """Build a multi-layer chained program: layers[i] is a dict with keys K,
    N, act_func, and optionally mult/shift/hold/seed. layers[0]'s K is the
    true input K (freshly staged activation); each later layer's K must equal
    the previous layer's N - its activation is the *same DDR3 bytes* the
    previous layer's store just wrote (Scratch A/B chaining, hand-built the
    same way TosaToMacaque.cpp's RescaleToMacaque would address it), not a
    fresh copy.

    Every non-final layer's N must fit one 14-wide tile: RescaleToMacaque
    rejects N-tiling *and* M-chunking an intermediate (layer-to-layer)
    producer (Scratch A/B assumes a single tile/chunk - see ROADMAP.md), so
    this keeps the hand-built stream a program the real compiler could
    actually emit, not just something the RTL happens to tolerate. Only the
    final layer may N-tile.

    Returns (instrs, ddr, verify) where `verify` covers *every* layer's
    output, not just the final one - so a failure localizes to the specific
    layer that produced wrong data instead of just "somewhere in the chain".
    """
    for i, layer in enumerate(layers):
        if i > 0:
            assert layer["K"] == layers[i - 1]["N"], (
                f"layer {i}'s K must equal layer {i - 1}'s N (chained activation shape)"
            )
        if i < len(layers) - 1:
            assert layer["N"] <= ARRAY, (
                f"layer {i} is non-final so its N must fit one tile (<= {ARRAY})"
            )

    M_tiles = make_tiles(M)

    addr_cursor = [0x001000]

    def alloc(n):
        a = addr_cursor[0]
        addr_cursor[0] = align_up(a + n, 8)
        return a

    instrs = []
    ddr = []
    verify = []

    prev_A = None
    prev_store_addr = None  # {mi: addr} of the previous layer's (single) N-tile

    for li, layer in enumerate(layers):
        K, N = layer["K"], layer["N"]
        act_func = layer.get("act_func", ACT_PASSTHROUGH)
        mult, shift = layer.get("mult", 1), layer.get("shift", 0)
        K_tiles = make_tiles(K)
        N_tiles = make_tiles(N)
        hold = layer.get("hold", False) and len(K_tiles) == 1

        if li == 0:
            A, W = gen_matrices(M, K, N, seed=layer.get("seed", 0))
        else:
            A = prev_A
            _, W = gen_matrices(M, K, N, seed=layer.get("seed", li))

        O = ref_matmul(A, W, M, K, N)

        # ---- this layer's weights ----
        weight_tile_addr = {}
        for ki, (ko, ks) in enumerate(K_tiles):
            for ni, (no, ns) in enumerate(N_tiles):
                rows = []
                for kk in range(ARRAY):
                    for nn in range(ARRAY):
                        k, n = ko + kk, no + nn
                        rows.append(s8(W[k][n]) if (k < K and n < N) else 0)
                addr = alloc(200)
                weight_tile_addr[(ki, ni)] = addr
                ddr.append((addr, rows))

        # ---- this layer's bias (zero, same convention as build()) ----
        bias_tile_addr = {}
        for ni, (no, ns) in enumerate(N_tiles):
            addr = alloc(56)
            bias_tile_addr[ni] = addr
            ddr.append((addr, flatten_i32([0] * ARRAY)))

        # ---- this layer's activation: fresh-staged for layer 0, chained
        # (reads the previous layer's own store address) otherwise ----
        act_tile_addr = {}  # (mi, ki) -> addr
        if li == 0:
            for mi, (mo, ms) in enumerate(M_tiles):
                for ki, (ko, ks) in enumerate(K_tiles):
                    rows = []
                    for mm in range(ms):
                        for kk in range(ARRAY):
                            k = ko + kk
                            rows.append(s8(A[mo + mm][k]) if k < K else 0)
                    addr = alloc(200)
                    act_tile_addr[(mi, ki)] = addr
                    ddr.append((addr, rows))
        else:
            for mi in range(len(M_tiles)):
                act_tile_addr[(mi, 0)] = prev_store_addr[mi]

        # ---- matmul/activate/store, one group per (N-tile, M-tile) ----
        store_addr = {}
        for ni, (no, ns) in enumerate(N_tiles):
            for mi, (mo, ms) in enumerate(M_tiles):
                for ki, (ko, ks) in enumerate(K_tiles):
                    held = hold and mi > 0
                    if not held:
                        instrs.append(
                            enc(
                                OP_LOAD_W,
                                ddr3_addr=weight_tile_addr[(ki, ni)],
                                byte_count=14 * ARRAY,
                            )
                        )
                        if ki == 0:
                            instrs.append(
                                enc(
                                    OP_LOAD_B,
                                    ddr3_addr=bias_tile_addr[ni],
                                    byte_count=ARRAY * 4,
                                )
                            )
                    instrs.append(
                        enc(
                            OP_LOAD_ACT,
                            ddr3_addr=act_tile_addr[(mi, ki)],
                            byte_count=ms * ARRAY,
                        )
                    )
                    instrs.append(
                        enc(
                            OP_MATMUL,
                            acc_mode=(0 if ki == 0 else 1),
                            target=(MATMUL_WEIGHT_HOLD if held else 0),
                            tile_params=ms,
                        )
                    )
                instrs.append(
                    enc(
                        OP_ACTIVATE,
                        target=act_func,
                        ddr3_addr=mult,
                        byte_count=shift,
                        tile_params=ms,
                    )
                )
                addr = alloc(200)
                if ni == 0:
                    store_addr[mi] = addr  # only N-tile 0 feeds a later layer
                instrs.append(enc(OP_STORE, ddr3_addr=addr, byte_count=ms * ARRAY))

                exp = []
                for mm in range(ms):
                    for nn in range(ARRAY):
                        m, n = mo + mm, no + nn
                        exp.append(
                            requantize_activate(O[m][n], mult, shift, act_func)
                            if n < N
                            else 0
                        )
                verify.append((addr, exp))

        # Dense (unpadded) quantized output, exactly the bytes the next
        # layer's chained load_input will read back - this becomes that
        # layer's own reference `A`.
        prev_A = [
            [requantize_activate(O[m][n], mult, shift, act_func) for n in range(N)]
            for m in range(M)
        ]
        prev_store_addr = store_addr

    return instrs, ddr, verify


def run_chain_scenario(port, name, M, layers):
    def describe(layer):
        act_name = ACT_FUNC_NAMES[layer.get("act_func", ACT_PASSTHROUGH)]
        hold_tag = "+hold" if layer.get("hold") else ""
        return f"K{layer['K']}N{layer['N']}{hold_tag}+{act_name}"

    layer_desc = " -> ".join(describe(layer) for layer in layers)
    instrs, ddr, verify = build_chain(M, layers)
    return run_program(port, f"{name} (M={M}: {layer_desc})", instrs, ddr, verify)


def main():
    ap = argparse.ArgumentParser(
        description="NPU matmul stress test: multi-tile K/M/N, weight-stationary "
        "M-streaming, ReLU/leaky-ReLU, and layer-to-layer chaining"
    )
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

    ok = True
    # Multi-tile K/M/N stress, no weight_hold (K=130 > 14, so hold never
    # applies - see build()'s scoping).
    ok &= run_scenario(port, "multi-tile stress", M, K, N)
    # K<=14 (single K-tile), M spanning 4 M-tiles: exercises weight-stationary
    # M-streaming (MATMUL target[0]) - the actual hardware validation for
    # ROADMAP.md's "stream the M dim through each loaded weight tile" item.
    ok &= run_scenario(
        port,
        "weight-hold M-streaming",
        HOLD_M,
        HOLD_K,
        HOLD_N,
        use_weight_hold=True,
        seed=1,
    )
    # Weight-hold combined with K-tiling (K>14): weight reloaded once per
    # K-tile per hold-batch instead of once per (K-tile, M-tile) pair, and
    # M=270 spans 2 hold-batches, exercising the batch-boundary reload too.
    ok &= run_scenario(
        port,
        "weight-hold + K-tiling",
        HOLD_KTILE_M,
        HOLD_KTILE_K,
        HOLD_KTILE_N,
        use_weight_hold=True,
        seed=10,
    )
    # Isolated activation-function checks: same shape as the stress test
    # (so sums are large enough to actually clamp/clip), but ReLU and
    # leaky-ReLU instead of passthrough - neither was exercised on real
    # hardware before.
    ok &= run_scenario(port, "relu activation", 20, 30, 40, act_func=ACT_RELU, seed=2)
    ok &= run_scenario(
        port, "leaky-relu activation", 20, 30, 40, act_func=ACT_LEAKY_RELU, seed=3
    )
    # 3-layer chain, no hold: exercises Scratch A/B addressing on real
    # hardware (never tested there before) with a different activation
    # function at each layer and N-tiling on the final layer's output.
    ok &= run_chain_scenario(
        port,
        "3-layer chain",
        20,
        [
            {"K": 14, "N": 14, "act_func": ACT_RELU, "seed": 4},
            {"K": 14, "N": 10, "act_func": ACT_LEAKY_RELU, "seed": 5},
            {"K": 10, "N": 28, "act_func": ACT_PASSTHROUGH, "seed": 6},
        ],
    )
    # Combined: weight-hold + chaining + every activation function together,
    # spanning 3 M-tiles (2 held reuses per layer) - closer to a real
    # workload than any single scenario above.
    ok &= run_chain_scenario(
        port,
        "combined hold+chain+activations",
        42,
        [
            {"K": 10, "N": 14, "act_func": ACT_RELU, "hold": True, "seed": 7},
            {"K": 14, "N": 12, "act_func": ACT_LEAKY_RELU, "hold": True, "seed": 8},
            {"K": 12, "N": 25, "act_func": ACT_PASSTHROUGH, "hold": True, "seed": 9},
        ],
    )

    print("\n=== summary ===")
    for name, passed, fails in SCENARIO_RESULTS:
        status = "PASS" if passed else f"FAIL ({fails} mismatches)"
        print(f"  {status:20s} {name}")
    num_failed = sum(1 for _, passed, _ in SCENARIO_RESULTS if not passed)
    print(
        f"{len(SCENARIO_RESULTS) - num_failed}/{len(SCENARIO_RESULTS)} scenarios passed"
    )

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
