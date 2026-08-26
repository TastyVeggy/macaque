import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, NextTimeStep, ReadOnly, RisingEdge

ARRAY_SIZE = 14


def golden_requantize(acc, m, shift, func):
    """Bit-match the activate_unit RTL. Bias is ALREADY in `acc`
    (the array seeds accumulators with bias), so ACTIVATE is pure requantize:
    scaled = acc*M (+ round-half-up), requant = scaled >>> shift (arith),
    func: 0=ReLU, 1=leaky-ReLU (alpha=1/16), 2=passthrough, clamp to [-128,127].
    """
    scaled = acc * m
    if shift > 0:
        scaled += 1 << (shift - 1)
    requant = scaled >> shift
    if func == 0:  # ReLU
        act = max(0, requant)
    elif func == 1:  # leaky-ReLU, alpha = 1/16
        act = requant if requant >= 0 else requant >> 4
    else:  # passthrough
        act = requant
    return max(-128, min(127, act))


def s8(v):
    """Interpret a raw 8-bit value as signed INT8 (qb_wdata is signed)."""
    return v - 256 if v >= 128 else v


async def reset(dut):
    dut.rst.value = 1
    dut.req.value = 0
    dut.tile_params.value = 0
    dut.scale_m.value = 0
    dut.scale_shift.value = 0
    dut.act_func.value = 0
    for i in range(ARRAY_SIZE):
        dut.ob_rdata[i].value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 2)


def start_bram_model(dut, ob_data):
    """Model the registered BRAM read exactly like bram_sdp / out_buffer:
    rdata(cycle N+1) = mem[raddr(cycle N)],. Reads the CURRENT ob_data dict
    (shared, updated by run_case) so one task serves all cases.
    """

    async def drive():
        pending = None
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            re = int(dut.ob_re.value)
            addr = int(dut.ob_raddr.value)
            await NextTimeStep()
            if pending is not None:
                ob = ob_data.get(pending, [0] * ARRAY_SIZE)
                for i in range(ARRAY_SIZE):
                    dut.ob_rdata[i].value = ob[i]
            pending = addr if re else pending

    cocotb.start_soon(drive())


async def run_case(dut, ob_rows, m, shift, func, num_rows):
    """ob_rows: list of `num_rows` rows, each 14 int32 lane values (the INT32
    accumulator output, bias already included). Verifies EVERY written row
    against the golden, using distinct per-row data so a misaligned/dropped/
    shifted row would be caught.
    """
    ob_data.clear()
    ob_data.update({r: ob_rows[r] for r in range(num_rows)})

    dut.scale_m.value = m
    dut.scale_shift.value = shift
    dut.act_func.value = func
    dut.tile_params.value = num_rows

    dut.req.value = 1
    await RisingEdge(dut.clk)
    dut.req.value = 0

    out = {}
    for _ in range(200):
        if int(dut.qb_we.value) == 1:
            addr = int(dut.qb_waddr.value)
            out[addr] = [s8(int(dut.qb_wdata[i].value)) for i in range(ARRAY_SIZE)]
        if int(dut.done.value) == 1:
            break
        await RisingEdge(dut.clk)

    assert dut.done.value, "activate_unit never asserted done"
    assert sorted(out.keys()) == list(range(num_rows)), (
        f"expected exactly rows 0..{num_rows - 1} written, got {sorted(out.keys())}"
    )

    for addr, lanes in out.items():
        for i, v in enumerate(lanes):
            expected = golden_requantize(ob_data[addr][i], m, shift, func)
            assert v == expected, (
                f"lane {i} row {addr}: got {v}, expected {expected} "
                f"(acc={ob_data[addr][i]} M={m} shift={shift} func={func})"
            )


ob_data = {}


@cocotb.test()
async def test_requantize_cases(dut):
    """Distinct per-row data (acc = base + 7*r) exercises row alignment
    end-to-end: exactly rows 0..num_rows-1, each with correct requantize math.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    start_bram_model(dut, ob_data)

    cases = [
        # (acc_base, M, shift, func, rows)   func 0=ReLU, 1=leaky, 2=passthrough
        (100, 8192, 13, 0, 4),  # ~ x*1.0
        (-100, 8192, 13, 0, 4),  # negatives → 0 (ReLU)
        (-100, 8192, 13, 2, 4),  # negatives passthrough → preserved
        (100, 8192, 13, 2, 4),  # positives passthrough → preserved
        (-100, 8192, 13, 1, 4),  # negatives leaky → x>>4 (alpha=1/16)
        (100, 8192, 13, 1, 4),  # positives leaky → unchanged
        (32768, 8192, 13, 0, 4),  # big positive → clamps to 127
        (123, 16384, 13, 0, 2),  # ~ x*2.0 → clamp
        (50, 1, 0, 0, 2),  # shift 0
        (255, 128, 7, 0, 4),  # (x*128 + 64) >> 7
    ]

    for acc_base, m, shift, func, rows in cases:
        ob_rows = [[acc_base + 7 * r for _ in range(ARRAY_SIZE)] for r in range(rows)]
        await run_case(dut, ob_rows, m, shift, func, rows)


@cocotb.test()
async def test_rounding_edge(dut):
    """round-half-up: (1*3 + 1) >> 1 = 2, not 1."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    start_bram_model(dut, ob_data)

    ob_rows = [[1] * ARRAY_SIZE for _ in range(2)]
    await run_case(dut, ob_rows, 3, 1, 0, 2)
    await run_case(dut, ob_rows, 3, 1, 2, 2)  # passthrough keeps 2


@cocotb.test()
async def test_single_row(dut):
    """num_rows == 1 boundary: exactly row 0 written, then done. Locks the
    pipeline-drain behavior at the minimum tile size."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    start_bram_model(dut, ob_data)

    # Distinct per-lane data so a wrong/aligned value would be caught.
    ob_rows = [[50 + i for i in range(ARRAY_SIZE)]]
    await run_case(dut, ob_rows, 8192, 13, 0, 1)

    # A second single-row case to confirm re-arming works cleanly.
    ob_rows2 = [[-40 + i for i in range(ARRAY_SIZE)]]
    await run_case(dut, ob_rows2, 8192, 13, 2, 1)
