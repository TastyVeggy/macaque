import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, NextTimeStep, ReadOnly, RisingEdge


def encode_instr(opcode, acc_mode=0, target=0, ddr3_addr=0, byte_count=0, tile_params=0):
    word = (opcode & 0xF) << 60
    word |= (acc_mode & 1) << 59
    word |= (target & 0x7) << 56
    word |= (ddr3_addr & 0xFFFFFFF) << 28
    word |= (byte_count & 0xFFFF) << 12
    word |= (tile_params & 0xFFF)
    return word


OP_HALT = 0x6
OP_SYNC = 0x6
OP_MATMUL = 0x3
OP_LOAD_W = 0x0
OP_LOAD_B = 0x1
OP_LOAD_I = 0x2
OP_STORE = 0x5
OP_ACT = 0x4


async def reset(dut):
    dut.rst.value = 1
    dut.start.value = 0
    dut.reset.value = 0
    dut.instr_base.value = 0
    dut.instr_len.value = 0
    dut.matmul_done.value = 0
    dut.activate_done.value = 1
    dut.load_done.value = 1
    dut.store_done.value = 1
    await ClockCycles(dut.clk, 3)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 2)


async def wait_for(dut, signal, value, timeout=200):
    for _ in range(timeout):
        if int(signal.value) == value:
            return True
        await RisingEdge(dut.clk)
    return False


def start_im_data_model(dut, program):
    """Model the registered BRAM read exactly like npu_top's instr_mem:

       im_data (cycle M+1) = instr_mem[im_addr (cycle M)]

    Sampled post-NBA via ReadOnly so it matches the DUT's always_ff sampling,
    then presented on the NEXT timestep — no cocotb read/write race.
    """
    async def drive_im_data():
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            addr = int(dut.im_addr.value) >> 3
            await NextTimeStep()
            if addr < len(program):
                dut.im_data.value = program[addr]
            else:
                dut.im_data.value = 0

    cocotb.start_soon(drive_im_data())


@cocotb.test()
async def test_halt_program(dut):
    """A single HALT instruction should complete."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    program = [encode_instr(OP_HALT)]
    start_im_data_model(dut, program)
    dut.instr_base.value = 0
    dut.instr_len.value = 1

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_for(dut, dut.done, 1, timeout=50), "Sequencer did not halt"
    assert not dut.busy.value, "Sequencer still busy after HALT"


@cocotb.test()
async def test_sequencer_ready_idle(dut):
    """ready should be high in IDLE state"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    assert dut.ready.value, "ready not high in IDLE"


@cocotb.test()
async def test_sequencer_runs_sync_only(dut):
    """A single SYNC instruction should complete (both lanes must reach barrier)"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    # SYNC goes to both FIFOs; fetch unit sees prog_end after it
    program = [encode_instr(OP_SYNC)]
    start_im_data_model(dut, program)
    dut.instr_base.value = 0
    dut.instr_len.value = 1

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_for(dut, dut.done, 1, timeout=100), "SYNC program did not complete"


@cocotb.test()
async def test_matmul_fsm_sequence(dut):
    """Verify the MATMUL FSM: LOAD* → MATMUL (gated on dep_tracker) → drain → idle"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    N = 4
    program = [
        encode_instr(OP_LOAD_W, ddr3_addr=0x100, byte_count=196),
        encode_instr(OP_LOAD_B, ddr3_addr=0x110, byte_count=56),
        encode_instr(OP_LOAD_I, ddr3_addr=0x120, byte_count=196),
        encode_instr(OP_MATMUL, tile_params=N),
    ]
    start_im_data_model(dut, program)
    dut.instr_base.value = 0
    dut.instr_len.value = 4
    dut.matmul_done.value = 0
    dut.load_done.value = 1

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Exactly 3 loads must be issued: LOAD_W → LOAD_B → LOAD_I, each once.
    # (Regression: the first LOAD used to be duplicated by a stale im_data.)
    loads = []
    for _ in range(30):
        if int(dut.load_req.value) == 1:
            addr = int(dut.ddr3_addr.value)
            if not loads or loads[-1] != addr:
                loads.append(addr)
            if len(loads) == 3:
                break
        await RisingEdge(dut.clk)
    assert loads == [0x100, 0x110, 0x120], f"loads issued: {loads}"

    # --- WEIGHT_LOAD ---
    assert await wait_for(dut, dut.wb_re, 1, timeout=100), "wb_re never asserted (dep_tracker may not be ready)"

    assert await wait_for(dut, dut.matmul_start, 1, timeout=20), "matmul_start never pulsed"
    await RisingEdge(dut.clk)
    assert int(dut.matmul_start.value) == 0, "matmul_start not 1-cycle"

    # --- ACT_FEED ---
    assert await wait_for(dut, dut.act_valid, 1, timeout=100), "never entered ACT_FEED"

    act_pulses = 0
    while int(dut.act_valid.value) == 1:
        act_pulses += 1
        await RisingEdge(dut.clk)
    assert act_pulses == N, f"act_valid pulses: expected {N}, got {act_pulses}"

    # --- WAIT_DRAIN ---
    assert int(dut.busy.value) == 1, "not busy during WAIT_DRAIN"
    await ClockCycles(dut.clk, 5)
    assert int(dut.busy.value) == 1, "sequencer exited WAIT_DRAIN while matmul_done low"

    # Simulate the drain
    dut.matmul_done.value = 1
    await ClockCycles(dut.clk, N)
    dut.matmul_done.value = 0

    # 4-instruction program (3 loads + matmul): should reach done
    assert await wait_for(dut, dut.done, 1, timeout=100), "program did not complete after drain"


@cocotb.test()
async def test_dma_lane_overlap_prefetch(dut):
    """
    Verify the DMA lane overlaps the NEXT tile's loads with the CURRENT MATMUL.

    Program: [LOAD_W, LOAD_B, LOAD_I, MATMUL] x2

    As soon as MATMUL 1 starts (compute lane toggles bank_sel), the DMA lane
    is free to issue tile-2's loads (into the inactive bank) WHILE the compute
    lane is still busy on MATMUL 1 — before its drain completes.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    N = 4
    program = [
        encode_instr(OP_LOAD_W, ddr3_addr=0x100, byte_count=196),
        encode_instr(OP_LOAD_B, ddr3_addr=0x110, byte_count=56),
        encode_instr(OP_LOAD_I, ddr3_addr=0x120, byte_count=196),
        encode_instr(OP_MATMUL, tile_params=N),
        encode_instr(OP_LOAD_W, ddr3_addr=0x200, byte_count=196),
        encode_instr(OP_LOAD_B, ddr3_addr=0x210, byte_count=56),
        encode_instr(OP_LOAD_I, ddr3_addr=0x220, byte_count=196),
        encode_instr(OP_MATMUL, tile_params=N),
    ]
    start_im_data_model(dut, program)
    dut.instr_base.value = 0
    dut.instr_len.value = 8
    dut.load_done.value = 1  # instant DMA response
    dut.matmul_done.value = 0

    # Collect all load_req addresses from the start (background).
    loads = []

    async def collect_loads():
        while True:
            if int(dut.load_req.value) == 1:
                addr = int(dut.ddr3_addr.value)
                if not loads or loads[-1] != addr:
                    loads.append(addr)
            await RisingEdge(dut.clk)

    cocotb.start_soon(collect_loads())

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # MATMUL 1 enters ACT_FEED (weights already loaded by the DMA lane)
    assert await wait_for(dut, dut.act_valid, 1, timeout=100), "MATMUL 1 never entered ACT_FEED"

    # Keep matmul_done LOW so the compute lane stays busy with MATMUL 1.
    await ClockCycles(dut.clk, 10)
    assert int(dut.busy.value) == 1, "compute lane finished MATMUL 1 too early"

    # The DMA lane must have issued tile-2's loads (0x200+) WHILE MATMUL 1
    # was in flight — that is the overlap.
    tile2_loads = [a for a in loads if a >= 0x200]
    assert tile2_loads == [0x200, 0x210, 0x220], f"tile-2 loads during MATMUL 1: {loads}"

    # Now release the drain for MATMUL 1
    dut.matmul_done.value = 1
    await ClockCycles(dut.clk, N)
    dut.matmul_done.value = 0

    # MATMUL 2 should start and the program complete
    assert await wait_for(dut, dut.act_valid, 1, timeout=100), "MATMUL 2 never started"
    while int(dut.act_valid.value) == 1:
        await RisingEdge(dut.clk)

    dut.matmul_done.value = 1
    await ClockCycles(dut.clk, N)
    dut.matmul_done.value = 0

    assert await wait_for(dut, dut.done, 1, timeout=100), "program did not complete"


@cocotb.test()
async def test_two_matmul_overlap(dut):
    """
    Two-tile program: both MATMULs execute, bank_sel toggles per tile,
    and the program reaches done.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    N = 2
    program = [
        encode_instr(OP_LOAD_W, ddr3_addr=0x100, byte_count=196),
        encode_instr(OP_LOAD_B, ddr3_addr=0x110, byte_count=56),
        encode_instr(OP_LOAD_I, ddr3_addr=0x120, byte_count=196),
        encode_instr(OP_MATMUL, tile_params=N),
        encode_instr(OP_LOAD_W, ddr3_addr=0x200, byte_count=196),
        encode_instr(OP_LOAD_B, ddr3_addr=0x210, byte_count=56),
        encode_instr(OP_LOAD_I, ddr3_addr=0x220, byte_count=196),
        encode_instr(OP_MATMUL, tile_params=N),
    ]
    start_im_data_model(dut, program)
    dut.instr_base.value = 0
    dut.instr_len.value = 8
    dut.load_done.value = 1
    dut.matmul_done.value = 0

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # First MATMUL's ACT_FEED
    assert await wait_for(dut, dut.act_valid, 1, timeout=100), "first MATMUL never started"
    while int(dut.act_valid.value) == 1:
        await RisingEdge(dut.clk)

    # Drive drain for first matmul
    dut.matmul_done.value = 1
    await ClockCycles(dut.clk, N)
    dut.matmul_done.value = 0

    # Second MATMUL's ACT_FEED
    assert await wait_for(dut, dut.act_valid, 1, timeout=100, ), "second MATMUL never started"
    while int(dut.act_valid.value) == 1:
        await RisingEdge(dut.clk)

    # Drive drain for second matmul
    dut.matmul_done.value = 1
    await ClockCycles(dut.clk, N)
    dut.matmul_done.value = 0

    assert await wait_for(dut, dut.done, 1, timeout=100), "program did not complete after 2 matmuls"


@cocotb.test()
async def test_load_store_mutual_exclusion(dut):
    """
    LOAD_* (DMA lane) and STORE (compute lane) share the single
    ddr3_addr/byte_count bus. The wrapper arbiter must guarantee:
      1. load_req and store_req are never asserted in the same cycle
      2. each request's ddr3_addr is the correct one on the shared bus
         (loads get the load address, stores get the store address)

    Requests are 1-cycle pulses, so they're sampled post-NBA (ReadOnly),
    which reliably sees every pulse.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    program = [
        encode_instr(OP_LOAD_W, ddr3_addr=0x100, byte_count=196),
        encode_instr(OP_STORE, ddr3_addr=0x400, byte_count=196),
        encode_instr(OP_LOAD_B, ddr3_addr=0x110, byte_count=56),
        encode_instr(OP_STORE, ddr3_addr=0x410, byte_count=56),
    ]
    start_im_data_model(dut, program)
    dut.instr_base.value = 0
    dut.instr_len.value = 4
    dut.load_done.value = 1
    dut.store_done.value = 1

    loads, stores, conflicts = [], [], []

    async def monitor():
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            lr = int(dut.load_req.value)
            sr = int(dut.store_req.value)
            addr = int(dut.ddr3_addr.value)
            if lr and sr:
                conflicts.append(addr)
            elif lr:
                loads.append(addr)
            elif sr:
                stores.append(addr)

    cocotb.start_soon(monitor())

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    assert await wait_for(dut, dut.done, 1, timeout=100), "program did not complete"
    await ClockCycles(dut.clk, 2)  # let the monitor drain its last sample

    assert conflicts == [], f"load_req and store_req asserted together: {conflicts}"
    assert loads == [0x100, 0x110], f"loads on shared bus: {loads}"
    assert stores == [0x400, 0x410], f"stores on shared bus: {stores}"


@cocotb.test()
async def test_acc_mode1_flags_error(dut):
    """
    K-tiling (acc_mode=1) is NOT yet implemented. A MATMUL with acc_mode=1 must
    assert the sticky `error` output (surfaced to STATUS bit 3) so the compiler
    can never silently get wrong numbers.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    # LOAD_* then a MATMUL with acc_mode=1 (K-tiling accumulate).
    program = [
        encode_instr(OP_LOAD_W, ddr3_addr=0x100, byte_count=196),
        encode_instr(OP_LOAD_B, ddr3_addr=0x110, byte_count=56),
        encode_instr(OP_LOAD_I, ddr3_addr=0x120, byte_count=196),
        encode_instr(OP_MATMUL, acc_mode=1, tile_params=4),
    ]
    start_im_data_model(dut, program)
    dut.instr_base.value = 0
    dut.instr_len.value = 4
    dut.load_done.value = 1
    dut.matmul_done.value = 0

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # The MATMUL with acc_mode=1 is decoded → error must assert and stay.
    assert await wait_for(dut, dut.error, 1, timeout=100), "error never asserted for acc_mode=1"
    await ClockCycles(dut.clk, 3)
    assert int(dut.error.value) == 1, "error should be sticky (held) after acc_mode=1 MATMUL"


@cocotb.test()
async def test_acc_mode0_no_error(dut):
    """A normal MATMUL (acc_mode=0) must NOT assert error."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    program = [
        encode_instr(OP_LOAD_W, ddr3_addr=0x100, byte_count=196),
        encode_instr(OP_LOAD_B, ddr3_addr=0x110, byte_count=56),
        encode_instr(OP_LOAD_I, ddr3_addr=0x120, byte_count=196),
        encode_instr(OP_MATMUL, acc_mode=0, tile_params=4),
    ]
    start_im_data_model(dut, program)
    dut.instr_base.value = 0
    dut.instr_len.value = 4
    dut.load_done.value = 1
    dut.matmul_done.value = 0

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    await ClockCycles(dut.clk, 5)
    assert int(dut.error.value) == 0, "error should not assert for acc_mode=0 MATMUL"