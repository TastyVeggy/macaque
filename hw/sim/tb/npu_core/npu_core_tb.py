import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

from matmul_helpers import (
    ARRAY,
    ACT_PASSTHROUGH,
    AxiRam,
    OP_ACTIVATE,
    OP_LOAD_ACT,
    OP_LOAD_B,
    OP_LOAD_W,
    OP_MATMUL,
    OP_STORE,
    build,
    enc,
    flatten_i32,
    ref_matmul,
    s8,
)


REG_CTRL = 0x00
REG_STATUS = 0x08
REG_INSTR_ADDR = 0x10
REG_INSTR_LEN = 0x18


async def reg_write(dut, addr, val):
    """Write a 64-bit word to the register bus (value in low 32 bits)."""
    dut.reg_addr.value = addr
    dut.reg_wdata.value = val
    dut.reg_we.value = 1
    await RisingEdge(dut.clk)
    dut.reg_we.value = 0
    await RisingEdge(dut.clk)


async def reg_read(dut, addr):
    """Read a 64-bit word from the register bus (combinational mux)."""
    dut.reg_addr.value = addr
    await RisingEdge(dut.clk)
    return int(dut.reg_rdata.value)


@cocotb.test()
async def test_npu_core_halt_only(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)

    # Pre-load a HALT instruction at byte address 0 via the imem write port
    dut.imem_we.value = 1
    dut.imem_waddr.value = 0
    dut.imem_wdata.value = 0x60000000_00000000
    await ClockCycles(dut.clk, 1)
    dut.imem_we.value = 0
    await ClockCycles(dut.clk, 1)

    # Write INSTR_ADDR = 0, INSTR_LEN = 1
    await reg_write(dut, REG_INSTR_ADDR, 0)
    await reg_write(dut, REG_INSTR_LEN, 1)

    # Verify the writes took effect
    addr = await reg_read(dut, REG_INSTR_ADDR)
    assert addr == 0, f"INSTR_ADDR: expected 0, got 0x{addr:08X}"
    length = await reg_read(dut, REG_INSTR_LEN)
    assert length == 1, f"INSTR_LEN: expected 1, got 0x{length:08X}"

    # Set CTRL.start
    await reg_write(dut, REG_CTRL, 0x0000_0001)

    # Poll for done
    for i in range(50):
        status = await reg_read(dut, REG_STATUS)
        cocotb.log.info(f"Poll {i}: STATUS=0x{status:08X} (ready={bool(status&1)} busy={bool(status&2)} done={bool(status&4)})")
        if bool(status & 0x4):
            break
        await ClockCycles(dut.clk, 10)
    else:
        assert False, "Timeout waiting for STATUS.done"

    assert bool(status & 0x4), f"STATUS.done not set: 0x{status:08X}"


@cocotb.test()
async def test_npu_core_reads_status(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)

    status = await reg_read(dut, REG_STATUS)
    assert bool(status & 0x1), f"ready bit not set: 0x{status:08X}"


def gen_matrices(M, K, N):
    A = [[((m + k) % 3) - 1 for k in range(K)] for m in range(M)]
    W = [[((k + n) % 3) - 1 for n in range(N)] for k in range(K)]
    return A, W


async def _reset(dut):
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)


async def _load_imem(dut, instrs):
    for i, w in enumerate(instrs):
        dut.imem_we.value = 1
        dut.imem_waddr.value = 8 * i
        dut.imem_wdata.value = w
        await RisingEdge(dut.clk)
    dut.imem_we.value = 0
    await RisingEdge(dut.clk)


def _dump_state(dut):
    def sig(path):
        try:
            v = dut._id(path, extended=False).value
            return int(v)
        except Exception:
            return "??"

    seq = "seq_inst"
    cocotb.log.info("=== DEBUG STATE DUMP ===")
    cocotb.log.info(f"fetch.state={sig(f'{seq}.fetch_inst.state')} "
                    f"busy={sig(f'{seq}.fetch_inst.busy')} halt={sig(f'{seq}.fetch_inst.halt')} "
                    f"pc={sig(f'{seq}.fetch_inst.pc')}")
    cocotb.log.info(f"dma_lane.state={sig(f'{seq}.dma_inst.state')} "
                    f"busy_r={sig(f'{seq}.dma_inst.busy_r')} fifo_empty={sig(f'{seq}.dma_fifo_empty')}")
    cocotb.log.info(f"comp_lane.state={sig(f'{seq}.comp_inst.state')} "
                    f"busy_r={sig(f'{seq}.comp_inst.busy_r')} fifo_empty={sig(f'{seq}.comp_fifo_empty')}")
    cocotb.log.info(f"dma_fifo(full={sig(f'{seq}.dma_fifo_full')} empty={sig(f'{seq}.dma_fifo_empty')}) "
                    f"comp_fifo(full={sig(f'{seq}.comp_fifo_full')} empty={sig(f'{seq}.comp_fifo_empty')})")
    cocotb.log.info(f"dma_unit.eng={sig('dma_unit_inst.eng')} "
                    f"adapter.state={sig('dma_unit_inst.u_adapter.state')} "
                    f"beats_left={sig('dma_unit_inst.beats_left')}")
    ad = "dma_unit_inst.u_adapter"
    cocotb.log.info(f"adapter: acc_fill={sig(f'{ad}.acc_fill')} "
                    f"row_index={sig(f'{ad}.row_index')} "
                    f"rows_total={sig(f'{ad}.load_rows_total')} "
                    f"target={sig(f'{ad}.target')} row_size={sig(f'{ad}.row_size')} "
                    f"read_pending={sig(f'{ad}.read_pending')} "
                    f"store_rows_read={sig(f'{ad}.store_rows_read')}")
    cocotb.log.info(f"dma_lane: issue_buf_type={sig(f'{seq}.dma_inst.issue_buf_type')} "
                    f"issue_bank={sig(f'{seq}.dma_inst.issue_loaded_bank')} "
                    f"load_req={sig(f'{seq}.dma_inst.load_req')}")
    cocotb.log.info(f"dma_lane.byte_count={sig(f'{seq}.dma_inst.byte_count')} "
                    f"ddr3_addr=0x{sig(f'{seq}.dma_inst.ddr3_addr'):x} "
                    f"dma_unit.cur_addr=0x{sig('dma_unit_inst.cur_addr'):x}")
    cocotb.log.info(f"dep: weight_slot={sig(f'{seq}.dep_tracker_inst.weight_slot')} "
                    f"bias_slot={sig(f'{seq}.dep_tracker_inst.bias_slot')} "
                    f"act_slot={sig(f'{seq}.dep_tracker_inst.act_slot')}")
    cocotb.log.info(f"dep: weight_rdy={sig(f'{seq}.dep_tracker_inst.weight_rdy')} "
                    f"bias_rdy={sig(f'{seq}.dep_tracker_inst.bias_rdy')} "
                    f"act_rdy={sig(f'{seq}.dep_tracker_inst.act_rdy')}")
    cocotb.log.info(f"dep: w_can_load={sig(f'{seq}.dep_tracker_inst.dma_weight_can_load')} "
                    f"b_can_load={sig(f'{seq}.dep_tracker_inst.dma_bias_can_load')} "
                    f"a_can_load={sig(f'{seq}.dep_tracker_inst.dma_act_can_load')}")
    cocotb.log.info(f"bank_sel: w={sig(f'{seq}.weight_bank_sel')} a={sig(f'{seq}.act_bank_sel')} "
                    f"b={sig(f'{seq}.bias_bank_sel')} out={sig(f'{seq}.out_bank_sel')} "
                    f"q={sig(f'{seq}.quant_bank_sel')}")
    cocotb.log.info(f"im_addr={sig(f'{seq}.im_addr')} im_data=0x{sig(f'{seq}.im_data'):x}"
                    if sig(f'{seq}.im_data') != "??" else f"im_addr={sig(f'{seq}.im_addr')}")


async def _monitor(dut):
    import collections
    seq = "seq_inst"
    prev = {}
    cycle = 0
    history = collections.deque(maxlen=400)

    def snap():
        def s(p):
            try:
                return int(dut._id(p, extended=False).value)
            except Exception:
                return None
        return {
            "eng": s("dma_unit_inst.eng"),
            "astate": s("dma_unit_inst.u_adapter.state"),
            "load_req": s(f"{seq}.load_req"),
            "store_req": s(f"{seq}.store_req"),
            "load_done": s(f"{seq}.load_done"),
            "store_done": s(f"{seq}.store_done"),
            "beats": s("dma_unit_inst.beats_left"),
            "row": s("dma_unit_inst.u_adapter.row_index"),
            "fill": s("dma_unit_inst.u_adapter.acc_fill"),
            "dlane": s(f"{seq}.dma_inst.state"),
            "clane": s(f"{seq}.comp_inst.state"),
        }

    try:
        while True:
            await RisingEdge(dut.clk)
            cycle += 1
            cur = snap()
            changes = [k for k in cur if cur[k] != prev.get(k)]
            if changes:
                history.append(f"[{cycle}] " + " ".join(f"{k}={cur[k]}" for k in changes))
                prev.update(cur)
    except Exception:
        pass
    finally:
        cocotb.log.info("=== MONITOR TAIL ===")
        for line in history:
            cocotb.log.info(line)


async def _run_matmul(dut, M, K, N, **build_kwargs):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    ram = AxiRam(dut)
    cocotb.start_soon(ram.run())
    cocotb.start_soon(_monitor(dut))

    A, W = gen_matrices(M, K, N)
    instrs, ddr, out = build(A, W, **build_kwargs)

    for addr, data in ddr:
        ram.write_bytes(addr, data)

    await _load_imem(dut, instrs)

    await reg_write(dut, REG_INSTR_ADDR, 0)
    await reg_write(dut, REG_INSTR_LEN, len(instrs))
    await reg_write(dut, REG_CTRL, 0x1)

    for i in range(20000):
        status = await reg_read(dut, REG_STATUS)
        if bool(status & 0x4):
            break
        if bool(status & 0x8):
            assert False, f"npu_error asserted (STATUS=0x{status:08X})"
        await ClockCycles(dut.clk, 10)
    else:
        _dump_state(dut)
        assert False, f"timeout waiting for done (STATUS=0x{status:08X})"

    fails = 0
    first_fails = []
    for addr, exp in out:
        got = ram.read_bytes(addr, len(exp))
        gs = [b - 256 if b >= 128 else b for b in got]
        for idx, (g, e) in enumerate(zip(gs, exp)):
            if g != e:
                fails += 1
                if len(first_fails) < 30:
                    first_fails.append((addr, idx, g, e))
        if M == 14 and N == 14:
            cocotb.log.info("=== got (row-major) ===")
            for r in range(len(exp) // 14):
                cocotb.log.info("  got[" + str(r) + "] = " + " ".join(f"{x:4d}" for x in gs[r*14:(r+1)*14]))
            cocotb.log.info("=== want ===")
            for r in range(len(exp) // 14):
                cocotb.log.info("  want[" + str(r) + "] = " + " ".join(f"{x:4d}" for x in exp[r*14:(r+1)*14]))

    for addr, idx, g, e in first_fails:
        row, col = idx // 14, idx % 14
        cocotb.log.info(f"  MISMATCH addr=0x{addr:x} idx={idx} (r{row},c{col}) got={g} want={e}")
    assert fails == 0, f"matmul {M}x{K}x{N} failed with {fails} mismatches"
    cocotb.log.info(f"matmul {M}x{K}x{N} PASS")


@cocotb.test()
async def test_matmul_single_tile(dut):
    await _run_matmul(dut, 14, 14, 14)


@cocotb.test()
async def test_matmul_full(dut):
    await _run_matmul(dut, 20, 30, 40)


@cocotb.test()
async def test_matmul_dma_burst_crosses_4k_boundary(dut):
    """dma_unit (hw/rtl/control/dma_unit.sv) clamps each AXI burst so it never
    crosses a 4KB address boundary (AXI4 forbids it - real slaves aren't
    required to handle a crossing burst correctly). No existing test's fixed
    DDR3 addresses ever land near one, so this deliberately places the single
    weight tile straddling 0x2000: 0x1FA0 (8-byte aligned) + 200 bytes runs to
    0x2068. AxiRam._check_4k asserts if dma_unit ever fails to split there.
    """
    await _run_matmul(dut, 14, 14, 14, weight_base=0x1FA0)


@cocotb.test()
async def test_matmul_weight_hold_m_streaming(dut):
    """MATMUL target[0] (weight_hold): stream several M-chunks through one
    already-loaded weight/bias bank without reloading, then drop back to a
    normal (unheld) load on the *other* bank to prove it's still usable.

    Hand-built instead of matmul_helpers.build() because build() always
    reloads weight/bias per M-chunk - this test needs a stream that
    deliberately doesn't, to exercise dep_tracker's held-bank bypass
    (hw/rtl/control/dep_tracker.sv) and compute_lane's conditional bank
    toggle (hw/rtl/control/compute_lane.sv).
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    ram = AxiRam(dut)
    cocotb.start_soon(ram.run())
    cocotb.start_soon(_monitor(dut))

    K = N = ARRAY
    W1 = [[((k + n) % 3) - 1 for n in range(N)] for k in range(K)]
    # Deliberately different from W1 so a stale-bank-1-reuse bug (RTL fails to
    # actually switch to the freshly-loaded bank 0) shows up as a mismatch
    # rather than accidentally computing the right answer anyway.
    W2 = [[((2 * k + n + 1) % 3) - 1 for n in range(N)] for k in range(K)]

    def gen_a_chunk(seed):
        return [[((m + k + seed) % 3) - 1 for k in range(K)] for m in range(ARRAY)]

    # Seeds 0,1,2,4 (not 0,1,2,3): the formula is periodic mod 3 in seed, so
    # seed=3 would silently produce the same matrix as seed=0, masking a
    # stale-data bug in chunk3 as an accidental pass.
    A0, A1, A2, A3 = (gen_a_chunk(s) for s in (0, 1, 2, 4))

    W1_ADDR, B1_ADDR = 0x1000, 0x6000
    W2_ADDR, B2_ADDR = 0x1200, 0x6100
    A0_ADDR, A1_ADDR, A2_ADDR, A3_ADDR = 0x8000, 0x8200, 0x8400, 0x8600
    OUT0_ADDR, OUT1_ADDR, OUT2_ADDR, OUT3_ADDR = 0x10000, 0x10200, 0x10400, 0x10600

    def weight_bytes(W):
        return [s8(W[k][n]) for k in range(ARRAY) for n in range(ARRAY)]

    def act_bytes(A):
        return [s8(A[m][k]) for m in range(ARRAY) for k in range(ARRAY)]

    for addr, data in [
        (W1_ADDR, weight_bytes(W1)),
        (B1_ADDR, flatten_i32([0] * ARRAY)),
        (W2_ADDR, weight_bytes(W2)),
        (B2_ADDR, flatten_i32([0] * ARRAY)),
        (A0_ADDR, act_bytes(A0)),
        (A1_ADDR, act_bytes(A1)),
        (A2_ADDR, act_bytes(A2)),
        (A3_ADDR, act_bytes(A3)),
    ]:
        ram.write_bytes(addr, data)

    instrs = [
        enc(OP_LOAD_W, ddr3_addr=W1_ADDR, byte_count=ARRAY * ARRAY),
        enc(OP_LOAD_B, ddr3_addr=B1_ADDR, byte_count=ARRAY * 4),
        enc(OP_LOAD_ACT, ddr3_addr=A0_ADDR, byte_count=ARRAY * ARRAY),
        enc(OP_MATMUL, acc_mode=0, target=0, tile_params=ARRAY),
        enc(OP_ACTIVATE, target=ACT_PASSTHROUGH, ddr3_addr=1, tile_params=ARRAY),
        enc(OP_STORE, ddr3_addr=OUT0_ADDR, byte_count=ARRAY * ARRAY),

        # Held: reuse bank 1's already-loaded W1/B1, no LOAD_W/LOAD_B.
        enc(OP_LOAD_ACT, ddr3_addr=A1_ADDR, byte_count=ARRAY * ARRAY),
        enc(OP_MATMUL, acc_mode=0, target=1, tile_params=ARRAY),
        enc(OP_ACTIVATE, target=ACT_PASSTHROUGH, ddr3_addr=1, tile_params=ARRAY),
        enc(OP_STORE, ddr3_addr=OUT1_ADDR, byte_count=ARRAY * ARRAY),

        # Held again, proving it's not just a one-shot bypass.
        enc(OP_LOAD_ACT, ddr3_addr=A2_ADDR, byte_count=ARRAY * ARRAY),
        enc(OP_MATMUL, acc_mode=0, target=1, tile_params=ARRAY),
        enc(OP_ACTIVATE, target=ACT_PASSTHROUGH, ddr3_addr=1, tile_params=ARRAY),
        enc(OP_STORE, ddr3_addr=OUT2_ADDR, byte_count=ARRAY * ARRAY),

        # Drop the hold: fresh load into the never-touched bank 0, proving it
        # still accepts a real load correctly after sitting untouched through
        # the whole held sequence.
        enc(OP_LOAD_W, ddr3_addr=W2_ADDR, byte_count=ARRAY * ARRAY),
        enc(OP_LOAD_B, ddr3_addr=B2_ADDR, byte_count=ARRAY * 4),
        enc(OP_LOAD_ACT, ddr3_addr=A3_ADDR, byte_count=ARRAY * ARRAY),
        enc(OP_MATMUL, acc_mode=0, target=0, tile_params=ARRAY),
        enc(OP_ACTIVATE, target=ACT_PASSTHROUGH, ddr3_addr=1, tile_params=ARRAY),
        enc(OP_STORE, ddr3_addr=OUT3_ADDR, byte_count=ARRAY * ARRAY),
    ]

    await _load_imem(dut, instrs)

    await reg_write(dut, REG_INSTR_ADDR, 0)
    await reg_write(dut, REG_INSTR_LEN, len(instrs))
    await reg_write(dut, REG_CTRL, 0x1)

    for i in range(20000):
        status = await reg_read(dut, REG_STATUS)
        if bool(status & 0x4):
            break
        if bool(status & 0x8):
            assert False, f"npu_error asserted (STATUS=0x{status:08X})"
        await ClockCycles(dut.clk, 10)
    else:
        _dump_state(dut)
        assert False, f"timeout waiting for done (STATUS=0x{status:08X})"

    expected = [
        (OUT0_ADDR, ref_matmul(A0, W1)),
        (OUT1_ADDR, ref_matmul(A1, W1)),
        (OUT2_ADDR, ref_matmul(A2, W1)),
        (OUT3_ADDR, ref_matmul(A3, W2)),
    ]

    fails = 0
    for addr, O in expected:
        exp = [O[m][n] for m in range(ARRAY) for n in range(ARRAY)]
        got = ram.read_bytes(addr, len(exp))
        gs = [b - 256 if b >= 128 else b for b in got]
        for idx, (g, e) in enumerate(zip(gs, exp)):
            if g != e:
                fails += 1
                row, col = idx // ARRAY, idx % ARRAY
                cocotb.log.info(
                    f"  MISMATCH addr=0x{addr:x} idx={idx} (r{row},c{col}) got={g} want={e}"
                )
    assert fails == 0, f"weight-hold M-streaming failed with {fails} mismatches"
    cocotb.log.info("weight-hold M-streaming PASS")


@cocotb.test()
async def test_matmul_weight_hold_with_k_tiling(dut):
    """MATMUL target[0] (weight_hold) combined with K-tiling: several
    M-chunks' partial sums held resident in out_buffer at once (via MATMUL's
    ddr3_addr[7:0] row-base and ACTIVATE's byte_count[12:5]/[13] row-base +
    bank-hold), so weight is reloaded once per K-tile instead of once per
    (K-tile, M-chunk) pair.

    Sequence, K=28 (2 K-tiles), M=42 (3 M-chunks of 14 rows), N=14:
      K-tile 0: real load_weight/load_bias, then 3 chunks - chunk 0 unheld
        (real load just happened), chunks 1-2 held (target[0]=1), each at
        its own mat_row_base (0, 14, 28), acc_mode=0 (seeds from bias).
      K-tile 1: real load_weight (fresh K-tile, not held), then the same
        3-chunk pattern again, acc_mode=1 (accumulates the row-base'd
        partial sum from K-tile 0).
      Drain: 3 ACTIVATEs, one per chunk, each at its own act_row_base;
        bank_hold=1 on the first two (more chunks still draining the same
        bank), bank_hold=0 on the last (toggles, same as an ordinary
        activate) - then 3 STOREs.
      Finally, one ordinary (unheld, un-row-based) matmul proves the system
      is still healthy afterward - dep_tracker's bank bookkeeping and
      out_bank_sel's toggle weren't left in a bad state by the held batch.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await _reset(dut)

    ram = AxiRam(dut)
    cocotb.start_soon(ram.run())
    cocotb.start_soon(_monitor(dut))

    K = 28
    N = ARRAY
    CHUNK = ARRAY  # 14-row M-chunks, one per hold-batch slot
    M_CHUNKS = 3
    M = CHUNK * M_CHUNKS  # 42

    W1 = [[((k + n) % 3) - 1 for n in range(N)] for k in range(K)]
    A = [[((m + k) % 3) - 1 for k in range(K)] for m in range(M)]
    O = ref_matmul(A, W1)

    # Sanity matmul afterward: deliberately different weight so a
    # stale-bank-reuse bug can't accidentally compute the right answer.
    W2 = [[((2 * k + n + 1) % 3) - 1 for n in range(N)] for k in range(ARRAY)]
    A_sanity = [[((m + k + 5) % 3) - 1 for k in range(ARRAY)] for m in range(ARRAY)]
    O_sanity = ref_matmul(A_sanity, W2)

    def weight_bytes(Wmat, k0):
        return [s8(Wmat[k0 + k][n]) for k in range(ARRAY) for n in range(ARRAY)]

    def act_bytes(Amat, m0, k0):
        return [s8(Amat[m0 + m][k0 + k]) for m in range(ARRAY) for k in range(ARRAY)]

    W0_ADDR, W1T_ADDR, B_ADDR = 0x1000, 0x1200, 0x6000
    A_ADDR = {(m, k): 0x8000 + 0x200 * (m * 2 + k) for m in range(M_CHUNKS) for k in range(2)}
    OUT_ADDR = {m: 0x10000 + 0x200 * m for m in range(M_CHUNKS)}
    W2_ADDR, B2_ADDR, A_SANITY_ADDR, OUT_SANITY_ADDR = 0x1400, 0x6100, 0x8c00, 0x10600

    for addr, data in [
        (W0_ADDR, weight_bytes(W1, 0)),
        (W1T_ADDR, weight_bytes(W1, ARRAY)),
        (B_ADDR, flatten_i32([0] * ARRAY)),
        (W2_ADDR, weight_bytes(W2, 0)),
        (B2_ADDR, flatten_i32([0] * ARRAY)),
        (A_SANITY_ADDR, act_bytes(A_sanity, 0, 0)),
    ] + [(A_ADDR[(m, k)], act_bytes(A, m * CHUNK, k * ARRAY))
        for m in range(M_CHUNKS) for k in range(2)]:
        ram.write_bytes(addr, data)

    def mat_row_base_addr(base):
        # MATMUL's row base lives in ddr3_addr[7:0] - enc() takes the raw
        # ddr3_addr value directly.
        return base

    def act_byte_count(shift, row_base, bank_hold):
        return (shift & 0x1F) | ((row_base & 0xFF) << 5) | ((bank_hold & 1) << 13)

    instrs = []
    for k in range(2):
        instrs.append(enc(OP_LOAD_W, ddr3_addr=(W0_ADDR if k == 0 else W1T_ADDR),
                          byte_count=ARRAY * ARRAY))
        if k == 0:
            instrs.append(enc(OP_LOAD_B, ddr3_addr=B_ADDR, byte_count=ARRAY * 4))
        for m in range(M_CHUNKS):
            held = m > 0
            instrs.append(enc(OP_LOAD_ACT, ddr3_addr=A_ADDR[(m, k)], byte_count=ARRAY * ARRAY))
            instrs.append(enc(OP_MATMUL, acc_mode=(0 if k == 0 else 1),
                              target=(1 if held else 0),
                              ddr3_addr=mat_row_base_addr(m * CHUNK),
                              tile_params=ARRAY))

    for m in range(M_CHUNKS):
        last = m == M_CHUNKS - 1
        # ddr3_addr=1: ACTIVATE's ddr3_addr field is act_scale_m (passthrough
        # multiplier) - leaving it 0 multiplies every result by zero.
        instrs.append(enc(OP_ACTIVATE, target=ACT_PASSTHROUGH, ddr3_addr=1,
                          byte_count=act_byte_count(0, m * CHUNK, bank_hold=0 if last else 1),
                          tile_params=ARRAY))
        instrs.append(enc(OP_STORE, ddr3_addr=OUT_ADDR[m], byte_count=ARRAY * ARRAY))

    # Sanity matmul: ordinary, unheld, row_base=0 - proves the system is
    # still healthy after the held batch (weight/bias/act banks and
    # out_bank_sel weren't left in a bad state).
    instrs.append(enc(OP_LOAD_W, ddr3_addr=W2_ADDR, byte_count=ARRAY * ARRAY))
    instrs.append(enc(OP_LOAD_B, ddr3_addr=B2_ADDR, byte_count=ARRAY * 4))
    instrs.append(enc(OP_LOAD_ACT, ddr3_addr=A_SANITY_ADDR, byte_count=ARRAY * ARRAY))
    instrs.append(enc(OP_MATMUL, acc_mode=0, target=0, ddr3_addr=0, tile_params=ARRAY))
    instrs.append(enc(OP_ACTIVATE, target=ACT_PASSTHROUGH, ddr3_addr=1,
                      byte_count=act_byte_count(0, 0, bank_hold=0), tile_params=ARRAY))
    instrs.append(enc(OP_STORE, ddr3_addr=OUT_SANITY_ADDR, byte_count=ARRAY * ARRAY))

    await _load_imem(dut, instrs)

    await reg_write(dut, REG_INSTR_ADDR, 0)
    await reg_write(dut, REG_INSTR_LEN, len(instrs))
    await reg_write(dut, REG_CTRL, 0x1)

    for i in range(20000):
        status = await reg_read(dut, REG_STATUS)
        if bool(status & 0x4):
            break
        if bool(status & 0x8):
            assert False, f"npu_error asserted (STATUS=0x{status:08X})"
        await ClockCycles(dut.clk, 10)
    else:
        _dump_state(dut)
        assert False, f"timeout waiting for done (STATUS=0x{status:08X})"

    fails = 0
    for m in range(M_CHUNKS):
        exp = [O[m * CHUNK + mm][n] for mm in range(ARRAY) for n in range(ARRAY)]
        got = ram.read_bytes(OUT_ADDR[m], len(exp))
        gs = [b - 256 if b >= 128 else b for b in got]
        for idx, (g, e) in enumerate(zip(gs, exp)):
            if g != e:
                fails += 1
                row, col = idx // ARRAY, idx % ARRAY
                cocotb.log.info(
                    f"  MISMATCH chunk{m} idx={idx} (r{row},c{col}) got={g} want={e}"
                )

    exp_sanity = [O_sanity[m][n] for m in range(ARRAY) for n in range(ARRAY)]
    got_sanity = ram.read_bytes(OUT_SANITY_ADDR, len(exp_sanity))
    gs_sanity = [b - 256 if b >= 128 else b for b in got_sanity]
    for idx, (g, e) in enumerate(zip(gs_sanity, exp_sanity)):
        if g != e:
            fails += 1
            row, col = idx // ARRAY, idx % ARRAY
            cocotb.log.info(f"  MISMATCH sanity idx={idx} (r{row},c{col}) got={g} want={e}")

    assert fails == 0, f"weight-hold+K-tiling failed with {fails} mismatches"
    cocotb.log.info("weight-hold+K-tiling PASS")
