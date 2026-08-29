import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, ReadOnly, NextTimeStep

OP_WEIGHT = 0
OP_BIAS = 1
OP_ACT = 2


async def reset(dut):
    dut.rst.value = 1
    dut.load_req.value = 0
    dut.store_req.value = 0
    dut.load_target.value = 0
    dut.byte_count.value = 0
    dut.load_valid_bytes_per_row.value = 0
    dut.load_input_rows.value = 0
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tdata.value = 0
    dut.m_axis_tready.value = 1
    await ClockCycles(dut.clk, 3)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 2)


def bytes_to_beats(data):
    beats = []
    for i in range(0, len(data), 8):
        chunk = data[i : i + 8]
        word = 0
        for j, b in enumerate(chunk):
            word |= b << (8 * j)
        beats.append(word)
    return beats


async def pulse(dut, sig, cycles=1):
    """Pulse a 1-bit signal high for the given cycles from the next active edge."""
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    sig.value = 1
    await RisingEdge(dut.clk)
    sig.value = 0


async def drive_load_stream(dut, data):
    """Drive bytes onto s_axis as 8-byte beats, waiting for backpressure each beat."""
    beats = bytes_to_beats(data)
    for k, beat in enumerate(beats):
        dut.s_axis_tdata.value = beat
        dut.s_axis_tvalid.value = 1
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            tready = int(dut.s_axis_tready.value)
            await NextTimeStep()
            if tready:
                break
        dut.s_axis_tvalid.value = 0
    dut.s_axis_tvalid.value = 0


async def load_until_done(dut, data, timeout=1000):
    """Drive the LOAD stream while concurrently watching for load_done.
    load_done is a 1-cycle pulse that may fire during/just after streaming, so
    the watcher runs in parallel with the driver."""
    done = []

    async def watch():
        for _ in range(timeout):
            await RisingEdge(dut.clk)
            await ReadOnly()
            d = int(dut.load_done.value)
            await NextTimeStep()
            if d:
                done.append(True)
                return

    watcher = cocotb.start_soon(watch())
    await drive_load_stream(dut, data)
    await watcher
    return bool(done)


def ref_weight_rows(byte_count):
    n_rows = byte_count // 14
    rows = []
    for r in range(n_rows):
        rows.append([(r * 14 + c) & 0xFF for c in range(14)])
    return rows


class LoadWriteMonitor:
    def __init__(self, dut, port):
        self.dut = dut
        self.port = port  # 'wb', 'ab' or 'bb'
        self.writes = []  # list of (waddr, [14] vec)

    async def run(self):
        d = self.dut
        while True:
            await RisingEdge(d.clk)
            await ReadOnly()
            we = int(getattr(d, f"{self.port}_dma_we").value)
            waddr = int(getattr(d, f"{self.port}_dma_waddr").value)
            await NextTimeStep()
            if we:
                vec = [
                    int(getattr(d, f"{self.port}_dma_wdata")[c].value)
                    for c in range(14)
                ]
                self.writes.append((waddr, vec))


# ---------------------------------------------------------------------------
# STORE helpers
# ---------------------------------------------------------------------------
async def respond_quant(dut, quant):
    """Continuous quant-buffer model: on each qb_dma_re pulse (1 cycle), deliver
    qb_dma_rdata = quant[raddr] for the 2 cycles after the request, matching the
    real bram_sdp read latency the adapter expects."""
    delivering = 0
    deliver_row = 0
    while True:
        await RisingEdge(dut.clk)
        await ReadOnly()
        re = int(dut.qb_dma_re.value)
        raddr = int(dut.qb_dma_raddr.value) if re else -1
        await NextTimeStep()
        if re:
            delivering = 2
            deliver_row = raddr
        if delivering:
            row = quant[deliver_row]
            for c in range(14):
                dut.qb_dma_rdata[c].value = row[c]
            delivering -= 1
        else:
            for c in range(14):
                dut.qb_dma_rdata[c].value = 0


async def capture_store(dut, emitted, last_seen):
    """Capture S2MM beats on a valid handshake (tvalid & tready) until TLAST."""
    while True:
        await RisingEdge(dut.clk)
        await ReadOnly()
        v = int(dut.m_axis_tvalid.value)
        ready = int(dut.m_axis_tready.value)
        last = int(dut.m_axis_tlast.value)
        tdata = int(dut.m_axis_tdata.value)
        await NextTimeStep()
        if v and ready:
            for b in range(8):
                emitted.append((tdata >> (8 * b)) & 0xFF)
            if last:
                last_seen.append(True)
                return


@cocotb.test()
async def test_load_weight_rows(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    bc = 196
    mon = LoadWriteMonitor(dut, "wb")
    cocotb.start_soon(mon.run())

    dut.load_target.value = OP_WEIGHT
    dut.byte_count.value = bc
    await pulse(dut, dut.load_req)
    data = [(i) & 0xFF for i in range(bc)]
    assert await load_until_done(dut, data), "load_done never asserted"
    await ClockCycles(dut.clk, 2)

    rows = ref_weight_rows(bc)
    assert len(mon.writes) == 14, f"expected 14 rows written, got {len(mon.writes)}"
    for waddr, vec in mon.writes[:14]:
        assert vec == rows[waddr], f"row {waddr}: got {vec}, expected {rows[waddr]}"


@cocotb.test()
async def test_load_straddle_row_boundary(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    bc = 28
    mon = LoadWriteMonitor(dut, "wb")
    cocotb.start_soon(mon.run())

    dut.load_target.value = OP_WEIGHT
    dut.byte_count.value = bc
    await pulse(dut, dut.load_req)
    data = [(i) & 0xFF for i in range(bc)]
    assert await load_until_done(dut, data), "load_done never asserted"
    await ClockCycles(dut.clk, 2)

    rows = ref_weight_rows(bc)
    assert len(mon.writes) == 2, f"expected 2 rows written, got {len(mon.writes)}"
    for waddr, vec in mon.writes:
        assert vec == rows[waddr], f"row {waddr}: got {vec}, expected {rows[waddr]}"


@cocotb.test()
async def test_load_bias(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    bc = 56
    base = 0x1000
    mon = LoadWriteMonitor(dut, "bb")
    cocotb.start_soon(mon.run())

    # bias byte layout: lane c = (base + c*4) as 32-bit LE at bytes [c*4..c*4+3]
    data = [0] * bc
    for c in range(14):
        v = base + c * 4
        for b in range(4):
            data[c * 4 + b] = (v >> (8 * b)) & 0xFF
    dut.load_target.value = OP_BIAS
    dut.byte_count.value = bc
    await pulse(dut, dut.load_req)
    assert await load_until_done(dut, data), "load_done never asserted"
    await ClockCycles(dut.clk, 2)

    assert len(mon.writes) == 1, f"expected 1 bias row, got {len(mon.writes)}"
    waddr, vec = mon.writes[0]
    assert waddr == 0, f"bias waddr should be 0, got {waddr}"
    for c in range(14):
        assert vec[c] == (base + c * 4), f"bias lane {c}: got {vec[c]:#x}"


@cocotb.test()
async def test_load_back_to_back(dut):
    """Two consecutive LOADs with no idle gap must both complete and land in the
    correct buffers (regression: the adapter FSM must return to IDLE after each
    transfer)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    wb_mon = LoadWriteMonitor(dut, "wb")
    bb_mon = LoadWriteMonitor(dut, "bb")
    cocotb.start_soon(wb_mon.run())
    cocotb.start_soon(bb_mon.run())

    # First: a weight LOAD (196 bytes -> 14 rows)
    wb_bc = 196
    dut.load_target.value = OP_WEIGHT
    dut.byte_count.value = wb_bc
    await pulse(dut, dut.load_req)
    wdata = [(i) & 0xFF for i in range(wb_bc)]
    assert await load_until_done(dut, wdata), "first LOAD never done"

    # Second: a bias LOAD immediately after (56 bytes -> 1 row)
    base = 0x2000
    bb_data = [0] * 56
    for c in range(14):
        v = base + c * 4
        for b in range(4):
            bb_data[c * 4 + b] = (v >> (8 * b)) & 0xFF
    dut.load_target.value = OP_BIAS
    dut.byte_count.value = 56
    await pulse(dut, dut.load_req)
    assert await load_until_done(dut, bb_data), "second LOAD never done"

    await ClockCycles(dut.clk, 2)

    assert len(wb_mon.writes) == 14, (
        f"expected 14 weight rows, got {len(wb_mon.writes)}"
    )
    assert len(bb_mon.writes) == 1, f"expected 1 bias row, got {len(bb_mon.writes)}"
    _, bvec = bb_mon.writes[0]
    for c in range(14):
        assert bvec[c] == (base + c * 4), f"bias lane {c}: got {bvec[c]:#x}"


@cocotb.test()
async def test_load_input_unpadded(dut):
    """Plain, non-padded LOAD_INPUT (load_valid_bytes_per_row=0): must behave
    exactly like the legacy weight/bias loads above - never exercised by any
    other test, since they only cover OP_WEIGHT/OP_BIAS."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    bc = 196  # 14 full rows
    mon = LoadWriteMonitor(dut, "ab")
    cocotb.start_soon(mon.run())

    dut.load_target.value = OP_ACT
    dut.byte_count.value = bc
    await pulse(dut, dut.load_req)
    data = [(i) & 0xFF for i in range(bc)]
    assert await load_until_done(dut, data), "load_done never asserted"
    await ClockCycles(dut.clk, 2)

    rows = ref_weight_rows(bc)
    assert len(mon.writes) == 14, f"expected 14 rows written, got {len(mon.writes)}"
    for waddr, vec in mon.writes:
        assert vec == rows[waddr], f"row {waddr}: got {vec}, expected {rows[waddr]}"


@cocotb.test()
async def test_load_input_padded_row(dut):
    """LOAD_INPUT with the new DMA zero-injection field set: K isn't a
    multiple of 14, so only the first `valid` bytes of every row are real -
    dense-packed in DDR3 back to back, no per-row padding gap, matching
    byte_count = rows * valid. The adapter must pop each row after just `valid`
    real bytes and zero-fill lanes [valid, 14) itself"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    rows = 3
    valid = 6
    mon = LoadWriteMonitor(dut, "ab")
    cocotb.start_soon(mon.run())

    row_real = [[1 + r * 10 + c for c in range(valid)] for r in range(rows)]
    data = [b for row in row_real for b in row]  # dense, no padding gap

    dut.load_target.value = OP_ACT
    dut.byte_count.value = rows * valid
    dut.load_valid_bytes_per_row.value = valid
    dut.load_input_rows.value = rows
    await pulse(dut, dut.load_req)
    assert await load_until_done(dut, data), "padded load never done"
    await ClockCycles(dut.clk, 2)

    assert len(mon.writes) == rows, f"expected {rows} rows written, got {len(mon.writes)}"
    for r, (waddr, vec) in enumerate(mon.writes):
        assert waddr == r, f"row {r}: waddr {waddr}"
        assert vec[:valid] == row_real[r], (
            f"row {r} real lanes: got {vec[:valid]}, expected {row_real[r]}"
        )
        assert vec[valid:] == [0] * (14 - valid), (
            f"row {r} padding lanes should be zero, got {vec[valid:]}"
        )


@cocotb.test()
async def test_store_roundtrip(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    n_rows = 2
    quant = [[(r * 14 + c) & 0xFF for c in range(14)] for r in range(n_rows)]
    expected = [quant[r][c] for r in range(n_rows) for c in range(14)]
    emitted = []
    last_seen = []

    cocotb.start_soon(respond_quant(dut, quant))

    dut.byte_count.value = n_rows * 14
    await pulse(dut, dut.store_req)

    await capture_store(dut, emitted, last_seen)

    # 28 real bytes -> 4 beats (24 + a 4-byte tail): first 28 match, tail pad zero.
    assert emitted[:28] == expected, (
        f"store mismatch: got {emitted[:28]}, expected {expected}"
    )
    assert len(last_seen) == 1, "exactly one TLAST expected"
    assert len(emitted) == 32, f"expected 4 beats (32 bytes), got {len(emitted)}"
    assert emitted[28:] == [0, 0, 0, 0], (
        f"tail pad bytes should be zero, got {emitted[28:]}"
    )


@cocotb.test()
async def test_store_backpressure(dut):
    """STORE under AXI backpressure: hold m_axis_tready low for stretches. The
    accumulator must not overflow (regression: unbounded reads when the DMA
    stalls) and the emitted stream must remain correct."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    n_rows = 14  # 196 bytes; more rows than ACC_BYTES/14 = 4, forcing fill+stall
    quant = [[(r * 14 + c) & 0xFF for c in range(14)] for r in range(n_rows)]
    expected = [quant[r][c] for r in range(n_rows) for c in range(14)]
    emitted = []
    last_seen = []

    async def flaky_ready():
        # alternate: 2 cycles ready, 6 cycles stalled
        while True:
            dut.m_axis_tready.value = 1
            await ClockCycles(dut.clk, 2)
            dut.m_axis_tready.value = 0
            await ClockCycles(dut.clk, 6)

    cocotb.start_soon(respond_quant(dut, quant))
    cocotb.start_soon(flaky_ready())

    dut.byte_count.value = n_rows * 14
    await pulse(dut, dut.store_req)

    await capture_store(dut, emitted, last_seen)

    assert emitted[:196] == expected, (
        f"store under backpressure mismatch: got {emitted[:196]}, expected {expected}"
    )
    assert len(last_seen) == 1, "exactly one TLAST expected"
