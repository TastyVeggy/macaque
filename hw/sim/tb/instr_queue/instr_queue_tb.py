import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

DEPTH = 8


async def reset(dut):
    dut.rst.value = 1
    dut.push.value = 0
    dut.push_data.value = 0
    dut.pop.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 2)


async def push_val(dut, v):
    dut.push_data.value = v
    dut.push.value = 1
    await RisingEdge(dut.clk)  # FIFO samples push at this edge
    dut.push.value = 0


async def pop_val(dut):
    """Read the current head value, then pop it. Returns the popped value."""
    await ClockCycles(dut.clk, 1)  # settle so any prior pop's head advance is visible
    v = int(dut.pop_data.value)  # mem[head]
    dut.pop.value = 1
    await RisingEdge(dut.clk)
    dut.pop.value = 0
    return v


async def flags(dut):
    """Advance past the last operation, then sample full/empty (settled)."""
    await ClockCycles(dut.clk, 1)
    return int(dut.full.value), int(dut.empty.value)


@cocotb.test()
async def test_reset_empty(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    f, e = await flags(dut)
    assert e == 1, "empty should be high after reset"
    assert f == 0, "full should be low after reset"


@cocotb.test()
async def test_fifo_order_and_drain(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    for v in range(DEPTH):
        await push_val(dut, v)
    f, e = await flags(dut)
    assert f == 1, "queue should be full after DEPTH pushes"
    assert e == 0

    for expect in range(DEPTH):
        got = await pop_val(dut)
        assert got == expect, f"pop: got {got}, expected {expect}"
    f, e = await flags(dut)
    assert e == 1, "queue should be empty after drain"


@cocotb.test()
async def test_full_at_capacity(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    for v in range(DEPTH):
        await push_val(dut, v)
    f, e = await flags(dut)
    assert f == 1

    # A 9th push must be ignored: full stays high.
    await push_val(dut, 0xDEAD)
    f, _ = await flags(dut)
    assert f == 1, "full should stay high after push-when-full"


@cocotb.test()
async def test_push_when_full_noop(dut):
    """At full, a pushed sentinel must not overwrite any entry."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    for v in range(DEPTH):
        await push_val(dut, v)
    f, _ = await flags(dut)
    assert f == 1

    await push_val(dut, 0xDEADBEEF)
    for expect in range(DEPTH):
        got = await pop_val(dut)
        assert got == expect, f"sentinel corrupted queue: got {got}, expected {expect}"
    f, e = await flags(dut)
    assert e == 1


@cocotb.test()
async def test_pop_when_empty_noop(dut):
    """Pop on empty must not advance the head (next push is the first value)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    f, e = await flags(dut)
    assert e == 1
    await pop_val(dut)  # ignored
    f, e = await flags(dut)
    assert e == 1, "empty should stay high after pop on empty"

    await push_val(dut, 42)
    assert await pop_val(dut) == 42, "head drifted after pop-on-empty"


@cocotb.test()
async def test_simultaneous_push_pop(dut):
    """push+pop in the same cycle: occupancy unchanged, order preserved."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    await push_val(dut, 0)
    await push_val(dut, 1)  # queue: [0, 1]

    dut.push_data.value = 2
    dut.push.value = 1
    dut.pop.value = 1
    await RisingEdge(dut.clk)  # pushes 2, pops 0 in the same edge
    dut.push.value = 0
    dut.pop.value = 0

    f, e = await flags(dut)
    assert e == 0
    assert f == 0  # occupancy unchanged (2)

    # queue now holds [1, 2]
    assert await pop_val(dut) == 1
    assert await pop_val(dut) == 2
    f, e = await flags(dut)
    assert e == 1


@cocotb.test()
async def test_wraparound_full(dut):
    """Fill → drain → refill: exercises RAM-index wrap and the 2^CW counter wrap."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    # First generation: 0..7 (tail goes 0..8)
    for v in range(DEPTH):
        await push_val(dut, v)
    f, _ = await flags(dut)
    assert f == 1
    for expect in range(DEPTH):
        assert await pop_val(dut) == expect
    _, e = await flags(dut)
    assert e == 1

    # Second generation: 8..15 — tail counter wraps at 2^CW (8 -> 16 mod 16 = 0),
    # RAM index wraps 7 -> 0 again.
    for v in range(DEPTH, 2 * DEPTH):
        await push_val(dut, v)
    f, _ = await flags(dut)
    assert f == 1
    for expect in range(DEPTH, 2 * DEPTH):
        assert await pop_val(dut) == expect
    _, e = await flags(dut)
    assert e == 1
