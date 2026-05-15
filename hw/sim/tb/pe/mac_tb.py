import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import random

PIPE_DEPTH = 3


async def reset(dut):
    dut.rst.value = 1
    dut.weight_load.value = 0
    dut.act_valid.value = 0
    dut.clear_acc.value = 0
    dut.act_in.value = 0
    dut.weight_in.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


def to_signed8(val):
    return val & 0xFF


def from_signed32(val):
    val = int(val)
    if val >= (1 << 31):
        val -= 1 << 32
    return val


async def stream_activations(dut, activations):
    """Stream activations and flush the pipeline with zeros."""
    for act in activations:
        dut.act_in.value = to_signed8(act)
        dut.act_valid.value = 1
        await RisingEdge(dut.clk)
    # keep valid high and then continue streaming zero so the last real product drains
    for _ in range(PIPE_DEPTH):
        dut.act_in.value = to_signed8(0)
        dut.act_valid.value = 1
        await RisingEdge(dut.clk)
    dut.act_valid.value = 0


async def load_weight(dut, weight):
    dut.weight_in.value = to_signed8(weight)
    dut.weight_load.value = 1
    await RisingEdge(dut.clk)
    dut.weight_load.value = 0


async def clear_acc(dut):
    """Assert clear_acc for one cycle. RSTP is synchronous so P clears
    on the rising edge while clear_acc is high."""
    dut.clear_acc.value = 1
    await RisingEdge(dut.clk)
    dut.clear_acc.value = 0
    await RisingEdge(dut.clk)  # one extra cycle for P to settle to 0


@cocotb.test()
async def test_single_mac(dut):
    """Single multiply-accumulate: weight=3, activation=4, expect 12"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, 3)
    await stream_activations(dut, [4])

    result = from_signed32(dut.acc_out.value)
    assert result == 12, f"Expected 12, got {result}"
    dut._log.info(f"Single MAC: PASS (got {result})")


@cocotb.test()
async def test_dot_product(dut):
    """Dot product of [1..14] with weight=3, expect 315"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, 3)

    activations = list(range(1, 15))
    expected = sum(3 * a for a in activations)
    await stream_activations(dut, activations)

    result = from_signed32(dut.acc_out.value)
    assert result == expected, f"Expected {expected}, got {result}"
    dut._log.info(f"Dot product: PASS (got {result})")


@cocotb.test()
async def test_signed_negative_weight(dut):
    """Negative weight × positive activation: -5 × 7 = -35"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, -5)
    await stream_activations(dut, [7])

    result = from_signed32(dut.acc_out.value)
    assert result == -35, f"Expected -35, got {result}"
    dut._log.info(f"Signed negative weight: PASS (got {result})")


@cocotb.test()
async def test_signed_both_negative(dut):
    """Both negative: -3 × -4 = 12"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, -3)
    await stream_activations(dut, [-4])

    result = from_signed32(dut.acc_out.value)
    assert result == 12, f"Expected 12, got {result}"
    dut._log.info(f"Both negative: PASS (got {result})")


@cocotb.test()
async def test_clear_accumulator(dut):
    """Accumulate, clear, verify zero, then accumulate again"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    await load_weight(dut, 10)
    await stream_activations(dut, [5])  # expect 50

    result = from_signed32(dut.acc_out.value)
    assert result == 50, f"Expected 50 before clear, got {result}"

    await clear_acc(dut)

    result = from_signed32(dut.acc_out.value)
    assert result == 0, f"Expected 0 after clear, got {result}"

    # Fresh accumulation after clear: 10 * 3 = 30
    await stream_activations(dut, [3])
    result = from_signed32(dut.acc_out.value)
    assert result == 30, f"Expected 30 after re-accumulate, got {result}"
    dut._log.info("Clear accumulator: PASS")


@cocotb.test()
async def test_saturation_boundary(dut):
    """Large values: stay within int32 range for 14 taps"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    # Worst case 8-bit: 127 * 127 * 14 = 225,778 — well within int32
    weight = 127
    activations = [127] * 14
    expected = sum(weight * a for a in activations)

    await load_weight(dut, weight)
    await stream_activations(dut, activations)

    result = from_signed32(dut.acc_out.value)
    assert result == expected, f"Expected {expected}, got {result}"
    dut._log.info(f"Saturation boundary: PASS (got {result})")


@cocotb.test()
async def test_random_dot_products(dut):
    """100 random dot products verified against Python reference"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    for trial in range(100):
        weight = random.randint(-128, 127)
        activations = [random.randint(-128, 127) for _ in range(14)]
        expected = sum(weight * a for a in activations)

        await clear_acc(dut)
        await load_weight(dut, weight)
        await stream_activations(dut, activations)

        result = from_signed32(dut.acc_out.value)
        assert result == expected, (
            f"Trial {trial}: weight={weight}, expected={expected}, got={result}"
        )

    dut._log.info("100 random dot products: PASS")
