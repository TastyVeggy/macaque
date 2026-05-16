import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import random

PIPE_DEPTH = 3


async def reset(dut):
    dut.rst.value = 1
    dut.weight_load.value = 0
    dut.weight_in.value = 0
    dut.input_valid.value = 0
    dut.act_in.value = 0
    dut.acc_in.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


def to_signed8(val):
    return val & 0xFF


def to_signed32(val):
    return val & 0xFFFFFFFF


def from_signed32(val):
    val = int(val)
    if val >= (1 << 31):
        val -= 1 << 32
    return val


async def stream(dut, activations, accumulations):
    """Stream activations and acc_ins together and flush pipeline with zeros."""
    assert len(activations) == len(accumulations), (
        "activations and acc_ins must be same length"
    )

    for act, acc in zip(activations, accumulations):
        dut.act_in.value = to_signed8(act)
        dut.acc_in.value = to_signed32(acc)
        dut.input_valid.value = 1
        await RisingEdge(dut.clk)
    # keep valid high and then continue streaming zero so the last real product drains
    for _ in range(PIPE_DEPTH):
        dut.act_in.value = 0
        dut.acc_in.value = 0
        dut.input_valid.value = 1
        await RisingEdge(dut.clk)
    dut.input_valid.value = 0


async def load_weight(dut, weight):
    dut.weight_in.value = to_signed8(weight)
    dut.weight_load.value = 1
    await RisingEdge(dut.clk)
    dut.weight_load.value = 0


@cocotb.test()
async def test_single_multiply(dut):
    """Single multiply: weight=3, act=4, acc_in=0 -> acc_out=12"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, 3)
    await stream(dut, [4], [0])

    result = from_signed32(dut.acc_out.value)
    assert result == 12, f"Expected 12, got {result}"
    dut._log.info(f"Single MAC: PASS (got {result})")


@cocotb.test()
async def test_single_mac(dut):
    """Single multiply-accumulate: weight=3, act=4, acc_in=100 → acc_out=112"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, 3)
    await stream(dut, [4], [100])

    result = from_signed32(dut.acc_out.value)
    assert result == 112, f"Expected 112, got {result}"
    dut._log.info(f"MAC with acc_in: PASS (got {result})")


@cocotb.test()
async def test_streaming_acc_in(dut):
    """Simulate a column of PEs: acc_in accumulates down the column.
    weight=3, activations=[1,2,4], acc_ins=[0,3,9] → acc_outs=[3,9,21]
    Each acc_in is the previous PE's acc_out (with PIPE_DEPTH delay in real array,
    but here we drive them directly to test the PE in isolation)."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, 3)

    activations = [1, 2, 4]
    acc_ins = [0, 3, 9]
    expected = [3, 9, 21]

    await stream(dut, activations, acc_ins)

    result = from_signed32(dut.acc_out.value)
    assert result == expected[-1], f"Expected {expected[-1]}, got {result}"
    dut._log.info(f"Streaming acc_in: PASS (got {result})")


@cocotb.test()
async def test_signed_negative_weight(dut):
    """Negative weight: weight=-5, act=7, acc_in=0 → acc_out=-35"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, -5)
    await stream(dut, [7], [0])

    result = from_signed32(dut.acc_out.value)
    assert result == -35, f"Expected -35, got {result}"
    dut._log.info(f"Signed negative weight: PASS (got {result})")


@cocotb.test()
async def test_signed_both_negative(dut):
    """Both negative: weight=-3, act=-4, acc_in=0 → acc_out=12"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, -3)
    await stream(dut, [-4], [0])

    result = from_signed32(dut.acc_out.value)
    assert result == 12, f"Expected 12, got {result}"
    dut._log.info(f"Both negative: PASS (got {result})")


@cocotb.test()
async def test_acc_in_passthrough(dut):
    """acc_in with zero weight contribution: weight=0 not possible (int8),
    use weight=1, act=0, acc_in=42 → acc_out=42 (passthrough)"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, 1)
    await stream(dut, [0], [42])

    result = from_signed32(dut.acc_out.value)
    assert result == 42, f"Expected 42, got {result}"
    dut._log.info(f"acc_in passthrough: PASS (got {result})")


@cocotb.test()
async def test_max_values(dut):
    """Max positive: weight=127, act=127, acc_in=0 → 16129"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, 127)
    await stream(dut, [127], [0])

    result = from_signed32(dut.acc_out.value)
    assert result == 16129, f"Expected 16129, got {result}"
    dut._log.info(f"Max values: PASS (got {result})")


@cocotb.test()
async def test_output_valid_timing(dut):
    """output_valid should go high exactly PIPE_DEPTH cycles after input_valid"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, 1)

    # Assert input_valid for one cycle
    dut.act_in.value = to_signed8(1)
    dut.acc_in.value = 0
    dut.input_valid.value = 1
    await RisingEdge(dut.clk)
    dut.input_valid.value = 0

    # output_valid should be low for PIPE_DEPTH-1 cycles
    for i in range(PIPE_DEPTH - 1):
        await RisingEdge(dut.clk)
        assert dut.output_valid.value == 0, (
            f"output_valid should be 0 at cycle {i + 1}, got {dut.output_valid.value}"
        )

    # Then high on cycle PIPE_DEPTH
    await RisingEdge(dut.clk)
    assert dut.output_valid.value == 1, (
        f"output_valid should be 1 at cycle {PIPE_DEPTH}, got {dut.output_valid.value}"
    )
    dut._log.info("output_valid timing: PASS")


@cocotb.test()
async def test_act_out_timing(dut):
    """act_out should be delayed by exactly PIPE_DEPTH cycles"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, 1)

    sentinel = 99
    dut.act_in.value = to_signed8(sentinel)
    dut.acc_in.value = 0
    dut.input_valid.value = 1
    await RisingEdge(dut.clk)
    dut.act_in.value = 0
    dut.input_valid.value = 0

    for _ in range(PIPE_DEPTH - 1):
        await RisingEdge(dut.clk)

    await RisingEdge(dut.clk)
    result = int(dut.act_out.value)
    assert result == sentinel, f"Expected act_out={sentinel}, got {result}"
    dut._log.info(f"act_out timing: PASS (got {result})")


@cocotb.test()
async def test_random_pe(dut):
    """100 random single-cycle PE operations"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    for trial in range(100):
        rng = random.Random(trial)
        weight = rng.randint(-128, 127)
        act = rng.randint(-128, 127)
        acc = rng.randint(-(2**31), 2**31 - 1)
        expected = acc + weight * act

        await load_weight(dut, weight)
        await stream(dut, [act], [acc])

        result = from_signed32(dut.acc_out.value)
        assert result == expected, (
            f"Trial {trial}: weight={weight} act={act} acc_in={acc} "
            f"expected={expected} got={result}"
        )

    dut._log.info("100 random PE ops: PASS")
