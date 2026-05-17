import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import random

PIPE_DEPTH = 3  # for acc_out
ACT_DELAY = 1  # for act_out


async def reset(dut):
    dut.rst.value = 1
    dut.weight_in.value = 0
    dut.weight_in_valid.value = 0
    dut.act_in.value = 0
    dut.act_in_valid.value = 0
    dut.acc_in.value = 0
    dut.acc_in_valid.value = 0
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


async def load_weight(dut, weight):
    dut.weight_in.value = to_signed8(weight)
    dut.weight_in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.weight_in_valid.value = 0


async def drive_one(dut, act, acc):
    """Drive act and acc for one cycle with both valids high"""
    dut.act_in.value = to_signed8(act)
    dut.acc_in.value = to_signed32(acc)
    dut.act_in_valid.value = 1
    dut.acc_in_valid.value = 1
    await RisingEdge(dut.clk)


async def drive_idle(dut):
    dut.act_in.value = 0
    dut.acc_in.value = 0
    dut.act_in_valid.value = 0
    dut.acc_in_valid.value = 0
    await RisingEdge(dut.clk)


async def stream_and_wait(dut, acts, accs):
    """Stream inputs then flush PIPE_DEPTH idle cycles to drain acc pipeline"""
    assert len(acts) == len(accs)
    for act, acc in zip(acts, accs):
        await drive_one(dut, act, acc)
    for _ in range(PIPE_DEPTH):
        await drive_idle(dut)


@cocotb.test()
async def test_act_out_is_1_cycle(dut):
    """act_out must appear exactly 1 cycle after act_in for fast east path"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, 1)

    sentinel = 0x5A
    dut.act_in.value = sentinel
    dut.act_in_valid.value = 1
    dut.acc_in.value = 0
    dut.acc_in_valid.value = 1
    await RisingEdge(dut.clk)

    dut.act_in.value = 0
    dut.act_in_valid.value = 0
    dut.acc_in_valid.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.act_out.value) == sentinel, (
        f"act_out: expected {sentinel:#x}, got {int(dut.act_out.value):#x}"
    )
    assert int(dut.act_out_valid.value) == 1, (
        f"act_out_valid: expected 1, got {int(dut.act_out_valid.value)}"
    )
    dut._log.info("act_out 1-cycle delay: PASS")

    await RisingEdge(dut.clk)
    assert int(dut.act_out_valid.value) == 0, (
        f"act_out_valid should clear, got {int(dut.act_out_valid.value)}"
    )
    dut._log.info("act_out_valid clears: PASS")


@cocotb.test()
async def test_act_out_not_3_cycles(dut):
    """Confirm act_out is not delayed 3 cycles"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, 1)

    sentinel = 0x7B
    dut.act_in.value = sentinel
    dut.act_in_valid.value = 1
    dut.acc_in.value = 0
    dut.acc_in_valid.value = 1
    await RisingEdge(dut.clk)  # cycle 0: captured
    dut.act_in.value = 0
    dut.act_in_valid.value = 0
    dut.acc_in_valid.value = 0

    await RisingEdge(dut.clk)  # cycle 1: act_out = sentinel
    got_cycle1 = int(dut.act_out.value)

    await RisingEdge(dut.clk)  # cycle 2: act_out = 0
    await RisingEdge(dut.clk)  # cycle 3: act_out = 0
    got_cycle3 = int(dut.act_out.value)

    assert got_cycle1 == sentinel, (
        f"act_out not at cycle 1: fast east broken! got {got_cycle1}"
    )
    assert got_cycle3 != sentinel, "act_out still showing at cycle 3: delay is 3 not 1!"
    dut._log.info(f"fast east: PASS (cycle1={got_cycle1:#x} cycle3={got_cycle3:#x})")


@cocotb.test()
async def test_act_vs_acc_valid(dut):
    """act_out_valid arrives 1 cycle after input, acc_out_valid arrives 3 cycles after"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, 1)

    dut.act_in.value = to_signed8(5)
    dut.act_in_valid.value = 1
    dut.acc_in.value = to_signed32(7)
    dut.acc_in_valid.value = 1
    await RisingEdge(dut.clk)  # cycle 0: inputs captured
    dut.act_in_valid.value = 0
    dut.acc_in_valid.value = 0
    dut.act_in.value = 0
    dut.acc_in.value = 0

    # Cycle 1: act_out_valid=1, acc_out_valid=0
    await RisingEdge(dut.clk)
    act_vld = int(dut.act_out_valid.value)
    acc_vld = int(dut.acc_out_valid.value)
    assert act_vld == 1, f"act_out_valid should be 1 at cycle 1, got {act_vld}"
    assert acc_vld == 0, f"acc_out_valid should be 0 at cycle 1, got {acc_vld}"
    dut._log.info(f"Cycle 1: act_out_valid={act_vld} acc_out_valid={acc_vld}")

    # Cycle 2: both 0
    await RisingEdge(dut.clk)
    act_vld = int(dut.act_out_valid.value)
    acc_vld = int(dut.acc_out_valid.value)
    assert act_vld == 0, f"act_out_valid should be 0 at cycle 2, got {act_vld}"
    assert acc_vld == 0, f"acc_out_valid should be 0 at cycle 2, got {acc_vld}"
    dut._log.info(f"Cycle 2: act_out_valid={act_vld} acc_out_valid={acc_vld}")

    # Cycle 3: acc_out_valid=1, result correct
    await RisingEdge(dut.clk)
    act_vld = int(dut.act_out_valid.value)
    acc_vld = int(dut.acc_out_valid.value)
    acc_result = from_signed32(dut.acc_out.value)
    assert act_vld == 0, f"act_out_valid should be 0 at cycle 3, got {act_vld}"
    assert acc_vld == 1, f"acc_out_valid should be 1 at cycle 3, got {acc_vld}"
    assert acc_result == 12, f"acc_out should be 1*5+7=12, got {acc_result}"
    dut._log.info(
        f"Cycle 3: act_out_valid={act_vld} acc_out_valid={acc_vld} acc_out={acc_result}"
    )
    dut._log.info("act/acc valid: PASS")


@cocotb.test()
async def test_back_to_back_act_passthrough(dut):
    """5 values back to back: each appears on act_out exactly 1 cycle later"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, 0)

    sent = [10, 20, 30, 40, 50]
    received = []

    # Drive value N and sample act_out of value N-1 on the same edge
    # Cycle 0: drive sent[0]
    # Cycle 1: drive sent[1], sample act_out = sent[0]
    # :
    # Cycle 5: drive nothing, sample act_out = sent[4]

    for i, val in enumerate(sent):
        dut.act_in.value = to_signed8(val)
        dut.act_in_valid.value = 1
        dut.acc_in.value = 0
        dut.acc_in_valid.value = 0
        await RisingEdge(dut.clk)
        # After this edge: act_out holds sent[i-1]
        if i > 0:
            received.append(int(dut.act_out.value))

    # Final cycle: deassert and capture last value
    dut.act_in.value = 0
    dut.act_in_valid.value = 0
    await RisingEdge(dut.clk)
    received.append(int(dut.act_out.value))

    dut._log.info(f"Sent:     {sent}")
    dut._log.info(f"Received: {received}")

    for i in range(len(sent)):
        assert received[i] == sent[i], (
            f"Position {i}: sent {sent[i]}, got {received[i]}"
        )
    dut._log.info("back-to-back act passthrough: PASS")


@cocotb.test()
async def test_acc_out_valid_is_pipe_depth(dut):
    """acc_out_valid must be delayed exactly PIPE_DEPTH cycles from acc_in_valid"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, 1)

    # Single cycle pulse
    dut.act_in.value = to_signed8(1)
    dut.act_in_valid.value = 1
    dut.acc_in.value = 0
    dut.acc_in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.act_in_valid.value = 0
    dut.acc_in_valid.value = 0
    dut.act_in.value = 0

    # acc_out_valid should be low for cycles 1 and 2
    for i in range(PIPE_DEPTH - 1):
        await RisingEdge(dut.clk)
        assert int(dut.acc_out_valid.value) == 0, (
            f"acc_out_valid high too early at cycle {i + 1}"
        )

    # High on cycle PIPE_DEPTH
    await RisingEdge(dut.clk)
    assert int(dut.acc_out_valid.value) == 1, (
        f"acc_out_valid not high at cycle {PIPE_DEPTH}"
    )
    dut._log.info(f"acc_out_valid PIPE_DEPTH={PIPE_DEPTH} delay: PASS")

    # Low again after
    await RisingEdge(dut.clk)
    assert int(dut.acc_out_valid.value) == 0, (
        "acc_out_valid did not clear after PIPE_DEPTH+1"
    )
    dut._log.info("acc_out_valid clears after pulse: PASS")


@cocotb.test()
async def test_single_mac(dut):
    """weight=3, act=4, acc_in=100 → acc_out=112"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)
    await load_weight(dut, 3)
    await stream_and_wait(dut, [4], [100])
    result = from_signed32(dut.acc_out.value)
    assert result == 112, f"Expected 112, got {result}"
    dut._log.info(f"single MAC: PASS (got {result})")


@cocotb.test()
async def test_random_mac(dut):
    """100 random signle-cycle MAC operations"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    for trial in range(100):
        rng = random.Random(trial)
        weight = rng.randint(-128, 127)
        act = rng.randint(-128, 127)
        acc = rng.randint(-(2**20), 2**20)
        expected = acc + weight * act

        await load_weight(dut, weight)
        await stream_and_wait(dut, [act], [acc])
        result = from_signed32(dut.acc_out.value)
        assert result == expected, (
            f"trial={trial} w={weight} a={act} acc={acc} expected={expected} got={result}"
        )

    dut._log.info("100 random MAC ops: PASS")
