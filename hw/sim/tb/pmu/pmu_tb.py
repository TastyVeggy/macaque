import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge


async def stable_cycles(dut, n):
    """Wait for n clock cycles and return the delta of pmu_cycles"""
    before = int(dut.pmu_cycles.value)
    await ClockCycles(dut.clk, n)
    after = int(dut.pmu_cycles.value)
    return after - before


async def reset(dut):
    dut.rst.value = 1
    dut.enable.value = 0
    dut.clear.value = 0
    dut.run_active.value = 0
    dut.mac_active.value = 0
    dut.stall.value = 0
    dut.dma_bytes_rd_this_cycle.value = 0
    dut.dma_bytes_wr_this_cycle.value = 0
    dut.frozen.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 5)


@cocotb.test()
async def test_pmu_counts_cycles_when_active(dut):
    """pmu_cycles increments 1/cycle when enable+run_active are high"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    dut.run_active.value = 1
    dut.enable.value = 1
    await RisingEdge(dut.clk)

    delta = await stable_cycles(dut, 10)
    assert delta == 10, f"pmu_cycles delta: expected 10, got {delta}"

    dut.mac_active.value = 1
    await RisingEdge(dut.clk)
    await ClockCycles(dut.clk, 5)
    assert int(dut.pmu_compute.value) == 5, (
        f"pmu_compute: expected 5, got {int(dut.pmu_compute.value)}"
    )

    dut.mac_active.value = 0
    dut.stall.value = 1
    await RisingEdge(dut.clk)
    await ClockCycles(dut.clk, 3)
    assert int(dut.pmu_stall.value) == 3, (
        f"pmu_stall: expected 3, got {int(dut.pmu_stall.value)}"
    )


@cocotb.test()
async def test_pmu_clear_resets_all(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    dut.run_active.value = 1
    dut.enable.value = 1
    dut.mac_active.value = 1
    await RisingEdge(dut.clk)
    await ClockCycles(dut.clk, 5)

    assert int(dut.pmu_cycles.value) > 0
    assert int(dut.pmu_compute.value) > 0

    dut.clear.value = 1
    await RisingEdge(dut.clk)
    dut.clear.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.pmu_cycles.value) == 0, f"not cleared: {int(dut.pmu_cycles.value)}"
    assert int(dut.pmu_compute.value) == 0, "pmu_compute not cleared"


@cocotb.test()
async def test_pmu_frozen_holds(dut):
    """Frozen signal prevents counting"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    dut.run_active.value = 1
    dut.enable.value = 1
    await RisingEdge(dut.clk)
    await ClockCycles(dut.clk, 5)
    val = int(dut.pmu_cycles.value)

    dut.frozen.value = 1
    await RisingEdge(dut.clk)
    delta = await stable_cycles(dut, 5)
    assert delta == 0, (
        f"pmu_cycles changed by {delta} while frozen (was {val}, now {int(dut.pmu_cycles.value)})"
    )


@cocotb.test()
async def test_pmu_disabled_when_enable_low(dut):
    """Counters do not increment when enable=0"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    dut.run_active.value = 1
    dut.enable.value = 0
    await RisingEdge(dut.clk)
    delta = await stable_cycles(dut, 5)
    assert delta == 0, f"pmu_cycles changed by {delta} while disabled"


@cocotb.test()
async def test_pmu_clear_while_running(dut):
    """Clear resets counters; counting resumes after"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    dut.run_active.value = 1
    dut.enable.value = 1
    await RisingEdge(dut.clk)
    await ClockCycles(dut.clk, 3)

    dut.clear.value = 1
    await RisingEdge(dut.clk)
    dut.clear.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.pmu_cycles.value) == 0, "not cleared after clear pulse"

    delta = await stable_cycles(dut, 5)
    assert delta == 5, f"pmu_cycles: expected 5 after resume, got {delta}"
