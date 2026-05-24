import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
from cocotbext.axi import AxiLiteMaster, AxiLiteBus

REG_CTRL = 0x00
REG_STATUS = 0x04
REG_INSTR_ADDR = 0x08
REG_INSTR_LEN = 0x0C
REG_PMU_CTRL = 0x10
REG_PMU_CYCLES_LO = 0x14
REG_PMU_CYCLES_HI = 0x18
REG_PMU_COMPUTE = 0x1C
REG_PMU_STALL = 0x20
REG_PMU_DMA_RD = 0x24
REG_PMU_DMA_WR = 0x28


def to_bytes(val: int) -> bytes:
    return val.to_bytes(4, "little")


def from_bytes(data: bytes) -> int:
    return int.from_bytes(data, "little")


async def axil_write(master, addr: int, val: int) -> None:
    await master.write(addr, to_bytes(val))


async def axil_read(master, addr: int) -> int:
    result = await master.read(addr, 4)
    return from_bytes(result.data)


async def reset(dut):
    dut.S_AXI_ARESETN.value = 0
    dut.npu_ready.value = 0
    dut.npu_busy.value = 0
    dut.npu_done.value = 0
    dut.npu_error.value = 0
    dut.pmu_cycles.value = 0
    dut.pmu_compute.value = 0
    dut.pmu_stall.value = 0
    dut.pmu_dma_bytes_rd.value = 0
    dut.pmu_dma_bytes_wr.value = 0
    await ClockCycles(dut.S_AXI_ACLK, 5)
    dut.S_AXI_ARESETN.value = 1
    await ClockCycles(dut.S_AXI_ACLK, 2)  # settle


def make_master(dut):
    return AxiLiteMaster(
        AxiLiteBus.from_prefix(dut, "S_AXI"),
        dut.S_AXI_ACLK,
        dut.S_AXI_ARESETN,
        reset_active_level=False,
    )


@cocotb.test()
async def test_reset_clears_rw_registers(dut):
    """All RW registers must read back as 0 immediately after reset"""
    cocotb.start_soon(Clock(dut.S_AXI_ACLK, 10, unit="ns").start())
    await reset(dut)
    m = make_master(dut)

    for name, addr in [
        ("REG_CTRL", REG_CTRL),
        ("REG_INSTR_ADDR", REG_INSTR_ADDR),
        ("REG_INSTR_LEN", REG_INSTR_LEN),
        ("REG_PMU_CTRL", REG_PMU_CTRL),
    ]:
        val = await axil_read(m, addr)
        assert val == 0, f"{name} not 0 after reset: 0x{val:08X}"


@cocotb.test()
async def test_rw_registers(dut):
    """
    Write a value, read it back via AXI, and check the hardware
    output port reflects it
    """
    cocotb.start_soon(Clock(dut.S_AXI_ACLK, 10, unit="ns").start())
    await reset(dut)
    m = make_master(dut)

    await axil_write(m, REG_INSTR_ADDR, 0xDEAD_BEEF)
    assert await axil_read(m, REG_INSTR_ADDR) == 0xDEAD_BEEF, (
        "REG_INSTR_ADDR AXI readback mismatch"
    )
    assert dut.instr_addr.value == 0xDEAD_BEEF, "instr_addr output port did not update"

    await axil_write(m, REG_INSTR_LEN, 0x0000_0040)
    assert await axil_read(m, REG_INSTR_LEN) == 0x0000_0040, (
        "REG_INSTR_LEN AXI readback mismatch"
    )
    assert dut.instr_len.value == 0x0000_0040, "instr_len output port did not update"

    # PMU_CTRL enable bit (bit 0 — level signal, must hold)
    await axil_write(m, REG_PMU_CTRL, 0x0000_0001)
    await RisingEdge(dut.S_AXI_ACLK)  # let combinational output settle
    assert dut.pmu_enable.value == 1, "pmu_enable output port not set"
    val = await axil_read(m, REG_PMU_CTRL)
    assert val & 0x1, f"REG_PMU_CTRL enable bit dropped: 0x{val:08X}"

    # when overwriting, last write wins
    await axil_write(m, REG_INSTR_ADDR, 0x5555_5555)
    assert await axil_read(m, REG_INSTR_ADDR) == 0x5555_5555, (
        "Second write to INSTR_ADDR did not take effect"
    )

    # Boundary: all-ones
    await axil_write(m, REG_INSTR_ADDR, 0xFFFF_FFFF)
    assert await axil_read(m, REG_INSTR_ADDR) == 0xFFFF_FFFF

    # Boundary: all-zeros
    await axil_write(m, REG_INSTR_ADDR, 0x0000_0000)
    assert await axil_read(m, REG_INSTR_ADDR) == 0x0000_0000


@cocotb.test()
async def test_w1c_npu_start(dut):
    """
    CTRL[0] = npu_start is write-1-to-clear.
    RTL auto-clears it every cycle, so by the time we read back
    it must already be 0.
    """
    cocotb.start_soon(Clock(dut.S_AXI_ACLK, 10, unit="ns").start())
    await reset(dut)
    m = make_master(dut)

    await axil_write(m, REG_CTRL, 0x0000_0001)  # set start
    await ClockCycles(dut.S_AXI_ACLK, 2)  # let auto-clear run

    val = await axil_read(m, REG_CTRL)
    assert not (val & 0x1), f"npu_start (CTRL[0]) did not self-clear: 0x{val:08X}"


@cocotb.test()
async def test_w1c_pmu_clear(dut):
    """PMU_CTRL[1] = pmu_clear is write-1-to-clear."""
    cocotb.start_soon(Clock(dut.S_AXI_ACLK, 10, unit="ns").start())
    await reset(dut)
    m = make_master(dut)

    # First set enable so we can check it survives the clear pulse
    await axil_write(m, REG_PMU_CTRL, 0x0000_0001)  # enable=1
    await axil_write(m, REG_PMU_CTRL, 0x0000_0003)  # enable=1, clear=1
    await ClockCycles(dut.S_AXI_ACLK, 2)

    val = await axil_read(m, REG_PMU_CTRL)
    assert not (val & 0x2), f"pmu_clear (PMU_CTRL[1]) did not self-clear: 0x{val:08X}"
    assert val & 0x1, (
        f"pmu_enable was incorrectly cleared alongside pmu_clear: 0x{val:08X}"
    )


@cocotb.test()
async def test_npu_reset_is_level_not_w1c(dut):
    """
    CTRL[1] = npu_reset is a level signal, NOT W1C.
    It must stay high until explicitly written to 0.
    """
    cocotb.start_soon(Clock(dut.S_AXI_ACLK, 10, unit="ns").start())
    await reset(dut)
    m = make_master(dut)

    await axil_write(m, REG_CTRL, 0x0000_0002)  # reset=1
    await ClockCycles(dut.S_AXI_ACLK, 2)

    assert dut.npu_reset.value == 1, "npu_reset deasserted unexpectedly"
    val = await axil_read(m, REG_CTRL)
    assert val & 0x2, f"CTRL[1] (npu_reset) dropped: 0x{val:08X}"

    # Now deassert it
    await axil_write(m, REG_CTRL, 0x0000_0000)
    await RisingEdge(dut.S_AXI_ACLK)
    assert dut.npu_reset.value == 0, "npu_reset did not deassert after write 0"


@cocotb.test()
async def test_w1c_and_level_in_same_write(dut):
    """
    Writing CTRL = 0x3 (start=1, reset=1) simultaneously:
    start must self-clear, reset must hold.
    """
    cocotb.start_soon(Clock(dut.S_AXI_ACLK, 10, unit="ns").start())
    await reset(dut)
    m = make_master(dut)

    await axil_write(m, REG_CTRL, 0x0000_0003)
    await ClockCycles(dut.S_AXI_ACLK, 2)

    assert dut.npu_reset.value == 1, "npu_reset should still be asserted"
    val = await axil_read(m, REG_CTRL)
    assert not (val & 0x1), "npu_start should have self-cleared"
    assert val & 0x2, "npu_reset should still be set"


@cocotb.test()
async def test_status_register(dut):
    """REG_STATUS must mirror npu_ready/busy/done/error exactly."""
    cocotb.start_soon(Clock(dut.S_AXI_ACLK, 10, unit="ns").start())
    await reset(dut)
    m = make_master(dut)

    # Pattern A: ready=1, busy=0, done=1, error=0  → 0b0101 = 0x5
    dut.npu_ready.value = 1
    dut.npu_busy.value = 0
    dut.npu_done.value = 1
    dut.npu_error.value = 0
    await RisingEdge(dut.S_AXI_ACLK)
    val = await axil_read(m, REG_STATUS)
    assert val == 0x5, f"STATUS pattern A: expected 0x5, got 0x{val:08X}"

    # Pattern B: all set → 0xF
    dut.npu_ready.value = 1
    dut.npu_busy.value = 1
    dut.npu_done.value = 1
    dut.npu_error.value = 1
    await RisingEdge(dut.S_AXI_ACLK)
    val = await axil_read(m, REG_STATUS)
    assert val == 0xF, f"STATUS pattern B: expected 0xF, got 0x{val:08X}"

    # Pattern C: all clear → 0x0
    dut.npu_ready.value = 0
    dut.npu_busy.value = 0
    dut.npu_done.value = 0
    dut.npu_error.value = 0
    await RisingEdge(dut.S_AXI_ACLK)
    val = await axil_read(m, REG_STATUS)
    assert val == 0x0, f"STATUS pattern C: expected 0x0, got 0x{val:08X}"


@cocotb.test()
async def test_pmu_registers(dut):
    """All PMU read-only counters must reflect the driven input values."""
    cocotb.start_soon(Clock(dut.S_AXI_ACLK, 10, unit="ns").start())
    await reset(dut)
    m = make_master(dut)

    # Drive a 64-bit cycle counter and verify the LO/HI split
    dut.pmu_cycles.value = 0xDEAD_BEEF_CAFE_0001
    dut.pmu_compute.value = 0xAABB_CCDD
    dut.pmu_stall.value = 0x1122_3344
    dut.pmu_dma_bytes_rd.value = 0xFACE_0001
    dut.pmu_dma_bytes_wr.value = 0xFACE_0002
    await RisingEdge(dut.S_AXI_ACLK)

    lo = await axil_read(m, REG_PMU_CYCLES_LO)
    hi = await axil_read(m, REG_PMU_CYCLES_HI)
    assert lo == 0xCAFE_0001, f"PMU_CYCLES_LO: expected 0xCAFE0001, got 0x{lo:08X}"
    assert hi == 0xDEAD_BEEF, f"PMU_CYCLES_HI: expected 0xDEADBEEF, got 0x{hi:08X}"

    assert await axil_read(m, REG_PMU_COMPUTE) == 0xAABB_CCDD
    assert await axil_read(m, REG_PMU_STALL) == 0x1122_3344
    assert await axil_read(m, REG_PMU_DMA_RD) == 0xFACE_0001
    assert await axil_read(m, REG_PMU_DMA_WR) == 0xFACE_0002

    # Verify counter updates live — change value, read again
    dut.pmu_compute.value = 0x0000_0001
    await RisingEdge(dut.S_AXI_ACLK)
    assert await axil_read(m, REG_PMU_COMPUTE) == 0x0000_0001, (
        "PMU_COMPUTE did not update when input changed"
    )


@cocotb.test()
async def test_write_to_ro_address_ignored(dut):
    """Writing to STATUS or any PMU counter must not change them."""
    cocotb.start_soon(Clock(dut.S_AXI_ACLK, 10, unit="ns").start())
    await reset(dut)
    m = make_master(dut)

    dut.npu_ready.value = 1
    dut.npu_busy.value = 0
    dut.npu_done.value = 0
    dut.npu_error.value = 0
    dut.pmu_compute.value = 0xDEAD_DEAD
    await RisingEdge(dut.S_AXI_ACLK)

    before_status = await axil_read(m, REG_STATUS)
    before_compute = await axil_read(m, REG_PMU_COMPUTE)

    # Attempt to overwrite
    await axil_write(m, REG_STATUS, 0xFFFF_FFFF)
    await axil_write(m, REG_PMU_COMPUTE, 0xFFFF_FFFF)

    assert await axil_read(m, REG_STATUS) == before_status, (
        "REG_STATUS was modified by a write"
    )
    assert await axil_read(m, REG_PMU_COMPUTE) == before_compute, (
        "REG_PMU_COMPUTE was modified by a write"
    )


@cocotb.test()
async def test_register_independence(dut):
    """Writing one RW register must not affect any other."""
    cocotb.start_soon(Clock(dut.S_AXI_ACLK, 10, unit="ns").start())
    await reset(dut)
    m = make_master(dut)

    await axil_write(m, REG_INSTR_ADDR, 0xAAAA_AAAA)
    await axil_write(m, REG_INSTR_LEN, 0xBBBB_BBBB)
    await axil_write(m, REG_PMU_CTRL, 0x0000_0001)

    # Overwrite INSTR_ADDR only
    await axil_write(m, REG_INSTR_ADDR, 0x1234_5678)

    assert await axil_read(m, REG_INSTR_LEN) == 0xBBBB_BBBB, (
        "Writing INSTR_ADDR corrupted INSTR_LEN"
    )
    assert await axil_read(m, REG_PMU_CTRL) & 0x1, (
        "Writing INSTR_ADDR corrupted PMU_CTRL"
    )
    assert await axil_read(m, REG_INSTR_ADDR) == 0x1234_5678, (
        "INSTR_ADDR did not update correctly"
    )


@cocotb.test()
async def test_unmapped_address_returns_zero(dut):
    """Addresses not in the register map must return 0 on read."""
    cocotb.start_soon(Clock(dut.S_AXI_ACLK, 10, unit="ns").start())
    await reset(dut)
    m = make_master(dut)

    for addr in (0x2C, 0x30, 0x3C):
        val = await axil_read(m, addr)
        assert val == 0, f"Unmapped 0x{addr:02X} returned 0x{val:08X}"
