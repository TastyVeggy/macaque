import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, ReadOnly, NextTimeStep


async def reset(dut):
    dut.clk_sa.value = 0
    dut.sa_we.value = 0
    dut.sa_waddr.value = 0
    dut.rd_re.value = 0
    dut.rd_raddr.value = 0
    dut.rd_sel.value = 0
    dut.out_bank_sel.value = 0
    for c in range(14):
        dut.sa_wdata[c].value = 0
    await ClockCycles(dut.clk_sa, 3)


async def write_row(dut, bank, addr, values):
    """Write one 14-lane row to the given bank via the SA write port."""
    dut.out_bank_sel.value = bank
    dut.sa_waddr.value = addr
    for c in range(14):
        dut.sa_wdata[c].value = values[c]
    dut.sa_we.value = 1
    await RisingEdge(dut.clk_sa)
    dut.sa_we.value = 0


async def read_row(dut, rd_sel, addr):
    """Issue a read and return the 14-lane vector (1-cycle BRAM latency)."""
    dut.rd_sel.value = rd_sel
    dut.rd_raddr.value = addr
    dut.rd_re.value = 1
    await RisingEdge(dut.clk_sa)
    dut.rd_re.value = 0
    await RisingEdge(dut.clk_sa)
    await ReadOnly()
    vec = [int(dut.rd_rdata[c].value) for c in range(14)]
    await NextTimeStep()
    return vec


@cocotb.test()
async def test_rd_sel_feedback(dut):
    """rd_sel=1 must read the ACTIVE (drain) bank; rd_sel=0 the INACTIVE bank.

    This is the tiled matmul feedback path: while the array drains bank `out_bank_sel`,
    the acc_mode=1 seed must read back that same bank's partial sums."""
    cocotb.start_soon(Clock(dut.clk_sa, 10, unit="ns").start())
    await reset(dut)

    bank0 = [(0 * 14 + c) & 0xFF for c in range(14)]
    bank1 = [(0x100 + 0 * 14 + c) & 0xFF for c in range(14)]

    # Fill bank0 row 0 and bank1 row 0 with distinct values.
    await write_row(dut, 0, 0, bank0)
    await write_row(dut, 1, 0, bank1)

    # With out_bank_sel=0: active bank = bank0, inactive = bank1.
    dut.out_bank_sel.value = 0

    fb = await read_row(dut, rd_sel=1, addr=0)  # active -> bank0
    act = await read_row(dut, rd_sel=0, addr=0)  # inactive -> bank1

    assert fb == bank0, f"rd_sel=1 (active) got {fb}, expected {bank0}"
    assert act == bank1, f"rd_sel=0 (inactive) got {act}, expected {bank1}"

    # Flip the bank: out_bank_sel=1 -> active = bank1, inactive = bank0.
    dut.out_bank_sel.value = 1
    fb = await read_row(dut, rd_sel=1, addr=0)
    act = await read_row(dut, rd_sel=0, addr=0)

    assert fb == bank1, f"rd_sel=1 (active, flipped) got {fb}, expected {bank1}"
    assert act == bank0, f"rd_sel=0 (inactive, flipped) got {act}, expected {bank0}"
