import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

from matmul_helpers import AxiRam, build, ref_matmul, s8


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
