from cocotb.triggers import Timer, RisingEdge
import cocotb
from cocotb.clock import Clock

CLK_FREQ = 50_000_000
BAUD_RATE = 115_200
BIT_PERIOD_NS = int(1e9 / BAUD_RATE)


async def uart_send_byte(dut, byte):
    dut.rx_pin.value = 0
    await Timer(BIT_PERIOD_NS, unit="ns")

    for i in range(8):
        dut.rx_pin.value = (byte >> i) & 1
        await Timer(BIT_PERIOD_NS, unit="ns")

    dut.rx_pin.value = 1
    await Timer(BIT_PERIOD_NS, unit="ns")


async def uart_recv_byte(dut):
    while dut.tx_pin.value == 1:
        await RisingEdge(dut.clk)
    await Timer(BIT_PERIOD_NS // 2, unit="ns")

    byte = 0
    for i in range(8):
        await Timer(BIT_PERIOD_NS, unit="ns")
        bit = int(dut.tx_pin.value)
        byte |= bit << i
    await Timer(BIT_PERIOD_NS, unit="ns")
    return byte


@cocotb.test()
async def test_uart_echo(dut):
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())

    dut.rst_n.value = 0
    dut.rx_pin.value = 1
    for _ in range(10):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1

    test_bytes = [0x41, 0x42, 0x55, 0xFF, 0x00]

    for sent in test_bytes:
        await uart_send_byte(dut, sent)
        received = await uart_recv_byte(dut)
        assert received == sent, (
            f"Echo mismatch: sent 0x{sent:02X}, got 0x{received:02X}"
        )
        dut._log.info(f"Echo: PASS (0x{sent:02X} echoed)")
