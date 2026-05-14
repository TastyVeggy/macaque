SIM = verilator

VERILOG_SOURCES = $(PWD)/../../rtl/pkg/npu_pkg.sv \
                  $(PWD)/../../rtl/uart/uart_rx.sv \
                  $(PWD)/../../rtl/uart/uart_tx.sv \
                  $(PWD)/../../rtl/uart/uart_top.sv \
                  $(PWD)/../../rtl/common/reset_bridge.sv

TOPLEVEL = uart_top

MODULE = uart_tb

include $(shell cocotb-config --makefiles)/Makefile.sim
