SIM = verilator
VERILOG_SOURCES += $(PWD)/../../../rtl/pkg/npu_pkg.sv \
                  $(PWD)/../../../rtl/pe/mac_unit.sv

TOPLEVEL = mac_unit
MODULE = mac_tb

include $(shell cocotb-config --makefiles)/Makefile.sim
