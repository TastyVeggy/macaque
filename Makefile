export ROOT_DIR := $(shell pwd)
BUILD_SW        := sw/build/debug
TOP             ?= npu_top

.PHONY: hw sw clean help sim_uart sim_pe test_sw

help:
	@echo "Targets:"
	@echo "  hw        - Build the hardware (Synthesis, Impl, Bitstream)"
	@echo "  hw TOP=x  - Build the hardware with specified top module"
	@echo "  sw 	   - Build the software stack"
	@echo "  clean     - Remove build artifacts"
	@echo "  test_sw   - Build and run C++ unit tests"
	@echo "  sim_uart  - Test the uart"

hw: 
	$(MAKE) -C hw TOP=$(TOP)

sw:
	cmake --preset debug -S sw
	cmake --build $(BUILD_SW)

sim_uart:
	$(MAKE) -C hw/sim/tb/uart

sim_pe:
	$(MAKE) -C hw/sim/tb/pe

test_sw: sw
	ctest --test-dir $(BUILD_SW) --output-on-failure

clean:
	$(MAKE) -C hw clean
	rm -rf sw/build

