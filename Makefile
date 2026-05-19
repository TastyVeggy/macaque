export ROOT_DIR := $(shell pwd)
BUILD_SW        := sw/build/debug
TOP             ?= npu_top

.PHONY: hw sw clean distclean help test_sw sim_hw

help:
	@echo "Targets:"
	@echo "  hw                            - Build the hardware (Synthesis, Impl, Bitstream)"
	@echo "  hw TOP=x                      - Build the hardware with specified top module"
	@echo "  sw                            - Build the software stack"
	@echo "  clean                         - Fast clean (keep compiled dependencies)"
	@echo "  distclean                     - Full clean"
	@echo "  test_sw                       - Build and run sw unit tests"
	@echo "  sim_hw TB=x MODE=cocotb|xsim  - Simulate hardware"

hw: 
	$(MAKE) -C hw TOP=$(TOP)

sim_hw:
	$(MAKE) sim_hw -C hw/sim TB=$(TB) MODE=$(MODE)

sw:
	cmake --preset debug -S sw
	cmake --build $(BUILD_SW)

test_sw: sw
	ctest --test-dir $(BUILD_SW) --output-on-failure

clean:
	$(MAKE) -C hw clean
ifneq ($(wildcard $(BUILD_SW)),)
	-cmake --build $(BUILD_SW) --target common/clean
endif

distclean:
	$(MAKE) -C hw clean
	rm -rf $(BUILD_SW)


