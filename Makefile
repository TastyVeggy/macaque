export ROOT_DIR := $(shell pwd)
BUILD_SW        := sw/build/debug
TOP             ?= npu_top
PORT            ?= /dev/ttyUSB0
PROJECT := macaque
PRESET ?= debug

.PHONY: hw sim-hw sw test-sw test-hw clean distclean

help:
	@echo "Targets:"
	@echo "  hw            - Build the hardware (Synthesis, Impl, Bitstream)"
	@echo "  hw TOP=x      - Build the hardware with specified top module"
	@echo "  test-hw TOP=x - Run the on-board host test for TOP (you should already build the hardware with the specific top module via hw TOP=x)"
	@echo "  sim-hw        - Run all simulated hardware tests"
	@echo "  sw            - Build the software stack"
	@echo "  test-sw       - Build and run sw unit tests"
	@echo "  clean         - Fast clean (keep compiled dependencies)"
	@echo "  distclean     - Full clean"

hw: 
	$(MAKE) -C hw TOP=$(TOP)

test-hw:
	$(MAKE) -C hw run-test TOP=$(TOP) PORT=$(PORT)

sim-hw:
	$(MAKE) -C hw/sim 

sw:
	$(MAKE) -C sw PRESET=$(PRESET)

test-sw:
	$(MAKE) -C sw test PRESET=$(PRESET)

clean:
	$(MAKE) -C hw clean 
	$(MAKE) -C sw clean

distclean: clean
	$(MAKE) -C sw distclean


