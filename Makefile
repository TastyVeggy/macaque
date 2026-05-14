export ROOT_DIR := $(shell pwd)

.PHONY: hw clean help sim_uart

help:
	@echo "Available targets:"
	@echo "  make hw    - Build the hardware (Synthesis, Impl, Bitstream)"
	@echo "  make clean - Remove build artifacts"
	@echo "  make sim_uart - Test the uart"

hw: 
	$(MAKE) -C hw

sim_uart:
	$(MAKE) -C hw/sim/tb

clean:
	$(MAKE) -C hw clean
