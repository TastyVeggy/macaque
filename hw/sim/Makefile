ROOT_DIR  ?= $(shell cd ../.. && pwd)
TB_DIR    := $(ROOT_DIR)/hw/sim/tb
RTL_DIR   := $(ROOT_DIR)/hw/rtl
WORK_DIR  := $(ROOT_DIR)/hw/sim/work

ALL_FOLDERS := $(notdir $(shell find $(TB_DIR) -maxdepth 1 -mindepth 1 -type d))

XSIM_TBS   := $(foreach tb,$(ALL_FOLDERS),$(if $(wildcard $(TB_DIR)/$(tb)/*.sv),$(tb)))
COCO_TBS   := $(foreach tb,$(ALL_FOLDERS),$(if $(wildcard $(TB_DIR)/$(tb)/*.py),$(tb)))

.PHONY: all run list clean

all: sim_hw

sim_hw:
	@if [ -z "$(TB)" ]; then \
		$(MAKE) help; \
	else \
		$(MAKE) execute_sim; \
	fi

help:
	@echo "Available XSIM (SystemVerilog) TBs:"
	@$(foreach tb,$(XSIM_TBS),echo "  - make sim_hw TB=$(tb) MODE=xsim";)
	@echo ""
	@echo "Available Cocotb (Python) TBs:"
	@$(foreach tb,$(COCO_TBS),echo "  - make sim_hw TB=$(tb) MODE=cocotb";)


execute_sim:
ifeq ($(MODE),xsim)
	@if echo "$(XSIM_TBS)" | grep -qw "$(TB)"; then \
		./run_sim.sh $(TB); \
	else \
		echo "ERROR: No .sv file found in $(TB_DIR)/$(TB)" ; exit 1; \
	fi
else ifeq ($(MODE),cocotb)
	@if echo "$(COCO_TBS)" | grep -qw "$(TB)"; then \
		$(MAKE) -f Makefile.cocotb RTL_DIR=$(RTL_DIR) TB_DIR=$(TB_DIR) TB=$(TB); \
	else \
		echo "ERROR: No .py file found in $(TB_DIR)/$(TB)"; exit 1; \
	fi
endif

clean:
	rm -rf $(WORK_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf *.log *.jou *.pb xsim.dir
