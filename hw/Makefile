ROOT_DIR ?= $(shell cd .. && pwd)
TOP      ?= npu_top
VIV_BIN    := vivado -mode batch -notrace
SCRIPT_DIR := scripts
BUILD_DIR  := build
RTL_DIR    := rtl
SIM_DIR    := sim
CONSTRS    := constrs
TOP_DIR    := $(RTL_DIR)/top
PYTHON     ?= python 			# python binary path
PORT       ?= /dev/ttyUSB0

# On-board host test: by convention hw/test/<TOP>/test.py (override HW_TEST).
HW_TEST    ?= test/$(TOP)/test.py

VALID_TOPS  := $(basename $(notdir $(shell find $(TOP_DIR) -name "*.sv")))
SYNTH_STAMP := $(BUILD_DIR)/.synth_$(TOP)
IMPL_STAMP  := $(BUILD_DIR)/.impl_$(TOP)
BIT_STAMP   := $(BUILD_DIR)/.bit_$(TOP)
PROGRAM_STAMP := $(BUILD_DIR)/.program_$(TOP)

define check_top
	@if ! echo "$(VALID_TOPS)" | tr ' ' '\n' | grep -qx "$(TOP)"; then \
		echo ""; \
		echo "ERROR: '$(TOP)' is not a valid top module."; \
		echo ""; \
		echo "Available top modules:"; \
		echo "$(VALID_TOPS)" | tr ' ' '\n' | sort | sed 's/^/  - /'; \
		echo ""; \
		echo "Usage: make hw TOP=<module>"; \
		echo ""; \
		exit 1; \
	fi
endef

SRCS := $(shell find $(RTL_DIR) -name "*.sv" -o -name "*.v")
XDC  := $(shell find $(CONSTRS) -name "*.xdc")
IPDEPS := $(shell find $(SCRIPT_DIR)/ip -type f) $(shell find ip -name "*.prj")
DEPS := $(SRCS) $(XDC) $(IPDEPS) $(SCRIPT_DIR)/top_config.tcl $(SCRIPT_DIR)/project.tcl

.PHONY: all build synth impl bitstream program run_test project validate sim tidy clean

all: program

$(SYNTH_STAMP): $(DEPS) | validate
	@mkdir -p $(BUILD_DIR)
	@rm -f $(IMPL_STAMP) $(BIT_STAMP) $(PROGRAM_STAMP)
	$(VIV_BIN) -source $(SCRIPT_DIR)/synth.tcl -tclargs $(ROOT_DIR) $(TOP)
	@touch $(SYNTH_STAMP)

$(IMPL_STAMP): $(SYNTH_STAMP) | validate
	@mkdir -p $(BUILD_DIR)
	@rm -f $(BIT_STAMP) $(PROGRAM_STAMP)
	$(VIV_BIN) -source $(SCRIPT_DIR)/impl.tcl -tclargs $(ROOT_DIR) $(TOP)
	@touch $(IMPL_STAMP)

$(BIT_STAMP): $(IMPL_STAMP) | validate
	@mkdir -p $(BUILD_DIR)
	@rm -f $(PROGRAM_STAMP)
	$(VIV_BIN) -source $(SCRIPT_DIR)/bitstream.tcl -tclargs $(ROOT_DIR) $(TOP)
	@touch $(BIT_STAMP)

$(PROGRAM_STAMP): $(BIT_STAMP) | validate
	@mkdir -p $(BUILD_DIR)
	@rm -f $(BUILD_DIR)/.program_*
	$(VIV_BIN) -source $(SCRIPT_DIR)/program.tcl -tclargs $(ROOT_DIR) $(TOP)
	@touch $(PROGRAM_STAMP)

build: $(BIT_STAMP)
synth: $(SYNTH_STAMP)
impl: $(IMPL_STAMP)
bitstream: $(BIT_STAMP)

program: $(PROGRAM_STAMP)

run-test: | validate
	@test -f "$(HW_TEST)" || { \
		echo ""; echo "ERROR: host script not found: $(HW_TEST)"; \
		echo "Set HW_TEST=path (relative to hw/) or PORT=..."; exit 1; }; \
	python3 $(HW_TEST) $(PORT)

project:
	$(VIV_BIN) -source $(SCRIPT_DIR)/project.tcl -tclargs $(ROOT_DIR) $(TOP)

sim:
	$(MAKE) -C sim 

validate:
	$(call check_top)

tidy:
	rm -rf .Xil *.log *.jou clockInfo.txt vivado_pid*.str xvlog.pb

clean: tidy
	rm -rf $(BUILD_DIR)
	$(MAKE) -C sim clean
