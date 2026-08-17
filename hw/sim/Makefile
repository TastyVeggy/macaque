ROOT_DIR  ?= $(shell cd ../.. && pwd)
TB_DIR    := $(ROOT_DIR)/hw/sim/tb
RTL_DIR   := $(ROOT_DIR)/hw/rtl
WORK_DIR  := $(ROOT_DIR)/hw/sim/work

ALL_FOLDERS := $(notdir $(shell find $(TB_DIR) -maxdepth 1 -mindepth 1 -type d))

XSIM_TBS   := $(foreach tb,$(ALL_FOLDERS),$(if $(wildcard $(TB_DIR)/$(tb)/*.sv),$(tb)))
COCO_TBS   := $(foreach tb,$(ALL_FOLDERS),$(if $(wildcard $(TB_DIR)/$(tb)/*.py),$(tb)))

.PHONY: all run run_all run_coco run_xsim list clean

all: run_all

sim_hw:
	@if [ -z "$(TB)" ]; then \
		$(MAKE) help; \
	else \
		$(MAKE) execute_sim; \
	fi

define run_summary
	echo ""; \
	echo "=== HW test summary ==="; \
	cat .run_all_results; \
	echo "================================="; \
	! grep -qE "FAIL=[1-9]|FAIL [1-9]|no Result line|rc=[1-9]" .run_all_results; \
	rm -f .run_all_results
endef

run_coco:
	@rm -f .run_all_results; \
	echo "=== Cocotb (Python) suites ==="; \
	for tb in $(COCO_TBS); do \
		out=$$($(MAKE) -s sim_hw TB=$$tb MODE=cocotb 2>&1); \
		summary=$$(printf '%s\n' "$$out" | grep -oE "TESTS=[0-9]+ PASS=[0-9]+ FAIL=[0-9]+ SKIP=[0-9]+" | tail -1); \
		if [ -z "$$summary" ]; then summary="TESTS=? PASS=? FAIL=? SKIP=?"; fi; \
		printf '%-18s %s\n' "[cocotb] $$tb" "$$summary" >> .run_all_results; \
		printf '%s' "$$out" > /tmp/run_all_coco_$$tb.log; \
	done; \
	$(run_summary)

run_xsim:
	@rm -f .run_all_results; \
	echo "=== XSIM (SystemVerilog) suites ==="; \
	for tb in $(XSIM_TBS); do \
		out=$$(./run_sim.sh $$tb 2>&1); rc=$$?; \
		res=$$(printf '%s\n' "$$out" | grep -oE "Result: *[0-9]+ PASS +[0-9]+ FAIL" | tail -1); \
		if [ -z "$$res" ]; then res="(no Result line, rc=$$rc)"; fi; \
		printf '%-18s %s\n' "[xsim] $$tb" "$$res" >> .run_all_results; \
		printf '%s' "$$out" > /tmp/run_all_xsim_$$tb.log; \
	done; \
	$(run_summary)

run_all:
	@$(MAKE) -s run_coco; \
	rc=$$?; \
	$(MAKE) -s run_xsim; \
	rc=$$(( rc + $$? )); \
	exit $$rc

list:
	@echo "Cocotb TBs: $(COCO_TBS)"
	@echo "XSIM TBs:   $(XSIM_TBS)"


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
