ifeq ($(ROOT_DIR),)
    ROOT_DIR := $(shell cd .. && pwd)
endif
export ROOT_DIR

VIV_BIN    := vivado -mode batch -notrace
SCRIPT_DIR := scripts
BUILD_DIR  := build
RTL_DIR    := rtl
SIM_DIR    := sim
CONSTRS    := constrs

all: program

# for dependency tracking
SRCS := $(shell find $(RTL_DIR) -name "*.sv" -o -name "*.v")
XDC  := $(shell find $(CONSTRS) -name "*.xdc")

# synthesis run if src files or constraints changed
$(BUILD_DIR)/post_synth.dcp: $(SRCS) $(XDC)
	@mkdir -p $(BUILD_DIR)
	$(VIV_BIN) -source $(SCRIPT_DIR)/synth.tcl -tclargs $(ROOT_DIR)

# run implementation if synth output changed
$(BUILD_DIR)/post_route.dcp: $(BUILD_DIR)/post_synth.dcp
	$(VIV_BIN) -source $(SCRIPT_DIR)/impl.tcl -tclargs $(ROOT_DIR)

# generate bitstream if post_route changed
$(BUILD_DIR)/$(proj).bit: $(BUILD_DIR)/post_route.dcp
	$(VIV_BIN) -source $(SCRIPT_DIR)/bitstream.tcl -tclargs $(ROOT_DIR)

synth: $(BUILD_DIR)/post_synth.dcp

impl: $(BUILD_DIR)/post_route.dcp

bitstream: $(BUILD_DIR)/$(proj).bit

project:
	$(VIV_BIN) -source $(SCRIPT_DIR)/project.tcl -tclargs $(ROOT_DIR)

program: bitstream
	$(VIV_BIN) -source $(SCRIPT_DIR)/program.tcl -tclargs $(ROOT_DIR)

tidy: 
	rm -rf .Xil *.log *.jou clockInfo.txt

clean: tidy
	rm -rf $(BUILD_DIR) $(SIM_DIR)/tb/sim_build
