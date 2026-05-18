# Macaque Hardware Simulation

This directory contains the verification environment for the Macaque hardware. The scripts support Xilinx XSim (SystemVerilog) and Cocotb (Python) simulation.

## Directory Structure & Naming
To add a new test, create a subfolder in hw/sim/tb/ following this specification:
* Folder Name: \<name\>
* XSim Testbench: \<name\>_tb.sv
* Cocotb Testbench: \<name\>_tb.py
* List of sources: sources.f (not inclusive of the testbench itself)
* Optional testbench config: tb.cfg 
    * TOP: dut for cocotb (default to `TB` passed in make command)
    * SIM: simulator for cocotb (default to verilator)
```
hw/sim/tb/
└── <name>/
    ├── <name>_tb.sv    # Required for MODE=xsim
    ├── <name>_tb.py    # Required for MODE=cocotb
    ├── sources.f       # include the sources you need (relative to hw/rtl)
    └── tb.cfg          # config for cocotb
```
## Running Simulations

The simulation environment automatically detects available tests based on the file extensions present in the tb/ subfolders.

Run make without arguments to see a menu of detected testbenches and their supported modes:
```bash
make
```

### Running a specific simulation
Use the `TB` variable to specify the folder name and `MODE` to choose the engine.

SystemVerilog (XSim):
```bash
make TB=systolic_array MODE=xsim
```

Python (Cocotb):
```bash
make TB=systolic_array MODE=cocotb
```

## Environment Setup
The simulation scripts rely on Xilinx Vivado tools.
* Recommended: source your Vivado settings64.sh before running. This sets the `XILINX_VIVADO` enviornment variable automatically. You will likely need it to run vivado anyways.
* Fallback: If the variable is not found, the script defaults to:
    `/opt/Xilinx/Vivado/2025.2` (as can be found in `run_sim.sh`)

Cocotb require Python and cocotb

## Cleanup
```bash
make clean
```

