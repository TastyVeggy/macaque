# Macaque

A vertically integrated custom AI accelerator and software toolchain stack. 

The design is implemented on the QMTECH Wukong V3 development board, which features a Xilinx Artix-7 XC7A100T FPGA.

---

## Hardware

| **Component** | **Part** |
|---|---|
| FPGA | Xilinx Artix-7 XC7A100T-1FGG676C |
| Board | QMTECH XC7A100T Wukong V3 |
| Clock | 50 MHz on-board crystal |
| DSP slices | 240 × DSP48E1 |
| Block RAM | 135 × BRAM36 (4,860 Kb) |
| DDR3 | 256 MB Micron MT41K128M16JT-125:K |
| Ethernet | Realtek RTL8211EG (1 Gbps) |
| UART | CH340N USB-UART bridge |

---

## Architecture

### NPU Core
A 14×14 weight-stationary systolic array computing INT8 matrix multiplication with INT32 accumulation.

---

## Build

### Prerequisites
- Vivado ML Edition v2025.2 (Artix-7 device support)
- CMake 3.28
- GCC or Clang with C++20 support

### Hardware
To do a full build and program the FPGA:
```bash
make hw
```

Alternatively, each individual stage can be carried out separately:
```bash
make -C hw synth
make -C hw impl
make -C hw bitstream
make -C hw program
```

#### Running testbenches
Refer to [README](hw/sim/README.md) in `hw/sim`.

### Software

To configure and build all C++ components
```bash
make sw
```

To run unit tests:
```bash
make test_sw
```

