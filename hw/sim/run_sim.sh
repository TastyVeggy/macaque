#!/bin/bash
set -e

VIVADO_BASE="${XILINX_VIVADO:-/opt/Xilinx/Vivado/2025.2}"
GLBL_PATH="$VIVADO_BASE/data/verilog/src/glbl.v"

TB_DIR=$1
SIM_DIR="$(cd "$(dirname "$0")" && pwd)"
HW_DIR="$(cd "$SIM_DIR/.." && pwd)"
RTL_DIR="$HW_DIR/rtl"
WORK_DIR="$SIM_DIR/work/$TB_DIR"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

ORIGINAL_FILELIST="$SIM_DIR/tb/$TB_DIR/sources.f"

RESOLVED_FILELIST="$WORK_DIR/resolved_sources.f"

sed "s|^[^#]|${RTL_DIR}/&|" "$ORIGINAL_FILELIST" > "$RESOLVED_FILELIST"

xvlog -sv \
    -d SYNTHESIS \
    -i "$SIM_DIR/../rtl/pkg" \
    -f "$RESOLVED_FILELIST" \
    $(find "$SIM_DIR/tb/$TB_DIR" -maxdepth 1 -name "*.sv" -type f) \
    $GLBL_PATH

xelab -debug typical \
    -timescale 1ns/1ps \
    -L work \
    -L unisims_ver \
    -L unimacro_ver \
    -L secureip \
    -s ${TB_DIR}_tb_sim \
    work.${TB_DIR}_tb \
    work.glbl

xsim ${TB_DIR}_tb_sim -runall
