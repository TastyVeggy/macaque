`timescale 1ns / 1ps
module ctrl_regs #(
    parameter int AW = 8  // register byte-offset width
) (
    // NPU control/status side
    output logic npu_start,  // 1-cycle pulse (write 1 to CTRL[0])
    output logic npu_reset,  // level (CTRL[1])

    input logic npu_busy,
    input logic npu_done,
    input logic npu_error,
    input logic npu_ready,

    output logic [31:0] instr_addr,
    output logic [31:0] instr_len,

    output logic pmu_enable,  // level (PMU_CTRL[0])
    output logic pmu_clear,   // 1-cycle pulse (write 1 to PMU_CTRL[1])

    input logic [63:0] pmu_cycles,
    input logic [31:0] pmu_compute,
    input logic [31:0] pmu_stall,
    input logic [31:0] pmu_dma_bytes_rd,
    input logic [31:0] pmu_dma_bytes_wr,

    // Host register bus
    input  logic          clk,
    input  logic          rst,        // active-high
    input  logic [AW-1:0] reg_addr,   // byte offset within the register region
    input  logic          reg_we,     // write strobe (1 cycle)
    input  logic [  63:0] reg_wdata,
    output logic [  63:0] reg_rdata   // combinational read mux
);

  logic [31:0] reg_ctrl;
  logic [31:0] reg_instr_addr;
  logic [31:0] reg_instr_len;
  logic [31:0] reg_pmu_ctrl;

  // Write path
  always_ff @(posedge clk) begin
    if (rst) begin
      reg_ctrl       <= '0;
      reg_instr_addr <= '0;
      reg_instr_len  <= '0;
      reg_pmu_ctrl   <= '0;
    end else begin
      reg_ctrl[0]     <= 1'b0;  // start: write-1-to-clear
      reg_pmu_ctrl[1] <= 1'b0;  // clear: write-1-to-clear

      if (reg_we) begin
        unique case (reg_addr)
          npu_pkg::REG_CTRL:       reg_ctrl <= reg_wdata[31:0];
          npu_pkg::REG_INSTR_ADDR: reg_instr_addr <= reg_wdata[31:0];
          npu_pkg::REG_INSTR_LEN:  reg_instr_len <= reg_wdata[31:0];
          npu_pkg::REG_PMU_CTRL:   reg_pmu_ctrl <= reg_wdata[31:0];
          default:                 ;  // read-only / reserved registers ignore writes
        endcase
      end
    end
  end

  // 1-cycle pulses for start / clear.
  logic npu_start_pulse;
  logic pmu_clear_pulse;
  always_ff @(posedge clk) begin
    if (rst) begin
      npu_start_pulse <= 1'b0;
      pmu_clear_pulse <= 1'b0;
    end else begin
      npu_start_pulse <= reg_we && (reg_addr == npu_pkg::REG_CTRL) && reg_wdata[0];
      pmu_clear_pulse <= reg_we && (reg_addr == npu_pkg::REG_PMU_CTRL) && reg_wdata[1];
    end
  end

  assign npu_start  = npu_start_pulse;
  assign pmu_clear  = pmu_clear_pulse;
  assign npu_reset  = reg_ctrl[1];
  assign instr_addr = reg_instr_addr;
  assign instr_len  = reg_instr_len;
  assign pmu_enable = reg_pmu_ctrl[0];

  // Read path
  always_comb begin
    unique case (reg_addr)
      npu_pkg::REG_CTRL:             reg_rdata = {32'b0, reg_ctrl};
      npu_pkg::REG_STATUS:           reg_rdata = {60'b0, npu_error, npu_done, npu_busy, npu_ready};
      npu_pkg::REG_INSTR_ADDR:       reg_rdata = {32'b0, reg_instr_addr};
      npu_pkg::REG_INSTR_LEN:        reg_rdata = {32'b0, reg_instr_len};
      npu_pkg::REG_PMU_CTRL:         reg_rdata = {32'b0, reg_pmu_ctrl};
      npu_pkg::REG_PMU_CYCLES:       reg_rdata = pmu_cycles;
      npu_pkg::REG_PMU_COMPUTE:      reg_rdata = {32'b0, pmu_compute};
      npu_pkg::REG_PMU_STALL:        reg_rdata = {32'b0, pmu_stall};
      npu_pkg::REG_PMU_DMA_BYTES_RD: reg_rdata = {32'b0, pmu_dma_bytes_rd};
      npu_pkg::REG_PMU_DMA_BYTES_WR: reg_rdata = {32'b0, pmu_dma_bytes_wr};
      default:                       reg_rdata = '0;
    endcase
  end

  // Upper 32 bits of the write word are reserved (registers are 32-bit).
  wire _unused = &{1'b0, reg_wdata[63:32], 1'b0};

endmodule
