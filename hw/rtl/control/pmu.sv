`timescale 1ns / 1ps

module pmu (
    input logic clk,
    input logic rst,

    input logic enable,
    input logic clear,

    // Event inputs
    input logic run_active,
    input logic mac_active,
    input logic stall,
    input logic [31:0] dma_bytes_rd_this_cycle,
    input logic [31:0] dma_bytes_wr_this_cycle,

    // Counter outputs
    output logic [63:0] pmu_cycles,
    output logic [31:0] pmu_compute,
    output logic [31:0] pmu_stall,
    output logic [31:0] pmu_dma_bytes_rd,
    output logic [31:0] pmu_dma_bytes_wr,

    input logic frozen
);

  always_ff @(posedge clk) begin
    if (rst || clear) begin
      pmu_cycles       <= '0;
      pmu_compute      <= '0;
      pmu_stall        <= '0;
      pmu_dma_bytes_rd <= '0;
      pmu_dma_bytes_wr <= '0;
    end else if (enable && run_active && !frozen) begin
      pmu_cycles <= pmu_cycles + 1;
      if (mac_active) pmu_compute <= pmu_compute + 1;
      if (stall) pmu_stall <= pmu_stall + 1;
      pmu_dma_bytes_rd <= pmu_dma_bytes_rd + dma_bytes_rd_this_cycle;
      pmu_dma_bytes_wr <= pmu_dma_bytes_wr + dma_bytes_wr_this_cycle;
    end
  end

endmodule
