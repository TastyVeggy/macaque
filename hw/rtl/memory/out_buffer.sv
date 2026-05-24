`timescale 1ns / 1ps

module out_buffer (
    // Syustolic array write from here (clk_sa domain)
    input logic                clk_sa,
    input logic                sa_we,
    input npu_pkg::bram_addr_t sa_waddr,
    input npu_pkg::acc_vec_t   sa_wdata,

    // DMA read to here (clk_ddr domain)
    input  logic                clk_ddr,
    input  logic                dma_re,
    input  npu_pkg::bram_addr_t dma_raddr,
    output npu_pkg::acc_vec_t   dma_rdata,

    // Bank select (clk_sa domain, synced internally to clk_ddr)
    // out_bank_sel=0: PE writes bank0, DMA flushes bank1
    // out_bank_sel=1: PE writes bank1, DMA flushes bank0
    input logic out_bank_sel
);

  npu_pkg::acc_vec_t rdata0;
  npu_pkg::acc_vec_t rdata1;

  // cross from clk_sa to clk_ddr
  logic out_bank_sel_sync;
  logic out_bank_sel_ddr;

  // BRAM read latency
  logic out_bank_sel_mux;

  always_ff @(posedge clk_ddr) begin
    out_bank_sel_sync <= out_bank_sel;
    out_bank_sel_ddr  <= out_bank_sel_sync;
    out_bank_sel_mux  <= out_bank_sel_ddr;
  end

  generate
    for (genvar i = 0; i < npu_pkg::ARRAY_SIZE; i++) begin : gen_col

      bram_sdp #(
          .DATA_WIDTH(npu_pkg::DTYPE_ACC_W)
      ) bank0 (
          .clk_a  (clk_sa),
          .we_a   (sa_we & (out_bank_sel == 1'b0)),
          .addr_a (sa_waddr),
          .wdata_a(sa_wdata[i]),
          .clk_b  (clk_ddr),
          .re_b   (dma_re & (out_bank_sel_ddr == 1'b1)),
          .addr_b (dma_raddr),
          .rdata_b(rdata0[i])
      );

      bram_sdp #(
          .DATA_WIDTH(npu_pkg::DTYPE_ACC_W)
      ) bank1 (
          .clk_a  (clk_sa),
          .we_a   (sa_we & (out_bank_sel == 1'b1)),
          .addr_a (sa_waddr),
          .wdata_a(sa_wdata[i]),
          .clk_b  (clk_ddr),
          .re_b   (dma_re & (out_bank_sel_ddr == 1'b0)),
          .addr_b (dma_raddr),
          .rdata_b(rdata1[i])
      );

      assign dma_rdata[i] = out_bank_sel_mux ? rdata0[i] : rdata1[i];

    end
  endgenerate

endmodule
