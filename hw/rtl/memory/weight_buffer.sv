`timescale 1ns / 1ps

module weight_buffer (
    // DMA write to here (clk_ddr domain)
    input logic                 clk_ddr,
    input logic                 dma_we,
    input npu_pkg::bram_addr_t  dma_waddr,
    input npu_pkg::weight_vec_t dma_wdata,

    // Syustolic array read from here (clk_sa domain)
    input  logic                 clk_sa,
    input  logic                 sa_re,
    input  npu_pkg::bram_addr_t  sa_raddr,
    output npu_pkg::weight_vec_t sa_rdata,

    // Bank select (clk_ddr domain, synced internally to clk_sa)
    // weight_bank_sel=0: SA reads bank0, DMA loads bank1
    // weight_bank_sel=1: SA reads bank1, DMA loads bank0
    input logic weight_bank_sel
);

  npu_pkg::weight_vec_t rdata0;
  npu_pkg::weight_vec_t rdata1;

  // cross from clk_ddr to clk_sa
  logic weight_bank_sel_sync;
  logic weight_bank_sel_sa;

  always_ff @(posedge clk_sa) begin
    weight_bank_sel_sync <= weight_bank_sel;
    weight_bank_sel_sa   <= weight_bank_sel_sync;
  end

  generate
    for (genvar i = 0; i < npu_pkg::ARRAY_SIZE; i++) begin : gen_col

      bram_sdp #(
          .DATA_WIDTH(npu_pkg::DTYPE_WEIGHT_W)
      ) bank0 (
          .clk_a  (clk_ddr),
          .we_a   (dma_we & (weight_bank_sel == 1'b1)),
          .addr_a (dma_waddr),
          .wdata_a(dma_wdata[i]),
          .clk_b  (clk_sa),
          .re_b   (sa_re & (weight_bank_sel_sa == 1'b0)),
          .addr_b (sa_raddr),
          .rdata_b(rdata0[i])
      );

      bram_sdp #(
          .DATA_WIDTH(npu_pkg::DTYPE_WEIGHT_W)
      ) bank1 (
          .clk_a  (clk_ddr),
          .we_a   (dma_we & (weight_bank_sel == 1'b0)),
          .addr_a (dma_waddr),
          .wdata_a(dma_wdata[i]),
          .clk_b  (clk_sa),
          .re_b   (sa_re & (weight_bank_sel_sa == 1'b1)),
          .addr_b (sa_raddr),
          .rdata_b(rdata1[i])
      );

      assign sa_rdata[i] = weight_bank_sel_sa ? rdata1[i] : rdata0[i];

    end
  endgenerate

endmodule
