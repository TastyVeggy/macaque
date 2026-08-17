`timescale 1ns / 1ps

module quant_buffer (
    // Activate unit writes here (clk_sa domain)
    input logic                clk_sa,
    input logic                sa_we,
    input npu_pkg::bram_addr_t sa_waddr,
    input npu_pkg::act_vec_t   sa_wdata,

    // DMA reads from here (clk_ddr domain).
    input  logic                clk_ddr,
    input  logic                dma_re,
    input  npu_pkg::bram_addr_t dma_raddr,
    output npu_pkg::act_vec_t   dma_rdata,

    // Bank select (clk_sa domain, synced internally to clk_ddr)
    // quant_bank_sel=0: Activate unit writes bank0, DMA stores bank1
    // quant_bank_sel=1: activate unit writes bank1, DMA stores bank0
    input logic quant_bank_sel
);

  npu_pkg::act_vec_t rdata0;
  npu_pkg::act_vec_t rdata1;

  // cross from clk_sa to clk_ddr
  logic quant_bank_sel_sync;
  logic quant_bank_sel_ddr;

  logic quant_bank_sel_mux;

  always_ff @(posedge clk_ddr) begin
    quant_bank_sel_sync <= quant_bank_sel;
    quant_bank_sel_ddr  <= quant_bank_sel_sync;
    quant_bank_sel_mux  <= quant_bank_sel_ddr;
  end

  generate
    for (genvar i = 0; i < npu_pkg::ARRAY_SIZE; i++) begin : gen_col

      bram_sdp #(
          .DATA_WIDTH(npu_pkg::DTYPE_ACT_W)
      ) bank0 (
          .clk_a  (clk_sa),
          .we_a   (sa_we & (quant_bank_sel == 1'b0)),
          .addr_a (sa_waddr),
          .wdata_a(sa_wdata[i]),
          .clk_b  (clk_ddr),
          .re_b   (dma_re & (quant_bank_sel_ddr == 1'b1)),
          .addr_b (dma_raddr),
          .rdata_b(rdata0[i])
      );

      bram_sdp #(
          .DATA_WIDTH(npu_pkg::DTYPE_ACT_W)
      ) bank1 (
          .clk_a  (clk_sa),
          .we_a   (sa_we & (quant_bank_sel == 1'b1)),
          .addr_a (sa_waddr),
          .wdata_a(sa_wdata[i]),
          .clk_b  (clk_ddr),
          .re_b   (dma_re & (quant_bank_sel_ddr == 1'b0)),
          .addr_b (dma_raddr),
          .rdata_b(rdata1[i])
      );

      assign dma_rdata[i] = quant_bank_sel_mux ? rdata0[i] : rdata1[i];

    end
  endgenerate

endmodule
