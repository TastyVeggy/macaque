`timescale 1ns / 1ps

module out_buffer (
    input logic                clk_sa,
    input logic                sa_we,
    input npu_pkg::bram_addr_t sa_waddr,
    input npu_pkg::acc_vec_t   sa_wdata,

    input  logic                rd_re,
    input  npu_pkg::bram_addr_t rd_raddr,
    output npu_pkg::acc_vec_t   rd_rdata,

    // Bank select (clk_sa domain)
    // out_bank_sel=0: array drains bank0, ACTIVATE reads bank1
    // out_bank_sel=1: array drains bank1, ACTIVATE reads bank0
    input logic out_bank_sel
);

  npu_pkg::acc_vec_t rdata0;
  npu_pkg::acc_vec_t rdata1;

  generate
    for (genvar i = 0; i < npu_pkg::ARRAY_SIZE; i++) begin : gen_col

      bram_sdp #(
          .DATA_WIDTH(npu_pkg::DTYPE_ACC_W)
      ) bank0 (
          .clk_a  (clk_sa),
          .we_a   (sa_we & (out_bank_sel == 1'b0)),
          .addr_a (sa_waddr),
          .wdata_a(sa_wdata[i]),
          .clk_b  (clk_sa),
          .re_b   (rd_re & (out_bank_sel == 1'b1)),
          .addr_b (rd_raddr),
          .rdata_b(rdata0[i])
      );

      bram_sdp #(
          .DATA_WIDTH(npu_pkg::DTYPE_ACC_W)
      ) bank1 (
          .clk_a  (clk_sa),
          .we_a   (sa_we & (out_bank_sel == 1'b1)),
          .addr_a (sa_waddr),
          .wdata_a(sa_wdata[i]),
          .clk_b  (clk_sa),
          .re_b   (rd_re & (out_bank_sel == 1'b0)),
          .addr_b (rd_raddr),
          .rdata_b(rdata1[i])
      );

      assign rd_rdata[i] = out_bank_sel ? rdata0[i] : rdata1[i];

    end
  endgenerate

endmodule
