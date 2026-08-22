`timescale 1ns / 1ps

module imem #(
    parameter int DEPTH  = npu_pkg::IMEM_DEPTH,
    parameter int ADDR_W = npu_pkg::IMEM_ADDR_W
) (
    // Host write and read here (clk_host domain)
    input  logic              clk_host,
    input  logic              we,
    input  logic [ADDR_W+2:0] waddr,     // byte address
    input  logic [      63:0] wdata,
    input  logic              re,
    input  logic [ADDR_W+2:0] raddr,     // byte address
    output logic [      63:0] rdata,

    // Sequencer read from here (clk_seq domain)
    input  logic              clk_seq,
    input  logic              seq_re,
    input  logic [ADDR_W+2:0] seq_raddr,  // byte address
    output logic [      63:0] seq_rdata
);

  // Host read-back: same clock as host write.
  bram_sdp #(
      .DATA_WIDTH(64),
      .DEPTH(DEPTH),
      .ADDR_W(ADDR_W)
  ) host_bram (
      .clk_a  (clk_host),
      .we_a   (we),
      .addr_a (waddr[ADDR_W+2:3]),
      .wdata_a(wdata),
      .clk_b  (clk_host),
      .re_b   (re),
      .addr_b (raddr[ADDR_W+2:3]),
      .rdata_b(rdata)
  );

  // Sequencer read: written on clk_host, read on clk_seq.
  bram_sdp #(
      .DATA_WIDTH(64),
      .DEPTH(DEPTH),
      .ADDR_W(ADDR_W)
  ) seq_bram (
      .clk_a  (clk_host),
      .we_a   (we),
      .addr_a (waddr[ADDR_W+2:3]),
      .wdata_a(wdata),
      .clk_b  (clk_seq),
      .re_b   (seq_re),
      .addr_b (seq_raddr[ADDR_W+2:3]),
      .rdata_b(seq_rdata)
  );

  // Low 3 bits of the byte addresses are unused (8-byte word alignment).
  wire _unused = &{1'b0, waddr[2:0], raddr[2:0], seq_raddr[2:0], 1'b0};

endmodule
