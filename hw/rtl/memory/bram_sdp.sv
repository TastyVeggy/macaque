`timescale 1ns / 1ps
// Simple-dual-port BRAM: write on port A, read on port B (independent clocks).
module bram_sdp #(
`ifdef VERILATOR
    parameter int DATA_WIDTH = 8,  // for lsp lol
    parameter int DEPTH      = npu_pkg::BRAM_DEPTH,
    parameter int ADDR_W     = npu_pkg::DTYPE_BRAM_ADDR_W
`else
    parameter int DATA_WIDTH,
    parameter int DEPTH      = npu_pkg::BRAM_DEPTH,
    parameter int ADDR_W     = npu_pkg::DTYPE_BRAM_ADDR_W
`endif
) (
    input logic                          clk_a,
    input logic                          we_a,
    input logic           [ADDR_W-1:0]   addr_a,
    input logic [DATA_WIDTH-1:0]         wdata_a,

    input  logic                         clk_b,
    input  logic                         re_b,
    input  logic          [ADDR_W-1:0]   addr_b,
    output logic [DATA_WIDTH-1:0]        rdata_b
);

  (* ram_style = "block" *)
  logic [DATA_WIDTH-1:0] mem[DEPTH];

  always_ff @(posedge clk_a) begin
    if (we_a) mem[addr_a] <= wdata_a;
  end

  always_ff @(posedge clk_b) begin
    if (re_b) rdata_b <= mem[addr_b];
  end

endmodule
