`timescale 1ns / 1ps
module bram_sdp #(
`ifdef VERILATOR
    parameter int DATA_WIDTH = 8  // for lsp lol
`else
    parameter int DATA_WIDTH
`endif
) (
    input logic                                 clk_a,
    input logic                                 we_a,
    input npu_pkg::bram_addr_t                  addr_a,
    input logic                [DATA_WIDTH-1:0] wdata_a,

    input  logic                                 clk_b,
    input  logic                                 re_b,
    input  npu_pkg::bram_addr_t                  addr_b,
    output logic                [DATA_WIDTH-1:0] rdata_b
);

  (* ram_style = "block" *)
  logic [DATA_WIDTH-1:0] mem[npu_pkg::BRAM_DEPTH];

  always_ff @(posedge clk_a) begin
    if (we_a) mem[addr_a] <= wdata_a;
  end

  always_ff @(posedge clk_b) begin
    if (re_b) rdata_b <= mem[addr_b];
  end

endmodule
