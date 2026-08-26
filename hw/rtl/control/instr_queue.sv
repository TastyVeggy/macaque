`timescale 1ns / 1ps

module instr_queue #(
    parameter int DEPTH = 8
) (
    input logic clk,
    input logic rst,

    input  logic        push,
    input  logic [63:0] push_data,
    output logic        full,

    input  logic        pop,
    output logic [63:0] pop_data,
    output logic        empty
);

  localparam int AW = $clog2(DEPTH);
  localparam int CW = $clog2(DEPTH + 1);

  logic [CW-1:0] head;
  logic [CW-1:0] tail;
  logic [63:0] mem[DEPTH];

  logic [CW-1:0] count;
  assign count    = tail - head;
  assign full     = (count == CW'(DEPTH));
  assign empty    = (count == 0);
  assign pop_data = mem[head[AW-1:0]];

  always_ff @(posedge clk) begin
    if (rst) begin
      head <= '0;
      tail <= '0;
    end else begin
      if (push && !full) begin
        mem[tail[AW-1:0]] <= push_data;
        tail <= tail + 1'b1;
      end
      if (pop && !empty) begin
        head <= head + 1'b1;
      end
    end
  end

endmodule
