`timescale 1ns / 1ps

module npu_top (
    input  logic clk,
    output logic led
);

  localparam int HalfPeriod = 25_000_000;

  logic [24:0] counter;

  logic led_reg = 1'b1;

  always_ff @(posedge clk) begin
    if (counter == HalfPeriod - 1) begin
      counter <= '0;
      led_reg <= ~led_reg;
    end else begin
      counter <= counter + 1'b1;
    end
  end

  assign led = led_reg;

endmodule

