`timescale 1ns / 1ps
module uart_top #(
    parameter int CLK_FREQ  = npu_pkg::SYS_CLK_FREQ,
    parameter int BAUD_RATE = npu_pkg::UART_BAUD_DEFAULT
) (
    input  logic clk,
    input  logic rst_n,
    input  logic uart_rx,
    output logic uart_tx
);
  logic [7:0] rx_data;
  logic       rx_valid;
  logic       tx_ready;

  logic       rst;

  reset_bridge reset_bridge_inst (
      .clk(clk),
      .rst_n(rst_n),
      .rst_high(rst)
  );

  uart_rx #(
      .CLK_FREQ (CLK_FREQ),
      .BAUD_RATE(BAUD_RATE)
  ) rx_inst (
      .clk     (clk),
      .rst     (rst),
      .rx_pin  (uart_rx),
      .rx_data (rx_data),
      .rx_valid(rx_valid)
  );

  uart_tx #(
      .CLK_FREQ (CLK_FREQ),
      .BAUD_RATE(BAUD_RATE)
  ) tx_inst (
      .clk     (clk),
      .rst     (rst),
      .tx_data (rx_data),
      .tx_valid(rx_valid),
      .tx_ready(tx_ready),
      .tx_pin  (uart_tx)
  );

endmodule
