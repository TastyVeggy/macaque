`timescale 1ns / 1ps
import npu_pkg::*;

module mac_top #(
    parameter int CLK_FREQ  = npu_pkg::SYS_CLK_FREQ,
    parameter int BAUD_RATE = npu_pkg::UART_BAUD_DEFAULT
) (
    input  logic clk,
    input  logic rst_n,
    input  logic rx_pin,
    output logic tx_pin
);

  logic rst;
  reset_bridge reset_bridge_inst (
      .clk     (clk),
      .rst_n   (rst_n),
      .rst_high(rst)
  );

  logic [7:0] rx_data;
  logic       rx_valid;
  logic [7:0] tx_data;
  logic       tx_valid;
  logic       tx_ready;

  uart_rx #(
      .CLK_FREQ (CLK_FREQ),
      .BAUD_RATE(BAUD_RATE)
  ) rx_inst (
      .clk     (clk),
      .rst     (rst),
      .rx_pin  (rx_pin),
      .rx_data (rx_data),
      .rx_valid(rx_valid)
  );

  uart_tx #(
      .CLK_FREQ (CLK_FREQ),
      .BAUD_RATE(BAUD_RATE)
  ) tx_inst (
      .clk     (clk),
      .rst     (rst),
      .tx_data (tx_data),
      .tx_valid(tx_valid),
      .tx_ready(tx_ready),
      .tx_pin  (tx_pin)
  );

  int8_t  weight_in;
  logic   weight_load;
  int8_t  act_in;
  int32_t acc_in;
  logic   input_valid;
  int32_t acc_out;
  logic   output_valid_unused;
  int8_t  act_out_unused;

  mac_unit mac_inst (
      .clk(clk),
      .rst(rst),

      .weight_in      (weight_in),
      .weight_load    (weight_load),
      .weight_out     (),
      .weight_load_out(),

      .input_valid (input_valid),
      .output_valid(output_valid_unused),

      .act_in (act_in),
      .act_out(act_out_unused),

      .acc_in (acc_in),
      .acc_out(acc_out)
  );

  // ------------------------------------------------------------
  // Protocol:
  // 0x01 <weight> load weight
  // 0x02 <act> <acc0><acc1><acc2><acc3> send one byte of activation then
  // one byte of accumulation
  //   0x03        read result (FPGA replies 4 bytes, LSB first)

  typedef enum logic [2:0] {
    IDLE,
    LOAD_WEIGHT,
    RECV_ACT,
    RECV_ACC,
    TX_LOAD,
    TX_BUSY
  } state_t;

  state_t state;

  logic [1:0] rx_cnt;
  logic [1:0] tx_cnt;

  logic [31:0] acc_buf;
  logic [31:0] acc_latched;

  always_ff @(posedge clk) begin
    weight_load <= 1'b0;
    input_valid <= 1'b0;
    tx_valid    <= 1'b0;

    if (rst) begin
      state       <= IDLE;
      rx_cnt      <= '0;
      tx_cnt      <= '0;
      weight_in   <= '0;
      act_in      <= '0;
      acc_in      <= '0;
      acc_buf     <= '0;
      acc_latched <= '0;
      tx_data     <= '0;
    end else begin
      case (state)
        IDLE:
        if (rx_valid) begin
          case (rx_data)
            8'h01: state <= LOAD_WEIGHT;

            8'h02: state <= RECV_ACT;

            8'h03: begin
              acc_latched <= acc_out;
              tx_cnt      <= 0;
              state       <= TX_LOAD;
            end
            default: state <= IDLE;
          endcase
        end

        LOAD_WEIGHT: begin
          if (rx_valid) begin
            weight_in   <= int8_t'(rx_data);
            weight_load <= 1'b1;
            state       <= IDLE;
          end
        end

        RECV_ACT: begin
          if (rx_valid) begin
            act_in <= int8_t'(rx_data);
            rx_cnt <= 0;
            state  <= RECV_ACC;
          end
        end

        RECV_ACC: begin
          if (rx_valid) begin

            acc_buf[rx_cnt*8+:8] <= rx_data;

            if (rx_cnt == 2'd3) begin
              acc_in      <= int32_t'({rx_data, acc_buf[23:16], acc_buf[15:8], acc_buf[7:0]});
              input_valid <= 1'b1;  // one-cycle pulse
              state       <= IDLE;
            end else begin
              rx_cnt <= rx_cnt + 1'b1;
            end

          end
        end

        TX_LOAD: begin
          if (tx_ready) begin
            tx_data  <= acc_latched[tx_cnt*8 +: 8];
            tx_valid <= 1'b1;
            state    <= TX_BUSY;
          end
        end

        TX_BUSY: begin
          if (!tx_ready) begin
            if (tx_cnt == 2'd3) begin
              state <= IDLE;  // all 4 bytes sent
            end else begin
              tx_cnt <= tx_cnt + 1'b1;
              state  <= TX_LOAD;  // send next byte
            end
          end
        end

        default: state <= IDLE;
      endcase
    end
  end

endmodule
