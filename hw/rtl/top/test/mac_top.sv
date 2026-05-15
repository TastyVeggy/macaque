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
  logic   act_valid;
  logic   clear_acc;
  int32_t acc_out;
  int8_t  act_out_unused;
  logic   act_valid_out_unused;

  mac_unit mac_inst (
      .clk          (clk),
      .rst          (rst),
      .weight_in    (weight_in),
      .weight_load  (weight_load),
      .act_in       (act_in),
      .act_valid    (act_valid),
      .act_out      (act_out_unused),
      .act_valid_out(act_valid_out_unused),
      .clear_acc    (clear_acc),
      .acc_out      (acc_out)
  );

  // Protocol:
  //   0x01 <byte> load weight
  //   0x02 <byte> send one activation
  //   0x03        read result (FPGA replies 4 bytes, LSB first)
  //   0x04        clear accumulator

  typedef enum logic [2:0] {
    IDLE,
    LOAD_WEIGHT,
    SEND_ACT,
    TX_LOAD,
    TX_BUSY,
    TX_DONE
  } state_t;

  state_t        state;
  logic   [ 1:0] byte_cnt;
  logic   [31:0] acc_reg;

  always_ff @(posedge clk) begin
    weight_load <= 1'b0;
    act_valid   <= 1'b0;
    clear_acc   <= 1'b0;
    tx_valid    <= 1'b0;

    if (rst) begin
      state     <= IDLE;
      byte_cnt  <= '0;
      acc_reg   <= '0;
      weight_in <= '0;
      act_in    <= '0;
      tx_data   <= '0;
    end else begin
      case (state)
        IDLE: begin
          if (rx_valid) begin
            case (rx_data)
              8'h01: state <= LOAD_WEIGHT;

              8'h02: state <= SEND_ACT;

              8'h03: begin
                acc_reg  <= acc_out;
                byte_cnt <= 2'b00;
                state    <= TX_LOAD;
              end

              8'h04: begin
                clear_acc <= 1'b1;
                state     <= IDLE;
              end

              default: state <= IDLE;
            endcase
          end
        end

        LOAD_WEIGHT: begin
          if (rx_valid) begin
            weight_in   <= int8_t'(rx_data);
            weight_load <= 1'b1;
            state       <= IDLE;
          end
        end

        SEND_ACT: begin
          if (rx_valid) begin
            act_in    <= int8_t'(rx_data);
            act_valid <= 1'b1;  // one-cycle pulse
            state     <= IDLE;
          end
        end

        TX_LOAD: begin
          if (tx_ready) begin
            tx_data  <= acc_reg[byte_cnt*8+:8];
            tx_valid <= 1'b1;
            state    <= TX_BUSY;
          end
        end

        TX_BUSY: begin
          if (!tx_ready) state <= TX_DONE;
        end

        TX_DONE: begin
          if (tx_ready) begin
            if (byte_cnt == 2'b11) begin
              state <= IDLE;  // all 4 bytes sent
            end else begin
              byte_cnt <= byte_cnt + 1'b1;
              state    <= TX_LOAD; // send next byte
            end
          end
        end

        default: state <= IDLE;
      endcase
    end
  end

endmodule
