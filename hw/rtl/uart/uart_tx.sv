`timescale 1ns / 1ps
module uart_tx #(
    parameter int CLK_FREQ  = npu_pkg::SYS_CLK_FREQ,
    parameter int BAUD_RATE = npu_pkg::UART_BAUD_DEFAULT
) (
    input logic clk,
    input logic rst,
    input logic [7:0] tx_data,
    input logic tx_valid,
    output logic tx_ready,
    output logic tx_pin
);
  localparam int BIT_PERIOD_INT = CLK_FREQ / BAUD_RATE;
  typedef logic [$clog2(BIT_PERIOD_INT)-1:0] uart_cnt_t;

  localparam uart_cnt_t BIT_PERIOD = uart_cnt_t'(BIT_PERIOD_INT);

  typedef enum logic [1:0] {
    IDLE  = 2'b00,
    START = 2'b01,
    DATA  = 2'b10,
    STOP  = 2'b11
  } state_t;

  state_t state;
  uart_cnt_t clk_cnt;
  logic [2:0] bit_cnt;
  logic [7:0] shift_reg;

  always_ff @(posedge clk) begin
    if (rst) begin
      state <= IDLE;
      clk_cnt <= '0;
      bit_cnt <= '0;
      shift_reg <= '0;
      tx_pin <= 1'b1;
      tx_ready <= 1'b1;
    end else begin
      case (state)
        IDLE: begin
          tx_pin   <= 1'b1;
          tx_ready <= 1'b1;
          clk_cnt  <= '0;
          bit_cnt  <= '0;
          if (tx_valid) begin
            shift_reg <= tx_data;
            tx_ready <= 1'b0;
            state <= START;
          end
        end

        START: begin
          tx_pin   <= 1'b0;
          tx_ready <= 1'b0;
          if (clk_cnt == BIT_PERIOD - 1) begin
            clk_cnt <= '0;
            state   <= DATA;
          end else begin
            clk_cnt <= clk_cnt + 1'b1;
          end
        end

        DATA: begin
          tx_pin   <= shift_reg[0];
          tx_ready <= 1'b0;
          if (clk_cnt == BIT_PERIOD - 1) begin
            clk_cnt <= '0;
            if (bit_cnt == 3'd7) begin
              state <= STOP;
            end else begin
              shift_reg <= {1'b0, shift_reg[7:1]};
              bit_cnt   <= bit_cnt + 1'b1;
            end
          end else begin
            clk_cnt <= clk_cnt + 1'b1;
          end
        end

        STOP: begin
          tx_pin   <= 1'b1;
          tx_ready <= 1'b0;
          if (clk_cnt == BIT_PERIOD - 1) begin
            state <= IDLE;
            tx_ready <= 1'b1;
          end else begin
            clk_cnt <= clk_cnt + 1'b1;
          end
        end

        default: state <= IDLE;
      endcase
    end

  end
endmodule
