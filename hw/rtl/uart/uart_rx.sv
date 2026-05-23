`timescale 1ns / 1ps
module uart_rx #(
    parameter int CLK_FREQ  = npu_pkg::SYS_CLK_FREQ,
    parameter int BAUD_RATE = npu_pkg::UART_BAUD_DEFAULT
) (
    input  logic       clk,
    input  logic       rst,
    input  logic       rx_pin,
    output logic [7:0] rx_data,
    output logic       rx_valid
);
  localparam int BitPeriodInt = CLK_FREQ / BAUD_RATE;
  typedef logic [$clog2(BitPeriodInt)-1:0] uart_cnt_t;

  localparam uart_cnt_t BitPeriod = uart_cnt_t'(BitPeriodInt);
  localparam uart_cnt_t HalfPeriod = BitPeriod / 2;


  logic rx_meta, rx_sync;
  logic rx_prev;

  always_ff @(posedge clk) begin
    if (rst) begin
      rx_meta <= 1'b1;
      rx_sync <= 1'b1;
    end else begin
      rx_meta <= rx_pin;
      rx_sync <= rx_meta;
    end
  end

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
      rx_data <= '0;
      rx_valid <= '0;
      rx_prev <= 1'b1;
    end else begin
      rx_prev  <= rx_sync;
      rx_valid <= 1'b0;

      case (state)
        IDLE: begin
          clk_cnt <= '0;
          bit_cnt <= '0;
          if (rx_prev == 1'b1 && rx_sync == 1'b0) begin
            state <= START;
          end
        end

        START: begin
          if (clk_cnt == HalfPeriod - 1'b1) begin
            clk_cnt <= '0;
            if (rx_sync == 1'b0) begin
              state <= DATA;
            end else begin
              state <= IDLE;
            end
          end else begin
            clk_cnt <= clk_cnt + 1'b1;
          end
        end

        DATA: begin
          if (clk_cnt == BitPeriod - 1'b1) begin
            clk_cnt   <= '0;
            shift_reg <= {rx_sync, shift_reg[7:1]};
            if (bit_cnt == 3'd7) begin
              state <= STOP;
            end else begin
              bit_cnt <= bit_cnt + 1'b1;
            end
          end else begin
            clk_cnt <= clk_cnt + 1'b1;
          end
        end

        STOP: begin
          if (clk_cnt == BitPeriod - 1'b1) begin
            rx_data <= shift_reg;
            rx_valid <= 1'b1;
            state <= IDLE;
          end else begin
            clk_cnt <= clk_cnt + 1'b1;
          end
        end

        default: state <= IDLE;

      endcase
    end
  end

endmodule
