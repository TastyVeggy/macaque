`timescale 1ns / 1ps

module fetch_unit (
    input logic clk,
    input logic rst,

    // Control
    input logic        start,
    input logic [31:0] instr_base,
    input logic [31:0] instr_len,

    // instruction memory
    output logic [31:0] im_addr,
    input  logic [63:0] im_data,

    output logic        dma_push,
    output logic [63:0] dma_push_data,
    input  logic        dma_fifo_full,

    output logic        comp_push,
    output logic [63:0] comp_push_data,
    input  logic        comp_fifo_full,

    // Status
    output logic busy,
    output logic halt
);

  typedef enum logic [1:0] {
    FET_IDLE,
    FET_ADDR,
    FET_WAIT,
    FET_PUSH
  } fet_state_t;

  fet_state_t state;
  logic [31:0] pc;
  logic [31:0] prog_end;

  assign prog_end = instr_base + instr_len * 8;

  // Decoded instruction
  npu_pkg::decoded_instr_t d_instr;
  assign d_instr = npu_pkg::decode_instr(im_data);

  always_ff @(posedge clk) begin
    if (rst) begin
      state          <= FET_IDLE;
      pc             <= '0;
      im_addr        <= '0;
      dma_push       <= '0;
      comp_push      <= '0;
      dma_push_data  <= '0;
      comp_push_data <= '0;
      busy           <= '0;
      halt           <= '0;
    end else begin
      dma_push  <= '0;
      comp_push <= '0;
      busy      <= '0;

      case (state)

        FET_IDLE: begin
          if (start) begin
            halt  <= '0; // new program clears the halt latch
            pc    <= instr_base;
            state <= FET_ADDR;
          end
        end

        FET_ADDR: begin
          if (pc >= prog_end) begin
            halt  <= '1;
            state <= FET_IDLE;
          end else begin
            im_addr <= pc;
            state   <= FET_WAIT;
          end
        end

        // BRAM is capturing instr_mem[pc] now so im_data updates next cycle.
        FET_WAIT: begin
          busy  <= '1;
          state <= FET_PUSH;
        end

        FET_PUSH: begin
          busy <= '1;
          case (d_instr.opcode)

            npu_pkg::OP_LOAD_WEIGHT, npu_pkg::OP_LOAD_BIAS, npu_pkg::OP_LOAD_INPUT: begin
              if (!dma_fifo_full) begin
                dma_push      <= '1;
                dma_push_data <= im_data;
                pc            <= pc + 8;
                state         <= FET_ADDR;
              end
              // else: stall
            end

            npu_pkg::OP_MATMUL, npu_pkg::OP_ACTIVATE, npu_pkg::OP_STORE: begin
              if (!comp_fifo_full) begin
                comp_push      <= '1;
                comp_push_data <= im_data;
                pc             <= pc + 8;
                state          <= FET_ADDR;
              end
              // else: stall
            end

            npu_pkg::OP_SYNC: begin
              // atomic push: both FIFOs must accept in the same cycle.
              // If either is full, then stall.
              if (!dma_fifo_full && !comp_fifo_full) begin
                dma_push       <= '1;
                dma_push_data  <= im_data;
                comp_push      <= '1;
                comp_push_data <= im_data;
                pc             <= pc + 8;
                state          <= FET_ADDR;
              end
              // else: stall
            end

            default: begin
              // treat as halt
              halt  <= '1;
              state <= FET_IDLE;
            end

          endcase
        end

        default: state <= FET_IDLE;

      endcase
    end
  end

endmodule

