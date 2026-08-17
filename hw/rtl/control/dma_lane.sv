`timescale 1ns / 1ps

module dma_lane (
    input logic clk,
    input logic rst,

    output logic        fifo_pop,
    input  logic [63:0] fifo_data,
    input  logic        fifo_empty,

    // Dep tracker: can-load flags
    input logic dma_weight_can_load,
    input logic dma_bias_can_load,
    input logic dma_act_can_load,

    // Dep tracker: load done notification (1 cycle pulse)
    output logic                  dma_load_done_notify,
    output npu_pkg::buffer_type_t dma_load_buf_type,
    output logic                  dma_load_bank,

    // Barrier
    output logic sync_reached,
    input  logic sync_release,

    // DMA channel
    output logic        load_req,
    output logic [ 2:0] load_target,
    output logic [27:0] ddr3_addr,
    output logic [15:0] byte_count,
    input  logic        load_done,

    // Bank selects (compute lane owns this, owns which bank the SA reads).
    // The DMA always loads the INACTIVE bank (~bank_sel) and never toggles.
    input logic comp_weight_bank_sel,
    input logic comp_act_bank_sel,
    input logic comp_bias_bank_sel,

    // Combinational "would-issue a load this cycle". Used by the wrapper to arbitrate the shared
    // DMA channel (compute defers a STORE while a LOAD is issuing).
    output logic dma_issue,

    output logic busy
);

  logic busy_r;

  npu_pkg::decoded_instr_t d_dma;
  assign d_dma = npu_pkg::decode_instr(fifo_data);

  logic dma_can_load_this;
  always_comb begin
    unique case (d_dma.opcode)
      npu_pkg::OP_LOAD_WEIGHT: dma_can_load_this = dma_weight_can_load;
      npu_pkg::OP_LOAD_BIAS:   dma_can_load_this = dma_bias_can_load;
      npu_pkg::OP_LOAD_INPUT:  dma_can_load_this = dma_act_can_load;
      default:                 dma_can_load_this = 1'b0;
    endcase
  end

  // Which inactive bank will be loaded
  logic cur_loaded_bank;
  always_comb begin
    unique case (d_dma.opcode)
      npu_pkg::OP_LOAD_WEIGHT: cur_loaded_bank = ~comp_weight_bank_sel;
      npu_pkg::OP_LOAD_BIAS:   cur_loaded_bank = ~comp_bias_bank_sel;
      npu_pkg::OP_LOAD_INPUT:  cur_loaded_bank = ~comp_act_bank_sel;
      default:                 cur_loaded_bank = 1'b0;
    endcase
  end

  // Would-issue a load this cycle (matches the DMA_IDLE load_req condition).
  // Combinational so the wrapper/compute lane can arbitrate in the same cycle.
  assign dma_issue =
      (state == DMA_IDLE) && !fifo_empty &&
      (d_dma.opcode inside {
        npu_pkg::OP_LOAD_WEIGHT,
        npu_pkg::OP_LOAD_BIAS,
        npu_pkg::OP_LOAD_INPUT}) &&
      dma_can_load_this;

  typedef enum logic [1:0] {
    DMA_IDLE,
    DMA_WAIT,
    DMA_SYNC
  } dma_state_t;

  dma_state_t            state;
  npu_pkg::buffer_type_t issue_buf_type;
  logic                  issue_loaded_bank;

  always_ff @(posedge clk) begin
    if (rst) begin
      state                <= DMA_IDLE;
      fifo_pop             <= '0;
      load_req             <= '0;
      load_target          <= '0;
      ddr3_addr            <= '0;
      byte_count           <= '0;
      dma_load_done_notify <= '0;
      dma_load_buf_type    <= npu_pkg::BUF_WEIGHT;
      dma_load_bank        <= '0;
      sync_reached         <= '0;
      busy_r               <= '0;
      issue_buf_type       <= npu_pkg::BUF_WEIGHT;
      issue_loaded_bank    <= '0;
    end else begin
      fifo_pop             <= '0;
      load_req             <= '0;
      sync_reached         <= '0;
      busy_r               <= '0;
      dma_load_done_notify <= '0;

      case (state)

        DMA_IDLE: begin
          if (!fifo_empty) begin
            if (d_dma.opcode == npu_pkg::OP_SYNC) begin
              fifo_pop     <= 1'b1;
              sync_reached <= '1;
              state        <= DMA_SYNC;
            end else if (d_dma.opcode inside {
              npu_pkg::OP_LOAD_WEIGHT,
              npu_pkg::OP_LOAD_BIAS,
              npu_pkg::OP_LOAD_INPUT}) begin

              if (dma_can_load_this) begin
                fifo_pop <= 1'b1;
                load_req <= 1'b1;
                load_target <= d_dma.target;
                ddr3_addr <= d_dma.ddr3_addr;
                byte_count <= d_dma.byte_count;
                busy_r <= 1'b1;
                issue_buf_type <= (d_dma.opcode == npu_pkg::OP_LOAD_WEIGHT)
                  ? npu_pkg::BUF_WEIGHT
                  : (d_dma.opcode == npu_pkg::OP_LOAD_BIAS)
                    ? npu_pkg::BUF_BIAS
                    : npu_pkg::BUF_ACT;
                issue_loaded_bank <= cur_loaded_bank;
                state <= DMA_WAIT;
              end else begin
                // Inactive bank is busy so stall
                busy_r <= 1'b1;
              end
            end
          end
        end

        DMA_WAIT: begin
          busy_r <= 1'b1;
          if (load_done) begin
            // Complete the load. The compute lane toggles bank_sel at the next
            // MATMUL start.
            dma_load_done_notify <= 1'b1;
            dma_load_buf_type    <= issue_buf_type;
            dma_load_bank        <= issue_loaded_bank;
            state                <= DMA_IDLE;
          end
        end

        DMA_SYNC: begin
          busy_r <= 1'b1;
          if (sync_release) begin
            state <= DMA_IDLE;  // already popped when latched in DMA_IDLE
          end
        end

        default: state <= DMA_IDLE;

      endcase
    end
  end

  assign busy = busy_r || !fifo_empty;

endmodule
