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
    output logic                                                load_req,
    output npu_pkg::buffer_type_t                               load_target,
    output logic                  [npu_pkg::ISA_DDR_ADDR_W-1:0] ddr3_addr,
    output logic                  [npu_pkg::ISA_BYTE_CNT_W-1:0] byte_count,

    output logic [3:0] load_valid_bytes_per_row,
    output logic [7:0] load_input_rows,

    input logic load_done,

    // Bank selects (compute lane owns this, owns which bank the SA reads).
    // The DMA always loads the INACTIVE bank (~bank_sel) and never toggles.
    input logic comp_weight_bank_sel,
    input logic comp_act_bank_sel,
    input logic comp_bias_bank_sel,

    // Combinational "would-issue a load this cycle". Used by the wrapper to arbitrate the shared
    // DMA channel (compute defers a STORE while a LOAD is issuing).
    output logic dma_issue,

    // Shared-channel arbitration: high while the compute lane is holding a STORE
    // request. The DMA lane defers its LOAD so the store request is not lost.
    input logic store_req,

    output logic busy,

    output logic stall
);

  logic busy_r;

  npu_pkg::opcode_t d_dma_opcode;
  assign d_dma_opcode = npu_pkg::decode_opcode(fifo_data);

  logic dma_can_load_this;
  always_comb begin
    unique case (d_dma_opcode)
      npu_pkg::OP_LOAD_WEIGHT: dma_can_load_this = dma_weight_can_load;
      npu_pkg::OP_LOAD_BIAS:   dma_can_load_this = dma_bias_can_load;
      npu_pkg::OP_LOAD_INPUT:  dma_can_load_this = dma_act_can_load;
      default:                 dma_can_load_this = 1'b0;
    endcase
  end

  // Which inactive bank will be loaded
  logic cur_loaded_bank;
  always_comb begin
    unique case (d_dma_opcode)
      npu_pkg::OP_LOAD_WEIGHT: cur_loaded_bank = ~comp_weight_bank_sel;
      npu_pkg::OP_LOAD_BIAS:   cur_loaded_bank = ~comp_bias_bank_sel;
      npu_pkg::OP_LOAD_INPUT:  cur_loaded_bank = ~comp_act_bank_sel;
      default:                 cur_loaded_bank = 1'b0;
    endcase
  end

  npu_pkg::buffer_type_t cur_buf_type;
  always_comb begin
    unique case (d_dma_opcode)
      npu_pkg::OP_LOAD_WEIGHT: cur_buf_type = npu_pkg::BUF_WEIGHT;
      npu_pkg::OP_LOAD_BIAS:   cur_buf_type = npu_pkg::BUF_BIAS;
      npu_pkg::OP_LOAD_INPUT:  cur_buf_type = npu_pkg::BUF_ACT;
      default:                 cur_buf_type = npu_pkg::BUF_WEIGHT;
    endcase
  end

  // Would-issue a load this cycle (matches the IDLE load_req condition).
  // Combinational so the wrapper/compute lane can arbitrate in the same cycle.
  assign dma_issue =
      (state == IDLE) && !fifo_empty &&
      (d_dma_opcode inside {
        npu_pkg::OP_LOAD_WEIGHT,
        npu_pkg::OP_LOAD_BIAS,
        npu_pkg::OP_LOAD_INPUT}) &&
      dma_can_load_this && !store_req;

  typedef enum logic [1:0] {
    IDLE,
    WAIT,
    NOTIFY,
    SYNC
  } state_t;

  state_t                state;
  npu_pkg::buffer_type_t issue_buf_type;
  logic                  issue_loaded_bank;

  always_ff @(posedge clk) begin
    if (rst) begin
      state                    <= IDLE;
      fifo_pop                 <= '0;
      load_req                 <= '0;
      load_target              <= npu_pkg::BUF_WEIGHT;
      ddr3_addr                <= '0;
      byte_count               <= '0;
      load_valid_bytes_per_row <= '0;
      load_input_rows          <= '0;
      dma_load_done_notify     <= '0;
      dma_load_buf_type        <= npu_pkg::BUF_WEIGHT;
      dma_load_bank            <= '0;
      sync_reached             <= '0;
      busy_r                   <= '0;
      issue_buf_type           <= npu_pkg::BUF_WEIGHT;
      issue_loaded_bank        <= '0;
    end else begin
      fifo_pop             <= '0;
      load_req             <= '0;
      sync_reached         <= '0;
      busy_r               <= '0;
      dma_load_done_notify <= '0;

      case (state)

        IDLE: begin
          if (!fifo_empty) begin
            if (d_dma_opcode == npu_pkg::OP_SYNC) begin
              fifo_pop     <= 1'b1;
              sync_reached <= '1;
              state        <= SYNC;
            end else if (d_dma_opcode inside {
              npu_pkg::OP_LOAD_WEIGHT,
              npu_pkg::OP_LOAD_BIAS,
              npu_pkg::OP_LOAD_INPUT}) begin

              if (dma_can_load_this && !store_req) begin
                automatic npu_pkg::load_instr_t d_load = npu_pkg::decode_load_instr(fifo_data);
                fifo_pop <= 1'b1;
                load_req <= 1'b1;
                load_target <= cur_buf_type;
                ddr3_addr <= d_load.ddr3_addr;
                byte_count <= d_load.byte_count;
                load_valid_bytes_per_row <= d_load.valid_bytes_per_row;
                load_input_rows <= d_load.input_rows;
                busy_r <= 1'b1;
                issue_buf_type <= cur_buf_type;
                issue_loaded_bank <= cur_loaded_bank;
                state <= WAIT;
              end else begin
                // Inactive bank is busy so stall
                busy_r <= 1'b1;
              end
            end
          end
        end

        WAIT: begin
          busy_r <= 1'b1;
          if (load_done) begin
            // Complete the load. The compute lane toggles bank_sel at the next
            // MATMUL start.
            dma_load_done_notify <= 1'b1;
            dma_load_buf_type    <= issue_buf_type;
            dma_load_bank        <= issue_loaded_bank;
            state                <= NOTIFY;
          end
        end

        NOTIFY: begin
          busy_r <= 1'b1;
          state  <= IDLE;
        end

        SYNC: begin
          busy_r <= 1'b1;
          if (sync_release) begin
            state <= IDLE;  // already popped when latched in IDLE
          end
        end

        default: state <= IDLE;

      endcase
    end
  end

  assign busy  = busy_r || !fifo_empty;
  assign stall = busy_r && (state != WAIT);

endmodule
