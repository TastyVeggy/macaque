`timescale 1ns / 1ps

module compute_lane (
    input logic clk,
    input logic rst,

    output logic        fifo_pop,
    input  logic [63:0] fifo_data,
    input  logic        fifo_empty,

    // Dep tracker: ready flags
    input logic weight_rdy,
    input logic act_rdy,
    input logic bias_rdy,

    // Dep tracker: notifications
    output logic comp_matmul_start_notify,
    output logic comp_matmul_drain_notify,

    output logic weight_bank_sel,
    output logic act_bank_sel,
    output logic bias_bank_sel,

    // Barrier
    output logic sync_reached,
    input  logic sync_release,

    // Shared DMA channel arbitration: high when the DMA lane is issuing a load
    // this cycle. A STORE must defer (retry next cycle) while it's high so the
    // single shared ddr3_addr/byte_count bus is never driven by both lanes.
    input logic dma_issue,

    // Systolic array interface
    output logic weight_valid,
    output logic act_valid,
    output logic bias_valid,
    input  logic matmul_done,

    // Buffer read control
    output logic                wb_re,
    output npu_pkg::bram_addr_t wb_raddr,
    output logic                ab_re,
    output npu_pkg::bram_addr_t ab_raddr,
    output logic                bb_re,
    output npu_pkg::bram_addr_t bb_raddr,

    // Out buffer partial-sum feedback read (acc_mode=1)
    output logic                ob_fb_re,
    output npu_pkg::bram_addr_t ob_fb_raddr,

    // ACTIVATE / STORE
    output logic                               activate_req,
    input  logic                               activate_done,
    output logic                               store_req,
    input  logic                               store_done,
    output logic [npu_pkg::ISA_DDR_ADDR_W-1:0] ddr3_addr,
    output logic [npu_pkg::ISA_BYTE_CNT_W-1:0] byte_count,

    // ACTIVATE requantize parameters
    output logic               [27:0] act_scale_m,
    output logic               [ 4:0] act_scale_shift,
    output npu_pkg::act_func_t        act_func,

    // Instruction fields
    output logic        acc_mode,
    output logic [11:0] tile_params,
    output logic        matmul_start,

    // Output bank management (owned by compute lane)
    output logic out_bank_sel,
    output logic quant_bank_sel,

    output logic error,

    // Status
    output logic busy,
    output logic mac_active
);

  logic busy_r;

  npu_pkg::decoded_instr_t d_comp;
  assign d_comp = npu_pkg::decode_instr(fifo_data);

  typedef enum logic [3:0] {
    IDLE,
    FETCH,
    MATMUL_WAIT,
    WEIGHT_LOAD,
    ACT_FEED,
    WAIT_DRAIN,
    ACTIVATE_WAIT,
    STORE_WAIT,
    SYNC_WAIT,
    HALT
  } state_t;

  state_t                                  state;

  logic   [npu_pkg::DTYPE_BRAM_ADDR_W-1:0] wl_count;
  logic   [npu_pkg::DTYPE_BRAM_ADDR_W-1:0] al_count;
  logic                                    drain_done;

  // Latched instruction fields consumed after the FIFO pop.
  logic   [                          11:0] cur_tile_params;
  logic                                    cur_acc_mode;

  always_ff @(posedge clk) begin
    if (rst) begin
      state                    <= IDLE;
      fifo_pop                 <= '0;
      weight_valid             <= '0;
      act_valid                <= '0;
      bias_valid               <= '0;
      wb_re                    <= '0;
      wb_raddr                 <= '0;
      ab_re                    <= '0;
      ab_raddr                 <= '0;
      bb_re                    <= '0;
      bb_raddr                 <= '0;
      ob_fb_re                 <= '0;
      ob_fb_raddr              <= '0;
      activate_req             <= '0;
      store_req                <= '0;
      ddr3_addr                <= '0;
      byte_count               <= '0;
      act_scale_m              <= '0;
      act_scale_shift          <= '0;
      act_func                 <= npu_pkg::ACT_RELU;
      acc_mode                 <= '0;
      tile_params              <= '0;
      matmul_start             <= '0;
      out_bank_sel             <= '0;
      quant_bank_sel           <= '0;
      weight_bank_sel          <= '0;
      act_bank_sel             <= '0;
      bias_bank_sel            <= '0;
      comp_matmul_start_notify <= '0;
      comp_matmul_drain_notify <= '0;
      sync_reached             <= '0;
      busy_r                   <= '0;
      mac_active               <= '0;
      error                    <= '0;
      wl_count                 <= '0;
      al_count                 <= '0;
      drain_done               <= '0;
      cur_tile_params          <= '0;
      cur_acc_mode             <= '0;

    end else begin
      fifo_pop                 <= '0;
      weight_valid             <= '0;
      act_valid                <= '0;
      bias_valid               <= '0;
      ob_fb_re                 <= '0;
      matmul_start             <= '0;
      activate_req             <= '0;
      store_req                <= '0;
      comp_matmul_start_notify <= '0;
      comp_matmul_drain_notify <= '0;
      sync_reached             <= '0;
      busy_r                   <= '0;
      mac_active               <= '0;

      case (state)

        IDLE: begin
          if (!fifo_empty) begin
            busy_r <= '1;
            case (d_comp.opcode)

              npu_pkg::OP_MATMUL: begin
                cur_tile_params <= d_comp.tile_params;
                cur_acc_mode    <= d_comp.acc_mode;
                fifo_pop        <= '1;
                state           <= MATMUL_WAIT;
              end

              npu_pkg::OP_ACTIVATE: begin
                automatic npu_pkg::act_instr_t d_act = npu_pkg::decode_act_instr(fifo_data);
                cur_tile_params <= 12'(d_act.act_num_rows);
                fifo_pop        <= '1;
                activate_req    <= '1;
                tile_params     <= 12'(d_act.act_num_rows);
                act_scale_m     <= d_act.act_scale_m;
                act_scale_shift <= d_act.act_scale_shift;
                act_func        <= d_act.act_func;
                out_bank_sel    <= ~out_bank_sel;
                state           <= ACTIVATE_WAIT;
              end

              npu_pkg::OP_STORE: begin
                if (!dma_issue) begin
                  // Channel free: issue the store (1-cycle request on the bus).
                  fifo_pop   <= '1;
                  store_req  <= '1;
                  ddr3_addr  <= d_comp.ddr3_addr;
                  byte_count <= d_comp.byte_count;
                  state      <= STORE_WAIT;
                end
                // else: DMA is issuing a load this cycle so stall
              end

              npu_pkg::OP_SYNC: begin
                fifo_pop     <= '1;
                sync_reached <= '1;
                state        <= SYNC_WAIT;
              end

              default: begin
                fifo_pop <= '1;
              end

            endcase
          end
        end

        MATMUL_WAIT: begin
          busy_r <= '1;
          // Gate on dep_tracker: the DMA-loaded bank (~bank_sel) must be LOADED.
          // bias only needed for first k-tile (acc_mode = 0), subsequent will seed from out buffer
          if (weight_rdy && act_rdy && (cur_acc_mode ? 1'b1 : bias_rdy)) begin
            comp_matmul_start_notify <= '1;
            weight_bank_sel          <= ~weight_bank_sel;
            act_bank_sel             <= ~act_bank_sel;
            bias_bank_sel            <= ~bias_bank_sel;
            wl_count                 <= '0;
            al_count                 <= '0;
            acc_mode                 <= cur_acc_mode;
            tile_params              <= cur_tile_params;
            state                    <= WEIGHT_LOAD;
          end
          // else: stall until buffers are loaded
        end

        WEIGHT_LOAD: begin
          busy_r <= '1;
          matmul_start <= (wl_count == 0);
          wb_re    <= 1'b1;
          wb_raddr <= wl_count;
          weight_valid <= (wl_count > 0);

          if (wl_count == npu_pkg::ARRAY_SIZE[npu_pkg::DTYPE_BRAM_ADDR_W-1:0]) begin
            wb_re    <= 1'b0;
            wl_count <= '0;
            state    <= ACT_FEED;
          end else begin
            wl_count <= wl_count + 1;
          end
        end

        ACT_FEED: begin
          busy_r <= '1;
          mac_active <= '1;
          ab_re <= (al_count < cur_tile_params[$clog2(npu_pkg::BRAM_DEPTH)-1:0]);
          ab_raddr <= al_count;
          bb_re <= (al_count < cur_tile_params[$clog2(npu_pkg::BRAM_DEPTH)-1:0]);
          bb_raddr <= al_count;
          ob_fb_re <= (al_count < cur_tile_params[$clog2(npu_pkg::BRAM_DEPTH)-1:0]) && cur_acc_mode;
          ob_fb_raddr <= al_count;
          act_valid <= (al_count > 0);
          bias_valid <= (al_count > 0);

          if (al_count == cur_tile_params[$clog2(npu_pkg::BRAM_DEPTH)-1:0]) begin
            ab_re    <= 1'b0;
            bb_re    <= 1'b0;
            ob_fb_re <= 1'b0;
            al_count <= '0;
            state    <= WAIT_DRAIN;
          end else begin
            al_count <= al_count + 1;
          end
        end

        WAIT_DRAIN: begin
          busy_r <= '1;
          if (matmul_done) drain_done <= 1'b1;
          if (drain_done && !matmul_done) begin
            drain_done               <= 1'b0;
            comp_matmul_drain_notify <= '1;
            state                    <= IDLE;
          end
        end

        ACTIVATE_WAIT: begin
          busy_r <= '1;
          if (activate_done) begin
            quant_bank_sel <= ~quant_bank_sel;
            state          <= IDLE;
          end
        end

        STORE_WAIT: begin
          busy_r <= '1;
          if (store_done) begin
            state <= IDLE;
          end
        end

        SYNC_WAIT: begin
          busy_r <= '1;
          if (sync_release) begin
            state <= IDLE;
          end
        end

        HALT: begin
        end

        default: state <= IDLE;

      endcase
    end
  end

  assign busy = busy_r || !fifo_empty;

endmodule
