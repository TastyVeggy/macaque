`timescale 1ns / 1ps


// Dependency tracker: per-buffer, per-bank slot
// - Tracks three buffers (weight, bias, activation), each with two banks.
// - Handles SYNC
module dep_tracker (
    input logic clk,
    input logic rst,

    // DMA lane to inform when a load just completed
    input logic                  dma_load_done_notify,
    input npu_pkg::buffer_type_t dma_load_buf_type,     // which buffer
    input logic                  dma_load_bank,         // which bank was loaded (the inactive one)

    // Compute lane to inform when MATMUL starting / drain done
    input logic comp_matmul_start_notify,
    input logic comp_matmul_drain_notify,
    input logic comp_weight_bank_sel,
    input logic comp_act_bank_sel,
    input logic comp_bias_bank_sel,

    input logic comp_weight_hold,

    // Dep tracker inform compute lane when each buffer is ready
    output logic weight_rdy,
    output logic act_rdy,
    output logic bias_rdy,

    // Dep tracker inform DMA lane when can load (inactive bank is EMPTY)
    output logic dma_weight_can_load,
    output logic dma_act_can_load,
    output logic dma_bias_can_load,

    // Barrier (SYNC)
    input  logic sync_reached_dma,
    input  logic sync_reached_comp,
    output logic sync_release
);

  npu_pkg::slot_state_t weight_slot[2];
  npu_pkg::slot_state_t bias_slot[2];
  npu_pkg::slot_state_t act_slot[2];

  assign weight_rdy          = comp_weight_hold
      ? 1'b1
      : (weight_slot[~comp_weight_bank_sel] == npu_pkg::SLOT_LOADED);
  assign bias_rdy            = comp_weight_hold
      ? 1'b1
      : (bias_slot[~comp_bias_bank_sel] == npu_pkg::SLOT_LOADED);
  assign act_rdy = (act_slot[~comp_act_bank_sel] == npu_pkg::SLOT_LOADED);

  assign dma_weight_can_load = (weight_slot[~comp_weight_bank_sel] == npu_pkg::SLOT_EMPTY);
  assign dma_bias_can_load = (bias_slot[~comp_bias_bank_sel] == npu_pkg::SLOT_EMPTY);
  assign dma_act_can_load = (act_slot[~comp_act_bank_sel] == npu_pkg::SLOT_EMPTY);

  // release when both lanes reached SYNC
  logic sync_dma, sync_comp;
  always_ff @(posedge clk) begin
    if (rst) begin
      sync_dma  <= '0;
      sync_comp <= '0;
    end else begin
      if (sync_reached_dma) sync_dma <= '1;
      if (sync_reached_comp) sync_comp <= '1;
      if (sync_dma && sync_comp) begin
        sync_dma  <= '0;
        sync_comp <= '0;
      end
    end
  end
  assign sync_release = (sync_dma && sync_comp);

  always_ff @(posedge clk) begin
    if (rst) begin
      weight_slot[0] <= npu_pkg::SLOT_EMPTY;
      weight_slot[1] <= npu_pkg::SLOT_EMPTY;
      bias_slot[0]   <= npu_pkg::SLOT_EMPTY;
      bias_slot[1]   <= npu_pkg::SLOT_EMPTY;
      act_slot[0]    <= npu_pkg::SLOT_EMPTY;
      act_slot[1]    <= npu_pkg::SLOT_EMPTY;
    end else begin
      if (dma_load_done_notify) begin
        unique case (dma_load_buf_type)
          npu_pkg::BUF_WEIGHT: weight_slot[dma_load_bank] <= npu_pkg::SLOT_LOADED;
          npu_pkg::BUF_BIAS:   bias_slot[dma_load_bank]   <= npu_pkg::SLOT_LOADED;
          npu_pkg::BUF_ACT:    act_slot[dma_load_bank]    <= npu_pkg::SLOT_LOADED;
          default: ;
        endcase
      end

      if (comp_matmul_start_notify) begin
        weight_slot[comp_weight_bank_sel] <= npu_pkg::SLOT_BUSY;
        bias_slot[comp_bias_bank_sel]     <= npu_pkg::SLOT_BUSY;
        act_slot[comp_act_bank_sel]       <= npu_pkg::SLOT_BUSY;
      end

      if (comp_matmul_drain_notify) begin
        weight_slot[comp_weight_bank_sel] <= npu_pkg::SLOT_EMPTY;
        bias_slot[comp_bias_bank_sel]     <= npu_pkg::SLOT_EMPTY;
        act_slot[comp_act_bank_sel]       <= npu_pkg::SLOT_EMPTY;
      end
    end
  end

endmodule
