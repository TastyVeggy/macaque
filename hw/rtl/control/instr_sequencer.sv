`timescale 1ns / 1ps

module instr_sequencer (
    input logic clk,
    input logic rst,

    input logic        start,
    input logic        reset,
    input logic [31:0] instr_base,
    input logic [31:0] instr_len,

    output logic busy,
    output logic done,
    output logic error,
    output logic ready,

    output logic [31:0] im_addr,
    input  logic [63:0] im_data,

    // MATMUL: to systolic array
    output logic weight_valid,
    output logic act_valid,
    output logic bias_valid,
    input  logic matmul_done,

    // ACTIVATE: to activation unit
    output logic activate_req,
    input  logic activate_done,

    // ACTIVATE requantize parameters
    output logic               [27:0] act_scale_m,
    output logic               [ 4:0] act_scale_shift,
    output npu_pkg::act_func_t        act_func,

    // LOAD/STORE: to DMA
    output logic        load_req,
    output logic        store_req,
    output logic [ 2:0] load_target,
    output logic [27:0] ddr3_addr,
    output logic [15:0] byte_count,
    output logic        acc_mode,
    output logic [11:0] tile_params,
    input  logic        load_done,
    input  logic        store_done,

    // Buffer read control during MATMUL
    output logic                wb_re,
    output npu_pkg::bram_addr_t wb_raddr,
    output logic                ab_re,
    output npu_pkg::bram_addr_t ab_raddr,
    output logic                bb_re,
    output npu_pkg::bram_addr_t bb_raddr,
    output logic                ob_fb_re,
    output npu_pkg::bram_addr_t ob_fb_raddr,

    // Bank select outputs
    output logic weight_bank_sel,
    output logic act_bank_sel,
    output logic bias_bank_sel,
    output logic out_bank_sel,
    output logic quant_bank_sel,

    // MATMUL start pulse
    output logic matmul_start,

    // PMU
    output logic pmu_freeze,
    output logic run_active,
    output logic mac_active,
    output logic stall
);

  // Fetch unit
  logic fet_busy;
  logic fet_halt;

  // DMA FIFO
  logic dma_push, dma_fifo_full, dma_fifo_empty, dma_fifo_pop;
  logic [63:0] dma_push_data, dma_fifo_data;

  // Compute FIFO
  logic comp_push, comp_fifo_full, comp_fifo_empty, comp_fifo_pop;
  logic [63:0] comp_push_data, comp_fifo_data;

  // DMA lane
  logic                  dma_lane_busy;
  logic                  dma_issue;  // DMA would-issue a load this cycle
  logic                  dma_load_done_notify;
  npu_pkg::buffer_type_t dma_load_buf_type;
  logic                  dma_load_bank;
  logic sync_reached_dma, sync_release;
  logic        dma_load_req;
  logic [ 2:0] dma_load_target;
  logic [27:0] dma_ddr3_addr;
  logic [15:0] dma_byte_count;

  // Compute lane
  logic        comp_lane_busy;
  logic        comp_mac_active;
  logic        comp_error;
  logic comp_matmul_start_notify, comp_matmul_drain_notify;
  logic        comp_store_req;
  logic [27:0] comp_ddr3_addr;
  logic [15:0] comp_byte_count;
  logic        sync_reached_comp;

  // Bank selects: owned by DMA lane
  logic int_weight_bank_sel, int_act_bank_sel, int_bias_bank_sel;

  // Dep tracker outputs
  logic weight_rdy, act_rdy, bias_rdy;
  logic dma_weight_can_load, dma_bias_can_load, dma_act_can_load;

  fetch_unit fetch_inst (
      .clk           (clk),
      .rst           (rst || reset),
      .start         (start),
      .instr_base    (instr_base),
      .instr_len     (instr_len),
      .im_addr       (im_addr),
      .im_data       (im_data),
      .dma_push      (dma_push),
      .dma_push_data (dma_push_data),
      .dma_fifo_full (dma_fifo_full),
      .comp_push     (comp_push),
      .comp_push_data(comp_push_data),
      .comp_fifo_full(comp_fifo_full),
      .busy          (fet_busy),
      .halt          (fet_halt)
  );

  instr_queue #(
      .DEPTH(8)
  ) dma_fifo (
      .clk      (clk),
      .rst      (rst || reset),
      .push     (dma_push),
      .push_data(dma_push_data),
      .full     (dma_fifo_full),
      .pop      (dma_fifo_pop),
      .pop_data (dma_fifo_data),
      .empty    (dma_fifo_empty)
  );

  instr_queue #(
      .DEPTH(8)
  ) comp_fifo (
      .clk      (clk),
      .rst      (rst || reset),
      .push     (comp_push),
      .push_data(comp_push_data),
      .full     (comp_fifo_full),
      .pop      (comp_fifo_pop),
      .pop_data (comp_fifo_data),
      .empty    (comp_fifo_empty)
  );

  dma_lane dma_inst (
      .clk                 (clk),
      .rst                 (rst || reset),
      .fifo_pop            (dma_fifo_pop),
      .fifo_data           (dma_fifo_data),
      .fifo_empty          (dma_fifo_empty),
      .dma_weight_can_load (dma_weight_can_load),
      .dma_bias_can_load   (dma_bias_can_load),
      .dma_act_can_load    (dma_act_can_load),
      .dma_load_done_notify(dma_load_done_notify),
      .dma_load_buf_type   (dma_load_buf_type),
      .dma_load_bank       (dma_load_bank),
      .sync_reached        (sync_reached_dma),
      .sync_release        (sync_release),
      .load_req            (dma_load_req),
      .load_target         (dma_load_target),
      .ddr3_addr           (dma_ddr3_addr),
      .byte_count          (dma_byte_count),
      .load_done           (load_done),
      .comp_weight_bank_sel(int_weight_bank_sel),
      .comp_act_bank_sel   (int_act_bank_sel),
      .comp_bias_bank_sel  (int_bias_bank_sel),
      .dma_issue           (dma_issue),
      .busy                (dma_lane_busy)
  );

  compute_lane comp_inst (
      .clk                     (clk),
      .rst                     (rst || reset),
      .fifo_pop                (comp_fifo_pop),
      .fifo_data               (comp_fifo_data),
      .fifo_empty              (comp_fifo_empty),
      .weight_rdy              (weight_rdy),
      .act_rdy                 (act_rdy),
      .bias_rdy                (bias_rdy),
      .comp_matmul_start_notify(comp_matmul_start_notify),
      .comp_matmul_drain_notify(comp_matmul_drain_notify),
      .weight_bank_sel         (int_weight_bank_sel),
      .act_bank_sel            (int_act_bank_sel),
      .bias_bank_sel           (int_bias_bank_sel),
      .sync_reached            (sync_reached_comp),
      .sync_release            (sync_release),
      .dma_issue               (dma_issue),
      .weight_valid            (weight_valid),
      .act_valid               (act_valid),
      .bias_valid              (bias_valid),
      .matmul_done             (matmul_done),
      .wb_re                   (wb_re),
      .wb_raddr                (wb_raddr),
      .ab_re                   (ab_re),
      .ab_raddr                (ab_raddr),
      .bb_re                   (bb_re),
      .bb_raddr                (bb_raddr),
      .ob_fb_re                (ob_fb_re),
      .ob_fb_raddr             (ob_fb_raddr),
      .activate_req            (activate_req),
      .activate_done           (activate_done),
      .act_scale_m             (act_scale_m),
      .act_scale_shift         (act_scale_shift),
      .act_func                (act_func),
      .store_req               (comp_store_req),
      .store_done              (store_done),
      .ddr3_addr               (comp_ddr3_addr),
      .byte_count              (comp_byte_count),
      .acc_mode                (acc_mode),
      .tile_params             (tile_params),
      .matmul_start            (matmul_start),
      .out_bank_sel            (out_bank_sel),
      .quant_bank_sel          (quant_bank_sel),
      .busy                    (comp_lane_busy),
      .mac_active              (comp_mac_active),
      .error                   (comp_error)
  );

  dep_tracker dep_tracker_inst (
      .clk                     (clk),
      .rst                     (rst || reset),
      .dma_load_done_notify    (dma_load_done_notify),
      .dma_load_buf_type       (dma_load_buf_type),
      .dma_load_bank           (dma_load_bank),
      .comp_matmul_start_notify(comp_matmul_start_notify),
      .comp_matmul_drain_notify(comp_matmul_drain_notify),
      .comp_weight_bank_sel    (int_weight_bank_sel),
      .comp_act_bank_sel       (int_act_bank_sel),
      .comp_bias_bank_sel      (int_bias_bank_sel),
      .weight_rdy              (weight_rdy),
      .act_rdy                 (act_rdy),
      .bias_rdy                (bias_rdy),
      .dma_weight_can_load     (dma_weight_can_load),
      .dma_bias_can_load       (dma_bias_can_load),
      .dma_act_can_load        (dma_act_can_load),
      .sync_reached_dma        (sync_reached_dma),
      .sync_reached_comp       (sync_reached_comp),
      .sync_release            (sync_release)
  );

  assign weight_bank_sel = int_weight_bank_sel;
  assign act_bank_sel    = int_act_bank_sel;
  assign bias_bank_sel   = int_bias_bank_sel;

  assign ready           = (start == 1'b0) && !fet_busy && !dma_lane_busy && !comp_lane_busy;
  assign busy            = fet_busy || dma_lane_busy || comp_lane_busy;
  assign done            = fet_halt && !dma_lane_busy && !comp_lane_busy;
  assign error           = comp_error;
  assign pmu_freeze      = done;
  assign run_active      = busy;
  assign mac_active      = comp_mac_active;
  assign stall           = dma_lane_busy || comp_lane_busy;

  // Mux shared DMA channel signals: load_req from DMA lane, store_req from compute lane
  assign load_req        = dma_load_req;
  assign store_req       = comp_store_req;
  assign load_target     = dma_load_target;
  assign ddr3_addr       = dma_load_req ? dma_ddr3_addr : comp_ddr3_addr;
  assign byte_count      = dma_load_req ? dma_byte_count : comp_byte_count;

endmodule
