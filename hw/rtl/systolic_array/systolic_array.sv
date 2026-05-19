`timescale 1ns / 1ps

import npu_pkg::*;

module systolic_array (
    input logic clk,
    input logic rst,

    // Weight configuration: Preload (ARRAY_SIZE x ARRAY_SIZE) weights during configuration phase. Flow north from bottom
    input int8_t weight_data [ARRAY_SIZE],
    input logic  weight_valid,

    // Activation inputs: Present one (ARRAY_SIZE sized) row of Matrix A (N x ARRAY_SIZE) every single cycle. Flows east from left
    input int8_t act_data [ARRAY_SIZE],
    input logic  act_valid,

    // Bias inputs: Present one (ARRAY_SIZE sized) row of bias matrix (N x ARRAY_SIZE) matching the current row of A. Flows south from top
    input int32_t bias_data [ARRAY_SIZE],
    input logic   bias_valid,

    // Outputs: (ARRAY_SIZE sized) row of the (N x ARRAY_SIZE) output matrix of AW
    // + B drop out back-to-back (after delay of DRAIN_LATENCY)
    output int32_t drain_data [ARRAY_SIZE],
    output logic   drain_valid,
    output logic   ready
);
  // Total latency from AFTER the cycle of initial input of Row 0 to
  // the cycle where there is valid output (at drain_data).
  //  Array Traversal: (ARRAY_SIZE * PE_LATENCY) cycles to flow South through DSPs.
  //  Skew then Deskew: (ARRAY_SIZE - 1) cycles for staggered input/output timing
  localparam int DRAIN_LATENCY = (ARRAY_SIZE * PE_LATENCY) + ARRAY_SIZE - 1;

  // Validate local timing logic against global package
  generate
    if (DRAIN_LATENCY != SYSTOLIC_ARRAY_LATENCY) begin : gen_array_latency_check
      COMPILE_ERROR_ARRAY_LATENCY_mismatch illegal_inst ();
      initial
        $fatal(1, "ARRAY_LATENCY must be %0d, got %0d", DRAIN_LATENCY, SYSTOLIC_ARRAY_LATENCY);
    end
  endgenerate

  // Activation input skewing: We delay the entrance into a leftmost pe of the systolic
  // grid that is at row r, corresponding to column r of the activation matrix by
  // r * PE_LATENCY
  int8_t act_skewed      [ARRAY_SIZE];
  logic  act_valid_skewed[ARRAY_SIZE];

  genvar r;
  generate
    for (r = 0; r < ARRAY_SIZE; r++) begin : gen_act_skew
      if (r == 0) begin : gen_act_first_row_no_skew
        assign act_skewed[0]       = act_data[0];
        assign act_valid_skewed[0] = act_valid;
      end else begin : gen_act_skew_main
        localparam int in_delay = r * PE_LATENCY;
        int8_t act_pipe[in_delay];
        logic  vld_pipe[in_delay];

        always_ff @(posedge clk) begin
          if (rst) begin
            act_pipe <= '{default: '0};
            vld_pipe <= '{default: '0};
          end else begin
            act_pipe[0] <= act_data[r];
            vld_pipe[0] <= act_valid;
            for (int i = 1; i < in_delay; i++) begin
              act_pipe[i] <= act_pipe[i-1];
              vld_pipe[i] <= vld_pipe[i-1];
            end
          end
        end
        assign act_skewed[r]       = act_pipe[in_delay-1];
        assign act_valid_skewed[r] = vld_pipe[in_delay-1];
      end
    end
  endgenerate

  // Bias input skewing: We delay column c of the weight matrix by c cycles
  int32_t bias_skewed[ARRAY_SIZE];
  logic bias_valid_skewed[ARRAY_SIZE];

  genvar b_col;
  generate
    for (b_col = 0; b_col < ARRAY_SIZE; b_col++) begin : gen_bias_skew
      if (b_col == 0) begin : gen_bias_first_col_no_skew
        assign bias_skewed[0] = bias_data[0];
        assign bias_valid_skewed[0] = bias_valid;
      end else begin : gen_bias_skew_main
        int32_t bias_pipe[b_col];
        logic   vld_pipe [b_col];

        always_ff @(posedge clk) begin
          if (rst) begin
            bias_pipe <= '{default: '0};
            vld_pipe  <= '{default: '0};
          end else begin
            bias_pipe[0] <= bias_data[b_col];
            vld_pipe[0]  <= bias_valid;
            for (int i = 1; i < b_col; i++) begin
              bias_pipe[i] <= bias_pipe[i-1];
              vld_pipe[i]  <= vld_pipe[i-1];
            end
          end
        end
        assign bias_skewed[b_col] = bias_pipe[b_col-1];
        assign bias_valid_skewed[b_col] = vld_pipe[b_col-1];
      end
    end
  endgenerate

  // Systolic interconnect

  // Horizontal wires
  int8_t  act_h         [  ARRAY_SIZE][ARRAY_SIZE+1];
  logic   act_h_valid   [  ARRAY_SIZE][ARRAY_SIZE+1];
  // Vertical wires
  int8_t  weight_v      [ARRAY_SIZE+1][  ARRAY_SIZE];
  logic   weight_v_valid[ARRAY_SIZE+1][  ARRAY_SIZE];

  int32_t acc_v         [ARRAY_SIZE+1][  ARRAY_SIZE];
  logic   acc_v_valid   [ARRAY_SIZE+1][  ARRAY_SIZE];


  // Drive boundaries with internally skewed input lines
  always_comb begin
    for (int row = 0; row < ARRAY_SIZE; row++) begin
      act_h[row][0]       = act_skewed[row];
      act_h_valid[row][0] = act_valid_skewed[row];
    end
    for (int col = 0; col < ARRAY_SIZE; col++) begin
      weight_v[ARRAY_SIZE][col]       = weight_data[col];
      weight_v_valid[ARRAY_SIZE][col] = weight_valid;
      // Feed the skewed biases down the top accumulator input lines
      acc_v[0][col]                   = bias_skewed[col];
      acc_v_valid[0][col]             = bias_valid_skewed[col];
    end
  end

  // Physical Grid Instantiation
  logic weight_hold[ARRAY_SIZE];

  genvar row, col;
  generate
    for (row = 0; row < ARRAY_SIZE; row++) begin : gen_row
      for (col = 0; col < ARRAY_SIZE; col++) begin : gen_col
        pe pe_inst (
            .clk             (clk),
            .rst             (rst),
            .weight_in       (weight_v[row+1][col]),
            .weight_in_valid (weight_v_valid[row+1][col]),
            .weight_out      (weight_v[row][col]),
            .weight_out_valid(weight_v_valid[row][col]),
            .weight_hold     (weight_hold[row]),
            .act_in          (act_h[row][col]),
            .act_in_valid    (act_h_valid[row][col]),
            .act_out         (act_h[row][col+1]),
            .act_out_valid   (act_h_valid[row][col+1]),
            .acc_in          (acc_v[row][col]),
            .acc_in_valid    (acc_v_valid[row][col]),
            .acc_out         (acc_v[row+1][col]),
            .acc_out_valid   (acc_v_valid[row+1][col])
        );
      end
    end
  endgenerate

  //Output deskewing: column c of output is delayed by ARRAY_SIZE - 1 - c
  int32_t deskewed_out[ARRAY_SIZE];
  logic   deskewed_vld[ARRAY_SIZE];

  genvar d_col;
  generate
    for (d_col = 0; d_col < ARRAY_SIZE; d_col++) begin : gen_output_deskew

      localparam int out_delay = (ARRAY_SIZE - 1) - d_col;
      if (out_delay == 0) begin : gen_output_last_col_no_skew
        assign deskewed_out[d_col] = acc_v[ARRAY_SIZE][d_col];
        assign deskewed_vld[d_col] = acc_v_valid[ARRAY_SIZE][d_col];
      end else begin : gen_output_skew_main
        int32_t out_pipe[out_delay];
        logic   vld_pipe[out_delay];

        always_ff @(posedge clk) begin
          if (rst) begin
            out_pipe <= '{default: '0};
            vld_pipe <= '{default: '0};
          end else begin
            out_pipe[0] <= acc_v[ARRAY_SIZE][d_col];
            vld_pipe[0] <= acc_v_valid[ARRAY_SIZE][d_col];
            for (int i = 1; i < out_delay; i++) begin
              out_pipe[i] <= out_pipe[i-1];
              vld_pipe[i] <= vld_pipe[i-1];
            end
          end
        end
        assign deskewed_out[d_col] = out_pipe[out_delay-1];
        assign deskewed_vld[d_col] = vld_pipe[out_delay-1];
      end
    end
  endgenerate

  assign drain_data = deskewed_out;
  // All valid column should be identical given the skews
  always_comb begin
    drain_valid = 1'b1;
    for (int i = 0; i < ARRAY_SIZE; i++) drain_valid &= deskewed_vld[i];
  end

  typedef enum logic {
    STANDARD,    // idling/computing
    WEIGHT_LOAD
  } state_t;

  state_t state;

  logic [$clog2(ARRAY_SIZE)-1:0] cfg_cnt;

  always_comb begin
    for (int row = 0; row < ARRAY_SIZE; row++) begin
      if (state == WEIGHT_LOAD) begin
        weight_hold[row] = (int'(cfg_cnt) < (int'(ARRAY_SIZE) - 1 - row));
      end else begin
        weight_hold[row] = 1'b1;
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      state   <= STANDARD;
      cfg_cnt <= '0;
      ready   <= 1'b1;
    end else begin
      case (state)
        STANDARD: begin
          ready <= 1'b1;
          if (weight_valid) begin
            ready   <= 1'b0;
            cfg_cnt <= 1;
            state   <= WEIGHT_LOAD;
          end
        end
        WEIGHT_LOAD: begin
          ready <= 1'b0;
          if (cfg_cnt == ARRAY_SIZE[$clog2(ARRAY_SIZE)-1:0] - 1) begin
            state <= STANDARD;
          end else begin
            cfg_cnt <= cfg_cnt + 1;
          end
        end
        default: state <= STANDARD;
      endcase
    end
  end

endmodule
