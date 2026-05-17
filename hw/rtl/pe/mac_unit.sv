`timescale 1ns / 1ps
import npu_pkg::*;

module mac_unit (
    input logic clk,
    input logic rst,

    // Weight loading chain: flows south during configuration
    input  int8_t weight_in,
    input  logic  weight_in_valid,
    output int8_t weight_out,
    output logic  weight_out_valid,

    // Activation stream: flows east during computation
    input  int8_t act_in,
    output int8_t act_out,
    input  logic  act_in_valid,
    output logic  act_out_valid,

    // Partial sum stream: flows south during computation
    input  int32_t acc_in,
    output int32_t acc_out,
    input  logic   acc_in_valid,
    output logic   acc_out_valid
);

  int8_t weight_reg;
  always_ff @(posedge clk) begin
    if (rst) begin
      weight_reg       <= '0;
      weight_out       <= '0;
      weight_out_valid <= '0;
    end else begin
      if (weight_in_valid) weight_reg <= weight_in;
      // during configuration phase to pass down the weights
      weight_out       <= weight_in;
      weight_out_valid <= weight_in_valid;
    end
  end

  // Pipeline depth
  //   Synthesis:   AREG(1) + MREG(1) + PREG(1)
  //   Simulation:  stage0  + stage1  + stage2
  localparam int PIPE_DEPTH = 3;

  // npu_pkg::MAC_LATENCY is the system-wide constant used by all other
  // modules that schedule around the PE pipeline. PIPE_DEPTH is the
  // ground truth defined here ijn accordance to the physical implementation.
  // They cannot be derived from each other directly (circular dependency),
  // so this generate block enforces agreement at elaboration time.
  generate
    if (MAC_LATENCY != PIPE_DEPTH) begin : gen_mac_latency_check
      // Kills synthesis
      COMPILE_ERROR_MAC_LATENCY_does_not_match_mac_implementation illegal_inst ();
      // Kills simulation
      initial $fatal(1, "MAC_LATENCY must be %0d, got %0d", PIPE_DEPTH, npu_pkg::MAC_LATENCY);
    end
  endgenerate

  // MAC_LATENCY directly affects this logic
  logic compute_enable;
  assign compute_enable = act_in_valid && acc_in_valid;

  // Pipeline delays for internal DSP stages
  logic compute_enable_d;
  logic compute_enable_d_d;

  always_ff @(posedge clk) begin
    if (rst) begin
      compute_enable_d   <= '0;
      compute_enable_d_d <= '0;
    end else begin
      compute_enable_d   <= compute_enable;
      compute_enable_d_d <= compute_enable_d;
    end
  end

`ifdef SYNTHESIS

  logic [29:0] dsp_a;
  logic [17:0] dsp_b;
  logic [47:0] dsp_p;

  assign dsp_a = 30'(signed'(weight_reg));
  assign dsp_b = 18'(signed'(act_in));

  logic [47:0] dsp_c;

  // Need delay one cycle
  always_ff @(posedge clk) begin
    if (rst) begin
      dsp_c <= '0;
    end else if (compute_enable) begin
      dsp_c <= 48'(signed'(acc_in));
    end
  end

  DSP48E1 #(
      .AREG         (1),
      .BREG         (1),
      .MREG         (1),
      .PREG         (1),
      .CREG         (1),
      .DREG         (0),
      .ADREG        (0),
      .USE_MULT     ("MULTIPLY"),
      .USE_DPORT    ("FALSE"),
      .OPMODEREG    (0),
      .ALUMODEREG   (0),
      .INMODEREG    (0),
      .CARRYINREG   (0),
      .CARRYINSELREG(0)
  ) dsp_inst (
      .CLK           (clk),
      .A             (dsp_a),
      .B             (dsp_b),
      .C             (dsp_c),
      .D             (25'b0),
      .ACIN          (30'b0),
      .BCIN          (18'b0),
      .PCIN          (48'b0),
      .MULTSIGNIN    (1'b0),
      .CARRYCASCIN   (1'b0),
      .CARRYIN       (1'b0),
      .CARRYINSEL    (3'b000),
      .OPMODE        (7'b011_01_01),
      .ALUMODE       (4'b0000),
      .INMODE        (5'b00000),
      .P             (dsp_p),
      .CEA1          (1'b0),
      .CEA2          (compute_enable),
      .CEB1          (1'b0),
      .CEB2          (compute_enable),
      .CEC           (compute_enable_d),
      .CED           (1'b0),
      .CEAD          (1'b0),
      .CEM           (compute_enable_d),
      .CEP           (compute_enable_d_d),
      .CEALUMODE     (1'b0),
      .CECTRL        (1'b0),
      .CEINMODE      (1'b0),
      .CECARRYIN     (1'b0),
      .RSTA          (rst),
      .RSTB          (rst),
      .RSTC          (rst),
      .RSTD          (rst),
      .RSTM          (rst),
      .RSTP          (rst),
      .RSTALUMODE    (rst),
      .RSTCTRL       (rst),
      .RSTINMODE     (rst),
      .RSTALLCARRYIN (rst),
      .ACOUT         (),
      .BCOUT         (),
      .PCOUT         (),
      .MULTSIGNOUT   (),
      .CARRYCASCOUT  (),
      .CARRYOUT      (),
      .OVERFLOW      (),
      .UNDERFLOW     (),
      .PATTERNDETECT (),
      .PATTERNBDETECT()
  );


  assign acc_out = dsp_p[ACC_WIDTH-1:0];

`else

  // Behavioural model for DSP slice
  //   Stage 0 (AREG/BREG/CREG): latch inputs
  //   Stage 1 (MREG):      multiply
  //   Stage 2 (PREG):      accumulate
  int8_t stage0_a, stage0_b;
  int32_t stage0_c;
  int32_t stage1_c;
  int16_t stage1_m;
  int32_t stage2_p;

  always_ff @(posedge clk) begin

    if (rst) begin
      stage0_a <= '0;
      stage0_b <= '0;
      stage0_c <= '0;
    end else if (compute_enable) begin
      stage0_a <= weight_reg;
      stage0_b <= act_in;
      stage0_c <= acc_in;
    end

    if (rst) begin
      stage1_m <= '0;
      stage1_c <= '0;
    end else if (compute_enable_d) begin
      stage1_m <= int16_t'(stage0_a) * int16_t'(stage0_b);
      stage1_c <= stage0_c;
    end

    if (rst) begin
      stage2_p <= '0;
    end else if (compute_enable_d_d) begin
      stage2_p <= stage1_c + int32_t'(stage1_m);
    end
  end

  assign acc_out = stage2_p;

`endif

  // quickly pump out the act_out, so can move fast in east direction
  int8_t act_reg;
  logic  act_valid_reg;
  always_ff @(posedge clk) begin
    act_reg <= rst ? '0 : act_in;
    act_valid_reg <= rst ? '0 : act_in_valid;
  end
  assign act_out = act_reg;
  assign act_out_valid = act_valid_reg;

  // delay acc_valid, so next PE receive accurately receive the DSP
  // accumulation
  logic acc_out_valid_pipe[PIPE_DEPTH];

  always_ff @(posedge clk) begin
    if (rst) begin
      foreach (acc_out_valid_pipe[i]) acc_out_valid_pipe[i] <= '0;
    end else begin
      acc_out_valid_pipe[0] <= acc_in_valid;
      for (int i = 1; i < PIPE_DEPTH; i++) begin
        acc_out_valid_pipe[i] <= acc_out_valid_pipe[i-1];
      end
    end
  end

  assign acc_out_valid = acc_out_valid_pipe[PIPE_DEPTH-1];

endmodule
