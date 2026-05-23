`timescale 1ns / 1ps
module pe (
    input logic clk,
    input logic rst,

    // Weight loading chain: flows south during configuration
    input npu_pkg::weight_t weight_in,
    input logic weight_in_valid,
    output npu_pkg::weight_t weight_out,
    output logic weight_out_valid,

    // If hold, the weights will be stored in the PE, and not flow out
    input logic weight_hold,

    // Activation stream: flows east during computation
    input npu_pkg::act_t act_in,
    input logic act_in_valid,
    output npu_pkg::act_t act_out,
    output logic act_out_valid,

    // Partial sum stream: flows south during computation
    input npu_pkg::acc_t acc_in,
    input logic acc_in_valid,
    output npu_pkg::acc_t acc_out,
    output logic acc_out_valid
);
  npu_pkg::weight_t weight_reg;
  always_ff @(posedge clk) begin
    if (rst) begin
      weight_reg       <= '0;
      weight_out       <= '0;
      weight_out_valid <= '0;
    end else begin
      // Latch the weight locally only if valid and NOT held
      if (weight_in_valid && !weight_hold) begin
        weight_reg <= weight_in;
      end

      // Weights will always flow
      weight_out       <= weight_in;
      weight_out_valid <= weight_in_valid;
    end
  end

  // Pipeline depth
  //   Synthesis:   AREG(1) + MREG(1) + PREG(1)
  //   Simulation:  stage0  + stage1  + stage2
  localparam int PipeDepth = 3;

  // npu_pkg::PE_LATENCY is the system-wide constant used by all other
  // modules that schedule around the PE pipeline. PipeDepth is the
  // ground truth defined here ijn accordance to the physical implementation.
  // They cannot be derived from each other directly (circular dependency),
  // so this generate block enforces agreement at elaboration time.
  generate
    import npu_pkg::PE_LATENCY;

    if (PE_LATENCY != PipeDepth) begin : gen_pe_latency_check
      // Kills synthesis
      COMPILE_ERROR_PE_LATENCY_mismatch illegal_inst ();
      // Kills simulation
      initial $fatal(1, "PE_LATENCY must be %0d, got %0d", PipeDepth, PE_LATENCY);
    end
  endgenerate

  // PE_LATENCY directly affects this logic
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


  assign acc_out = dsp_p[npu_pkg::DTYPE_ACC_W-1:0];

`else

  // Behavioural model for DSP slice
  //   Stage 0 (AREG/BREG/CREG): latch inputs
  //   Stage 1 (MREG):      multiply
  //   Stage 2 (PREG):      accumulate
  npu_pkg::weight_t stage0_a;
  npu_pkg::act_t stage0_b;
  npu_pkg::acc_t stage0_c;
  npu_pkg::acc_t stage1_c;
  npu_pkg::product_t stage1_m;
  npu_pkg::acc_t stage2_p;

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
      stage1_m <= npu_pkg::product_t'(stage0_a) * npu_pkg::product_t'(stage0_b);
      stage1_c <= stage0_c;
    end

    if (rst) begin
      stage2_p <= '0;
    end else if (compute_enable_d_d) begin
      stage2_p <= stage1_c + npu_pkg::acc_t'(stage1_m);
    end
  end

  assign acc_out = stage2_p;

`endif

  // quickly pump out the act_out, so can move fast in east direction
  npu_pkg::act_t act_reg;
  logic act_valid_reg;
  always_ff @(posedge clk) begin
    act_reg <= rst ? '0 : act_in;
    act_valid_reg <= rst ? '0 : act_in_valid;
  end
  assign act_out = act_reg;
  assign act_out_valid = act_valid_reg;

  // delay acc_valid, so next PE receive accurately receive the DSP
  // accumulation
  logic acc_out_valid_pipe[PipeDepth];

  always_ff @(posedge clk) begin
    if (rst) begin
      foreach (acc_out_valid_pipe[i]) acc_out_valid_pipe[i] <= '0;
    end else begin
      acc_out_valid_pipe[0] <= acc_in_valid;
      for (int i = 1; i < PipeDepth; i++) begin
        acc_out_valid_pipe[i] <= acc_out_valid_pipe[i-1];
      end
    end
  end

  assign acc_out_valid = acc_out_valid_pipe[PipeDepth-1];

endmodule
