`timescale 1ns / 1ps
import npu_pkg::*;

module mac_unit (
    input  logic   clk,
    input  logic   rst,
    input  int8_t  weight_in,
    input  logic   weight_load,
    input  int8_t  act_in,
    input  logic   act_valid,
    output int8_t  act_out,
    output logic   act_valid_out,
    input  logic   clear_acc,
    output int32_t acc_out
);

  int8_t weight_reg;
  always_ff @(posedge clk) begin
    if (rst) weight_reg <= '0;
    else if (weight_load) weight_reg <= weight_in;
  end

  //   Synthesis:   AREG(1) + MREG(1) + PREG(1)
  //   Simulation:  stage0  + stage1  + stage2
  localparam int PIPE_DEPTH = 3;

`ifdef SYNTHESIS

  logic [29:0] dsp_a;
  logic [17:0] dsp_b;
  logic [47:0] dsp_p;

  assign dsp_a = 30'(signed'(weight_reg));
  assign dsp_b = 18'(signed'(act_in));

  DSP48E1 #(
      .AREG         (1),
      .BREG         (1),
      .MREG         (1),
      .PREG         (1),
      .CREG         (0),
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
      .C             (48'b0),
      .D             (25'b0),
      .ACIN          (30'b0),
      .BCIN          (18'b0),
      .PCIN          (48'b0),
      .MULTSIGNIN    (1'b0),
      .CARRYCASCIN   (1'b0),
      .CARRYIN       (1'b0),
      .CARRYINSEL    (3'b000),
      .OPMODE        (7'b010_01_01),
      .ALUMODE       (4'b0000),
      .INMODE        (5'b00000),
      .P             (dsp_p),
      .CEA1          (1'b0),
      .CEA2          (act_valid),
      .CEB1          (1'b0),
      .CEB2          (act_valid),
      .CEC           (1'b0),
      .CED           (1'b0),
      .CEAD          (1'b0),
      .CEM           (act_valid),
      .CEP           (act_valid),
      .CEALUMODE     (1'b0),
      .CECTRL        (1'b0),
      .CEINMODE      (1'b0),
      .CECARRYIN     (1'b0),
      .RSTA          (rst),
      .RSTB          (rst),
      .RSTC          (rst),
      .RSTD          (rst),
      .RSTM          (rst),
      .RSTP          (rst | clear_acc),
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
  //   Stage 0 (AREG/BREG): latch inputs
  //   Stage 1 (MREG):      multiply
  //   Stage 2 (PREG):      accumulate
  int8_t stage0_a, stage0_b;
  int16_t stage1_m;
  int32_t stage2_p;

  always_ff @(posedge clk) begin
    if (rst) begin
      stage0_a <= '0;
      stage0_b <= '0;
    end else if (act_valid) begin
      stage0_a <= weight_reg;
      stage0_b <= act_in;
    end

    if (rst) begin
      stage1_m <= '0;
    end else if (act_valid) begin
      stage1_m <= int16_t'(stage0_a) * int16_t'(stage0_b);
    end

    if (rst | clear_acc) begin
      stage2_p <= '0;
    end else if (act_valid) begin
      stage2_p <= stage2_p + int32_t'(stage1_m);
    end
  end

  assign acc_out = stage2_p;

`endif

  // delay activation, so next PE receive the activation in a manner that is
  // synchronised with the DSP accumulation
  int8_t act_pipe[PIPE_DEPTH];
  logic  val_pipe[PIPE_DEPTH];

  always_ff @(posedge clk) begin
    if (rst) begin
      foreach (act_pipe[i]) act_pipe[i] <= '0;
      foreach (val_pipe[i]) val_pipe[i] <= '0;
    end else begin
      act_pipe[0] <= act_in;
      val_pipe[0] <= act_valid;
      for (int i = 1; i < PIPE_DEPTH; i++) begin
        act_pipe[i] <= act_pipe[i-1];
        val_pipe[i] <= val_pipe[i-1];
      end
    end
  end

  assign act_out       = act_pipe[PIPE_DEPTH-1];
  assign act_valid_out = val_pipe[PIPE_DEPTH-1];

endmodule
