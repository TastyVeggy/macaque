`timescale 1ns / 1ps

// Handle activation function and requantisation from INT32 to INT8
module activate_unit (
    input logic clk,
    input logic rst,

    // Control from sequencer
    input  logic        req,         // pulse/level to start
    output logic        done,        // held high while completed until req deasserts
    input  logic [11:0] tile_params, // number of output rows (M dimension)

    // Requantize parameters
    input logic               [27:0] scale_m,      // 28-bit unsigned fixed-point M
    input logic               [ 4:0] scale_shift,  // right shift, 0..31
    input npu_pkg::act_func_t        act_func,     // activation-function selector

    // out_buffer row this M-chunk's accumulator starts at (0 for every
    // matmul pattern except weight-hold combined with K-tiling, which keeps
    // several chunks resident in out_buffer at once).
    input npu_pkg::bram_addr_t row_base,

    // Out buffer read (INT32 accumulator input)
    output logic                ob_re,
    output npu_pkg::bram_addr_t ob_raddr,
    input  npu_pkg::acc_vec_t   ob_rdata,

    // Quant buffer write (INT8 output)
    output logic                qb_we,
    output npu_pkg::bram_addr_t qb_waddr,
    output npu_pkg::act_vec_t   qb_wdata
);

  typedef enum logic [1:0] {
    ACT_IDLE,
    ACT_PIPELINE,
    ACT_DRAIN,
    ACT_DONE
  } state_t;

  state_t state;

  logic [npu_pkg::DTYPE_BRAM_ADDR_W-1:0] act_count;
  logic [npu_pkg::DTYPE_BRAM_ADDR_W-1:0] num_rows;

  assign num_rows = tile_params[npu_pkg::DTYPE_BRAM_ADDR_W-1:0];

  logic signed [63:0] prod_r   [npu_pkg::ARRAY_SIZE];  // Stage 1 out
  logic signed [63:0] requant_r[npu_pkg::ARRAY_SIZE];  // Stage 2 out

  logic valid_a, valid_b;  // token cascade
  logic [npu_pkg::DTYPE_BRAM_ADDR_W-1:0] row_a, row_b;  // carried row index
  logic s1_consume;  // a row is being consumed this cycle

  assign s1_consume = (state == ACT_PIPELINE) && (act_count > 0 && act_count <= num_rows);

  logic signed [63:0] requant_next[npu_pkg::ARRAY_SIZE];
  npu_pkg::act_vec_t qb_wdata_next;
  logic signed [63:0] act_out;  // per-lane scratch, fully assigned each iteration

  always_comb begin
    for (int i = 0; i < npu_pkg::ARRAY_SIZE; i++) begin
      // Stage 2: round-half-up then arithmetic right shift.
      requant_next[i] = (scale_shift > 0)
        ? (prod_r[i] + (64'sd1 << (scale_shift - 1))) >>> scale_shift
        : prod_r[i] >>> scale_shift;

      // Stage 3: activation function + clamp to signed INT8.
      unique case (act_func)
        npu_pkg::ACT_RELU: act_out = (requant_r[i] < 0) ? '0 : requant_r[i];
        npu_pkg::ACT_LEAKY_RELU:
        act_out = (requant_r[i] < 0) ? (requant_r[i] >>> npu_pkg::LEAKY_RELU_SHIFT) : requant_r[i];
        default: act_out = requant_r[i];
      endcase
      if (act_out > 127) qb_wdata_next[i] = 8'sd127;
      else if (act_out < -128) qb_wdata_next[i] = -8'sd128;
      else qb_wdata_next[i] = act_out[7:0];
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      state     <= ACT_IDLE;
      done      <= '0;
      ob_re     <= '0;
      ob_raddr  <= '0;
      act_count <= '0;
      qb_we     <= '0;
      qb_waddr  <= '0;
      valid_a   <= '0;
      valid_b   <= '0;
      row_a     <= '0;
      row_b     <= '0;
      for (int i = 0; i < npu_pkg::ARRAY_SIZE; i++) begin
        prod_r[i]    <= '0;
        requant_r[i] <= '0;
        qb_wdata[i]  <= '0;
      end
    end else begin

      // Stage 1 (mult): prod_r = acc * M
      if (s1_consume) begin
        for (int i = 0; i < npu_pkg::ARRAY_SIZE; i++)
        prod_r[i] <= $signed(ob_rdata[i]) * $signed({4'b0, scale_m});
      end
      valid_a <= s1_consume;
      row_a   <= act_count - 1'b1;

      // Stage 2 (round+shift): register the combinational requant_next.
      if (valid_a) begin
        for (int i = 0; i < npu_pkg::ARRAY_SIZE; i++) requant_r[i] <= requant_next[i];
      end
      valid_b <= valid_a;
      row_b   <= row_a;

      // Stage 3: activation + clamp, then write (register the combinational next).
      qb_we   <= valid_b;
      if (valid_b) begin
        qb_waddr <= row_b;
        for (int i = 0; i < npu_pkg::ARRAY_SIZE; i++) qb_wdata[i] <= qb_wdata_next[i];
      end

      if (state == ACT_PIPELINE && act_count < num_rows) ob_raddr <= row_base + act_count + 1'b1;

      done <= '0;
      case (state)

        ACT_IDLE: begin
          if (req) begin
            act_count <= '0;
            ob_re     <= 1'b1;
            ob_raddr  <= row_base;
            state     <= ACT_PIPELINE;
          end
        end

        ACT_PIPELINE: begin
          if (act_count < num_rows) begin
            act_count <= act_count + 1'b1;
          end else begin
            ob_re <= 1'b0;  // stop reading
            state <= ACT_DRAIN;
          end
        end

        ACT_DRAIN: begin
          // Wait until the final rows have flowed out of all 3 pipeline stages.
          if (!valid_a && !valid_b && !qb_we) begin
            state <= ACT_DONE;
          end
        end

        ACT_DONE: begin
          done <= 1'b1;  // held until req deasserts
          if (!req) begin
            state <= ACT_IDLE;
          end
        end

        default: state <= ACT_IDLE;

      endcase
    end
  end

endmodule

