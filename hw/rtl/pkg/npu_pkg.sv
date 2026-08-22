package npu_pkg;
  parameter int ARRAY_SIZE = 14;
  parameter int BRAM_DEPTH = 256;
  parameter int BANK_SEL_SETTLE_CYCLES = 8;
  parameter int SYS_CLK_FREQ = 50_000_000;
  parameter int UART_BAUD_DEFAULT = 115_200;

  parameter int MIG_MEMCLK_FREQ = 325_000_000;
  parameter int MIG_PHY_RATIO   = 4;
  parameter int MIG_UI_CLK_FREQ = MIG_MEMCLK_FREQ / MIG_PHY_RATIO;

  // In systolic array
  parameter int DTYPE_WEIGHT_W = 8;
  parameter int DTYPE_ACT_W = 8;
  parameter int DTYPE_PRODUCT_W = DTYPE_WEIGHT_W + DTYPE_ACT_W;
  parameter int DTYPE_BIAS_W = 32;
  parameter int DTYPE_ACC_W = 32;

  typedef logic signed [DTYPE_WEIGHT_W-1:0] weight_t;
  typedef logic signed [DTYPE_ACT_W-1:0] act_t;
  typedef logic signed [DTYPE_PRODUCT_W-1:0] product_t;
  typedef logic signed [DTYPE_BIAS_W-1:0] bias_t;
  typedef logic signed [DTYPE_ACC_W-1:0] acc_t;  // essentially output

  typedef weight_t weight_vec_t[ARRAY_SIZE];
  typedef act_t act_vec_t[ARRAY_SIZE];
  typedef bias_t bias_vec_t[ARRAY_SIZE];
  typedef acc_t acc_vec_t[ARRAY_SIZE];  // essentially output

  // Bytes per lane-packed buffer row (canonical DDR3 <-> buffer layout).
  parameter int INT8_ROW_BYTES = ARRAY_SIZE * (DTYPE_WEIGHT_W / 8);  // 14 (weight/act/quant)
  parameter int INT32_ROW_BYTES = ARRAY_SIZE * (DTYPE_BIAS_W / 8);   // 56 (bias/out/acc)

  // BRAM
  parameter int DTYPE_BRAM_ADDR_W = $clog2(BRAM_DEPTH);

  typedef logic signed [DTYPE_BRAM_ADDR_W-1:0] bram_addr_t;

  // Processing element: Total cycles from input to valid acc_out
  parameter int PE_LATENCY = 3;
  // Systolic array: Total cycles from initial input row to first valid drain_data.
  parameter int SYSTOLIC_ARRAY_LATENCY = ARRAY_SIZE * PE_LATENCY + ARRAY_SIZE - 1;

  typedef enum logic [3:0] {
    OP_LOAD_WEIGHT = 4'h0,
    OP_LOAD_BIAS   = 4'h1,
    OP_LOAD_INPUT  = 4'h2,
    OP_MATMUL      = 4'h3,
    OP_ACTIVATE    = 4'h4,
    OP_STORE       = 4'h5,
    OP_SYNC        = 4'h6
  } opcode_t;

  // ISA instruction field positions (per SPEC_ISA.md §1.1)
  parameter int ISA_OPCODE_W = 4;
  parameter int ISA_OPCODE_L = 60;
  parameter int ISA_ACC_MODE_B = 59;
  parameter int ISA_TARGET_H = 58;
  parameter int ISA_TARGET_L = 56;
  parameter int ISA_TARGET_W = 3;
  parameter int ISA_DDR_ADDR_H = 55;
  parameter int ISA_DDR_ADDR_L = 28;
  parameter int ISA_DDR_ADDR_W = 28;
  parameter int ISA_BYTE_CNT_H = 27;
  parameter int ISA_BYTE_CNT_L = 12;
  parameter int ISA_BYTE_CNT_W = 16;
  parameter int ISA_TILE_H = 11;
  parameter int ISA_TILE_L = 0;
  parameter int ISA_TILE_W = 12;

  // Decoded instruction struct (generic view, used by all lanes).
  // For ACTIVATE (OP_ACTIVATE) use the NAMED view decode_act_instr()/act_instr_t.
  typedef struct packed {
    opcode_t                   opcode;
    logic                      acc_mode;
    logic [ISA_TARGET_W-1:0]   target;
    logic [ISA_DDR_ADDR_W-1:0] ddr3_addr;
    logic [ISA_BYTE_CNT_W-1:0] byte_count;
    logic [ISA_TILE_W-1:0]     tile_params;
  } decoded_instr_t;

  function automatic decoded_instr_t decode_instr(logic [63:0] raw);
    decoded_instr_t d;
    d.opcode      = opcode_t'(raw[ISA_OPCODE_L+:ISA_OPCODE_W]);
    d.acc_mode    = raw[ISA_ACC_MODE_B];
    d.target      = raw[ISA_TARGET_H:ISA_TARGET_L];
    d.ddr3_addr   = raw[ISA_DDR_ADDR_H:ISA_DDR_ADDR_L];
    d.byte_count  = raw[ISA_BYTE_CNT_H:ISA_BYTE_CNT_L];
    d.tile_params = raw[ISA_TILE_H:ISA_TILE_L];
    return d;
  endfunction

  typedef enum logic [2:0] {
    ACT_RELU        = 3'd0,
    ACT_LEAKY_RELU  = 3'd1,
    ACT_PASSTHROUGH = 3'd2
  } act_func_t;

  // Leaky-ReLU negative slope: alpha = 2^-LEAKY_RELU_SHIFT
  parameter int LEAKY_RELU_SHIFT = 4;

  // Named view of an ACTIVATE instruction. These are the SAME
  // bits as the generic decoded_instr_t fields, just given opcode-appropriate
  // names. decode_act_instr() extracts them.
  typedef struct packed {
    opcode_t     opcode;           // [63:60]
    logic        acc_mode;         // [59] (unused by ACTIVATE)
    act_func_t   act_func;         // [58:56] (target)
    logic [27:0] act_scale_m;      // [55:28] (ddr3_addr)
    logic [10:0] leaky_shift;      // [27:17] (byte_count[15:5]) — per-instruction
                                   //          leaky slope (α = 2^-leaky_shift).
                                   //          NOT yet implemented: hardware uses
                                   //          the global npu_pkg::LEAKY_RELU_SHIFT.
    logic [4:0]  act_scale_shift;  // [16:12] (byte_count[4:0])
    logic [3:0]  reserved;         // [11:8]  (tile_params[11:8]) — reserved/unallocated.
    logic [7:0]  act_num_rows;     // [ 7:0]  (tile_params[7:0])
  } act_instr_t;

  function automatic act_instr_t decode_act_instr(logic [63:0] raw);
    act_instr_t a;
    a.opcode          = opcode_t'(raw[ISA_OPCODE_L+:ISA_OPCODE_W]);
    a.acc_mode        = raw[ISA_ACC_MODE_B];
    a.act_func        = act_func_t'(raw[ISA_TARGET_H:ISA_TARGET_L]);
    a.act_scale_m     = raw[ISA_DDR_ADDR_H:ISA_DDR_ADDR_L];
    a.leaky_shift     = raw[ISA_BYTE_CNT_H-:11];
    a.act_scale_shift = raw[ISA_BYTE_CNT_L+:5];
    a.reserved        = raw[ISA_TILE_H-:4];
    a.act_num_rows    = raw[ISA_TILE_L+:8];
    return a;
  endfunction

  // Control register map
  // Byte offsets from REG_BASE, 64-bit registers
  parameter logic [31:0] REG_BASE         = 32'h4000_0000;
  parameter logic [ 7:0] REG_CTRL         = 8'h00;
  parameter logic [ 7:0] REG_STATUS       = 8'h08;
  parameter logic [ 7:0] REG_INSTR_ADDR   = 8'h10;
  parameter logic [ 7:0] REG_INSTR_LEN    = 8'h18;
  parameter logic [ 7:0] REG_PMU_CTRL     = 8'h20;
  parameter logic [ 7:0] REG_PMU_CYCLES   = 8'h28;
  parameter logic [ 7:0] REG_PMU_COMPUTE  = 8'h30;
  parameter logic [ 7:0] REG_PMU_STALL    = 8'h38;
  parameter logic [ 7:0] REG_PMU_DMA_BYTES_RD = 8'h40;
  parameter logic [ 7:0] REG_PMU_DMA_BYTES_WR = 8'h48;

  // Dual-lane sequencer: per-bank slot state for dep_tracker
  typedef enum logic [1:0] {
    SLOT_EMPTY,   // available for DMA to load
    SLOT_LOADED,  // DMA finished; ready for compute to read
    SLOT_BUSY     // compute is reading
  } slot_state_t;

  // Buffer type selector (identifies which of the 3 load targets)
  typedef enum logic [1:0] {
    BUF_WEIGHT = 2'd0,
    BUF_BIAS   = 2'd1,
    BUF_ACT    = 2'd2
  } buffer_type_t;

endpackage

