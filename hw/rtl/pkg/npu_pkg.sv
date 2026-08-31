package npu_pkg;
  parameter int ARRAY_SIZE = 14;
  parameter int BRAM_DEPTH = 256;
  parameter int BANK_SEL_SETTLE_CYCLES = 8;
  parameter int SYS_CLK_FREQ = 50_000_000;
  parameter int UART_BAUD_DEFAULT = 115_200;

  parameter int MIG_MEMCLK_FREQ = 333_333_333;
  parameter int MIG_PHY_RATIO = 4;
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
  parameter int INT32_ROW_BYTES = ARRAY_SIZE * (DTYPE_BIAS_W / 8);  // 56 (bias/out/acc)

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

  parameter int ISA_OPCODE_W = 4;
  parameter int ISA_OPCODE_L = 60;

  function automatic opcode_t decode_opcode(logic [63:0] raw);
    return opcode_t'(raw[ISA_OPCODE_L+:ISA_OPCODE_W]);
  endfunction

  parameter int ISA_DDR_ADDR_L = 28;
  parameter int ISA_DDR_ADDR_W = 28;
  parameter int ISA_BYTE_CNT_L = 12;
  parameter int ISA_BYTE_CNT_W = 16;

  typedef struct packed {
    logic [ISA_DDR_ADDR_W-1:0] ddr3_addr;   // [55:28]
    logic [ISA_BYTE_CNT_W-1:0] byte_count;  // [27:12]
  } mem_instr_t;

  function automatic mem_instr_t decode_mem_instr(logic [63:0] raw);
    mem_instr_t m;
    m.ddr3_addr  = raw[ISA_DDR_ADDR_L+:ISA_DDR_ADDR_W];
    m.byte_count = raw[ISA_BYTE_CNT_L+:ISA_BYTE_CNT_W];
    return m;
  endfunction

  typedef struct packed {
    logic [ISA_DDR_ADDR_W-1:0] ddr3_addr;            // [55:28]
    logic [ISA_BYTE_CNT_W-1:0] byte_count;           // [27:12]
    logic [3:0]                valid_bytes_per_row;  // [11:8]
    logic [7:0]                input_rows;           // [7:0]
  } load_instr_t;

  function automatic load_instr_t decode_load_instr(logic [63:0] raw);
    load_instr_t l;
    l.ddr3_addr           = raw[ISA_DDR_ADDR_L+:ISA_DDR_ADDR_W];
    l.byte_count          = raw[ISA_BYTE_CNT_L+:ISA_BYTE_CNT_W];
    l.valid_bytes_per_row = raw[11:8];
    l.input_rows          = raw[7:0];
    return l;
  endfunction

  typedef struct packed {
    logic       acc_mode;      // [59] K-tile accumulate flag
    logic       weight_hold;   // [56]
    logic [7:0] mat_row_base;  // [35:28]
    logic [7:0] tile_params;   // [7:0]
  } mat_instr_t;

  function automatic mat_instr_t decode_mat_instr(logic [63:0] raw);
    mat_instr_t m;
    m.acc_mode     = raw[59];
    m.weight_hold  = raw[56];
    m.mat_row_base = raw[35:28];
    m.tile_params  = raw[7:0];
    return m;
  endfunction

  typedef enum logic [2:0] {
    ACT_RELU        = 3'd0,
    ACT_LEAKY_RELU  = 3'd1,
    ACT_PASSTHROUGH = 3'd2
  } act_func_t;

  // Leaky-ReLU negative slope: alpha = 2^-LEAKY_RELU_SHIFT
  parameter int LEAKY_RELU_SHIFT = 4;

  // [38:26] are spare (TODO: a future per-instruction leaky-ReLU
  // shift)
  typedef struct packed {
    act_func_t   act_func;         // [58:56]
    logic [16:0] act_scale_m;      // [55:39]
    logic        act_bank_hold;    // [25]
    logic [7:0]  act_row_base;     // [24:17]
    logic [4:0]  act_scale_shift;  // [16:12]
    logic [7:0]  act_num_rows;     // [7:0]
  } act_instr_t;

  function automatic act_instr_t decode_act_instr(logic [63:0] raw);
    act_instr_t a;
    a.act_func        = act_func_t'(raw[58:56]);
    a.act_scale_m     = raw[55:39];
    a.act_bank_hold   = raw[25];
    a.act_row_base    = raw[24:17];
    a.act_scale_shift = raw[16:12];
    a.act_num_rows    = raw[7:0];
    return a;
  endfunction

  // Control register map
  // Byte offsets from REG_BASE, 64-bit registers
  parameter logic [31:0] REG_BASE = 32'h4000_0000;
  parameter logic [7:0] REG_CTRL = 8'h00;
  parameter logic [7:0] REG_STATUS = 8'h08;
  parameter logic [7:0] REG_INSTR_ADDR = 8'h10;
  parameter logic [7:0] REG_INSTR_LEN = 8'h18;
  parameter logic [7:0] REG_PMU_CTRL = 8'h20;
  parameter logic [7:0] REG_PMU_CYCLES = 8'h28;
  parameter logic [7:0] REG_PMU_COMPUTE = 8'h30;
  parameter logic [7:0] REG_PMU_STALL = 8'h38;
  parameter logic [7:0] REG_PMU_DMA_BYTES_RD = 8'h40;
  parameter logic [7:0] REG_PMU_DMA_BYTES_WR = 8'h48;

  // Instruction memory
  parameter int IMEM_DEPTH = 4096;
  parameter int IMEM_ADDR_W = $clog2(IMEM_DEPTH);  // word index width
  parameter logic [31:0] IMEM_BASE = 32'h5000_0000;

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
