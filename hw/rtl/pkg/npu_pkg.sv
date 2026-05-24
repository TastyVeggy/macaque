package npu_pkg;

  parameter int ARRAY_SIZE = 14;
  parameter int BRAM_DEPTH = 256;
  parameter int BANK_SEL_SETTLE_CYCLES = 8;
  parameter int SYS_CLK_FREQ = 50_000_000;
  parameter int UART_BAUD_DEFAULT = 115_200;

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


  // BRAM
  parameter int DTYPE_BRAM_ADDR_W = $clog2(BRAM_DEPTH);

  typedef logic signed [DTYPE_BRAM_ADDR_W-1:0] bram_addr_t;

  // Processing element: Total cycles from input to valid acc_out
  parameter int PE_LATENCY = 3;
  // Systolic array: Total cycles from initial input row to first valid drain_data.
  parameter int SYSTOLIC_ARRAY_LATENCY = ARRAY_SIZE * PE_LATENCY + ARRAY_SIZE - 1;

  typedef enum logic [3:0] {
    OP_LOAD_WEIGHT = 4'h0,
    OP_LOAD_INPUT  = 4'h1,
    OP_MATMUL      = 4'h2,
    OP_ACTIVATE    = 4'h3,
    OP_STORE       = 4'h4,
    OP_DMA_RD      = 4'h5,
    OP_DMA_WR      = 4'h6,
    OP_SYNC        = 4'h7
  } opcode_t;

endpackage
