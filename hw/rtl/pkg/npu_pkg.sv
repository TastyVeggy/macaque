package npu_pkg;

  localparam int DATA_WIDTH = 8;
  localparam int ACC_WIDTH = 32;

  typedef logic signed [DATA_WIDTH-1:0] int8_t;
  typedef logic signed [DATA_WIDTH*2-1:0] int16_t;
  typedef logic signed [ACC_WIDTH-1:0] int32_t;

  localparam int SYS_CLK_FREQ = 50_000_000;
  localparam int UART_BAUD_DEFAULT = 115_200;

  localparam int ARRAY_SIZE = 16;

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
