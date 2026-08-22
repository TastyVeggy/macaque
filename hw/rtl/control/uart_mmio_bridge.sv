`timescale 1ns / 1ps

// Protocol (little-endian):
//   'S' (0x53)                   : status read  -> StatusW bits, zero-padded to
//                                                  a whole number of bytes
//   'W' (0x57) + addr[4] + d[N]  : write        -> 1 byte 0x00 (ack)
//   'R' (0x52) + addr[4]         : read         -> N bytes d (N = DataW/8)
module uart_mmio_bridge #(
    parameter int AddrW    = 28,                         // ddr3 address width
    parameter int DataW    = 64,                         // data width
    parameter int IdW      = 4,                          // axi id width
    parameter int StatusW  = 2,                          // number of status bits returned by 'S'
    parameter int ClkFreq  = npu_pkg::SYS_CLK_FREQ,
    parameter int BaudRate = npu_pkg::UART_BAUD_DEFAULT
) (
    input logic clk,
    input logic rst,

    // UART
    input  logic rx_pin,
    output logic tx_pin,

    // Status (comes from ddr3 mig)
    input logic [StatusW-1:0] status_in,

    output logic [      7:0] reg_addr,
    output logic             reg_we,
    output logic [DataW-1:0] reg_wdata,
    input  logic [DataW-1:0] reg_rdata,

    // Instruction memory
    output logic                            im_wr_en,
    output logic [npu_pkg::IMEM_ADDR_W+2:0] im_wr_addr,
    output logic [               DataW-1:0] im_wr_data,
    output logic                            im_re,
    output logic [npu_pkg::IMEM_ADDR_W+2:0] im_rd_addr,
    input  logic [               DataW-1:0] im_rd_data,

    // AXI4 master (single-beat)
    output logic [  IdW-1:0] m_axi_awid,
    output logic [AddrW-1:0] m_axi_awaddr,
    output logic [      7:0] m_axi_awlen,
    output logic [      2:0] m_axi_awsize,
    output logic [      1:0] m_axi_awburst,
    output logic [      0:0] m_axi_awlock,
    output logic [      3:0] m_axi_awcache,
    output logic [      2:0] m_axi_awprot,
    output logic [      3:0] m_axi_awqos,
    output logic             m_axi_awvalid,
    input  logic             m_axi_awready,

    output logic [  DataW-1:0] m_axi_wdata,
    output logic [DataW/8-1:0] m_axi_wstrb,
    output logic               m_axi_wlast,
    output logic               m_axi_wvalid,
    input  logic               m_axi_wready,

    input  logic [IdW-1:0] m_axi_bid,
    input  logic [    1:0] m_axi_bresp,
    input  logic           m_axi_bvalid,
    output logic           m_axi_bready,

    output logic [  IdW-1:0] m_axi_arid,
    output logic [AddrW-1:0] m_axi_araddr,
    output logic [      7:0] m_axi_arlen,
    output logic [      2:0] m_axi_arsize,
    output logic [      1:0] m_axi_arburst,
    output logic [      0:0] m_axi_arlock,
    output logic [      3:0] m_axi_arcache,
    output logic [      2:0] m_axi_arprot,
    output logic [      3:0] m_axi_arqos,
    output logic             m_axi_arvalid,
    input  logic             m_axi_arready,

    input  logic [  IdW-1:0] m_axi_rid,
    input  logic [DataW-1:0] m_axi_rdata,
    input  logic [      1:0] m_axi_rresp,
    input  logic             m_axi_rlast,
    input  logic             m_axi_rvalid,
    output logic             m_axi_rready
);

  localparam int DataBytes = DataW / 8;  // data bytes per beat
  localparam int AxiSize = $clog2(DataBytes);  // AXI size
  localparam int ByteCntW = $clog2((DataBytes > 4) ? DataBytes : 4);  // covers 4-byte addr + data
  localparam int StatusBytes = (StatusW + 7) / 8;  // bytes returned by 'S'

  // Opcodes
  localparam logic [7:0] OpStatus = 8'h53;  // 'S'
  localparam logic [7:0] OpWrite = 8'h57;  // 'W'
  localparam logic [7:0] OpRead = 8'h52;  // 'R'

  logic [7:0] rx_data;
  logic       rx_valid;
  uart_rx #(
      .CLK_FREQ (ClkFreq),
      .BAUD_RATE(BaudRate)
  ) u_rx (
      .clk(clk),
      .rst(rst),
      .rx_pin(rx_pin),
      .rx_data(rx_data),
      .rx_valid(rx_valid)
  );

  logic [7:0] tx_data;
  logic       tx_valid;
  logic       tx_ready;
  uart_tx #(
      .CLK_FREQ (ClkFreq),
      .BAUD_RATE(BaudRate)
  ) u_tx (
      .clk(clk),
      .rst(rst),
      .tx_data(tx_data),
      .tx_valid(tx_valid),
      .tx_ready(tx_ready),
      .tx_pin(tx_pin)
  );

  typedef enum logic [3:0] {
    IDLE,
    W_ADDR,       // capture 4 address bytes
    W_DATA,       // capture DataW/8 data bytes
    REG_W,        // register write
    IMEM_W,       // instruction-memory write
    AXI_W_AW,     // write address channel
    AXI_W_W,      // write data channel
    AXI_W_B,      // write response channel
    R_ADDR,       // capture 4 address bytes
    R_DECODE,     // address settled; pick reg / imem / axi
    REG_R,        // register read (simple bus)
    IMEM_R,       // instruction-memory read (assert im_re)
    IMEM_R_WAIT,  // bram read latency; data ready next cycle
    AXI_R_AR,     // read address channel
    AXI_R_R,      // read data channel
    TX            // send response bytes back over UART
  } state_t;
  state_t                state;

  logic   [        31:0] addr;  // captured address; low AddrW bits drive AXI
  logic   [   DataW-1:0] wdata;  // captured write data
  logic   [ByteCntW-1:0] byte_cnt;  // bytes captured in the current phase
  logic   [   DataW-1:0] resp;  // response payload (status/ack/read data)
  logic   [         3:0] resp_len;  // number of response bytes to send
  logic   [         3:0] resp_idx;  // next response byte index

  always_ff @(posedge clk) begin
    if (rst) begin
      state         <= IDLE;
      addr          <= '0;
      wdata         <= '0;
      byte_cnt      <= '0;
      resp          <= '0;
      resp_len      <= '0;
      resp_idx      <= '0;
      tx_valid      <= '0;
      tx_data       <= '0;
      m_axi_awvalid <= '0;
      m_axi_wvalid  <= '0;
      m_axi_bready  <= '0;
      m_axi_arvalid <= '0;
      m_axi_rready  <= '0;
      reg_we        <= '0;
      im_wr_en      <= '0;
      im_re         <= '0;
    end else begin
      tx_valid      <= '0;
      m_axi_awvalid <= '0;
      m_axi_wvalid  <= '0;
      m_axi_bready  <= '0;
      m_axi_arvalid <= '0;
      m_axi_rready  <= '0;
      reg_we        <= '0;
      im_wr_en      <= '0;
      im_re         <= '0;

      case (state)
        IDLE: begin
          byte_cnt <= '0;
          resp_idx <= '0;
          if (rx_valid) begin
            unique case (rx_data)
              OpStatus: begin
                resp     <= DataW'(status_in);
                resp_len <= StatusBytes[3:0];
                state    <= TX;
              end
              OpWrite: begin
                addr     <= '0;
                byte_cnt <= '0;
                state    <= W_ADDR;
              end
              OpRead: begin
                addr     <= '0;
                byte_cnt <= '0;
                state    <= R_ADDR;
              end
              default: state <= IDLE;
            endcase
          end
        end

        // capture 4 address bytes (little-endian)
        W_ADDR: begin
          if (rx_valid) begin
            addr[byte_cnt*8+:8] <= rx_data;
            if (byte_cnt == 3) begin
              byte_cnt <= '0;
              state    <= W_DATA;
            end else begin
              byte_cnt <= byte_cnt + 1'b1;
            end
          end
        end

        W_DATA: begin
          if (rx_valid) begin
            wdata[byte_cnt*8+:8] <= rx_data;
            if (byte_cnt == ByteCntW'(DataBytes - 1)) begin
              if (reg_access) state <= REG_W;
              else if (imem_access) state <= IMEM_W;
              else state <= AXI_W_AW;
            end else begin
              byte_cnt <= byte_cnt + 1'b1;
            end
          end
        end

        REG_W: begin
          reg_we   <= 1'b1;
          resp     <= '0;
          resp_len <= 4'd1;
          resp_idx <= '0;
          state    <= TX;
        end

        IMEM_W: begin
          im_wr_en <= 1'b1;
          resp     <= '0;
          resp_len <= 4'd1;
          resp_idx <= '0;
          state    <= TX;
        end

        AXI_W_AW: begin
          m_axi_awvalid <= 1'b1;
          if (m_axi_awvalid && m_axi_awready) state <= AXI_W_W;
        end

        AXI_W_W: begin
          m_axi_wvalid <= 1'b1;
          if (m_axi_wvalid && m_axi_wready) state <= AXI_W_B;
        end

        AXI_W_B: begin
          m_axi_bready <= 1'b1;
          if (m_axi_bvalid && m_axi_bready) begin
            // bresp ignored for the ack (always 0x00)
            // TODO: surface an error byte
            resp     <= '0;
            resp_len <= 4'd1;
            resp_idx <= '0;
            state    <= TX;
          end
        end

        R_ADDR: begin
          if (rx_valid) begin
            addr[byte_cnt*8+:8] <= rx_data;
            if (byte_cnt == 3) begin
              state <= R_DECODE;
            end else begin
              byte_cnt <= byte_cnt + 1'b1;
            end
          end
        end

        R_DECODE: begin
          if (reg_access) state <= REG_R;
          else if (imem_access) state <= IMEM_R;
          else state <= AXI_R_AR;
        end

        REG_R: begin
          resp     <= reg_rdata;
          resp_len <= DataBytes[3:0];
          resp_idx <= '0;
          state    <= TX;
        end

        IMEM_R: begin
          if (im_re) begin
            state <= IMEM_R_WAIT;  // bram read in flight; data ready next cycle
          end else begin
            im_re <= 1'b1;
          end
        end

        IMEM_R_WAIT: begin
          resp     <= im_rd_data;
          resp_len <= DataBytes[3:0];
          resp_idx <= '0;
          state    <= TX;
        end

        AXI_R_AR: begin
          m_axi_arvalid <= 1'b1;
          if (m_axi_arvalid && m_axi_arready) state <= AXI_R_R;
        end

        AXI_R_R: begin
          m_axi_rready <= 1'b1;
          if (m_axi_rvalid && m_axi_rready) begin
            resp     <= m_axi_rdata;
            resp_len <= DataBytes[3:0];
            resp_idx <= '0;
            state    <= TX;
          end
        end

        TX: begin
          tx_data  <= resp[resp_idx*8+:8];
          tx_valid <= 1'b1;
          if (tx_valid && tx_ready) begin
            if (resp_idx == (resp_len - 4'd1)) begin
              state <= IDLE;
            end else begin
              resp_idx <= resp_idx + 1'b1;
            end
          end
        end

        default: state <= IDLE;
      endcase
    end
  end

  assign m_axi_awid    = IdW'(0);
  assign m_axi_awaddr  = addr[AddrW-1:0];
  assign m_axi_awlen   = 8'd0;
  assign m_axi_awsize  = AxiSize[2:0];
  assign m_axi_awburst = 2'b01;   // INCR
  assign m_axi_awlock  = 1'b0;
  assign m_axi_awcache = 4'd0;
  assign m_axi_awprot  = 3'd0;
  assign m_axi_awqos   = 4'd0;

  assign m_axi_wdata   = wdata;
  assign m_axi_wstrb   = {DataBytes{1'b1}};
  assign m_axi_wlast   = 1'b1;

  assign m_axi_arid    = IdW'(0);
  assign m_axi_araddr  = addr[AddrW-1:0];
  assign m_axi_arlen   = 8'd0;
  assign m_axi_arsize  = AxiSize[2:0];
  assign m_axi_arburst = 2'b01;   // INCR
  assign m_axi_arlock  = 1'b0;
  assign m_axi_arcache = 4'd0;
  assign m_axi_arprot  = 3'd0;
  assign m_axi_arqos   = 4'd0;

  // MMIO region decode
  logic reg_access;
  logic imem_access;
  assign reg_access  = (addr[31:24] == npu_pkg::REG_BASE[31:24]);
  assign imem_access = (addr[31:24] == npu_pkg::IMEM_BASE[31:24]);
  assign reg_addr    = addr[7:0];
  assign reg_wdata   = wdata;
  assign im_wr_addr  = addr[npu_pkg::IMEM_ADDR_W+2:0];
  assign im_rd_addr  = addr[npu_pkg::IMEM_ADDR_W+2:0];
  assign im_wr_data  = wdata;

  wire _unused = &{1'b0, m_axi_bid, m_axi_bresp, m_axi_rid, m_axi_rresp, m_axi_rlast, 1'b0};

  if (AddrW < 32) begin : gen_unused_addr
    wire _unused_addr = &{1'b0, addr[31:AddrW], 1'b0};
  end

endmodule
