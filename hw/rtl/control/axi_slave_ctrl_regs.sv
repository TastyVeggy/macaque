`timescale 1ns / 1ps

module axi_slave_ctrl_regs #(
    parameter int C_S_AXI_DATA_WIDTH = 32,
    parameter int C_S_AXI_ADDR_WIDTH = 6
) (
    output logic npu_start,
    output logic npu_reset,

    input logic npu_busy,
    input logic npu_done,
    input logic npu_error,
    input logic npu_ready,

    output logic [31:0] instr_addr,

    output logic [31:0] instr_len,

    output logic pmu_enable,
    output logic pmu_clear,

    input logic [63:0] pmu_cycles,

    input logic [31:0] pmu_compute,

    input logic [31:0] pmu_stall,

    input logic [31:0] pmu_dma_bytes_rd,

    input logic [31:0] pmu_dma_bytes_wr,

    input  logic                            S_AXI_ACLK,
    input  logic                            S_AXI_ARESETN,
    input  logic [  C_S_AXI_ADDR_WIDTH-1:0] S_AXI_AWADDR,
    input  logic [                     2:0] S_AXI_AWPROT,
    input  logic                            S_AXI_AWVALID,
    output logic                            S_AXI_AWREADY,
    input  logic [  C_S_AXI_DATA_WIDTH-1:0] S_AXI_WDATA,
    input  logic [C_S_AXI_DATA_WIDTH/8-1:0] S_AXI_WSTRB,
    input  logic                            S_AXI_WVALID,
    output logic                            S_AXI_WREADY,
    output logic [                     1:0] S_AXI_BRESP,
    output logic                            S_AXI_BVALID,
    input  logic                            S_AXI_BREADY,
    input  logic [  C_S_AXI_ADDR_WIDTH-1:0] S_AXI_ARADDR,
    input  logic [                     2:0] S_AXI_ARPROT,
    input  logic                            S_AXI_ARVALID,
    output logic                            S_AXI_ARREADY,
    output logic [  C_S_AXI_DATA_WIDTH-1:0] S_AXI_RDATA,
    output logic [                     1:0] S_AXI_RRESP,
    output logic                            S_AXI_RVALID,
    input  logic                            S_AXI_RREADY
);

  // Register map
  // 0x00 REG_CTRL          RW [0]=start(w1c) [1]=reset(level)
  // 0x04 REG_STATUS        RO [0]=ready [1]=busy [2]=done [3]=error
  // 0x08 REG_INSTR_ADDR    RW
  // 0x0C REG_INSTR_LEN     RW
  // 0x10 REG_PMU_CTRL      RW [0]=enable [1]=clear(w1c)
  // 0x14 REG_PMU_CYCLES_LO RO
  // 0x18 REG_PMU_CYCLES_HI RO
  // 0x1C REG_PMU_COMPUTE   RO
  // 0x20 REG_PMU_STALL     RO
  // 0x24 REG_PMU_DMA_RD    RO
  // 0x28 REG_PMU_DMA_WR    RO

  localparam int AddrLSB = (C_S_AXI_DATA_WIDTH / 32) + 1;
  localparam int AddrBits = 4;

  logic [                  31:0] reg_ctrl;
  logic [                  31:0] reg_instr_addr;
  logic [                  31:0] reg_instr_len;
  logic [                  31:0] reg_pmu_ctrl;

  // Write channel handshake
  logic                          aw_active;
  logic [C_S_AXI_ADDR_WIDTH-1:0] aw_addr_lat;

  // capture write address when handshake completes
  always_ff @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
      aw_active   <= 1'b0;
      aw_addr_lat <= '0;
    end else begin
      if (S_AXI_AWVALID && S_AXI_AWREADY) begin
        aw_addr_lat <= S_AXI_AWADDR;
        aw_active   <= ~(S_AXI_WVALID && S_AXI_WREADY);
      end else if (S_AXI_WVALID && S_AXI_WREADY) begin
        aw_active <= 1'b0;
      end
    end
  end

  // accept address and data simultaneously
  assign S_AXI_AWREADY = ~aw_active;
  assign S_AXI_WREADY  = S_AXI_AWVALID || aw_active;  // always ready to accept data

  logic                          wr_en;
  logic [C_S_AXI_ADDR_WIDTH-1:0] wr_addr;

  assign wr_en   = S_AXI_WVALID && S_AXI_WREADY;
  assign wr_addr = aw_active ? aw_addr_lat : S_AXI_AWADDR;

  always_ff @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
      S_AXI_BVALID <= 1'b0;
      S_AXI_BRESP  <= 2'b00;
    end else begin
      if (wr_en && !S_AXI_BVALID) S_AXI_BVALID <= 1'b1;
      else if (S_AXI_BREADY) S_AXI_BVALID <= 1'b0;
    end
  end

  task automatic write_word(ref logic [31:0] reg_out, input logic [31:0] wdata,
                            input logic [C_S_AXI_DATA_WIDTH/8-1:0] wstrb);
    if (wstrb[0]) reg_out[7:0] = wdata[7:0];
    if (wstrb[1]) reg_out[15:8] = wdata[15:8];
    if (wstrb[2]) reg_out[23:16] = wdata[23:16];
    if (wstrb[3]) reg_out[31:24] = wdata[31:24];
  endtask

  always_ff @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
      reg_ctrl       <= '0;
      reg_instr_addr <= '0;
      reg_instr_len  <= '0;
      reg_pmu_ctrl   <= '0;
    end else begin
      reg_ctrl[0]     <= 1'b0;  // npu_start  w1c
      reg_pmu_ctrl[1] <= 1'b0;  // pmu_clear  w1c

      if (wr_en) begin
        case (wr_addr[AddrLSB+AddrBits-1:AddrLSB])
          4'h0:    write_word(reg_ctrl, S_AXI_WDATA, S_AXI_WSTRB);
          4'h2:    write_word(reg_instr_addr, S_AXI_WDATA, S_AXI_WSTRB);
          4'h3:    write_word(reg_instr_len, S_AXI_WDATA, S_AXI_WSTRB);
          4'h4:    write_word(reg_pmu_ctrl, S_AXI_WDATA, S_AXI_WSTRB);
          default: ;  // read-only regs, ignore writes
        endcase
      end
    end
  end

  logic npu_start_pulse;
  logic pmu_clear_pulse;
  always_ff @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
      npu_start_pulse <= 1'b0;
      pmu_clear_pulse <= 1'b0;
    end else begin
      npu_start_pulse <= (wr_en && wr_addr[AddrLSB+AddrBits-1:AddrLSB] == 4'h0 && S_AXI_WDATA[0]);
      pmu_clear_pulse <= (wr_en && wr_addr[AddrLSB+AddrBits-1:AddrLSB] == 4'h4 && S_AXI_WDATA[1]);
    end
  end
  assign npu_start  = npu_start_pulse;
  assign pmu_clear  = pmu_clear_pulse;

  assign npu_reset  = reg_ctrl[1];
  assign instr_addr = reg_instr_addr;
  assign instr_len  = reg_instr_len;
  assign pmu_enable = reg_pmu_ctrl[0];

  logic [C_S_AXI_ADDR_WIDTH-1:0] rd_addr_lat;

  typedef enum logic [1:0] {
    RD_IDLE = 2'b00,
    RD_DATA = 2'b01
  } rd_state_t;

  rd_state_t rd_state;

  always_ff @(posedge S_AXI_ACLK) begin
    if (!S_AXI_ARESETN) begin
      rd_state      <= RD_IDLE;
      S_AXI_ARREADY <= 1'b1;
      S_AXI_RVALID  <= 1'b0;
      S_AXI_RRESP   <= 2'b00;
      rd_addr_lat   <= '0;
    end else begin
      case (rd_state)

        RD_IDLE: begin
          if (S_AXI_ARVALID && S_AXI_ARREADY) begin
            rd_addr_lat   <= S_AXI_ARADDR;
            S_AXI_ARREADY <= 1'b0;
            S_AXI_RVALID  <= 1'b1;
            rd_state      <= RD_DATA;
          end
        end

        RD_DATA: begin
          if (S_AXI_RVALID && S_AXI_RREADY) begin
            S_AXI_RVALID  <= 1'b0;
            S_AXI_ARREADY <= 1'b1;
            rd_state      <= RD_IDLE;
          end
        end

        default: begin
          rd_state      <= RD_IDLE;
          S_AXI_ARREADY <= 1'b1;
          S_AXI_RVALID  <= 1'b0;
        end

      endcase
    end
  end

  always_comb begin
    case (rd_addr_lat[AddrLSB+AddrBits-1:AddrLSB])
      4'h0: S_AXI_RDATA = reg_ctrl;
      4'h1: S_AXI_RDATA = {28'b0, npu_error, npu_done, npu_busy, npu_ready};
      4'h2: S_AXI_RDATA = reg_instr_addr;
      4'h3: S_AXI_RDATA = reg_instr_len;
      4'h4: S_AXI_RDATA = reg_pmu_ctrl;
      4'h5: S_AXI_RDATA = pmu_cycles[31:0];
      4'h6: S_AXI_RDATA = pmu_cycles[63:32];
      4'h7: S_AXI_RDATA = pmu_compute;
      4'h8: S_AXI_RDATA = pmu_stall;
      4'h9: S_AXI_RDATA = pmu_dma_bytes_rd;
      4'hA: S_AXI_RDATA = pmu_dma_bytes_wr;
      default: S_AXI_RDATA = 32'h0;
    endcase
  end

endmodule
