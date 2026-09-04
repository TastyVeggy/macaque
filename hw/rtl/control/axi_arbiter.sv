`timescale 1ns / 1ps

module axi_arbiter #(
    parameter int ADDR_W = 28,
    parameter int ID_W   = 4,
    parameter int DATA_W = 64
) (
    input logic clk,
    input logic rst,

    // Slave port 0 - master 0 connects here (higher priority)
    input  logic [    ID_W-1:0] s0_awid,
    input  logic [  ADDR_W-1:0] s0_awaddr,
    input  logic [         7:0] s0_awlen,
    input  logic [         2:0] s0_awsize,
    input  logic [         1:0] s0_awburst,
    input  logic [         0:0] s0_awlock,
    input  logic [         3:0] s0_awcache,
    input  logic [         2:0] s0_awprot,
    input  logic [         3:0] s0_awqos,
    input  logic                s0_awvalid,
    output logic                s0_awready,
    input  logic [  DATA_W-1:0] s0_wdata,
    input  logic [DATA_W/8-1:0] s0_wstrb,
    input  logic                s0_wlast,
    input  logic                s0_wvalid,
    output logic                s0_wready,
    output logic [    ID_W-1:0] s0_bid,
    output logic [         1:0] s0_bresp,
    output logic                s0_bvalid,
    input  logic                s0_bready,
    input  logic [    ID_W-1:0] s0_arid,
    input  logic [  ADDR_W-1:0] s0_araddr,
    input  logic [         7:0] s0_arlen,
    input  logic [         2:0] s0_arsize,
    input  logic [         1:0] s0_arburst,
    input  logic [         0:0] s0_arlock,
    input  logic [         3:0] s0_arcache,
    input  logic [         2:0] s0_arprot,
    input  logic [         3:0] s0_arqos,
    input  logic                s0_arvalid,
    output logic                s0_arready,
    output logic [    ID_W-1:0] s0_rid,
    output logic [  DATA_W-1:0] s0_rdata,
    output logic [         1:0] s0_rresp,
    output logic                s0_rlast,
    output logic                s0_rvalid,
    input  logic                s0_rready,

    // Slave port 1 - master 1 connects here
    input  logic [    ID_W-1:0] s1_awid,
    input  logic [  ADDR_W-1:0] s1_awaddr,
    input  logic [         7:0] s1_awlen,
    input  logic [         2:0] s1_awsize,
    input  logic [         1:0] s1_awburst,
    input  logic [         0:0] s1_awlock,
    input  logic [         3:0] s1_awcache,
    input  logic [         2:0] s1_awprot,
    input  logic [         3:0] s1_awqos,
    input  logic                s1_awvalid,
    output logic                s1_awready,
    input  logic [  DATA_W-1:0] s1_wdata,
    input  logic [DATA_W/8-1:0] s1_wstrb,
    input  logic                s1_wlast,
    input  logic                s1_wvalid,
    output logic                s1_wready,
    output logic [    ID_W-1:0] s1_bid,
    output logic [         1:0] s1_bresp,
    output logic                s1_bvalid,
    input  logic                s1_bready,
    input  logic [    ID_W-1:0] s1_arid,
    input  logic [  ADDR_W-1:0] s1_araddr,
    input  logic [         7:0] s1_arlen,
    input  logic [         2:0] s1_arsize,
    input  logic [         1:0] s1_arburst,
    input  logic [         0:0] s1_arlock,
    input  logic [         3:0] s1_arcache,
    input  logic [         2:0] s1_arprot,
    input  logic [         3:0] s1_arqos,
    input  logic                s1_arvalid,
    output logic                s1_arready,
    output logic [    ID_W-1:0] s1_rid,
    output logic [  DATA_W-1:0] s1_rdata,
    output logic [         1:0] s1_rresp,
    output logic                s1_rlast,
    output logic                s1_rvalid,
    input  logic                s1_rready,

    // Master port - drives MIG's s_axi slave interface
    output logic [    ID_W-1:0] m_awid,
    output logic [  ADDR_W-1:0] m_awaddr,
    output logic [         7:0] m_awlen,
    output logic [         2:0] m_awsize,
    output logic [         1:0] m_awburst,
    output logic [         0:0] m_awlock,
    output logic [         3:0] m_awcache,
    output logic [         2:0] m_awprot,
    output logic [         3:0] m_awqos,
    output logic                m_awvalid,
    input  logic                m_awready,
    output logic [  DATA_W-1:0] m_wdata,
    output logic [DATA_W/8-1:0] m_wstrb,
    output logic                m_wlast,
    output logic                m_wvalid,
    input  logic                m_wready,
    input  logic [    ID_W-1:0] m_bid,
    input  logic [         1:0] m_bresp,
    input  logic                m_bvalid,
    output logic                m_bready,
    output logic [    ID_W-1:0] m_arid,
    output logic [  ADDR_W-1:0] m_araddr,
    output logic [         7:0] m_arlen,
    output logic [         2:0] m_arsize,
    output logic [         1:0] m_arburst,
    output logic [         0:0] m_arlock,
    output logic [         3:0] m_arcache,
    output logic [         2:0] m_arprot,
    output logic [         3:0] m_arqos,
    output logic                m_arvalid,
    input  logic                m_arready,
    input  logic [    ID_W-1:0] m_rid,
    input  logic [  DATA_W-1:0] m_rdata,
    input  logic [         1:0] m_rresp,
    input  logic                m_rlast,
    input  logic                m_rvalid,
    output logic                m_rready
);

  // Write arbitration
  logic wr_busy;  // a write is in progress (AW granted, B not yet done)
  logic wr_owner;  // 0 = master 0, 1 = master 1

  // Priority: master 0 wins if both request while idle
  wire  aw_req = !wr_busy && (s0_awvalid || s1_awvalid);
  wire  aw_sel = s0_awvalid ? 1'b0 : 1'b1;  // master 0 priority

  assign m_awvalid = wr_busy ? 1'b0 : (s0_awvalid || s1_awvalid);
  assign m_awaddr  = (aw_sel == 1'b1) ? s1_awaddr : s0_awaddr;
  assign m_awid    = (aw_sel == 1'b1) ? s1_awid   : s0_awid;
  assign m_awlen   = (aw_sel == 1'b1) ? s1_awlen  : s0_awlen;
  assign m_awsize  = (aw_sel == 1'b1) ? s1_awsize : s0_awsize;
  assign m_awburst = (aw_sel == 1'b1) ? s1_awburst: s0_awburst;
  assign m_awlock  = (aw_sel == 1'b1) ? s1_awlock : s0_awlock;
  assign m_awcache = (aw_sel == 1'b1) ? s1_awcache: s0_awcache;
  assign m_awprot  = (aw_sel == 1'b1) ? s1_awprot : s0_awprot;
  assign m_awqos   = (aw_sel == 1'b1) ? s1_awqos  : s0_awqos;

  assign s0_awready = !wr_busy && (aw_sel == 1'b0) && m_awready;
  assign s1_awready = !wr_busy && (aw_sel == 1'b1) && m_awready;

  // W follows the granted owner
  assign m_wdata  = (wr_owner == 1'b1) ? s1_wdata  : s0_wdata;
  assign m_wstrb  = (wr_owner == 1'b1) ? s1_wstrb  : s0_wstrb;
  assign m_wlast  = (wr_owner == 1'b1) ? s1_wlast  : s0_wlast;
  assign m_wvalid = (wr_owner == 1'b1) ? s1_wvalid : s0_wvalid;
  assign s0_wready = (wr_owner == 1'b0) ? m_wready : 1'b0;
  assign s1_wready = (wr_owner == 1'b1) ? m_wready : 1'b0;

  // B goes back to the owner
  assign s0_bvalid = (wr_owner == 1'b0) ? m_bvalid : 1'b0;
  assign s1_bvalid = (wr_owner == 1'b1) ? m_bvalid : 1'b0;
  assign s0_bid    = m_bid;
  assign s1_bid    = m_bid;
  assign s0_bresp  = m_bresp;
  assign s1_bresp  = m_bresp;
  assign m_bready  = (wr_owner == 1'b1) ? s1_bready : s0_bready;

  always_ff @(posedge clk) begin
    if (rst) begin
      wr_busy  <= 1'b0;
      wr_owner <= 1'b0;
    end else begin
      if (aw_req && m_awready) begin
        wr_busy  <= 1'b1;
        wr_owner <= aw_sel;
      end else if (wr_busy && m_bvalid && m_bready) begin
        wr_busy <= 1'b0;
      end
    end
  end

  //Read arbitration
  logic rd_busy;
  logic rd_owner;

  wire  ar_req = !rd_busy && (s0_arvalid || s1_arvalid);
  wire  ar_sel = s0_arvalid ? 1'b0 : 1'b1;

  assign m_arvalid = rd_busy ? 1'b0 : (s0_arvalid || s1_arvalid);
  assign m_araddr  = (ar_sel == 1'b1) ? s1_araddr : s0_araddr;
  assign m_arid    = (ar_sel == 1'b1) ? s1_arid   : s0_arid;
  assign m_arlen   = (ar_sel == 1'b1) ? s1_arlen  : s0_arlen;
  assign m_arsize  = (ar_sel == 1'b1) ? s1_arsize : s0_arsize;
  assign m_arburst = (ar_sel == 1'b1) ? s1_arburst: s0_arburst;
  assign m_arlock  = (ar_sel == 1'b1) ? s1_arlock : s0_arlock;
  assign m_arcache = (ar_sel == 1'b1) ? s1_arcache: s0_arcache;
  assign m_arprot  = (ar_sel == 1'b1) ? s1_arprot : s0_arprot;
  assign m_arqos   = (ar_sel == 1'b1) ? s1_arqos  : s0_arqos;

  assign s0_arready = !rd_busy && (ar_sel == 1'b0) && m_arready;
  assign s1_arready = !rd_busy && (ar_sel == 1'b1) && m_arready;

  // R goes back to the owner
  assign s0_rvalid = (rd_owner == 1'b0) ? m_rvalid : 1'b0;
  assign s1_rvalid = (rd_owner == 1'b1) ? m_rvalid : 1'b0;
  assign s0_rid    = m_rid;
  assign s1_rid    = m_rid;
  assign s0_rdata  = m_rdata;
  assign s1_rdata  = m_rdata;
  assign s0_rresp  = m_rresp;
  assign s1_rresp  = m_rresp;
  assign s0_rlast  = m_rlast;
  assign s1_rlast  = m_rlast;
  assign m_rready  = (rd_owner == 1'b1) ? s1_rready : s0_rready;

  always_ff @(posedge clk) begin
    if (rst) begin
      rd_busy  <= 1'b0;
      rd_owner <= 1'b0;
    end else begin
      if (ar_req && m_arready) begin
        rd_busy  <= 1'b1;
        rd_owner <= ar_sel;
      end else if (rd_busy && m_rvalid && m_rready && m_rlast) begin
        rd_busy <= 1'b0;
      end
    end
  end

endmodule
