`timescale 1ns / 1ps

module axi_arbiter #(
    parameter int ADDR_W = 28,
    parameter int ID_W   = 4,
    parameter int DATA_W = 64
) (
    input logic clk,
    input logic rst,

    // -Master 0 (higher priority)
    input  logic [    ID_W-1:0] m0_awid,
    input  logic [  ADDR_W-1:0] m0_awaddr,
    input  logic [         7:0] m0_awlen,
    input  logic [         2:0] m0_awsize,
    input  logic [         1:0] m0_awburst,
    input  logic [         0:0] m0_awlock,
    input  logic [         3:0] m0_awcache,
    input  logic [         2:0] m0_awprot,
    input  logic [         3:0] m0_awqos,
    input  logic                m0_awvalid,
    output logic                m0_awready,
    input  logic [  DATA_W-1:0] m0_wdata,
    input  logic [DATA_W/8-1:0] m0_wstrb,
    input  logic                m0_wlast,
    input  logic                m0_wvalid,
    output logic                m0_wready,
    output logic [    ID_W-1:0] m0_bid,
    output logic [         1:0] m0_bresp,
    output logic                m0_bvalid,
    input  logic                m0_bready,
    input  logic [    ID_W-1:0] m0_arid,
    input  logic [  ADDR_W-1:0] m0_araddr,
    input  logic [         7:0] m0_arlen,
    input  logic [         2:0] m0_arsize,
    input  logic [         1:0] m0_arburst,
    input  logic [         0:0] m0_arlock,
    input  logic [         3:0] m0_arcache,
    input  logic [         2:0] m0_arprot,
    input  logic [         3:0] m0_arqos,
    input  logic                m0_arvalid,
    output logic                m0_arready,
    output logic [    ID_W-1:0] m0_rid,
    output logic [  DATA_W-1:0] m0_rdata,
    output logic [         1:0] m0_rresp,
    output logic                m0_rlast,
    output logic                m0_rvalid,
    input  logic                m0_rready,

    // Master 1
    input  logic [    ID_W-1:0] m1_awid,
    input  logic [  ADDR_W-1:0] m1_awaddr,
    input  logic [         7:0] m1_awlen,
    input  logic [         2:0] m1_awsize,
    input  logic [         1:0] m1_awburst,
    input  logic [         0:0] m1_awlock,
    input  logic [         3:0] m1_awcache,
    input  logic [         2:0] m1_awprot,
    input  logic [         3:0] m1_awqos,
    input  logic                m1_awvalid,
    output logic                m1_awready,
    input  logic [  DATA_W-1:0] m1_wdata,
    input  logic [DATA_W/8-1:0] m1_wstrb,
    input  logic                m1_wlast,
    input  logic                m1_wvalid,
    output logic                m1_wready,
    output logic [    ID_W-1:0] m1_bid,
    output logic [         1:0] m1_bresp,
    output logic                m1_bvalid,
    input  logic                m1_bready,
    input  logic [    ID_W-1:0] m1_arid,
    input  logic [  ADDR_W-1:0] m1_araddr,
    input  logic [         7:0] m1_arlen,
    input  logic [         2:0] m1_arsize,
    input  logic [         1:0] m1_arburst,
    input  logic [         0:0] m1_arlock,
    input  logic [         3:0] m1_arcache,
    input  logic [         2:0] m1_arprot,
    input  logic [         3:0] m1_arqos,
    input  logic                m1_arvalid,
    output logic                m1_arready,
    output logic [    ID_W-1:0] m1_rid,
    output logic [  DATA_W-1:0] m1_rdata,
    output logic [         1:0] m1_rresp,
    output logic                m1_rlast,
    output logic                m1_rvalid,
    input  logic                m1_rready,

    // Slave (MIG s_axi)
    output logic [    ID_W-1:0] s_awid,
    output logic [  ADDR_W-1:0] s_awaddr,
    output logic [         7:0] s_awlen,
    output logic [         2:0] s_awsize,
    output logic [         1:0] s_awburst,
    output logic [         0:0] s_awlock,
    output logic [         3:0] s_awcache,
    output logic [         2:0] s_awprot,
    output logic [         3:0] s_awqos,
    output logic                s_awvalid,
    input  logic                s_awready,
    output logic [  DATA_W-1:0] s_wdata,
    output logic [DATA_W/8-1:0] s_wstrb,
    output logic                s_wlast,
    output logic                s_wvalid,
    input  logic                s_wready,
    input  logic [    ID_W-1:0] s_bid,
    input  logic [         1:0] s_bresp,
    input  logic                s_bvalid,
    output logic                s_bready,
    output logic [    ID_W-1:0] s_arid,
    output logic [  ADDR_W-1:0] s_araddr,
    output logic [         7:0] s_arlen,
    output logic [         2:0] s_arsize,
    output logic [         1:0] s_arburst,
    output logic [         0:0] s_arlock,
    output logic [         3:0] s_arcache,
    output logic [         2:0] s_arprot,
    output logic [         3:0] s_arqos,
    output logic                s_arvalid,
    input  logic                s_arready,
    input  logic [    ID_W-1:0] s_rid,
    input  logic [  DATA_W-1:0] s_rdata,
    input  logic [         1:0] s_rresp,
    input  logic                s_rlast,
    input  logic                s_rvalid,
    output logic                s_rready
);

  // Write arbitration
  logic wr_busy;  // a write is in progress (AW granted, B not yet done)
  logic wr_owner;  // 0 = master 0, 1 = master 1

  // Priority: master 0 wins if both request while idle
  wire  aw_req = !wr_busy && (m0_awvalid || m1_awvalid);
  wire  aw_sel = m0_awvalid ? 1'b0 : 1'b1;  // master 0 priority

  assign s_awvalid = wr_busy ? 1'b0 : (m0_awvalid || m1_awvalid);
  assign s_awaddr  = (aw_sel == 1'b1) ? m1_awaddr : m0_awaddr;
  assign s_awid    = (aw_sel == 1'b1) ? m1_awid   : m0_awid;
  assign s_awlen   = (aw_sel == 1'b1) ? m1_awlen  : m0_awlen;
  assign s_awsize  = (aw_sel == 1'b1) ? m1_awsize : m0_awsize;
  assign s_awburst = (aw_sel == 1'b1) ? m1_awburst: m0_awburst;
  assign s_awlock  = (aw_sel == 1'b1) ? m1_awlock : m0_awlock;
  assign s_awcache = (aw_sel == 1'b1) ? m1_awcache: m0_awcache;
  assign s_awprot  = (aw_sel == 1'b1) ? m1_awprot : m0_awprot;
  assign s_awqos   = (aw_sel == 1'b1) ? m1_awqos  : m0_awqos;

  assign m0_awready = !wr_busy && (aw_sel == 1'b0) && s_awready;
  assign m1_awready = !wr_busy && (aw_sel == 1'b1) && s_awready;

  // W follows the granted owner
  assign s_wdata  = (wr_owner == 1'b1) ? m1_wdata  : m0_wdata;
  assign s_wstrb  = (wr_owner == 1'b1) ? m1_wstrb  : m0_wstrb;
  assign s_wlast  = (wr_owner == 1'b1) ? m1_wlast  : m0_wlast;
  assign s_wvalid = (wr_owner == 1'b1) ? m1_wvalid : m0_wvalid;
  assign m0_wready = (wr_owner == 1'b0) ? s_wready : 1'b0;
  assign m1_wready = (wr_owner == 1'b1) ? s_wready : 1'b0;

  // B goes back to the owner
  assign m0_bvalid = (wr_owner == 1'b0) ? s_bvalid : 1'b0;
  assign m1_bvalid = (wr_owner == 1'b1) ? s_bvalid : 1'b0;
  assign m0_bid    = s_bid;
  assign m1_bid    = s_bid;
  assign m0_bresp  = s_bresp;
  assign m1_bresp  = s_bresp;
  assign s_bready  = (wr_owner == 1'b1) ? m1_bready : m0_bready;

  always_ff @(posedge clk) begin
    if (rst) begin
      wr_busy  <= 1'b0;
      wr_owner <= 1'b0;
    end else begin
      if (aw_req && s_awready) begin
        wr_busy  <= 1'b1;
        wr_owner <= aw_sel;
      end else if (wr_busy && s_bvalid && s_bready) begin
        wr_busy <= 1'b0;
      end
    end
  end

  //Read arbitration
  logic rd_busy;
  logic rd_owner;

  wire  ar_req = !rd_busy && (m0_arvalid || m1_arvalid);
  wire  ar_sel = m0_arvalid ? 1'b0 : 1'b1;

  assign s_arvalid = rd_busy ? 1'b0 : (m0_arvalid || m1_arvalid);
  assign s_araddr  = (ar_sel == 1'b1) ? m1_araddr : m0_araddr;
  assign s_arid    = (ar_sel == 1'b1) ? m1_arid   : m0_arid;
  assign s_arlen   = (ar_sel == 1'b1) ? m1_arlen  : m0_arlen;
  assign s_arsize  = (ar_sel == 1'b1) ? m1_arsize : m0_arsize;
  assign s_arburst = (ar_sel == 1'b1) ? m1_arburst: m0_arburst;
  assign s_arlock  = (ar_sel == 1'b1) ? m1_arlock : m0_arlock;
  assign s_arcache = (ar_sel == 1'b1) ? m1_arcache: m0_arcache;
  assign s_arprot  = (ar_sel == 1'b1) ? m1_arprot : m0_arprot;
  assign s_arqos   = (ar_sel == 1'b1) ? m1_arqos  : m0_arqos;

  assign m0_arready = !rd_busy && (ar_sel == 1'b0) && s_arready;
  assign m1_arready = !rd_busy && (ar_sel == 1'b1) && s_arready;

  // R goes back to the owner
  assign m0_rvalid = (rd_owner == 1'b0) ? s_rvalid : 1'b0;
  assign m1_rvalid = (rd_owner == 1'b1) ? s_rvalid : 1'b0;
  assign m0_rid    = s_rid;
  assign m1_rid    = s_rid;
  assign m0_rdata  = s_rdata;
  assign m1_rdata  = s_rdata;
  assign m0_rresp  = s_rresp;
  assign m1_rresp  = s_rresp;
  assign m0_rlast  = s_rlast;
  assign m1_rlast  = s_rlast;
  assign s_rready  = (rd_owner == 1'b1) ? m1_rready : m0_rready;

  always_ff @(posedge clk) begin
    if (rst) begin
      rd_busy  <= 1'b0;
      rd_owner <= 1'b0;
    end else begin
      if (ar_req && s_arready) begin
        rd_busy  <= 1'b1;
        rd_owner <= ar_sel;
      end else if (rd_busy && s_rvalid && s_rready && s_rlast) begin
        rd_busy <= 1'b0;
      end
    end
  end

endmodule
