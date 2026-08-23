`timescale 1ns / 1ps

module npu_top (
    input logic clk,
    input logic rst_n,

    input  logic uart_rx,
    output logic uart_tx,

    output logic led_calib,
    output logic led_locked,

    inout  [15:0] ddr3_dq,
    inout  [ 1:0] ddr3_dqs_n,
    inout  [ 1:0] ddr3_dqs_p,
    output [13:0] ddr3_addr,
    output [ 2:0] ddr3_ba,
    output        ddr3_ras_n,
    output        ddr3_cas_n,
    output        ddr3_we_n,
    output        ddr3_reset_n,
    output [ 0:0] ddr3_ck_p,
    output [ 0:0] ddr3_ck_n,
    output [ 0:0] ddr3_cke,
    output [ 1:0] ddr3_dm,
    output [ 0:0] ddr3_odt
);

  // Clocking: 50 MHz -> 200 MHz sys_clk_i
  logic sys_clk_i;
  logic clk_wiz_locked;
  clk_wiz_0 u_clk_wiz (
      .clk_in1 (clk),
      .clk_out1(sys_clk_i),
      .reset   (1'b0),
      .locked  (clk_wiz_locked)
  );

  logic        ui_clk;
  logic        ui_clk_sync_rst;
  logic        mmcm_locked;
  logic        init_calib_complete;
  logic        aresetn;

  // AXI4 wires (arbiter slave side -> MIG s_axi)
  logic [ 3:0] m_awid;
  logic [27:0] m_awaddr;
  logic [ 7:0] m_awlen;
  logic [ 2:0] m_awsize;
  logic [ 1:0] m_awburst;
  logic [ 0:0] m_awlock;
  logic [ 3:0] m_awcache;
  logic [ 2:0] m_awprot;
  logic [ 3:0] m_awqos;
  logic        m_awvalid;
  logic        m_awready;
  logic [63:0] m_wdata;
  logic [ 7:0] m_wstrb;
  logic        m_wlast;
  logic        m_wvalid;
  logic        m_wready;
  logic [ 3:0] m_bid;
  logic [ 1:0] m_bresp;
  logic        m_bvalid;
  logic        m_bready;
  logic [ 3:0] m_arid;
  logic [27:0] m_araddr;
  logic [ 7:0] m_arlen;
  logic [ 2:0] m_arsize;
  logic [ 1:0] m_arburst;
  logic [ 0:0] m_arlock;
  logic [ 3:0] m_arcache;
  logic [ 2:0] m_arprot;
  logic [ 3:0] m_arqos;
  logic        m_arvalid;
  logic        m_arready;
  logic [ 3:0] m_rid;
  logic [63:0] m_rdata;
  logic [ 1:0] m_rresp;
  logic        m_rlast;
  logic        m_rvalid;
  logic        m_rready;

  mig_7series_0 u_mig (
      .ddr3_dq            (ddr3_dq),
      .ddr3_dqs_n         (ddr3_dqs_n),
      .ddr3_dqs_p         (ddr3_dqs_p),
      .ddr3_addr          (ddr3_addr),
      .ddr3_ba            (ddr3_ba),
      .ddr3_ras_n         (ddr3_ras_n),
      .ddr3_cas_n         (ddr3_cas_n),
      .ddr3_we_n          (ddr3_we_n),
      .ddr3_reset_n       (ddr3_reset_n),
      .ddr3_ck_p          (ddr3_ck_p),
      .ddr3_ck_n          (ddr3_ck_n),
      .ddr3_cke           (ddr3_cke),
      .ddr3_dm            (ddr3_dm),
      .ddr3_odt           (ddr3_odt),
      .sys_clk_i          (sys_clk_i),
      .ui_clk             (ui_clk),
      .ui_clk_sync_rst    (ui_clk_sync_rst),
      .mmcm_locked        (mmcm_locked),
      .aresetn            (aresetn),
      .app_sr_req         (1'b0),
      .app_ref_req        (1'b0),
      .app_zq_req         (1'b0),
      .app_sr_active      (),
      .app_ref_ack        (),
      .app_zq_ack         (),
      .s_axi_awid         (m_awid),
      .s_axi_awaddr       (m_awaddr),
      .s_axi_awlen        (m_awlen),
      .s_axi_awsize       (m_awsize),
      .s_axi_awburst      (m_awburst),
      .s_axi_awlock       (m_awlock),
      .s_axi_awcache      (m_awcache),
      .s_axi_awprot       (m_awprot),
      .s_axi_awqos        (m_awqos),
      .s_axi_awvalid      (m_awvalid),
      .s_axi_awready      (m_awready),
      .s_axi_wdata        (m_wdata),
      .s_axi_wstrb        (m_wstrb),
      .s_axi_wlast        (m_wlast),
      .s_axi_wvalid       (m_wvalid),
      .s_axi_wready       (m_wready),
      .s_axi_bready       (m_bready),
      .s_axi_bid          (m_bid),
      .s_axi_bresp        (m_bresp),
      .s_axi_bvalid       (m_bvalid),
      .s_axi_arid         (m_arid),
      .s_axi_araddr       (m_araddr),
      .s_axi_arlen        (m_arlen),
      .s_axi_arsize       (m_arsize),
      .s_axi_arburst      (m_arburst),
      .s_axi_arlock       (m_arlock),
      .s_axi_arcache      (m_arcache),
      .s_axi_arprot       (m_arprot),
      .s_axi_arqos        (m_arqos),
      .s_axi_arvalid      (m_arvalid),
      .s_axi_arready      (m_arready),
      .s_axi_rready       (m_rready),
      .s_axi_rid          (m_rid),
      .s_axi_rdata        (m_rdata),
      .s_axi_rresp        (m_rresp),
      .s_axi_rlast        (m_rlast),
      .s_axi_rvalid       (m_rvalid),
      .init_calib_complete(init_calib_complete),
      .device_temp_i      (12'h000),
      .device_temp        (),
      .sys_rst            (1'b1)
  );

  assign aresetn = ~ui_clk_sync_rst;

  logic rst_sync1, rst_sync;
  always_ff @(posedge ui_clk) begin
    rst_sync1 <= ~rst_n;
    rst_sync  <= rst_sync1;
  end
  wire                             sys_rst = rst_sync | ui_clk_sync_rst;

  logic [                     7:0] reg_addr;
  logic                            reg_we;
  logic [                    63:0] reg_wdata;
  logic [                    63:0] reg_rdata;

  logic                            im_wr_en;
  logic [npu_pkg::IMEM_ADDR_W+2:0] im_wr_addr;
  logic [                    63:0] im_wr_data;
  logic                            im_re;
  logic [npu_pkg::IMEM_ADDR_W+2:0] im_rd_addr;
  logic [                    63:0] im_rd_data;

  // bridge AXI master (arbiter m0)
  logic [                     3:0] b_awid;
  logic [                    27:0] b_awaddr;
  logic [                     7:0] b_awlen;
  logic [                     2:0] b_awsize;
  logic [                     1:0] b_awburst;
  logic [                     0:0] b_awlock;
  logic [                     3:0] b_awcache;
  logic [                     2:0] b_awprot;
  logic [                     3:0] b_awqos;
  logic                            b_awvalid;
  logic                            b_awready;
  logic [                    63:0] b_wdata;
  logic [                     7:0] b_wstrb;
  logic                            b_wlast;
  logic                            b_wvalid;
  logic                            b_wready;
  logic [                     3:0] b_bid;
  logic [                     1:0] b_bresp;
  logic                            b_bvalid;
  logic                            b_bready;
  logic [                     3:0] b_arid;
  logic [                    27:0] b_araddr;
  logic [                     7:0] b_arlen;
  logic [                     2:0] b_arsize;
  logic [                     1:0] b_arburst;
  logic [                     0:0] b_arlock;
  logic [                     3:0] b_arcache;
  logic [                     2:0] b_arprot;
  logic [                     3:0] b_arqos;
  logic                            b_arvalid;
  logic                            b_arready;
  logic [                     3:0] b_rid;
  logic [                    63:0] b_rdata;
  logic [                     1:0] b_rresp;
  logic                            b_rlast;
  logic                            b_rvalid;
  logic                            b_rready;

  // npu_core AXI master (arbiter m1)
  logic [                     3:0] n_awid;
  logic [                    27:0] n_awaddr;
  logic [                     7:0] n_awlen;
  logic [                     2:0] n_awsize;
  logic [                     1:0] n_awburst;
  logic [                     0:0] n_awlock;
  logic [                     3:0] n_awcache;
  logic [                     2:0] n_awprot;
  logic [                     3:0] n_awqos;
  logic                            n_awvalid;
  logic                            n_awready;
  logic [                    63:0] n_wdata;
  logic [                     7:0] n_wstrb;
  logic                            n_wlast;
  logic                            n_wvalid;
  logic                            n_wready;
  logic [                     3:0] n_bid;
  logic [                     1:0] n_bresp;
  logic                            n_bvalid;
  logic                            n_bready;
  logic [                     3:0] n_arid;
  logic [                    27:0] n_araddr;
  logic [                     7:0] n_arlen;
  logic [                     2:0] n_arsize;
  logic [                     1:0] n_arburst;
  logic [                     0:0] n_arlock;
  logic [                     3:0] n_arcache;
  logic [                     2:0] n_arprot;
  logic [                     3:0] n_arqos;
  logic                            n_arvalid;
  logic                            n_arready;
  logic [                     3:0] n_rid;
  logic [                    63:0] n_rdata;
  logic [                     1:0] n_rresp;
  logic                            n_rlast;
  logic                            n_rvalid;
  logic                            n_rready;

  uart_mmio_bridge #(
      .ClkFreq(npu_pkg::MIG_UI_CLK_FREQ),
      .StatusW(2)
  ) u_bridge (
      .clk          (ui_clk),
      .rst          (sys_rst),
      .rx_pin       (uart_rx),
      .tx_pin       (uart_tx),
      .status_in    ({init_calib_complete, mmcm_locked}),
      .reg_addr     (reg_addr),
      .reg_we       (reg_we),
      .reg_wdata    (reg_wdata),
      .reg_rdata    (reg_rdata),
      .im_wr_en     (im_wr_en),
      .im_wr_addr   (im_wr_addr),
      .im_wr_data   (im_wr_data),
      .im_re        (im_re),
      .im_rd_addr   (im_rd_addr),
      .im_rd_data   (im_rd_data),
      .m_axi_awid   (b_awid),
      .m_axi_awaddr (b_awaddr),
      .m_axi_awlen  (b_awlen),
      .m_axi_awsize (b_awsize),
      .m_axi_awburst(b_awburst),
      .m_axi_awlock (b_awlock),
      .m_axi_awcache(b_awcache),
      .m_axi_awprot (b_awprot),
      .m_axi_awqos  (b_awqos),
      .m_axi_awvalid(b_awvalid),
      .m_axi_awready(b_awready),
      .m_axi_wdata  (b_wdata),
      .m_axi_wstrb  (b_wstrb),
      .m_axi_wlast  (b_wlast),
      .m_axi_wvalid (b_wvalid),
      .m_axi_wready (b_wready),
      .m_axi_bid    (b_bid),
      .m_axi_bresp  (b_bresp),
      .m_axi_bvalid (b_bvalid),
      .m_axi_bready (b_bready),
      .m_axi_arid   (b_arid),
      .m_axi_araddr (b_araddr),
      .m_axi_arlen  (b_arlen),
      .m_axi_arsize (b_arsize),
      .m_axi_arburst(b_arburst),
      .m_axi_arlock (b_arlock),
      .m_axi_arcache(b_arcache),
      .m_axi_arprot (b_arprot),
      .m_axi_arqos  (b_arqos),
      .m_axi_arvalid(b_arvalid),
      .m_axi_arready(b_arready),
      .m_axi_rid    (b_rid),
      .m_axi_rdata  (b_rdata),
      .m_axi_rresp  (b_rresp),
      .m_axi_rlast  (b_rlast),
      .m_axi_rvalid (b_rvalid),
      .m_axi_rready (b_rready)
  );

  npu_core u_npu (
      .clk          (ui_clk),
      .rst_n        (~sys_rst),
      .reg_addr     (reg_addr),
      .reg_we       (reg_we),
      .reg_wdata    (reg_wdata),
      .reg_rdata    (reg_rdata),
      .imem_we      (im_wr_en),
      .imem_waddr   (im_wr_addr),
      .imem_wdata   (im_wr_data),
      .imem_re      (im_re),
      .imem_raddr   (im_rd_addr),
      .imem_rdata   (im_rd_data),
      .m_axi_awid   (n_awid),
      .m_axi_awaddr (n_awaddr),
      .m_axi_awlen  (n_awlen),
      .m_axi_awsize (n_awsize),
      .m_axi_awburst(n_awburst),
      .m_axi_awlock (n_awlock),
      .m_axi_awcache(n_awcache),
      .m_axi_awprot (n_awprot),
      .m_axi_awqos  (n_awqos),
      .m_axi_awvalid(n_awvalid),
      .m_axi_awready(n_awready),
      .m_axi_wdata  (n_wdata),
      .m_axi_wstrb  (n_wstrb),
      .m_axi_wlast  (n_wlast),
      .m_axi_wvalid (n_wvalid),
      .m_axi_wready (n_wready),
      .m_axi_bid    (n_bid),
      .m_axi_bresp  (n_bresp),
      .m_axi_bvalid (n_bvalid),
      .m_axi_bready (n_bready),
      .m_axi_arid   (n_arid),
      .m_axi_araddr (n_araddr),
      .m_axi_arlen  (n_arlen),
      .m_axi_arsize (n_arsize),
      .m_axi_arburst(n_arburst),
      .m_axi_arlock (n_arlock),
      .m_axi_arcache(n_arcache),
      .m_axi_arprot (n_arprot),
      .m_axi_arqos  (n_arqos),
      .m_axi_arvalid(n_arvalid),
      .m_axi_arready(n_arready),
      .m_axi_rid    (n_rid),
      .m_axi_rdata  (n_rdata),
      .m_axi_rresp  (n_rresp),
      .m_axi_rlast  (n_rlast),
      .m_axi_rvalid (n_rvalid),
      .m_axi_rready (n_rready)
  );

  axi_arbiter u_arbiter (
      .clk(ui_clk),
      .rst(sys_rst),

      .m0_awid   (b_awid),
      .m0_awaddr (b_awaddr),
      .m0_awlen  (b_awlen),
      .m0_awsize (b_awsize),
      .m0_awburst(b_awburst),
      .m0_awlock (b_awlock),
      .m0_awcache(b_awcache),
      .m0_awprot (b_awprot),
      .m0_awqos  (b_awqos),
      .m0_awvalid(b_awvalid),
      .m0_awready(b_awready),
      .m0_wdata  (b_wdata),
      .m0_wstrb  (b_wstrb),
      .m0_wlast  (b_wlast),
      .m0_wvalid (b_wvalid),
      .m0_wready (b_wready),
      .m0_bid    (b_bid),
      .m0_bresp  (b_bresp),
      .m0_bvalid (b_bvalid),
      .m0_bready (b_bready),
      .m0_arid   (b_arid),
      .m0_araddr (b_araddr),
      .m0_arlen  (b_arlen),
      .m0_arsize (b_arsize),
      .m0_arburst(b_arburst),
      .m0_arlock (b_arlock),
      .m0_arcache(b_arcache),
      .m0_arprot (b_arprot),
      .m0_arqos  (b_arqos),
      .m0_arvalid(b_arvalid),
      .m0_arready(b_arready),
      .m0_rid    (b_rid),
      .m0_rdata  (b_rdata),
      .m0_rresp  (b_rresp),
      .m0_rlast  (b_rlast),
      .m0_rvalid (b_rvalid),
      .m0_rready (b_rready),

      .m1_awid   (n_awid),
      .m1_awaddr (n_awaddr),
      .m1_awlen  (n_awlen),
      .m1_awsize (n_awsize),
      .m1_awburst(n_awburst),
      .m1_awlock (n_awlock),
      .m1_awcache(n_awcache),
      .m1_awprot (n_awprot),
      .m1_awqos  (n_awqos),
      .m1_awvalid(n_awvalid),
      .m1_awready(n_awready),
      .m1_wdata  (n_wdata),
      .m1_wstrb  (n_wstrb),
      .m1_wlast  (n_wlast),
      .m1_wvalid (n_wvalid),
      .m1_wready (n_wready),
      .m1_bid    (n_bid),
      .m1_bresp  (n_bresp),
      .m1_bvalid (n_bvalid),
      .m1_bready (n_bready),
      .m1_arid   (n_arid),
      .m1_araddr (n_araddr),
      .m1_arlen  (n_arlen),
      .m1_arsize (n_arsize),
      .m1_arburst(n_arburst),
      .m1_arlock (n_arlock),
      .m1_arcache(n_arcache),
      .m1_arprot (n_arprot),
      .m1_arqos  (n_arqos),
      .m1_arvalid(n_arvalid),
      .m1_arready(n_arready),
      .m1_rid    (n_rid),
      .m1_rdata  (n_rdata),
      .m1_rresp  (n_rresp),
      .m1_rlast  (n_rlast),
      .m1_rvalid (n_rvalid),
      .m1_rready (n_rready),

      .s_awid   (m_awid),
      .s_awaddr (m_awaddr),
      .s_awlen  (m_awlen),
      .s_awsize (m_awsize),
      .s_awburst(m_awburst),
      .s_awlock (m_awlock),
      .s_awcache(m_awcache),
      .s_awprot (m_awprot),
      .s_awqos  (m_awqos),
      .s_awvalid(m_awvalid),
      .s_awready(m_awready),
      .s_wdata  (m_wdata),
      .s_wstrb  (m_wstrb),
      .s_wlast  (m_wlast),
      .s_wvalid (m_wvalid),
      .s_wready (m_wready),
      .s_bid    (m_bid),
      .s_bresp  (m_bresp),
      .s_bvalid (m_bvalid),
      .s_bready (m_bready),
      .s_arid   (m_arid),
      .s_araddr (m_araddr),
      .s_arlen  (m_arlen),
      .s_arsize (m_arsize),
      .s_arburst(m_arburst),
      .s_arlock (m_arlock),
      .s_arcache(m_arcache),
      .s_arprot (m_arprot),
      .s_arqos  (m_arqos),
      .s_arvalid(m_arvalid),
      .s_arready(m_arready),
      .s_rid    (m_rid),
      .s_rdata  (m_rdata),
      .s_rresp  (m_rresp),
      .s_rlast  (m_rlast),
      .s_rvalid (m_rvalid),
      .s_rready (m_rready)
  );

  assign led_calib  = ~init_calib_complete;
  assign led_locked = ~mmcm_locked;

endmodule
