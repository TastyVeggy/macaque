`timescale 1ns / 1ps

module uart_mmio_top (
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

  // AXI4 master (from the bridge) -> MIG s_axi
  logic [ 3:0] axi_awid;
  logic [27:0] axi_awaddr;
  logic [ 7:0] axi_awlen;
  logic [ 2:0] axi_awsize;
  logic [ 1:0] axi_awburst;
  logic [ 0:0] axi_awlock;
  logic [ 3:0] axi_awcache;
  logic [ 2:0] axi_awprot;
  logic [ 3:0] axi_awqos;
  logic        axi_awvalid;
  logic        axi_awready;
  logic [63:0] axi_wdata;
  logic [ 7:0] axi_wstrb;
  logic        axi_wlast;
  logic        axi_wvalid;
  logic        axi_wready;
  logic [ 3:0] axi_bid;
  logic [ 1:0] axi_bresp;
  logic        axi_bvalid;
  logic        axi_bready;
  logic [ 3:0] axi_arid;
  logic [27:0] axi_araddr;
  logic [ 7:0] axi_arlen;
  logic [ 2:0] axi_arsize;
  logic [ 1:0] axi_arburst;
  logic [ 0:0] axi_arlock;
  logic [ 3:0] axi_arcache;
  logic [ 2:0] axi_arprot;
  logic [ 3:0] axi_arqos;
  logic        axi_arvalid;
  logic        axi_arready;
  logic [ 3:0] axi_rid;
  logic [63:0] axi_rdata;
  logic [ 1:0] axi_rresp;
  logic        axi_rlast;
  logic        axi_rvalid;
  logic        axi_rready;

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
      .s_axi_awid         (axi_awid),
      .s_axi_awaddr       (axi_awaddr),
      .s_axi_awlen        (axi_awlen),
      .s_axi_awsize       (axi_awsize),
      .s_axi_awburst      (axi_awburst),
      .s_axi_awlock       (axi_awlock),
      .s_axi_awcache      (axi_awcache),
      .s_axi_awprot       (axi_awprot),
      .s_axi_awqos        (axi_awqos),
      .s_axi_awvalid      (axi_awvalid),
      .s_axi_awready      (axi_awready),
      .s_axi_wdata        (axi_wdata),
      .s_axi_wstrb        (axi_wstrb),
      .s_axi_wlast        (axi_wlast),
      .s_axi_wvalid       (axi_wvalid),
      .s_axi_wready       (axi_wready),
      .s_axi_bready       (axi_bready),
      .s_axi_bid          (axi_bid),
      .s_axi_bresp        (axi_bresp),
      .s_axi_bvalid       (axi_bvalid),
      .s_axi_arid         (axi_arid),
      .s_axi_araddr       (axi_araddr),
      .s_axi_arlen        (axi_arlen),
      .s_axi_arsize       (axi_arsize),
      .s_axi_arburst      (axi_arburst),
      .s_axi_arlock       (axi_arlock),
      .s_axi_arcache      (axi_arcache),
      .s_axi_arprot       (axi_arprot),
      .s_axi_arqos        (axi_arqos),
      .s_axi_arvalid      (axi_arvalid),
      .s_axi_arready      (axi_arready),
      .s_axi_rready       (axi_rready),
      .s_axi_rid          (axi_rid),
      .s_axi_rdata        (axi_rdata),
      .s_axi_rresp        (axi_rresp),
      .s_axi_rlast        (axi_rlast),
      .s_axi_rvalid       (axi_rvalid),
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
  wire                             bridge_rst = rst_sync | ui_clk_sync_rst;

  // register bus
  logic [                     7:0] reg_addr;
  logic                            reg_we;
  logic [                    63:0] reg_wdata;
  logic [                    63:0] reg_rdata;

  // instruction memor
  logic                            im_wr_en;
  logic [npu_pkg::IMEM_ADDR_W+2:0] im_wr_addr;
  logic [                    63:0] im_wr_data;
  logic                            im_re;
  logic [npu_pkg::IMEM_ADDR_W+2:0] im_rd_addr;
  logic [                    63:0] im_rd_data;

  uart_mmio_bridge #(
      .ClkFreq(npu_pkg::MIG_UI_CLK_FREQ),
      .StatusW(2)
  ) u_bridge (
      .clk          (ui_clk),
      .rst          (bridge_rst),
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
      .m_axi_awid   (axi_awid),
      .m_axi_awaddr (axi_awaddr),
      .m_axi_awlen  (axi_awlen),
      .m_axi_awsize (axi_awsize),
      .m_axi_awburst(axi_awburst),
      .m_axi_awlock (axi_awlock),
      .m_axi_awcache(axi_awcache),
      .m_axi_awprot (axi_awprot),
      .m_axi_awqos  (axi_awqos),
      .m_axi_awvalid(axi_awvalid),
      .m_axi_awready(axi_awready),
      .m_axi_wdata  (axi_wdata),
      .m_axi_wstrb  (axi_wstrb),
      .m_axi_wlast  (axi_wlast),
      .m_axi_wvalid (axi_wvalid),
      .m_axi_wready (axi_wready),
      .m_axi_bid    (axi_bid),
      .m_axi_bresp  (axi_bresp),
      .m_axi_bvalid (axi_bvalid),
      .m_axi_bready (axi_bready),
      .m_axi_arid   (axi_arid),
      .m_axi_araddr (axi_araddr),
      .m_axi_arlen  (axi_arlen),
      .m_axi_arsize (axi_arsize),
      .m_axi_arburst(axi_arburst),
      .m_axi_arlock (axi_arlock),
      .m_axi_arcache(axi_arcache),
      .m_axi_arprot (axi_arprot),
      .m_axi_arqos  (axi_arqos),
      .m_axi_arvalid(axi_arvalid),
      .m_axi_arready(axi_arready),
      .m_axi_rid    (axi_rid),
      .m_axi_rdata  (axi_rdata),
      .m_axi_rresp  (axi_rresp),
      .m_axi_rlast  (axi_rlast),
      .m_axi_rvalid (axi_rvalid),
      .m_axi_rready (axi_rready)
  );

  // Control registers
  logic npu_start, npu_reset, pmu_enable, pmu_clear;
  logic [31:0] instr_addr, instr_len;
  ctrl_regs u_ctrl_regs (
      .clk(ui_clk),
      .rst(bridge_rst),
      .reg_addr(reg_addr),
      .reg_we(reg_we),
      .reg_wdata(reg_wdata),
      .reg_rdata(reg_rdata),

      .npu_start (npu_start),
      .npu_reset (npu_reset),
      .instr_addr(instr_addr),
      .instr_len (instr_len),
      .pmu_enable(pmu_enable),
      .pmu_clear (pmu_clear),

      .npu_busy(1'b0),
      .npu_done(1'b0),
      .npu_error(1'b0),
      .npu_ready(1'b1),
      .pmu_cycles(64'd0),
      .pmu_compute(32'd0),
      .pmu_stall(32'd0),
      .pmu_dma_bytes_rd(32'd0),
      .pmu_dma_bytes_wr(32'd0)
  );

  // Instruction memory
  logic [63:0] seq_rdata;
  imem u_imem (
      .clk_host(ui_clk),
      .we(im_wr_en),
      .waddr(im_wr_addr),
      .wdata(im_wr_data),
      .re(im_re),
      .raddr(im_rd_addr),
      .rdata(im_rd_data),
      .clk_seq(ui_clk),
      .seq_re(1'b0),
      .seq_raddr('0),
      .seq_rdata(seq_rdata)
  );

  // Unused NPU control outputs) + sequencer read data.
  wire _unused = &{1'b0, npu_start, npu_reset, instr_addr, instr_len,
                        pmu_enable, pmu_clear, seq_rdata, 1'b0};

  assign led_calib  = ~init_calib_complete;
  assign led_locked = ~mmcm_locked;

endmodule
