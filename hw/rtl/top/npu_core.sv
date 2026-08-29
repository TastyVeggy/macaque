`timescale 1ns / 1ps

module npu_core (
    input logic clk,
    input logic rst_n,

    input  logic                            imem_we,
    input  logic [npu_pkg::IMEM_ADDR_W+2:0] imem_waddr,  // byte address
    input  logic [                    63:0] imem_wdata,
    input  logic                            imem_re,
    input  logic [npu_pkg::IMEM_ADDR_W+2:0] imem_raddr,  // byte address
    output logic [                    63:0] imem_rdata,

    input  logic [ 7:0] reg_addr,
    input  logic        reg_we,
    input  logic [63:0] reg_wdata,
    output logic [63:0] reg_rdata,

    // AXI4 master (DMA -> MIG/DDR3)
    output logic [ 3:0] m_axi_awid,
    output logic [27:0] m_axi_awaddr,
    output logic [ 7:0] m_axi_awlen,
    output logic [ 2:0] m_axi_awsize,
    output logic [ 1:0] m_axi_awburst,
    output logic [ 0:0] m_axi_awlock,
    output logic [ 3:0] m_axi_awcache,
    output logic [ 2:0] m_axi_awprot,
    output logic [ 3:0] m_axi_awqos,
    output logic        m_axi_awvalid,
    input  logic        m_axi_awready,
    output logic [63:0] m_axi_wdata,
    output logic [ 7:0] m_axi_wstrb,
    output logic        m_axi_wlast,
    output logic        m_axi_wvalid,
    input  logic        m_axi_wready,
    input  logic [ 3:0] m_axi_bid,
    input  logic [ 1:0] m_axi_bresp,
    input  logic        m_axi_bvalid,
    output logic        m_axi_bready,
    output logic [ 3:0] m_axi_arid,
    output logic [27:0] m_axi_araddr,
    output logic [ 7:0] m_axi_arlen,
    output logic [ 2:0] m_axi_arsize,
    output logic [ 1:0] m_axi_arburst,
    output logic [ 0:0] m_axi_arlock,
    output logic [ 3:0] m_axi_arcache,
    output logic [ 2:0] m_axi_arprot,
    output logic [ 3:0] m_axi_arqos,
    output logic        m_axi_arvalid,
    input  logic        m_axi_arready,
    input  logic [ 3:0] m_axi_rid,
    input  logic [63:0] m_axi_rdata,
    input  logic [ 1:0] m_axi_rresp,
    input  logic        m_axi_rlast,
    input  logic        m_axi_rvalid,
    output logic        m_axi_rready
);

  logic rst;
  reset_bridge rst_bridge_inst (
      .clk(clk),
      .rst_n(rst_n),
      .rst_high(rst)
  );
  logic seq_rst;

  logic npu_start, npu_reset;
  assign seq_rst = rst || npu_reset;
  logic [31:0] instr_addr, instr_len;
  logic pmu_enable, pmu_clear;
  logic npu_busy, npu_done, npu_error, npu_ready;
  logic [63:0] pmu_cycles;
  logic [31:0] pmu_compute, pmu_stall, pmu_dma_bytes_rd, pmu_dma_bytes_wr;

  ctrl_regs regs_inst (
      .npu_start,
      .npu_reset,
      .npu_busy,
      .npu_done,
      .npu_error,
      .npu_ready,
      .instr_addr,
      .instr_len,
      .pmu_enable,
      .pmu_clear,
      .pmu_cycles,
      .pmu_compute,
      .pmu_stall,
      .pmu_dma_bytes_rd,
      .pmu_dma_bytes_wr,
      .clk      (clk),
      .rst      (rst),
      .reg_addr (reg_addr),
      .reg_we   (reg_we),
      .reg_wdata(reg_wdata),
      .reg_rdata(reg_rdata)
  );

  logic [31:0] im_addr;
  logic [63:0] im_data;
  imem u_imem (
      .clk_host (clk),
      .we       (imem_we),
      .waddr    (imem_waddr),
      .wdata    (imem_wdata),
      .re       (imem_re),
      .raddr    (imem_raddr),
      .rdata    (imem_rdata),
      .clk_seq  (clk),
      .seq_re   (1'b1),
      .seq_raddr(im_addr[npu_pkg::IMEM_ADDR_W+2:0]),
      .seq_rdata(im_data)
  );

  logic start_pulse;
  logic load_req, store_req, load_done, store_done;
  logic activate_req, activate_done;
  logic matmul_done;
  logic weight_valid, act_valid, bias_valid;
  logic wb_re, ab_re, bb_re, ob_fb_re;
  logic [npu_pkg::DTYPE_BRAM_ADDR_W-1:0] wb_raddr, ab_raddr, bb_raddr, ob_fb_raddr;
  logic weight_bank_sel, act_bank_sel, bias_bank_sel;
  logic out_bank_sel, quant_bank_sel;
  logic pmu_freeze, run_active, mac_active, stall, matmul_start;
  logic acc_mode;  // 1 = K-tiling accumulate (seed array from out_buffer partial sums)
  logic [11:0] tile_params;
  logic [16:0] act_scale_m;
  logic [4:0] act_scale_shift;
  npu_pkg::act_func_t act_func;
  npu_pkg::bram_addr_t mat_row_base, act_row_base;
  logic act_bank_hold;
  logic [npu_pkg::ISA_DDR_ADDR_W-1:0] seq_ddr3_addr;
  logic [npu_pkg::ISA_BYTE_CNT_W-1:0] seq_byte_count;
  npu_pkg::buffer_type_t load_target;
  logic [3:0] load_valid_bytes_per_row;
  logic [7:0] load_input_rows;

  assign start_pulse = npu_start;

  instr_sequencer seq_inst (
      .clk,
      .rst       (seq_rst),
      .start     (start_pulse),
      .reset     (npu_reset),
      .instr_base(instr_addr),
      .instr_len (instr_len),
      .busy      (npu_busy),
      .done      (npu_done),
      .error     (npu_error),
      .ready     (npu_ready),
      .im_addr,
      .im_data,
      .weight_valid,
      .act_valid,
      .bias_valid,
      .matmul_done,
      .activate_req,
      .activate_done,
      .load_req,
      .store_req,
      .load_target,
      .ddr3_addr (seq_ddr3_addr),
      .byte_count(seq_byte_count),
      .load_valid_bytes_per_row,
      .load_input_rows,
      .acc_mode,
      .tile_params,
      .act_scale_m,
      .act_scale_shift,
      .act_func,
      .mat_row_base,
      .act_row_base,
      .act_bank_hold,
      .load_done,
      .store_done,
      .wb_re,
      .wb_raddr,
      .ab_re,
      .ab_raddr,
      .bb_re,
      .bb_raddr,
      .ob_fb_re,
      .ob_fb_raddr,
      .matmul_start,
      .weight_bank_sel,
      .act_bank_sel,
      .bias_bank_sel,
      .out_bank_sel,
      .quant_bank_sel,
      .pmu_freeze,
      .run_active,
      .mac_active,
      .stall
  );

  npu_pkg::weight_vec_t wb_rdata;
  weight_buffer wb_inst (
      .clk_ddr  (clk),
      .clk_sa   (clk),
      .dma_we   (wb_dma_we),
      .dma_waddr(wb_dma_waddr),
      .dma_wdata(wb_dma_wdata),
      .sa_re    (wb_re),
      .sa_raddr (wb_raddr),
      .sa_rdata (wb_rdata),
      .weight_bank_sel
  );

  npu_pkg::act_vec_t ab_rdata;
  activation_buffer ab_inst (
      .clk_ddr  (clk),
      .clk_sa   (clk),
      .dma_we   (ab_dma_we),
      .dma_waddr(ab_dma_waddr),
      .dma_wdata(ab_dma_wdata),
      .sa_re    (ab_re),
      .sa_raddr (ab_raddr),
      .sa_rdata (ab_rdata),
      .act_bank_sel
  );

  npu_pkg::bias_vec_t bb_rdata;
  bias_buffer bb_inst (
      .clk_ddr  (clk),
      .clk_sa   (clk),
      .dma_we   (bb_dma_we),
      .dma_waddr(bb_dma_waddr),
      .dma_wdata(bb_dma_wdata),
      .sa_re    (seq_inst.bb_re),
      .sa_raddr (seq_inst.bb_raddr),
      .sa_rdata (bb_rdata),
      .bias_bank_sel
  );

  npu_pkg::acc_vec_t ob_rdata;
  logic [npu_pkg::DTYPE_BRAM_ADDR_W-1:0] drain_addr;
  logic array_drain_valid;
  npu_pkg::acc_vec_t array_drain_data;

  logic activate_ob_re;
  logic [npu_pkg::DTYPE_BRAM_ADDR_W-1:0] activate_ob_raddr;
  logic activate_qb_we;
  logic [npu_pkg::DTYPE_BRAM_ADDR_W-1:0] activate_qb_waddr;
  npu_pkg::act_vec_t activate_qb_wdata;

  always_ff @(posedge clk) begin
    if (rst || seq_rst) drain_addr <= '0;
    else if (matmul_start) drain_addr <= mat_row_base;
    else if (array_drain_valid) drain_addr <= drain_addr + 1;
  end

  // Out-buffer read port is shared between ACTIVATE (rd_sel=0, or rd_sel=1
  // when act_bank_hold) and the acc_mode=1 partial-sum feedback
  // (rd_sel=1).
  out_buffer ob_inst (
      .clk_sa  (clk),
      .sa_we   (array_drain_valid),
      .sa_waddr(drain_addr),
      .sa_wdata(array_drain_data),
      .rd_re   (activate_ob_re | ob_fb_re),
      .rd_raddr(ob_fb_re ? ob_fb_raddr : activate_ob_raddr),
      .rd_sel  (ob_fb_re | act_bank_hold),
      .rd_rdata(ob_rdata),
      .out_bank_sel
  );

  quant_buffer qb_inst (
      .clk_sa   (clk),
      .clk_ddr  (clk),
      .sa_we    (activate_qb_we),
      .sa_waddr (activate_qb_waddr),
      .sa_wdata (activate_qb_wdata),
      .dma_re   (qb_dma_re),
      .dma_raddr(qb_dma_raddr),
      .dma_rdata(qb_dma_rdata),
      .quant_bank_sel
  );

  npu_pkg::bias_vec_t array_bias_data;
  always_comb begin
    for (int i = 0; i < npu_pkg::ARRAY_SIZE; i++)
    array_bias_data[i] = acc_mode ? ob_rdata[i] : bb_rdata[i];
  end

  systolic_array array_inst (
      .clk,
      .rst        (seq_rst),
      .weight_data(wb_rdata),
      .weight_valid,
      .act_data   (ab_rdata),
      .act_valid,
      .bias_data  (array_bias_data),
      .bias_valid,
      .drain_data (array_drain_data),
      .drain_valid(array_drain_valid),
      .ready      ()
  );

  assign matmul_done = array_drain_valid;

  logic                        wb_dma_we;
  npu_pkg::bram_addr_t         wb_dma_waddr;
  npu_pkg::weight_vec_t        wb_dma_wdata;
  logic                        ab_dma_we;
  npu_pkg::bram_addr_t         ab_dma_waddr;
  npu_pkg::act_vec_t           ab_dma_wdata;
  logic                        bb_dma_we;
  npu_pkg::bram_addr_t         bb_dma_waddr;
  npu_pkg::bias_vec_t          bb_dma_wdata;
  logic                        qb_dma_re;
  npu_pkg::bram_addr_t         qb_dma_raddr;
  npu_pkg::act_vec_t           qb_dma_rdata;
  logic                 [31:0] dma_bytes_rd_this_cycle;
  logic                 [31:0] dma_bytes_wr_this_cycle;

  dma_unit dma_unit_inst (
      .clk,
      .rst       (rst),
      .load_req,
      .store_req,
      .load_target,
      .byte_count(seq_byte_count),
      .ddr3_addr (seq_ddr3_addr),
      .load_valid_bytes_per_row,
      .load_input_rows,
      .load_done,
      .store_done,
      .dma_bytes_rd_this_cycle,
      .dma_bytes_wr_this_cycle,

      .m_axi_awid   (m_axi_awid),
      .m_axi_awaddr (m_axi_awaddr),
      .m_axi_awlen  (m_axi_awlen),
      .m_axi_awsize (m_axi_awsize),
      .m_axi_awburst(m_axi_awburst),
      .m_axi_awlock (m_axi_awlock),
      .m_axi_awcache(m_axi_awcache),
      .m_axi_awprot (m_axi_awprot),
      .m_axi_awqos  (m_axi_awqos),
      .m_axi_awvalid(m_axi_awvalid),
      .m_axi_awready(m_axi_awready),
      .m_axi_wdata  (m_axi_wdata),
      .m_axi_wstrb  (m_axi_wstrb),
      .m_axi_wlast  (m_axi_wlast),
      .m_axi_wvalid (m_axi_wvalid),
      .m_axi_wready (m_axi_wready),
      .m_axi_bid    (m_axi_bid),
      .m_axi_bresp  (m_axi_bresp),
      .m_axi_bvalid (m_axi_bvalid),
      .m_axi_bready (m_axi_bready),
      .m_axi_arid   (m_axi_arid),
      .m_axi_araddr (m_axi_araddr),
      .m_axi_arlen  (m_axi_arlen),
      .m_axi_arsize (m_axi_arsize),
      .m_axi_arburst(m_axi_arburst),
      .m_axi_arlock (m_axi_arlock),
      .m_axi_arcache(m_axi_arcache),
      .m_axi_arprot (m_axi_arprot),
      .m_axi_arqos  (m_axi_arqos),
      .m_axi_arvalid(m_axi_arvalid),
      .m_axi_arready(m_axi_arready),
      .m_axi_rid    (m_axi_rid),
      .m_axi_rdata  (m_axi_rdata),
      .m_axi_rresp  (m_axi_rresp),
      .m_axi_rlast  (m_axi_rlast),
      .m_axi_rvalid (m_axi_rvalid),
      .m_axi_rready (m_axi_rready),

      .wb_dma_we   (wb_dma_we),
      .wb_dma_waddr(wb_dma_waddr),
      .wb_dma_wdata(wb_dma_wdata),
      .ab_dma_we   (ab_dma_we),
      .ab_dma_waddr(ab_dma_waddr),
      .ab_dma_wdata(ab_dma_wdata),
      .bb_dma_we   (bb_dma_we),
      .bb_dma_waddr(bb_dma_waddr),
      .bb_dma_wdata(bb_dma_wdata),
      .qb_dma_re   (qb_dma_re),
      .qb_dma_raddr(qb_dma_raddr),
      .qb_dma_rdata(qb_dma_rdata)
  );

  activate_unit activate_inst (
      .clk,
      .rst        (seq_rst),
      .req        (activate_req),
      .done       (activate_done),
      .tile_params,
      .scale_m    (act_scale_m),
      .scale_shift(act_scale_shift),
      .act_func   (act_func),
      .row_base   (act_row_base),
      .ob_re      (activate_ob_re),
      .ob_raddr   (activate_ob_raddr),
      .ob_rdata,
      .qb_we      (activate_qb_we),
      .qb_waddr   (activate_qb_waddr),
      .qb_wdata   (activate_qb_wdata)
  );

  pmu pmu_inst (
      .clk,
      .rst                    (rst),
      .enable                 (pmu_enable),
      .clear                  (pmu_clear),
      .run_active,
      .mac_active,
      .stall,
      .dma_bytes_rd_this_cycle(dma_bytes_rd_this_cycle),
      .dma_bytes_wr_this_cycle(dma_bytes_wr_this_cycle),
      .pmu_cycles,
      .pmu_compute,
      .pmu_stall,
      .pmu_dma_bytes_rd,
      .pmu_dma_bytes_wr,
      .frozen                 (pmu_freeze)
  );

endmodule
