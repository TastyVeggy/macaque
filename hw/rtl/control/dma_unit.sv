`timescale 1ns / 1ps

module dma_unit (
    input logic clk,
    input logic rst,

    // Sequencer request interface (load/store mutually exclusive)
    input  logic                                                load_req,
    input  logic                                                store_req,
    input  npu_pkg::buffer_type_t                               load_target,
    input  logic                  [npu_pkg::ISA_BYTE_CNT_W-1:0] byte_count,
    input  logic                  [npu_pkg::ISA_DDR_ADDR_W-1:0] ddr3_addr,
    output logic                                                load_done,
    output logic                                                store_done,

    // Bytes transferred this cycle (for the PMU DMA counters)
    output logic [31:0] dma_bytes_rd_this_cycle,
    output logic [31:0] dma_bytes_wr_this_cycle,

    // AXI4 master (burst-capable, 64-bit beats) -> MIG s_axi
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
    output logic        m_axi_rready,

    // Weight buffer write port
    output logic                 wb_dma_we,
    output npu_pkg::bram_addr_t  wb_dma_waddr,
    output npu_pkg::weight_vec_t wb_dma_wdata,

    // Activation buffer write port
    output logic                ab_dma_we,
    output npu_pkg::bram_addr_t ab_dma_waddr,
    output npu_pkg::act_vec_t   ab_dma_wdata,

    // Bias buffer write port
    output logic                bb_dma_we,
    output npu_pkg::bram_addr_t bb_dma_waddr,
    output npu_pkg::bias_vec_t  bb_dma_wdata,

    // Quant buffer read port
    output logic                qb_dma_re,
    output npu_pkg::bram_addr_t qb_dma_raddr,
    input  npu_pkg::act_vec_t   qb_dma_rdata
);

  logic [63:0] s_axis_tdata;
  logic        s_axis_tvalid;
  logic        s_axis_tready;
  logic [63:0] m_axis_tdata;
  logic        m_axis_tvalid;
  logic        m_axis_tready;
  logic        m_axis_tlast;

  logic a_load_done, a_store_done;

  stream_buffer_adapter u_adapter (
      .clk,
      .rst,
      .load_req,
      .store_req,
      .load_target,
      .byte_count,
      .load_done (a_load_done),
      .store_done(a_store_done),
      .s_axis_tdata,
      .s_axis_tvalid,
      .s_axis_tready,
      .m_axis_tdata,
      .m_axis_tvalid,
      .m_axis_tready,
      .m_axis_tlast,
      .wb_dma_we,
      .wb_dma_waddr,
      .wb_dma_wdata,
      .ab_dma_we,
      .ab_dma_waddr,
      .ab_dma_wdata,
      .bb_dma_we,
      .bb_dma_waddr,
      .bb_dma_wdata,
      .qb_dma_re,
      .qb_dma_raddr,
      .qb_dma_rdata
  );

  localparam int MaxBurstBeats = 16;

  typedef enum logic [2:0] {
    IDLE,
    LD_AR,    // LOAD: issue read address (this burst)
    LD_R,     // LOAD: wait read data (one beat)
    LD_PUSH,  // LOAD: push beat to adapter
    ST_WAIT,  // STORE: wait adapter beat (first beat of a new burst, or mid-burst)
    ST_AW,    // STORE: issue write address (this burst)
    ST_W,     // STORE: write data (one beat)
    ST_B      // STORE: wait write response (this burst)
  } state_t;
  state_t        state;

  logic   [27:0] cur_addr;
  logic   [12:0] beats_left;
  logic   [ 4:0] burst_beats_left;  // remaining beats within current axi burst
  logic   [63:0] rd_data;
  logic   [63:0] wr_data;
  logic          wr_last;  // last beat of the whole store
  logic          need_aw;  // STORE: next beat starts a fresh burst (needs a new AW)

  logic   [12:0] bytes_to_4k_boundary;
  logic   [ 9:0] beats_to_4k_boundary;
  logic   [ 4:0] burst_len;

  assign bytes_to_4k_boundary = 13'h1000 - {1'b0, cur_addr[11:0]};
  assign beats_to_4k_boundary = bytes_to_4k_boundary[12:3];

  always_comb begin
    burst_len = (beats_left > 13'(MaxBurstBeats)) ? 5'(MaxBurstBeats) : 5'(beats_left);
    if (10'(burst_len) > beats_to_4k_boundary) burst_len = 5'(beats_to_4k_boundary);
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      state            <= IDLE;
      cur_addr         <= '0;
      beats_left       <= '0;
      burst_beats_left <= '0;
      rd_data          <= '0;
      wr_data          <= '0;
      wr_last          <= 1'b0;
      need_aw          <= 1'b0;
      m_axi_arvalid    <= '0;
      m_axi_rready     <= '0;
      m_axi_awvalid    <= '0;
      m_axi_wvalid     <= '0;
      m_axi_bready     <= '0;
      s_axis_tvalid    <= '0;
      m_axis_tready    <= '0;
    end else begin
      m_axi_arvalid <= '0;
      m_axi_rready  <= '0;
      m_axi_awvalid <= '0;
      m_axi_wvalid  <= '0;
      m_axi_bready  <= '0;
      s_axis_tvalid <= '0;
      m_axis_tready <= '0;

      case (state)
        IDLE: begin
          if (load_req) begin
            cur_addr   <= ddr3_addr;
            beats_left <= 13'((byte_count + 16'd7) >> 3);
            state      <= LD_AR;
          end else if (store_req) begin
            cur_addr   <= ddr3_addr;
            beats_left <= 13'((byte_count + 16'd7) >> 3);
            wr_last    <= 1'b0;
            need_aw    <= 1'b1;
            state      <= ST_WAIT;
          end
        end

        LD_AR: begin
          if (m_axi_arvalid && m_axi_arready) begin
            m_axi_arvalid    <= 1'b0;
            burst_beats_left <= burst_len;
            state            <= LD_R;
          end else begin
            m_axi_arvalid <= 1'b1;
          end
        end

        LD_R: begin
          if (m_axi_rvalid && m_axi_rready) begin
            m_axi_rready     <= 1'b0;
            rd_data          <= m_axi_rdata;
            burst_beats_left <= burst_beats_left - 5'd1;
            state            <= LD_PUSH;
          end else begin
            m_axi_rready <= 1'b1;
          end
        end

        LD_PUSH: begin
          if (s_axis_tvalid && s_axis_tready) begin
            s_axis_tvalid <= 1'b0;
            cur_addr      <= cur_addr + 28'd8;
            if (beats_left == 13'd1) begin
              beats_left <= 13'd0;
              state      <= IDLE;
            end else begin
              beats_left <= beats_left - 13'd1;
              state      <= (burst_beats_left == 5'd0) ? LD_AR : LD_R;
            end
          end else begin
            s_axis_tvalid <= 1'b1;
          end
        end

        ST_WAIT: begin
          if (m_axis_tvalid && m_axis_tready) begin
            m_axis_tready <= 1'b0;
            wr_data       <= m_axis_tdata;
            wr_last       <= m_axis_tlast;
            if (need_aw) begin
              burst_beats_left <= burst_len;
              state            <= ST_AW;
            end else begin
              state <= ST_W;
            end
          end else begin
            m_axis_tready <= 1'b1;
          end
        end

        ST_AW: begin
          if (m_axi_awvalid && m_axi_awready) begin
            m_axi_awvalid <= 1'b0;
            need_aw       <= 1'b0;
            state         <= ST_W;
          end else begin
            m_axi_awvalid <= 1'b1;
          end
        end

        ST_W: begin
          if (m_axi_wvalid && m_axi_wready) begin
            m_axi_wvalid <= 1'b0;
            cur_addr     <= cur_addr + 28'd8;
            beats_left   <= beats_left - 13'd1;
            if (burst_beats_left == 5'd1) begin
              state <= ST_B;
            end else begin
              burst_beats_left <= burst_beats_left - 5'd1;
              state            <= ST_WAIT;
            end
          end else begin
            m_axi_wvalid <= 1'b1;
          end
        end

        ST_B: begin
          if (m_axi_bvalid && m_axi_bready) begin
            m_axi_bready <= 1'b0;
            if (wr_last) begin
              state <= IDLE;
            end else begin
              need_aw <= 1'b1;
              state   <= ST_WAIT;
            end
          end else begin
            m_axi_bready <= 1'b1;
          end
        end

        default: state <= IDLE;
      endcase
    end
  end

  assign s_axis_tdata = rd_data;

  assign m_axi_araddr = cur_addr;
  assign m_axi_awaddr = cur_addr;
  assign m_axi_wdata  = wr_data;
  assign m_axi_wstrb  = 8'hFF;
  assign m_axi_wlast  = (burst_beats_left == 5'd1);

  assign m_axi_awid    = 4'd0;
  assign m_axi_awlen   = (state == ST_AW) ? (8'(burst_beats_left) - 8'd1) : 8'd0;
  assign m_axi_awsize  = 3'b011;
  assign m_axi_awburst = 2'b01;
  assign m_axi_awlock  = 1'b0;
  assign m_axi_awcache = 4'd0;
  assign m_axi_awprot  = 3'd0;
  assign m_axi_awqos   = 4'd0;

  assign m_axi_arid    = 4'd0;
  assign m_axi_arlen   = (state == LD_AR) ? (8'(burst_len) - 8'd1) : 8'd0;
  assign m_axi_arsize  = 3'b011;
  assign m_axi_arburst = 2'b01;
  assign m_axi_arlock  = 1'b0;
  assign m_axi_arcache = 4'd0;
  assign m_axi_arprot  = 3'd0;
  assign m_axi_arqos   = 4'd0;

  // Burst master doesn't check the response ID/RESP fields, or cross-check
  // RLAST against its own beat counter (AXI4 guarantees a compliant slave
  // returns exactly ARLEN+1 beats with RLAST on the last one).
  wire _unused = &{1'b0, m_axi_bid, m_axi_bresp, m_axi_rid, m_axi_rresp, m_axi_rlast, 1'b0};

  assign load_done = a_load_done;
  assign store_done = (state == ST_B) && m_axi_bvalid && m_axi_bready && wr_last;

  // 8 bytes per completed beat
  assign dma_bytes_rd_this_cycle = (state == LD_R) && m_axi_rvalid && m_axi_rready ? 32'd8 : 32'd0;
  assign dma_bytes_wr_this_cycle = (state == ST_W) && m_axi_wvalid && m_axi_wready ? 32'd8 : 32'd0;

endmodule
