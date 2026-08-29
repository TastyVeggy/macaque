`timescale 1ns / 1ps

module stream_buffer_adapter (
    input logic clk,
    input logic rst,

    // Request from instruction sequencer (shared bus, mutually exclusive).
    input logic load_req,
    input logic store_req,
    input npu_pkg::buffer_type_t load_target,
    input logic [npu_pkg::ISA_BYTE_CNT_W-1:0] byte_count,
    // LOAD_INPUT-only DMA zero-injection sizing
    // every row of this transfer is the real input and 
    // byte_count is the normal padded transfer size, 
    // same as every other load
    input logic [3:0] load_valid_bytes_per_row,
    input logic [7:0] load_input_rows,
    output logic load_done,
    output logic store_done,

    // AXI4-Stream slave (MM2S load path: DDR -> buffer)
    input  logic [63:0] s_axis_tdata,
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,

    // AXI4-Stream master (S2MM store path: buffer -> DDR)
    output logic [63:0] m_axis_tdata,
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready,
    output logic        m_axis_tlast,

    // Weight buffer DMA write port
    output logic                 wb_dma_we,
    output npu_pkg::bram_addr_t  wb_dma_waddr,
    output npu_pkg::weight_vec_t wb_dma_wdata,

    // Activation buffer DMA write port
    output logic                ab_dma_we,
    output npu_pkg::bram_addr_t ab_dma_waddr,
    output npu_pkg::act_vec_t   ab_dma_wdata,

    // Bias buffer DMA write port
    output logic                bb_dma_we,
    output npu_pkg::bram_addr_t bb_dma_waddr,
    output npu_pkg::bias_vec_t  bb_dma_wdata,

    // Quant buffer DMA read port
    output logic                qb_dma_re,
    output npu_pkg::bram_addr_t qb_dma_raddr,
    input  npu_pkg::act_vec_t   qb_dma_rdata
);
  import npu_pkg::ARRAY_SIZE;
  import npu_pkg::BUF_ACT;
  import npu_pkg::BUF_BIAS;
  import npu_pkg::BUF_WEIGHT;

  localparam logic [15:0] Int8RowBytes = 16'(npu_pkg::INT8_ROW_BYTES);
  localparam logic [15:0] Int32RowBytes = 16'(npu_pkg::INT32_ROW_BYTES);

  // Sized fill thresholds
  localparam logic [8:0] LoadMaxFill = 9'(Int32RowBytes - 8);
  localparam logic [8:0] StoreMaxFill = 9'(Int32RowBytes - Int8RowBytes);

  typedef enum logic [1:0] {
    IDLE = 2'd0,
    LOAD_ST = 2'd1,
    STORE_ST = 2'd2
  } state_t;
  state_t state;

  npu_pkg::buffer_type_t target;
  logic [8:0] row_size;  // 14 (int8) or 56 (int32)
  logic [8:0] pop_threshold;  // row_size normally; valid_bytes_per_row when padding
  logic pad_active;  // zero-injection active for this transfer
  logic [3:0] valid_bytes_reg;  // registered load_valid_bytes_per_row
  logic [15:0] load_rows_total;  // rows covering byte_count for LOAD
  logic [15:0] row_index;  // current buffer address

  logic [15:0] store_rows_total;
  logic [15:0] store_rows_read;

  logic [Int32RowBytes*8-1:0] acc;
  logic [8:0] acc_fill;

  logic load_accept;
  logic store_accept;
  logic row_ready;
  logic all_rows_read;
  // A read is outstanding from request until its data is appended (qb_valid).
  logic read_pending;
  logic store_need_read;

  assign load_accept = (state == LOAD_ST) && s_axis_tvalid && s_axis_tready;
  assign store_accept = (state == STORE_ST) && m_axis_tvalid && m_axis_tready;
  assign row_ready = (acc_fill >= pop_threshold) && (row_index < load_rows_total);
  assign all_rows_read = (store_rows_read >= store_rows_total);
  assign store_need_read = (state == STORE_ST) && (store_rows_read < store_rows_total) &&
                           !read_pending && (acc_fill <= StoreMaxFill);

  // quant read data valid: 2 cycles after the request (req reg + BRAM)
  logic qb_re_r, qb_valid;

  // Load accumulator
  logic [Int32RowBytes*8-1:0] l_next_acc;
  logic [8:0] l_next_fill;
  always_comb begin
    l_next_fill = acc_fill;
    l_next_acc  = acc;
    if (row_ready) begin
      l_next_acc = acc >> (pop_threshold * 8);
      if (load_accept) begin
        l_next_acc[(acc_fill-pop_threshold)*8+:64] = s_axis_tdata;
        l_next_fill = acc_fill + 9'd8 - pop_threshold;
      end else begin
        l_next_fill = acc_fill - pop_threshold;
      end
    end else if (load_accept) begin
      l_next_acc = acc;
      l_next_acc[acc_fill*8+:64] = s_axis_tdata;
      l_next_fill = acc_fill + 9'd8;
    end
  end

  // Store accumulator
  logic [Int32RowBytes*8-1:0] s_next_acc;
  logic [8:0] s_next_fill;
  always_comb begin
    s_next_fill = acc_fill;
    s_next_acc  = acc;
    if (qb_valid) begin
      // append one 14-byte quant row at the current fill offset
      for (int c = 0; c < npu_pkg::ARRAY_SIZE; c++) s_next_acc[acc_fill*8+c*8+:8] = qb_dma_rdata[c];
      s_next_fill = acc_fill + 9'(Int8RowBytes);
    end
    if (store_accept) begin
      // emit 8 bytes from the head
      s_next_acc = s_next_acc >> 64;
      if (qb_valid) s_next_fill = acc_fill + 9'(Int8RowBytes) - 9'd8;
      else s_next_fill = (acc_fill >= 9'd8) ? (acc_fill - 9'd8) : 9'd0;
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      state            <= IDLE;
      target           <= npu_pkg::BUF_WEIGHT;
      row_size         <= 9'(Int8RowBytes);
      pop_threshold    <= 9'(Int8RowBytes);
      pad_active       <= 1'b0;
      valid_bytes_reg  <= '0;
      load_rows_total  <= '0;
      row_index        <= '0;
      store_rows_total <= '0;
      store_rows_read  <= '0;
      acc              <= '0;
      acc_fill         <= '0;
      qb_re_r          <= 1'b0;
      qb_valid         <= 1'b0;
      read_pending     <= 1'b0;
      wb_dma_we        <= 1'b0;
      ab_dma_we        <= 1'b0;
      bb_dma_we        <= 1'b0;
    end else begin
      wb_dma_we <= 1'b0;
      ab_dma_we <= 1'b0;
      bb_dma_we <= 1'b0;
      qb_re_r   <= 1'b0;
      qb_valid  <= qb_re_r;

      case (state)
        IDLE: begin
          if (load_req || store_req) begin
            target <= load_req ? load_target : BUF_ACT;
            if (load_req && (load_target == BUF_BIAS)) begin
              row_size        <= 9'(Int32RowBytes);
              pop_threshold   <= 9'(Int32RowBytes);
              pad_active      <= 1'b0;
              valid_bytes_reg <= '0;
              load_rows_total <= (byte_count + Int32RowBytes - 16'd1) / Int32RowBytes;
            end else begin
              row_size <= 9'(Int8RowBytes);
              if (load_req && (load_target == BUF_ACT) && (load_valid_bytes_per_row != '0)) begin
                // DMA zero-injection
                pad_active      <= 1'b1;
                pop_threshold   <= {5'b0, load_valid_bytes_per_row};
                valid_bytes_reg <= load_valid_bytes_per_row;
                load_rows_total <= {8'b0, load_input_rows};
              end else begin
                pad_active      <= 1'b0;
                pop_threshold   <= 9'(Int8RowBytes);
                valid_bytes_reg <= '0;
                load_rows_total <= (byte_count + Int8RowBytes - 16'd1) / Int8RowBytes;
              end
            end
            row_index        <= '0;
            store_rows_total <= (byte_count + Int8RowBytes - 16'd1) / Int8RowBytes;
            store_rows_read  <= '0;
            acc              <= '0;
            acc_fill         <= '0;
            state            <= load_req ? LOAD_ST : STORE_ST;
          end
        end

        LOAD_ST: begin
          acc      <= l_next_acc;
          acc_fill <= l_next_fill;
          if (row_ready) begin
            if (target == BUF_WEIGHT) begin
              wb_dma_we    <= 1'b1;
              wb_dma_waddr <= npu_pkg::bram_addr_t'(row_index);
              wb_dma_wdata <= wb_data_c;
            end else if (target == BUF_ACT) begin
              ab_dma_we    <= 1'b1;
              ab_dma_waddr <= npu_pkg::bram_addr_t'(row_index);
              ab_dma_wdata <= ab_data_c;
            end else begin  // BUF_BIAS
              bb_dma_we    <= 1'b1;
              bb_dma_waddr <= npu_pkg::bram_addr_t'(row_index);
              bb_dma_wdata <= bb_data_c;
            end
            row_index <= row_index + 16'd1;
          end
          if (row_index >= load_rows_total) state <= IDLE;
        end

        STORE_ST: begin
          qb_dma_raddr <= npu_pkg::bram_addr_t'(store_rows_read);
          if (store_need_read) begin
            read_pending <= 1'b1;
            qb_re_r      <= 1'b1;
          end
          acc      <= s_next_acc;
          acc_fill <= s_next_fill;
          if (qb_valid) begin
            read_pending    <= 1'b0;
            store_rows_read <= store_rows_read + 16'd1;
          end
          if (all_rows_read && (acc_fill == 9'd0) && !store_req) state <= IDLE;
        end

        default: ;
      endcase
    end
  end

  // These are taken at the extraction cycle so the write data
  // matches the row addressed by we/waddr, even though `acc` shifts in the
  // same cycle.
  npu_pkg::weight_vec_t wb_data_c;
  npu_pkg::act_vec_t    ab_data_c;
  npu_pkg::bias_vec_t   bb_data_c;
  always_comb begin
    wb_data_c = '{default: '0};
    ab_data_c = '{default: '0};
    bb_data_c = '{default: '0};
    if (target == BUF_WEIGHT)
      for (int c = 0; c < ARRAY_SIZE; c++) wb_data_c[c] = npu_pkg::weight_t'(acc[c*8+:8]);
    else if (target == BUF_ACT)
      for (int c = 0; c < ARRAY_SIZE; c++)
      ab_data_c[c] = (pad_active && (c >= int'(valid_bytes_reg))) ?
            npu_pkg::act_t'(8'h0) : npu_pkg::act_t'(acc[c*8+:8]);
    else if (target == BUF_BIAS)
      for (int c = 0; c < ARRAY_SIZE; c++) bb_data_c[c] = npu_pkg::bias_t'(acc[c*32+:32]);
  end

  always_comb begin
    m_axis_tdata = acc[63:0];
    m_axis_tvalid = (state == STORE_ST) &&
                    ( (acc_fill >= 9'd8) || (all_rows_read && (acc_fill > 9'd0)) );
    m_axis_tlast = (state == STORE_ST) && all_rows_read && (acc_fill > 9'd0) && (acc_fill <= 9'd8);
  end

  assign qb_dma_re = qb_re_r;

  assign s_axis_tready = (state == LOAD_ST) && (acc_fill <= LoadMaxFill);
  assign load_done = (state == LOAD_ST) && (row_index >= load_rows_total);
  assign store_done = (state == STORE_ST) && all_rows_read && (acc_fill == 9'd0);

endmodule
