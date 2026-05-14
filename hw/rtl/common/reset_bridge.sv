module reset_bridge (
    input  logic clk,
    input  logic rst_n,
    output logic rst_high
);
  logic [1:0] sync_reg;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sync_reg <= 2'b11;
    end else begin
      sync_reg <= {sync_reg[0], 1'b0};
    end
  end

  assign rst_high = sync_reg[1];
endmodule
