`timescale 1ns / 1ps

module pe_tb ();

  logic clk, rst;
  npu_pkg::weight_t weight_in;
  logic weight_in_valid;
  npu_pkg::weight_t weight_out;
  logic weight_out_valid;
  logic weight_hold;
  npu_pkg::act_t act_in;
  logic act_in_valid;
  npu_pkg::act_t act_out;
  logic act_out_valid;
  npu_pkg::acc_t acc_in;
  logic acc_in_valid;
  npu_pkg::acc_t acc_out;
  logic acc_out_valid;

  pe dut (.*);

  initial clk = 0;
  always #10 clk = ~clk;

  task automatic tick();
    @(posedge clk);
    #1;
  endtask

  task automatic load_weight(input npu_pkg::weight_t w);
    weight_hold = 0;
    weight_in = w;
    weight_in_valid = 1;
    tick();
    weight_in_valid = 0;
    weight_in = 0;
  endtask

  task automatic drive_pe(input npu_pkg::act_t a, input npu_pkg::acc_t c);
    weight_hold = 1;
    act_in = a;
    acc_in = c;
    act_in_valid = 1;
    acc_in_valid = 1;
    tick();
    act_in_valid = 0;
    acc_in_valid = 0;
    act_in = 0;
    acc_in = 0;
    // act_out should be now valid
    // acc_out should be valid 2 more ticks from here
  endtask

  task automatic reset();
    rst = 1;
    weight_hold = 0;
    weight_in = 0;
    weight_in_valid = 0;
    act_in = 0;
    act_in_valid    = 0;
    acc_in = 0;
    acc_in_valid    = 0;
    repeat (10) @(posedge clk);
    rst = 0;
    tick();
  endtask

  int pass_cnt, fail_cnt;

  task automatic check(input string name, input int got, input int exp);
    if (got === exp) begin
      $display("  PASS | %-40s | got=%0d", name, got);
      pass_cnt++;
    end else begin
      $display("  FAIL | %-40s | exp=%0d got=%0d", name, exp, got);
      fail_cnt++;
    end
  endtask

  task automatic check_valid(input string name, input logic got, input logic exp);
    if (got === exp) begin
      $display("  PASS | %-40s | got=%0b", name, got);
      pass_cnt++;
    end else begin
      $display("  FAIL | %-40s | exp=%0b got=%0b", name, exp, got);
      fail_cnt++;
    end
  endtask

  int   stream_count;
  int   stream_expected[3];
  event stream_done;

  always @(posedge clk) begin
    if (acc_out_valid && stream_count < 3) begin
      $display("  Stream[%0d]: got=%0d exp=%0d %s", stream_count, $signed(acc_out),
               stream_expected[stream_count], ($signed(acc_out)
               === stream_expected[stream_count]) ? "PASS" : "FAIL");
      stream_count <= stream_count + 1;
      if (stream_count == 2)->stream_done;
    end
  end

  initial begin
    pass_cnt = 0;
    fail_cnt = 0;
    stream_count = 0;

    reset();

    // Test 1: Weight load and basic MAC
    $display("\nTest 1: weight=3, act=4, acc_in=100 → 112");
    load_weight(8'sd3);
    repeat (2) tick();  // let weight settle

    drive_pe(8'sd4, 32'sd100);
    // #1 after posedge: act_out valid now
    check_valid("act_out_valid high", act_out_valid, 1'b1);
    check("act_out = 4", int'($signed(act_out)), 4);
    check_valid("acc_out_valid low", acc_out_valid, 1'b0);

    tick();  // cycle +2
    check_valid("acc_out_valid low+2", acc_out_valid, 1'b0);

    tick();  // cycle +3 - acc_out valid
    check_valid("acc_out_valid high", acc_out_valid, 1'b1);
    check("acc_out = 112", int'($signed(acc_out)), 112);

    tick();  // cycle +4
    check_valid("acc_out_valid clears", acc_out_valid, 1'b0);

    repeat (3) tick();

    // Test 2: act_out is 1 cycle
    $display("\nTest 2: act_out 1-cycle delay");
    reset();
    load_weight(8'sd1);
    repeat (2) tick();

    drive_pe(8'b0101_1010, 32'sd0);
    // #1 after posedge: act_out valid now
    check("act_out=0x5A at +1", int'(act_out), 8'b0101_1010);
    check_valid("act_out_valid=1 +1", act_out_valid, 1'b1);
    check_valid("acc_out_valid=0 +1", acc_out_valid, 1'b0);

    tick();  // cycle +2
    check("act_out=0 at +2", int'(act_out), 0);
    check_valid("act_out_valid=0 +2", act_out_valid, 1'b0);
    check_valid("acc_out_valid=0 +2", acc_out_valid, 1'b0);

    tick();  // cylce +3
    check_valid("acc_out_valid=1 +3", acc_out_valid, 1'b1);

    repeat (3) tick();

    // Test 3: 5 values back to back
    $display("\nTest 3: back-to-back act passthrough");
    reset();
    load_weight(8'sd0);
    repeat (2) tick();

    begin
      automatic npu_pkg::act_t s[5] = '{10, 20, 30, 40, 50};
      for (int i = 0; i < 5; i++) begin
        act_in = s[i];
        act_in_valid = 1;
        acc_in = 0;
        acc_in_valid = 0;
        tick();  // posedge captures s[i], #1 act_out=s[i]
        check($sformatf("passthrough[%0d]", i), int'(act_out), int'(s[i]));
        check_valid($sformatf("pass_vld[%0d]", i), act_out_valid, 1'b1);
      end
      act_in_valid = 0;
    end

    repeat (5) tick();

    // Test 4: streaming
    $display("\nTest 4: streaming 3 ops weight=3");
    reset();
    load_weight(8'sd3);
    repeat (2) tick();

    // 1*3+10=13, 2*3+10=16, 3*3+10=19
    stream_count = 0;
    stream_expected[0] = 13;
    stream_expected[1] = 16;
    stream_expected[2] = 19;

    drive_pe(8'sd1, 32'sd10);
    drive_pe(8'sd2, 32'sd10);
    drive_pe(8'sd3, 32'sd10);

    @stream_done;

    repeat (5) tick();

    // Test 5: signed
    $display("\nTest 5: signed");
    reset();
    load_weight(-8'sd5);
    repeat (2) tick();
    drive_pe(8'sd7, 32'sd0);
    tick();
    tick();  // wait 3 cycles
    check("weight=-5 act=7: -35", int'($signed(acc_out)), -35);

    repeat (3) tick();

    // Test 6: Verify weight_hold blocks updating internal register, but allows chain flow
    $display("\nTest 6: weight_hold validation");
    reset();
    load_weight(8'sd10);  // Latch initial weight = 10
    repeat (2) tick();

    // Turn on hold, while passing a new weight down the chain
    weight_hold = 1;
    weight_in = 8'sd99;
    weight_in_valid = 1;
    tick();

    // Check that the new weight still propagates downstream on the 1-cycle pipeline
    check("weight_out passthrough", int'($signed(weight_out)), 99);
    check_valid("weight_out_valid passthrough", weight_out_valid, 1'b1);

    weight_in = 0;
    weight_in_valid = 0;

    // Perform computation to confirm the active internal weight is still 10, NOT 99
    drive_pe(8'sd5, 32'sd0);  // 10 * 5 + 0 = 50
    tick();
    tick();
    check("Retained weight evaluation (exp 50)", int'($signed(acc_out)), 50);

    $display("\nResult: %0d PASS  %0d FAIL", pass_cnt, fail_cnt);
    $finish;
  end

endmodule
