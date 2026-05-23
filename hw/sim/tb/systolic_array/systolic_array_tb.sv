`timescale 1ns / 1ps

import npu_pkg::*;


module systolic_array_tb;

  logic clk, rst;

  weight_vec_t weight_data;
  logic weight_valid;

  act_vec_t act_data;
  logic act_valid;

  bias_vec_t bias_data;
  logic bias_valid;

  acc_vec_t drain_data;
  logic drain_valid;

  logic ready;

  systolic_array dut (.*);

  initial clk = 0;
  always #10 clk = ~clk;

  typedef int mat_t[ARRAY_SIZE][ARRAY_SIZE];
  typedef int vec_t[ARRAY_SIZE];

  `define ASSERT(cond, msg) \
  do begin \
    if (!(cond)) begin \
      $error("FAIL  %s  (@%0t)", msg, $time); \
      $finish; \
    end \
  end while (0)

  function automatic vec_t golden(input mat_t W, input vec_t a, input vec_t bias);
    vec_t o;
    for (int col = 0; col < ARRAY_SIZE; col++) begin
      o[col] = bias[col];
      for (int row = 0; row < ARRAY_SIZE; row++) o[col] += W[row][col] * a[row];
    end
    return o;
  endfunction

  function automatic vec_t zero_vec();
    vec_t v;
    foreach (v[i]) v[i] = 0;
    return v;
  endfunction

  function automatic int ri8(input int lo, input int hi);
    return lo + int'($urandom % unsigned'(hi - lo + 1));
  endfunction

  function automatic int ri32(input int lo, input int hi);
    longint range = longint'(hi) - longint'(lo) + 1;
    return lo + int'($urandom % unsigned'(range));
  endfunction

  function automatic vec_t rand_vec(input int lo, input int hi);
    vec_t v;
    foreach (v[i]) v[i] = ri8(lo, hi);
    return v;
  endfunction

  function automatic mat_t rand_mat(input int lo, input int hi);
    mat_t m;
    for (int r = 0; r < ARRAY_SIZE; r++) for (int c = 0; c < ARRAY_SIZE; c++) m[r][c] = ri8(lo, hi);
    return m;
  endfunction

  function automatic mat_t fill_mat(input int val);
    mat_t m;
    for (int r = 0; r < ARRAY_SIZE; r++) for (int c = 0; c < ARRAY_SIZE; c++) m[r][c] = val;
    return m;
  endfunction

  task automatic reset();
    rst          <= 1;
    weight_valid <= 0;
    act_valid    <= 0;
    bias_valid   <= 0;
    foreach (weight_data[i]) weight_data[i] <= '0;
    foreach (act_data[i]) act_data[i] <= '0;
    foreach (bias_data[i]) bias_data[i] <= '0;
    repeat (4) @(posedge clk);
    rst <= 0;
    @(posedge clk);
    `ASSERT(ready, "ready must be high after reset");
  endtask

  task automatic load_weights(input mat_t W);
    while (!ready) @(posedge clk);

    for (int row = 0; row < ARRAY_SIZE; row++) begin
      foreach (weight_data[col]) weight_data[col] <= W[row][col];
      weight_valid <= 1;
      @(posedge clk);
    end
    weight_valid <= 0;
    foreach (weight_data[i]) weight_data[i] <= '0;

    while (!ready) @(posedge clk);
  endtask

  task automatic drive_row(input vec_t a, input vec_t b);
    foreach (act_data[i]) act_data[i] <= a[i];
    foreach (bias_data[i]) bias_data[i] <= b[i];
    act_valid  <= 1;
    bias_valid <= 1;
    @(posedge clk);
    act_valid  <= 0;
    bias_valid <= 0;
    foreach (act_data[i]) act_data[i] <= '0;
    foreach (bias_data[i]) bias_data[i] <= '0;
  endtask

  task automatic collect(input int timeout, output vec_t result);
    int cnt = 0;
    @(posedge clk);
    while (!drain_valid) begin
      `ASSERT(cnt < timeout, $sformatf("drain_valid never asserted (waited %0d cycles)", timeout));
      cnt++;
      @(posedge clk);
    end
    foreach (result[c]) result[c] = int'(drain_data[c]);
  endtask

  task automatic run_batch(input vec_t acts[], input vec_t biases[], output vec_t results[]);
    int N = acts.size();
    results = new[N];

    fork
      begin : streamer
        for (int i = 0; i < N; i++) drive_row(acts[i], biases[i]);
      end
      begin : collector
        collect(SYSTOLIC_ARRAY_LATENCY + 1, results[0]);
        for (int i = 1; i < N; i++)
        collect(1, results[i]);  // back-to-back; allow 1 extra cycle of slack
      end
    join
  endtask

  task automatic check_batch(input mat_t W, input vec_t acts[], input vec_t biases[],
                             input vec_t results[], input string tag);
    for (int i = 0; i < acts.size(); i++) begin
      vec_t exp = golden(W, acts[i], biases[i]);
      for (int c = 0; c < ARRAY_SIZE; c++)
      `ASSERT(results[i][c] === exp[c], $sformatf(
              "%s  row=%0d col=%0d  expected=%0d  got=%0d", tag, i, c, exp[c], results[i][c]));
    end
    $display("PASS  %s", tag);
  endtask

  task automatic test_drain_latency();
    mat_t W;
    vec_t a;
    int   cycles;
    for (int r = 0; r < ARRAY_SIZE; r++) for (int c = 0; c < ARRAY_SIZE; c++) W[r][c] = 1;
    for (int i = 0; i < ARRAY_SIZE; i++) a[i] = 1;

    reset();
    load_weights(W);
    drive_row(a, zero_vec());

    cycles = 0;
    while (!drain_valid) begin
      cycles++;
      `ASSERT(cycles <= SYSTOLIC_ARRAY_LATENCY, $sformatf(
              "drain_valid too late: %0d > %0d", cycles, SYSTOLIC_ARRAY_LATENCY));
      @(posedge clk);
    end
    `ASSERT(cycles === SYSTOLIC_ARRAY_LATENCY, $sformatf(
            "wrong latency: expected %0d got %0d", SYSTOLIC_ARRAY_LATENCY, cycles));
    $display("PASS  Test drain_latency (%0d cycles)", cycles);
  endtask

  task automatic test_drain_valid_deasserts();
    mat_t W;
    vec_t result;
    for (int r = 0; r < ARRAY_SIZE; r++) for (int c = 0; c < ARRAY_SIZE; c++) W[r][c] = 1;

    reset();
    load_weights(W);
    drive_row(rand_vec(-10, 10), zero_vec());
    collect(SYSTOLIC_ARRAY_LATENCY + 2, result);

    @(posedge clk);
    `ASSERT(!drain_valid, "drain_valid did not deassert after single-row drain");
    $display("PASS  Test drain_valid_deasserts");
  endtask

  task automatic test_ready_during_weight_load();
    mat_t W = rand_mat(-10, 10);

    reset();
    while (!ready) @(posedge clk);

    foreach (weight_data[col]) weight_data[col] <= W[0][col];
    weight_valid <= 1;
    @(posedge clk);  // FSM latches weight_valid
    @(posedge clk);  // ready<=0 now visible
    `ASSERT(!ready, "ready should be low during weight load");

    for (int row = 1; row < ARRAY_SIZE; row++) begin
      foreach (weight_data[col]) weight_data[col] <= W[row][col];
      @(posedge clk);
    end
    weight_valid <= 0;
    foreach (weight_data[i]) weight_data[i] <= '0;

    while (!ready) @(posedge clk);
    `ASSERT(ready, "ready should reassert after weight load");
    $display("PASS  test ready_during_weight_load");
  endtask

  task automatic test_zero_weights();
    mat_t W;
    vec_t acts[], biases[], results[];
    for (int r = 0; r < ARRAY_SIZE; r++) for (int c = 0; c < ARRAY_SIZE; c++) W[r][c] = 0;

    acts = new[1];
    biases = new[1];
    acts[0] = rand_vec(-10, 10);
    biases[0] = zero_vec();

    reset();
    load_weights(W);
    run_batch(acts, biases, results);
    check_batch(W, acts, biases, results, "Test zero_weights");
  endtask

  task automatic test_zero_activations();
    mat_t W = rand_mat(-10, 10);
    vec_t acts[], biases[], results[];
    int N = 5;

    acts   = new[N];
    biases = new[N];
    for (int i = 0; i < N; i++) begin
      acts[i]   = zero_vec();
      biases[i] = zero_vec();
    end

    reset();
    load_weights(W);
    run_batch(acts, biases, results);
    check_batch(W, acts, biases, results, "Test zero_activations");
  endtask

  task automatic test_identity_weights();
    mat_t W;
    vec_t acts[], biases[], results[];
    for (int r = 0; r < ARRAY_SIZE; r++)
      for (int c = 0; c < ARRAY_SIZE; c++) W[r][c] = (r == c) ? 1 : 0;

    acts   = new[1];
    biases = new[1];
    for (int i = 0; i < ARRAY_SIZE; i++) acts[0][i] = i + 1;
    biases[0] = zero_vec();

    reset();
    load_weights(W);
    run_batch(acts, biases, results);
    check_batch(W, acts, biases, results, "Test identity_weights");
  endtask

  task automatic test_single_row_no_bias();
    mat_t W = rand_mat(-5, 5);
    vec_t acts[], biases[], results[];

    acts = new[1];
    biases = new[1];
    acts[0] = rand_vec(-5, 5);
    biases[0] = zero_vec();

    reset();
    load_weights(W);
    run_batch(acts, biases, results);
    check_batch(W, acts, biases, results, "Test single_row_no_bias");
  endtask

  task automatic test_single_row_with_bias();
    mat_t W = rand_mat(-5, 5);
    vec_t acts[], biases[], results[];

    acts = new[1];
    biases = new[1];
    acts[0] = rand_vec(-5, 5);
    for (int c = 0; c < ARRAY_SIZE; c++) biases[0][c] = ri32(-100, 100);

    reset();
    load_weights(W);
    run_batch(acts, biases, results);
    check_batch(W, acts, biases, results, "Test single_row_with_bias");
  endtask

  task automatic test_multi_row_no_bias();
    localparam int N = 8;
    mat_t W = rand_mat(-5, 5);
    vec_t acts[], biases[], results[];

    acts   = new[N];
    biases = new[N];
    for (int i = 0; i < N; i++) begin
      acts[i]   = rand_vec(-5, 5);
      biases[i] = zero_vec();
    end

    reset();
    load_weights(W);
    run_batch(acts, biases, results);
    check_batch(W, acts, biases, results, "Test multi_row_no_bias");
  endtask

  task automatic test_multi_row_with_bias();
    localparam int N = 6;
    mat_t W = rand_mat(-5, 5);
    vec_t acts[], biases[], results[];

    acts   = new[N];
    biases = new[N];
    for (int i = 0; i < N; i++) begin
      acts[i] = rand_vec(-5, 5);
      for (int c = 0; c < ARRAY_SIZE; c++) biases[i][c] = ri32(-200, 200);
    end

    reset();
    load_weights(W);
    run_batch(acts, biases, results);
    check_batch(W, acts, biases, results, "Test multi_row_with_bias");
  endtask

  task automatic test_bias_only();
    localparam int N = 4;
    mat_t W;
    vec_t acts[], biases[], results[];
    for (int r = 0; r < ARRAY_SIZE; r++) for (int c = 0; c < ARRAY_SIZE; c++) W[r][c] = 0;

    acts   = new[N];
    biases = new[N];
    for (int i = 0; i < N; i++) begin
      acts[i] = rand_vec(-10, 10);
      for (int c = 0; c < ARRAY_SIZE; c++) biases[i][c] = ri32(-500, 500);
    end

    reset();
    load_weights(W);
    run_batch(acts, biases, results);
    check_batch(W, acts, biases, results, "Test bias_only");
  endtask

  task automatic test_weight_reload();
    mat_t W1 = rand_mat(-5, 5);
    mat_t W2 = rand_mat(-5, 5);
    vec_t acts[], biases[], results[];

    acts = new[1];
    biases = new[1];
    acts[0] = rand_vec(-5, 5);
    biases[0] = zero_vec();

    reset();
    load_weights(W1);
    run_batch(acts, biases, results);
    check_batch(W1, acts, biases, results, "Test weight_reload (1st weights)");

    acts[0] = rand_vec(-5, 5);
    load_weights(W2);
    run_batch(acts, biases, results);
    check_batch(W2, acts, biases, results, "Test weight_reload (2nd weights)");
  endtask

  task automatic test_max_positive();
    mat_t W;
    vec_t acts[], biases[], results[];
    for (int r = 0; r < ARRAY_SIZE; r++) for (int c = 0; c < ARRAY_SIZE; c++) W[r][c] = 127;

    acts   = new[1];
    biases = new[1];
    for (int i = 0; i < ARRAY_SIZE; i++) acts[0][i] = 127;
    biases[0] = zero_vec();

    reset();
    load_weights(W);
    run_batch(acts, biases, results);
    check_batch(W, acts, biases, results, "Test max_positive");
  endtask

  task automatic test_max_negative();
    mat_t W;
    vec_t acts[], biases[], results[];
    for (int r = 0; r < ARRAY_SIZE; r++) for (int c = 0; c < ARRAY_SIZE; c++) W[r][c] = -128;

    acts   = new[1];
    biases = new[1];
    for (int i = 0; i < ARRAY_SIZE; i++) acts[0][i] = -128;
    biases[0] = zero_vec();

    reset();
    load_weights(W);
    run_batch(acts, biases, results);
    check_batch(W, acts, biases, results, "Test max_negative");
  endtask

  task automatic test_checkerboard_weights();
    localparam int N = 5;
    mat_t W;
    vec_t acts[], biases[], results[];
    for (int r = 0; r < ARRAY_SIZE; r++)
      for (int c = 0; c < ARRAY_SIZE; c++) W[r][c] = ((r + c) % 2 == 0) ? 1 : -1;

    acts   = new[N];
    biases = new[N];
    for (int i = 0; i < N; i++) begin
      acts[i]   = rand_vec(-10, 10);
      biases[i] = zero_vec();
    end

    reset();
    load_weights(W);
    run_batch(acts, biases, results);
    check_batch(W, acts, biases, results, "Test checkerboard_weights");
  endtask

  task automatic test_single_nonzero_weight();
    mat_t W;
    vec_t acts[], biases[], results[];

    for (int trow = 0; trow < ARRAY_SIZE; trow += 4) begin
      int tcol = (trow * 3 + 1) % ARRAY_SIZE;
      for (int r = 0; r < ARRAY_SIZE; r++) for (int c = 0; c < ARRAY_SIZE; c++) W[r][c] = 0;
      W[trow][tcol] = 5;

      acts = new[1];
      biases = new[1];
      acts[0] = rand_vec(-10, 10);
      biases[0] = zero_vec();

      reset();
      load_weights(W);
      run_batch(acts, biases, results);
      check_batch(W, acts, biases, results, $sformatf("Test single_nonzero [%0d][%0d]", trow, tcol
                  ));
    end
  endtask

  task automatic test_all_ones_accumulation();
    localparam int N = 5;
    mat_t W;
    vec_t acts[], biases[], results[];
    for (int r = 0; r < ARRAY_SIZE; r++) for (int c = 0; c < ARRAY_SIZE; c++) W[r][c] = 1;

    acts   = new[N];
    biases = new[N];
    for (int i = 0; i < N; i++) begin
      acts[i]   = rand_vec(-10, 10);
      biases[i] = zero_vec();
    end

    reset();
    load_weights(W);
    run_batch(acts, biases, results);
    check_batch(W, acts, biases, results, "Test all_ones_accumulation");

    for (int i = 0; i < N; i++) begin
      int row_sum = 0;
      for (int r = 0; r < ARRAY_SIZE; r++) row_sum += acts[i][r];
      for (int c = 0; c < ARRAY_SIZE; c++)
      `ASSERT(results[i][c] === row_sum, $sformatf(
              "Test: output[%0d][%0d]=%0d != sum=%0d", i, c, results[i][c], row_sum));
    end
    $display("PASS  Test all_ones_accumulation (sum check)");
  endtask

  task automatic test_large_bias_dominates();
    mat_t W = rand_mat(-1, 1);
    vec_t acts[], biases[], results[];

    acts = new[1];
    biases = new[1];
    acts[0] = rand_vec(-1, 1);
    for (int c = 0; c < ARRAY_SIZE; c++) biases[0][c] = ri32(10_000, 100_000);

    reset();
    load_weights(W);
    run_batch(acts, biases, results);
    check_batch(W, acts, biases, results, "Test large_bias_dominates");
  endtask

  task automatic test_negative_bias();
    mat_t W = rand_mat(-5, 5);
    vec_t acts[], biases[], results[];

    acts = new[1];
    biases = new[1];
    acts[0] = rand_vec(-5, 5);
    for (int c = 0; c < ARRAY_SIZE; c++) biases[0][c] = ri32(-100_000, -10_000);

    reset();
    load_weights(W);
    run_batch(acts, biases, results);
    check_batch(W, acts, biases, results, "Test negative_bias");
  endtask

  task automatic test_large_batch();
    localparam int N = 32;
    mat_t W = rand_mat(-3, 3);
    vec_t acts[], biases[], results[];

    acts   = new[N];
    biases = new[N];
    for (int i = 0; i < N; i++) begin
      acts[i] = rand_vec(-3, 3);
      for (int c = 0; c < ARRAY_SIZE; c++) biases[i][c] = ri32(-50, 50);
    end

    reset();
    load_weights(W);
    run_batch(acts, biases, results);
    check_batch(W, acts, biases, results, "Test large_batch");
  endtask

  task automatic test_reset_clears_pipeline();
    mat_t W1 = rand_mat(-5, 5);
    mat_t W2 = rand_mat(-5, 5);
    vec_t acts[], biases[], results[];

    reset();
    load_weights(W1);
    drive_row(rand_vec(-5, 5), zero_vec());  // dirty data enters pipeline
    repeat (PE_LATENCY) @(posedge clk);  // part-way through
    reset();  // hard reset

    acts = new[1];
    biases = new[1];
    acts[0] = rand_vec(-5, 5);
    biases[0] = zero_vec();

    load_weights(W2);
    run_batch(acts, biases, results);
    check_batch(W2, acts, biases, results, "Test reset_clears_pipeline");
  endtask

  task automatic test_random_stress();
    localparam int NUM_TRIALS = 20;

    for (int trial = 0; trial < NUM_TRIALS; trial++) begin
      mat_t W = rand_mat(-8, 8);
      int   N = 1 + int'($urandom % 8);
      vec_t acts[], biases[], results[];

      acts   = new[N];
      biases = new[N];
      for (int i = 0; i < N; i++) begin
        acts[i] = rand_vec(-8, 8);
        for (int c = 0; c < ARRAY_SIZE; c++) biases[i][c] = ri32(-200, 200);
      end

      reset();
      load_weights(W);
      run_batch(acts, biases, results);
      check_batch(W, acts, biases, results, $sformatf("Test random_stress trial=%0d N=%0d", trial, N
                  ));
    end
  endtask

  initial begin
    rst          = 1;
    weight_valid = 0;
    act_valid    = 0;
    bias_valid   = 0;
    foreach (weight_data[i]) weight_data[i] = '0;
    foreach (act_data[i]) act_data[i] = '0;
    foreach (bias_data[i]) bias_data[i] = '0;

    test_drain_latency();
    test_drain_valid_deasserts();
    test_ready_during_weight_load();
    test_zero_weights();
    test_zero_activations();
    test_identity_weights();
    test_single_row_no_bias();
    test_single_row_with_bias();
    test_multi_row_no_bias();
    test_multi_row_with_bias();
    test_bias_only();
    test_weight_reload();
    test_max_positive();
    test_max_negative();
    test_checkerboard_weights();
    test_single_nonzero_weight();
    test_all_ones_accumulation();
    test_large_bias_dominates();
    test_negative_bias();
    test_large_batch();
    test_reset_clears_pipeline();
    test_random_stress();
    $display("All Passed");
    $finish;
  end

endmodule
