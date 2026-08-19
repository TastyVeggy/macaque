import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import random

ARRAY_SIZE = 14
PE_LATENCY = 3
DRAIN_LATENCY = ARRAY_SIZE * PE_LATENCY + ARRAY_SIZE - 1


async def reset(dut):
    dut.rst.value = 1
    dut.weight_valid.value = 0
    dut.act_valid.value = 0
    dut.bias_valid.value = 0
    for i in range(ARRAY_SIZE):
        dut.weight_data[i].value = 0
        dut.act_data[i].value = 0
        dut.bias_data[i].value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


def to_signed8(val):
    return int(val) & 0xFF


def to_signed32(val):
    return int(val) & 0xFFFFFFFF


def from_signed32(val):
    val = int(val)
    if val >= (1 << 31):
        val -= 1 << 32
    return val


def ref_matmul_matrix_with_col(W, a, bias=None):
    """out[col] = sum_row(W[row][col] * a[row]) + bias[col]"""
    out = [0] * ARRAY_SIZE
    for col in range(ARRAY_SIZE):
        for row in range(ARRAY_SIZE):
            out[col] += W[row][col] * a[row]
        if bias is not None:
            out[col] += bias[col]
    return out


def ref_matmul(W, A, B=None):
    """returns A x W + B"""
    results = []
    for i, a in enumerate(A):
        b = B[i] if B is not None else None
        results.append(ref_matmul_matrix_with_col(W, a, b))
    return results


def rand_matrix(rows, cols, lo=-10, hi=10):
    return [[random.randint(lo, hi) for _ in range(cols)] for _ in range(rows)]


def rand_vec(n, lo=-10, hi=10):
    return [random.randint(lo, hi) for _ in range(n)]


async def load_weights(dut, W):
    while not dut.ready.value:
        await RisingEdge(dut.clk)

    for row in range(ARRAY_SIZE):  # 13 down to 0
        for col in range(ARRAY_SIZE):
            dut.weight_data[col].value = to_signed8(W[row][col])
        dut.weight_valid.value = 1
        await RisingEdge(dut.clk)

    dut.weight_valid.value = 0
    for col in range(ARRAY_SIZE):
        dut.weight_data[col].value = 0

    while not dut.ready.value:
        await RisingEdge(dut.clk)


def dump_weights(dut):
    """Print the weight_reg stored in each PE."""
    print("Weight registers (PE[row][col]):")
    for row in range(ARRAY_SIZE):
        row_vals = []
        for col in range(ARRAY_SIZE):
            w = int(dut.gen_row[row].gen_col[col].pe_inst.weight_reg.value)
            if w >= 128:
                w -= 256
            row_vals.append(w)
        print(f"  row {row:2d}: {row_vals}")


async def stream_row(dut, act_row, bias_row=None):
    """Present one activation and bias row for one cycle."""
    if bias_row is None:
        bias_row = [0] * ARRAY_SIZE
    for r in range(ARRAY_SIZE):
        dut.act_data[r].value = to_signed8(act_row[r])
    dut.act_valid.value = 1

    for c in range(ARRAY_SIZE):
        dut.bias_data[c].value = to_signed32(bias_row[c])
    dut.bias_valid.value = 1

    await RisingEdge(dut.clk)

    dut.act_valid.value = 0
    dut.bias_valid.value = 0
    for r in range(ARRAY_SIZE):
        dut.act_data[r].value = 0
    for c in range(ARRAY_SIZE):
        dut.bias_data[c].value = 0


async def collect_result(dut, expected_ready_cycles=1):
    for _ in range(expected_ready_cycles):
        await RisingEdge(dut.clk)
    assert dut.drain_valid.value == 1, (
        f"drain_valid is not asserted after {expected_ready_cycles} since the previous result is collected"
    )

    return [from_signed32(dut.drain_data[c].value) for c in range(ARRAY_SIZE)]


async def stream_n_and_collect(dut, act_rows: list[list[int]], bias_rows=None):
    """
    Stream N rows and collect N results concurrently.
    First result arrives after DRAIN_LATENCY cycles;
    each subsequent result arrives 1 cycle later.
    """
    N = len(act_rows)
    if bias_rows is not None:
        assert len(bias_rows) == N

    results = []

    async def streamer():
        for i in range(N):
            bias = bias_rows[i] if bias_rows is not None else None
            await stream_row(dut, act_rows[i], bias)

    async def collector():
        # First result need wait full drain
        # add 1 cause the DRAIN_LATENCY is calculated from
        # after the cycle involving the input being given
        results.append(await collect_result(dut, DRAIN_LATENCY + 1))
        # Subsequent results should then be 1 cycle apart
        for _ in range(N - 1):
            results.append(await collect_result(dut, 1))

    stream_coro = cocotb.start_soon(streamer())
    await collector()
    await stream_coro

    return results


@cocotb.test()
async def test_drain_latency(dut):
    """drain_valid must fire exactly DRAIN_LATENCY cycles after first act_valid."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    W = [[1] * ARRAY_SIZE for _ in range(ARRAY_SIZE)]
    act = [1] * ARRAY_SIZE
    bias = [0] * ARRAY_SIZE
    await load_weights(dut, W)

    assert dut.ready.value == 1, "ready not high before stream"
    # Drive act_valid for one cycle
    await stream_row(dut, act, bias)

    # Count exactly
    cycles = 0
    while not dut.drain_valid.value:
        await RisingEdge(dut.clk)
        cycles += 1
        assert cycles <= DRAIN_LATENCY, (
            f"drain_valid too late: {cycles} > {DRAIN_LATENCY}"
        )

    assert cycles == DRAIN_LATENCY, (
        f"drain latency wrong: expected {DRAIN_LATENCY}, got {cycles}"
    )
    dut._log.info(f"test_drain_latency: PASS (exactly {cycles} cycles)")


@cocotb.test()
async def test_zero_weights(dut):
    """All-zero weights: output must be zero regardless of activation (no bias)"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    W = [[0] * ARRAY_SIZE for _ in range(ARRAY_SIZE)]
    await load_weights(dut, W)

    act = [rand_vec(ARRAY_SIZE)]
    result = await stream_n_and_collect(dut, act)

    assert result == [[0] * ARRAY_SIZE], f"Expected all zeros, got {result}"
    dut._log.info("all-zero weights: PASS")


@cocotb.test()
async def test_zero_activation(dut):
    """All-zero activation: output must be zero regardless of activation (no bias)"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    W = rand_matrix(ARRAY_SIZE, ARRAY_SIZE)
    await load_weights(dut, W)

    act = [[0] * ARRAY_SIZE for _ in range(5)]
    result = await stream_n_and_collect(dut, act)

    assert result == [[0] * ARRAY_SIZE for _ in range(5)], (
        f"Expected all zeros, got {result}"
    )
    dut._log.info("all-zero weights: PASS")


@cocotb.test()
async def test_identity_weights(dut):
    """Identity weight matrix: out[col] = act[col]."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    W = [[1 if r == c else 0 for c in range(ARRAY_SIZE)] for r in range(ARRAY_SIZE)]
    await load_weights(dut, W)

    act = [[i + 1 for i in range(ARRAY_SIZE)]]
    result = await stream_n_and_collect(dut, act)

    expected = ref_matmul(W, act)
    assert result == expected, (
        f"Identity weight mismatch:  expected {expected}, got {result}"
    )
    dut._log.info(f"identity weights: PASS (Got: {result})")


@cocotb.test()
async def test_single_row_no_bias(dut):
    """Single activation row, no bias: verify matmul correctness"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    W = rand_matrix(ARRAY_SIZE, ARRAY_SIZE, lo=-5, hi=5)
    act = [rand_vec(ARRAY_SIZE, lo=-5, hi=5)]
    await load_weights(dut, W)

    result = await stream_n_and_collect(dut, act)
    expected = ref_matmul(W, act)

    assert result == expected, (
        f"Single-row matmul mismatch: expected {expected} got {result}"
    )
    dut._log.info("Single row no bias: PASS")


@cocotb.test()
async def test_single_row_with_bias(dut):
    """Single activation row with bias: verify matmul + bias correctness"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    W = rand_matrix(ARRAY_SIZE, ARRAY_SIZE, lo=-5, hi=5)
    act = [rand_vec(ARRAY_SIZE, lo=-5, hi=5)]
    bias = [rand_vec(ARRAY_SIZE, lo=-100, hi=100)]
    await load_weights(dut, W)

    result = await stream_n_and_collect(dut, act, bias)
    expected = ref_matmul(W, act, bias)

    assert result == expected, (
        f"Single-row+bias mismatch: expected {expected} got {result}"
    )
    dut._log.info("Single row with bias: PASS")


@cocotb.test()
async def test_multi_row_no_bias(dut):
    """Multiple activation rows with no bias: verify pipelined matmul correctness"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    N = 8
    W = rand_matrix(ARRAY_SIZE, ARRAY_SIZE, lo=-5, hi=5)
    act = [rand_vec(ARRAY_SIZE, lo=-5, hi=5) for _ in range(N)]
    await load_weights(dut, W)

    result = await stream_n_and_collect(dut, act)
    expected = ref_matmul(W, act)

    assert result == expected, (
        f"Multi-row (no bias) mismatch: expected {expected}, got {result}"
    )
    dut._log.info(f"Multi-row no bias ({N} rows): PASS")


@cocotb.test()
async def test_multi_row_with_bias(dut):
    """Multiple activation rows each with a different bias vector"""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    N = 6
    W = rand_matrix(ARRAY_SIZE, ARRAY_SIZE, lo=-5, hi=5)
    act = [rand_vec(ARRAY_SIZE, lo=-5, hi=5) for _ in range(N)]
    bias = [rand_vec(ARRAY_SIZE, lo=-200, hi=200) for _ in range(N)]
    await load_weights(dut, W)

    result = await stream_n_and_collect(dut, act, bias)
    expected = ref_matmul(W, act, bias)

    assert result == expected, (
        f"Multi-row+bias mismatch: expected {expected}, got {result}"
    )
    dut._log.info(f"Multi-row with bias ({N} rows): PASS")


@cocotb.test()
async def test_bias_only(dut):
    """Zero weights + non-zero bias: output must equal bias regardless of activation."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    W = [[0] * ARRAY_SIZE for _ in range(ARRAY_SIZE)]
    await load_weights(dut, W)

    N = 4
    act = [rand_vec(ARRAY_SIZE) for _ in range(N)]
    bias = [rand_vec(ARRAY_SIZE, lo=-500, hi=500) for _ in range(N)]

    result = await stream_n_and_collect(dut, act, bias)

    expected = [b[:] for b in bias]
    assert result == expected, f"Bias-only mismatch: expected {expected}, got {result}"
    dut._log.info("Bias-only (zero weights): PASS")


@cocotb.test()
async def test_weight_reload(dut):
    """Load two different weight matrices back-to-back; results must match each W."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    W1 = rand_matrix(ARRAY_SIZE, ARRAY_SIZE, lo=-5, hi=5)
    W2 = rand_matrix(ARRAY_SIZE, ARRAY_SIZE, lo=-5, hi=5)
    act1 = [rand_vec(ARRAY_SIZE, lo=-5, hi=5)]
    act2 = [rand_vec(ARRAY_SIZE, lo=-5, hi=5)]

    # First pass with W1
    await load_weights(dut, W1)
    result1 = await stream_n_and_collect(dut, act1)
    expected1 = ref_matmul(W1, act1)
    assert result1 == expected1, (
        f"Weight reload (W1) mismatch: expected {expected1}, got {result1}"
    )

    # Reload W2 and re-run
    await load_weights(dut, W2)
    result2 = await stream_n_and_collect(dut, act2)
    expected2 = ref_matmul(W2, act2)
    assert result2 == expected2, (
        f"Weight reload (W2) mismatch: expected {expected2}, got {result2}"
    )

    dut._log.info("Weight reload: PASS")


@cocotb.test()
async def test_max_positive_weights_and_acts(dut):
    """All weights and activations at +127 (int8 max): verify no silent overflow."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    W = [[127] * ARRAY_SIZE for _ in range(ARRAY_SIZE)]
    act = [[127] * ARRAY_SIZE]
    await load_weights(dut, W)

    result = await stream_n_and_collect(dut, act)
    expected = ref_matmul(W, act)

    assert result == expected, (
        f"Max-positive mismatch: expected {expected}, got {result}"
    )
    dut._log.info(f"Max positive weights/acts: PASS (output[0][0]={result[0][0]})")


@cocotb.test()
async def test_max_negative_weights_and_acts(dut):
    """All weights and activations at -128 (int8 min): result must be positive."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    W = [[-128] * ARRAY_SIZE for _ in range(ARRAY_SIZE)]
    act = [[-128] * ARRAY_SIZE]
    await load_weights(dut, W)

    result = await stream_n_and_collect(dut, act)
    expected = ref_matmul(W, act)

    assert result == expected, (
        f"Max-negative mismatch: expected {expected}, got {result}"
    )
    dut._log.info(f"Max negative weights/acts: PASS (output[0][0]={result[0][0]})")


@cocotb.test()
async def test_mixed_sign_weights(dut):
    """Alternating +/-1 weight matrix with random activations."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    # Checkerboard of +1 / -1
    W = [
        [(1 if (r + c) % 2 == 0 else -1) for c in range(ARRAY_SIZE)]
        for r in range(ARRAY_SIZE)
    ]
    act = [rand_vec(ARRAY_SIZE, lo=-10, hi=10) for _ in range(5)]
    await load_weights(dut, W)

    result = await stream_n_and_collect(dut, act)
    expected = ref_matmul(W, act)

    assert result == expected, (
        f"Mixed-sign weight mismatch:\n  expected {expected}\n  got      {result}"
    )
    dut._log.info("Mixed-sign weights (checkerboard): PASS")


@cocotb.test()
async def test_single_nonzero_weight(dut):
    """Only one PE has a non-zero weight; verifies column routing."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    for target_row in range(0, ARRAY_SIZE, 4):  # spot-check rows 0, 4, 8, 12
        await reset(dut)

        W = [[0] * ARRAY_SIZE for _ in range(ARRAY_SIZE)]
        target_col = (target_row * 3 + 1) % ARRAY_SIZE
        W[target_row][target_col] = 5

        act_val = rand_vec(ARRAY_SIZE, lo=-10, hi=10)
        act = [act_val]
        await load_weights(dut, W)

        result = await stream_n_and_collect(dut, act)
        expected = ref_matmul(W, act)

        assert result == expected, (
            f"Single non-zero PE [{target_row}][{target_col}] mismatch: "
            f"expected {expected}, got {result}"
        )

    dut._log.info("Single non-zero weight (column routing): PASS")


@cocotb.test()
async def test_large_batch(dut):
    """Stream a large batch (32 rows) to stress-test pipelining."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    N = 32
    W = rand_matrix(ARRAY_SIZE, ARRAY_SIZE, lo=-3, hi=3)
    act = [rand_vec(ARRAY_SIZE, lo=-3, hi=3) for _ in range(N)]
    bias = [rand_vec(ARRAY_SIZE, lo=-50, hi=50) for _ in range(N)]
    await load_weights(dut, W)

    result = await stream_n_and_collect(dut, act, bias)
    expected = ref_matmul(W, act, bias)

    assert result == expected, (
        f"Large-batch mismatch at one or more rows (first bad index: "
        f"{next(i for i, (r, e) in enumerate(zip(result, expected)) if r != e)})"
    )
    dut._log.info(f"Large batch ({N} rows) with bias: PASS")


@cocotb.test()
async def test_reset_clears_output(dut):
    """Assert rst mid-stream; after reset and re-load, output must be correct."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    W = rand_matrix(ARRAY_SIZE, ARRAY_SIZE, lo=-5, hi=5)
    act_dirty = [rand_vec(ARRAY_SIZE)]
    await load_weights(dut, W)

    # Fire one activation then immediately reset before drain_valid arrives
    await stream_row(dut, act_dirty[0])
    await ClockCycles(dut.clk, PE_LATENCY)  # part-way through pipeline
    await reset(dut)  # hard reset

    # Load fresh weights and run a clean computation
    W2 = rand_matrix(ARRAY_SIZE, ARRAY_SIZE, lo=-5, hi=5)
    act_clean = [rand_vec(ARRAY_SIZE, lo=-5, hi=5)]
    await load_weights(dut, W2)

    result = await stream_n_and_collect(dut, act_clean)
    expected = ref_matmul(W2, act_clean)

    assert result == expected, (
        f"Post-reset matmul mismatch: expected {expected}, got {result}"
    )
    dut._log.info("Reset clears pipeline: PASS")


@cocotb.test()
async def test_drain_valid_deasserts(dut):
    """drain_valid must deassert the cycle after the last result is read."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    W = [[1] * ARRAY_SIZE for _ in range(ARRAY_SIZE)]
    await load_weights(dut, W)

    act = [rand_vec(ARRAY_SIZE)]
    await stream_row(dut, act[0])

    # Wait for drain_valid
    for _ in range(DRAIN_LATENCY + 2):
        await RisingEdge(dut.clk)
        if dut.drain_valid.value:
            break
    else:
        assert False, "drain_valid never asserted"

    # The very next cycle it should deassert (single row, no more data)
    await RisingEdge(dut.clk)
    assert dut.drain_valid.value == 0, (
        "drain_valid did not deassert after single-row drain"
    )
    dut._log.info("drain_valid deasserts after last result: PASS")


@cocotb.test()
async def test_consecutive_back_to_back_throughput(dut):
    """
    Stream rows back-to-back with no gaps; verify every output arrives
    exactly one cycle after the previous one (pipeline stays full).
    """
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    N = 10
    W = rand_matrix(ARRAY_SIZE, ARRAY_SIZE, lo=-4, hi=4)
    act = [rand_vec(ARRAY_SIZE, lo=-4, hi=4) for _ in range(N)]
    await load_weights(dut, W)

    results = await stream_n_and_collect(dut, act)
    expected = ref_matmul(W, act)

    assert results == expected, (
        f"Back-to-back throughput mismatch:\n  expected {expected}\n  got      {results}"
    )
    dut._log.info(f"Back-to-back throughput ({N} rows, no gaps): PASS")


@cocotb.test()
async def test_ready_deasserts_during_weight_load(dut):
    """ready must be low while weights are being loaded."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    assert dut.ready.value == 1, "ready should be high after reset"

    W = rand_matrix(ARRAY_SIZE, ARRAY_SIZE)

    while not dut.ready.value:
        await RisingEdge(dut.clk)

    # Drive weight_valid for the first row.
    # ready is a registered output: the FSM sets ready<=0 on the rising edge
    # where weight_valid is seen, so it only reflects low one cycle later.
    for col in range(ARRAY_SIZE):
        dut.weight_data[col].value = to_signed8(W[ARRAY_SIZE - 1][col])
    dut.weight_valid.value = 1
    await RisingEdge(dut.clk)  # FSM latches weight_valid, begins WEIGHT_LOAD
    await RisingEdge(dut.clk)  # ready<=0 now visible on outputs

    assert dut.ready.value == 0, "ready should be deasserted during weight loading"

    for row in range(ARRAY_SIZE - 2, -1, -1):
        for col in range(ARRAY_SIZE):
            dut.weight_data[col].value = to_signed8(W[row][col])
        dut.weight_valid.value = 1
        await RisingEdge(dut.clk)

    dut.weight_valid.value = 0
    for col in range(ARRAY_SIZE):
        dut.weight_data[col].value = 0

    while not dut.ready.value:
        await RisingEdge(dut.clk)

    assert dut.ready.value == 1, "ready should re-assert after weight load"
    dut._log.info("ready deasserts during weight load: PASS")


@cocotb.test()
async def test_all_ones_weights_accumulation(dut):
    """
    All-ones weight matrix: each output column = sum of all activations.
    Verifies the accumulator tree sums every row contribution.
    """
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    W = [[1] * ARRAY_SIZE for _ in range(ARRAY_SIZE)]
    await load_weights(dut, W)

    N = 5
    act = [rand_vec(ARRAY_SIZE, lo=-10, hi=10) for _ in range(N)]
    result = await stream_n_and_collect(dut, act)
    expected = ref_matmul(W, act)

    for i, (r, e) in enumerate(zip(result, expected)):
        assert r == e, f"All-ones accumulation row {i}: expected {e}, got {r}"
        # Sanity: every element of one row must equal sum(act[i])
        row_sum = sum(act[i])
        for col, val in enumerate(r):
            assert val == row_sum, (
                f"All-ones: output[{i}][{col}]={val} != sum(act[{i}])={row_sum}"
            )

    dut._log.info("All-ones weights accumulation: PASS")


@cocotb.test()
async def test_large_bias_dominates(dut):
    """Very large bias with tiny weights; output must be dominated by bias."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    W = rand_matrix(ARRAY_SIZE, ARRAY_SIZE, lo=-1, hi=1)
    await load_weights(dut, W)

    act = [rand_vec(ARRAY_SIZE, lo=-1, hi=1)]
    bias = [rand_vec(ARRAY_SIZE, lo=10_000, hi=100_000)]

    result = await stream_n_and_collect(dut, act, bias)
    expected = ref_matmul(W, act, bias)

    assert result == expected, f"Large-bias mismatch: expected {expected}, got {result}"
    dut._log.info("Large bias dominates: PASS")


@cocotb.test()
async def test_negative_bias(dut):
    """Large negative bias: verifies 32-bit signed arithmetic for bias."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    W = rand_matrix(ARRAY_SIZE, ARRAY_SIZE, lo=-5, hi=5)
    await load_weights(dut, W)

    act = [rand_vec(ARRAY_SIZE, lo=-5, hi=5)]
    bias = [rand_vec(ARRAY_SIZE, lo=-100_000, hi=-10_000)]

    result = await stream_n_and_collect(dut, act, bias)
    expected = ref_matmul(W, act, bias)

    assert result == expected, (
        f"Negative-bias mismatch: expected {expected}, got {result}"
    )
    dut._log.info("Negative bias: PASS")


@cocotb.test()
async def test_random_stress(dut):
    """Randomised stress: 20 independent (weight, act, bias) triples."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    NUM_TRIALS = 20
    for trial in range(NUM_TRIALS):
        W = rand_matrix(ARRAY_SIZE, ARRAY_SIZE, lo=-8, hi=8)
        N = random.randint(1, 8)
        act = [rand_vec(ARRAY_SIZE, lo=-8, hi=8) for _ in range(N)]
        bias = [rand_vec(ARRAY_SIZE, lo=-200, hi=200) for _ in range(N)]

        await load_weights(dut, W)
        result = await stream_n_and_collect(dut, act, bias)
        expected = ref_matmul(W, act, bias)

        assert result == expected, (
            f"Stress trial {trial} failed:\n"
            f"  W={W}\n  act={act}\n  bias={bias}\n"
            f"  expected={expected}\n  got={result}"
        )

    dut._log.info(f"Random stress ({NUM_TRIALS} trials): PASS")


@cocotb.test()
async def test_tiled_matmul_accumulate(dut):
    """acc_mode=1 (tiled matmul) semantics: out = previous_partial + A*W, no bias.

    Tile 1 (acc_mode=0): O0 = B0 + A0*W0   (accumulator seed = bias)
    Tile 2 (acc_mode=1): O1 = O0 + A1*W1   (accumulator seed = O0 partial sums)
    """
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    await reset(dut)

    N = 4
    W0 = rand_matrix(ARRAY_SIZE, ARRAY_SIZE, lo=-8, hi=8)
    W1 = rand_matrix(ARRAY_SIZE, ARRAY_SIZE, lo=-8, hi=8)
    A0 = [rand_vec(ARRAY_SIZE, lo=-8, hi=8) for _ in range(N)]
    A1 = [rand_vec(ARRAY_SIZE, lo=-8, hi=8) for _ in range(N)]
    B0 = rand_vec(ARRAY_SIZE, lo=-200, hi=200)

    # Tile 1: seed = bias (acc_mode=0)
    await load_weights(dut, W0)
    partial = await stream_n_and_collect(dut, A0, bias_rows=[B0] * N)
    expected_partial = ref_matmul(W0, A0, [B0] * N)
    assert partial == expected_partial, (
        f"Tile 1 (acc_mode=0) mismatch: expected {expected_partial}, got {partial}"
    )

    # Tile 2: seed = tile-1 partial sums (acc_mode=1 accumulate)
    await load_weights(dut, W1)
    result = await stream_n_and_collect(dut, A1, bias_rows=partial)
    expected = ref_matmul(W1, A1, partial)
    assert result == expected, (
        f"Tile 2 (acc_mode=1) accumulate mismatch:\n"
        f"  W1={W1}\n  A1={A1}\n  partial={partial}\n"
        f"  expected={expected}\n  got={result}"
    )

    dut._log.info("Tiled matmul accumulate (acc_mode=0 then acc_mode=1): PASS")
