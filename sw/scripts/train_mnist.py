#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import struct
import urllib.request

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def load_mnist(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "mnist.npz")
    if not os.path.exists(path):
        print(f"downloading MNIST to {path} ...")
        urllib.request.urlretrieve(MNIST_URL, path)
    with np.load(path) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]
    return x_train, y_train, x_test, y_test


def init_layer(fan_in, fan_out, rng):
    # He init, appropriate for a ReLU hidden layer.
    w = rng.standard_normal((fan_in, fan_out)).astype(np.float32) * np.sqrt(2.0 / fan_in)
    b = np.zeros(fan_out, dtype=np.float32)
    return w, b


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train(x_train, y_train, x_test, y_test, hidden, epochs, batch_size, lr, seed=0):
    rng = np.random.default_rng(seed)
    n_in, n_hidden, n_out = 784, hidden, 10

    w1, b1 = init_layer(n_in, n_hidden, rng)
    w2, b2 = init_layer(n_hidden, n_out, rng)

    x_train = x_train.reshape(-1, n_in).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, n_in).astype(np.float32) / 255.0
    n = x_train.shape[0]

    y_onehot = np.zeros((n, n_out), dtype=np.float32)
    y_onehot[np.arange(n), y_train] = 1.0

    for epoch in range(epochs):
        perm = rng.permutation(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = x_train[idx], y_onehot[idx]
            b = xb.shape[0]

            z1 = xb @ w1 + b1
            a1 = np.maximum(z1, 0.0)
            z2 = a1 @ w2 + b2
            probs = softmax(z2)

            loss = -np.sum(yb * np.log(np.clip(probs, 1e-9, 1.0))) / b
            total_loss += loss * b

            dz2 = (probs - yb) / b
            dw2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)
            da1 = dz2 @ w2.T
            dz1 = da1 * (z1 > 0)
            dw1 = xb.T @ dz1
            db1 = dz1.sum(axis=0)

            w2 -= lr * dw2
            b2 -= lr * db2
            w1 -= lr * dw1
            b1 -= lr * db1

        z1 = x_test @ w1 + b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ w2 + b2
        acc = (z2.argmax(axis=1) == y_test).mean()
        print(f"epoch {epoch + 1}/{epochs}: loss={total_loss / n:.4f}  test_acc={acc:.4f}")

    return w1, b1, w2, b2


def quantize_multiplier(real_multiplier, mult_bits=17, shift_bits=5):
    """Approximates real_multiplier as mult / 2**shift, matching ACTIVATE's
    integer requantization (scale_m is a 17-bit unsigned fixed-point value,
    shift a 5-bit field - see sw/common/include/macaque/common/isa.hpp).
    Picks the largest safe shift to maximize precision.

    mult_bits is 17, not wider, so the hardware multiply's scale_m operand
    fits a DSP48E1's 18-bit signed B port in one chunk instead of two -
    widening this reintroduces the DSP pressure that caused the ACTIVATE
    timing violation (see hw/rtl/control/activate_unit.sv).
    """
    assert real_multiplier > 0
    max_mult = (1 << mult_bits) - 1
    max_shift = (1 << shift_bits) - 1
    shift = 0
    m = real_multiplier
    while m < (1 << (mult_bits - 1)) and shift < max_shift:
        m *= 2.0
        shift += 1
    mult = int(round(m))
    if mult > max_mult:
        mult >>= 1
        shift -= 1
    return mult, shift


def quantize_symmetric(x, bits=7):
    """Per-tensor symmetric (zero_point=0) quantization - this compiler only
    supports a_zp == b_zp == 0 (matchMatmulOperands rejects anything else),
    so no asymmetric/zero-point-shifted scheme is usable here.
    """
    scale = float(np.max(np.abs(x))) / ((1 << bits) - 1)
    if scale == 0.0:
        scale = 1.0
    q = np.round(x / scale).astype(np.int32)
    q = np.clip(q, -((1 << bits) - 1), (1 << bits) - 1)
    return q.astype(np.int8), scale


def requantize(acc_i32, mult, shift, relu=False):
    """Bit-for-bit the same integer math ACTIVATE performs (see
    hw/test/npu_top/test.py's requantize_activate and
    hw/rtl/control/activate_unit.sv): round-half-up, then optionally clip negatives to zero before the
    final INT8 clamp.
    """
    scaled = acc_i32.astype(np.int64) * mult
    if shift > 0:
        scaled += 1 << (shift - 1)
    requant = scaled >> shift
    if relu:
        requant = np.maximum(requant, 0)
    return np.clip(requant, -128, 127).astype(np.int8)


def quantize_model(w1, b1, w2, b2, x_train):
    input_scale = 1.0 / 127.0  # inputs are normalized to [0, 1], never negative

    w1_q, w1_scale = quantize_symmetric(w1)
    w2_q, w2_scale = quantize_symmetric(w2)

    bias1_scale = input_scale * w1_scale
    b1_q = np.round(b1 / bias1_scale).astype(np.int32)

    # Calibrate activation scales from a sample of real training data, run
    # through the *exact* real-ReLU-quantized layer-1 math (matching what
    # gets deployed - see requantize's relu=True), so act1_scale reflects
    # reality rather than the idealized float activation range.
    sample = x_train[:2000].reshape(-1, 784).astype(np.float32) / 255.0
    x_q = np.round(sample / input_scale).astype(np.int32)
    x_q = np.clip(x_q, -127, 127)

    acc1 = x_q.astype(np.int64) @ w1_q.astype(np.int64) + b1_q  # int32-range accumulator
    act1_scale = float(np.max(np.abs(np.maximum(sample @ w1 + b1, 0.0)))) / 127.0
    real_mult1 = bias1_scale / act1_scale
    mult1, shift1 = quantize_multiplier(real_mult1)
    a1_q = requantize(acc1, mult1, shift1, relu=True)

    w2_q_i64 = w2_q.astype(np.int64)
    bias2_scale = act1_scale * w2_scale
    b2_q = np.round(b2 / bias2_scale).astype(np.int32)

    acc2 = a1_q.astype(np.int64) @ w2_q_i64 + b2_q
    # Calibrate output_scale from the actual quantized-so-far logits (not the
    # float model's own logits, which use a different effective scale).
    output_scale = float(np.max(np.abs(acc2))) * bias2_scale / 127.0
    if output_scale == 0.0:
        output_scale = 1.0
    real_mult2 = bias2_scale / output_scale
    mult2, shift2 = quantize_multiplier(real_mult2)

    return {
        "input_scale": input_scale,
        "w1_q": w1_q, "w1_scale": w1_scale, "b1_q": b1_q,
        "mult1": mult1, "shift1": shift1, "act1_scale": act1_scale,
        "w2_q": w2_q, "w2_scale": w2_scale, "b2_q": b2_q,
        "mult2": mult2, "shift2": shift2, "output_scale": output_scale,
    }


def quantized_forward(x_u8, q):
    """Runs the exact int8 pipeline the compiler will deploy (real ReLU on
    the hidden layer, linear/Passthrough on the logits) - the ground truth
    this script's own emitted reference.json and macaque-lower's output
    must both agree with.
    """
    x_q = np.round(x_u8.astype(np.float32) / 255.0 / q["input_scale"]).astype(np.int32)
    x_q = np.clip(x_q, -127, 127)

    acc1 = x_q.astype(np.int64) @ q["w1_q"].astype(np.int64) + q["b1_q"]
    a1_q = requantize(acc1, q["mult1"], q["shift1"], relu=True)

    acc2 = a1_q.astype(np.int64) @ q["w2_q"].astype(np.int64) + q["b2_q"]
    out_q = requantize(acc2, q["mult2"], q["shift2"], relu=False)
    return x_q.astype(np.int8), a1_q, out_q


def _nest(flat, shape):
    """Builds MLIR's required nested-bracket literal syntax for a
    multi-dimensional dense tensor - a flat list is only valid for rank-1;
    every other rank must nest brackets matching the tensor's own shape
    exactly, confirmed the hard way (macaque-translate rejected a flat list
    for a rank-3 tensor: "inferred shape of elements literal ... does not
    match type").
    """
    if len(shape) == 1:
        return "[" + ", ".join(str(int(v)) for v in flat) + "]"
    stride = 1
    for d in shape[1:]:
        stride *= d
    return "[" + ", ".join(
        _nest(flat[i * stride:(i + 1) * stride], shape[1:]) for i in range(shape[0])
    ) + "]"


def dense_literal(values, elem_ty, shape):
    flat = list(values)
    body = _nest(flat, shape)
    shape_str = "x".join(str(d) for d in shape)
    return f"dense<{body}> : tensor<{shape_str}x{elem_ty}>"


def emit_layer(idx, in_dim, out_dim, w_q, b_q, mult, shift, input_val, rows, out_lines, ctr,
              relu=False):
    """Emits one matmul(+bias)+rescale layer, optionally followed by the
    tosa.clamp(0, 127) RescaleToMacaque detects as a fused ReLU (see
    matchReluClamp in TilingCommon.hpp) - the compiler maps that pattern to
    ACTIVATE's act_func=Relu instead of the default Passthrough. Returns the
    SSA value name of this layer's real logical output (the clamp's result
    if relu=True, else the rescale's own result) for the next layer (or
    final store) to consume - using the wrong one here would silently break
    chaining, since the next layer's matmul operand must be whichever value
    actually exists in the IR.
    """

    def fresh(name):
        ctr[0] += 1
        return f"%{name}{ctr[0]}"

    # w_q is [K, N] (numpy's default row-major .reshape(-1) already gives
    # K-outer, N-inner ordering - exactly what a tensor<1xKxN> dense literal
    # expects. No transpose here - transposing was an earlier bug caught
    # before this was ever run.
    w_name = fresh(f"w{idx}_")
    out_lines.append(
        f'  {w_name} = "tosa.const"() {{values = '
        f"{dense_literal(w_q.reshape(-1), 'i8', (1, in_dim, out_dim))}}} "
        f": () -> tensor<1x{in_dim}x{out_dim}xi8>"
    )

    b_name = fresh(f"b{idx}_")
    out_lines.append(
        f'  {b_name} = "tosa.const"() {{values = '
        f"{dense_literal(b_q, 'i32', (1, 1, out_dim))}}} "
        f": () -> tensor<1x1x{out_dim}xi32>"
    )

    mm_name = fresh(f"mm{idx}_")
    out_lines.append(
        f"  {mm_name} = tosa.matmul {input_val}, {w_name}, %aZp, %bZp : "
        f"(tensor<1x{rows}x{in_dim}xi8>, tensor<1x{in_dim}x{out_dim}xi8>, "
        f"tensor<1xi8>, tensor<1xi8>) -> tensor<1x{rows}x{out_dim}xi32>"
    )

    add_name = fresh(f"add{idx}_")
    out_lines.append(
        f"  {add_name} = tosa.add {mm_name}, {b_name} : "
        f"(tensor<1x{rows}x{out_dim}xi32>, tensor<1x1x{out_dim}xi32>) -> "
        f"tensor<1x{rows}x{out_dim}xi32>"
    )

    mult_name = fresh(f"mult{idx}_")
    out_lines.append(
        f'  {mult_name} = "tosa.const"() {{values = dense<{mult}> : tensor<1xi32>}} '
        ": () -> tensor<1xi32>"
    )
    shift_name = fresh(f"shift{idx}_")
    out_lines.append(
        f'  {shift_name} = "tosa.const"() {{values = dense<{shift}> : tensor<1xi8>}} '
        ": () -> tensor<1xi8>"
    )

    rescale_name = fresh(f"rescale{idx}_")
    out_lines.append(
        f"  {rescale_name} = tosa.rescale {add_name}, {mult_name}, {shift_name}, %inZp32, %aZp "
        "{scale32 = true, rounding_mode = SINGLE_ROUND, per_channel = false, "
        "input_unsigned = false, output_unsigned = false} : "
        f"(tensor<1x{rows}x{out_dim}xi32>, tensor<1xi32>, tensor<1xi8>, "
        f"tensor<1xi32>, tensor<1xi8>) -> tensor<1x{rows}x{out_dim}xi8>"
    )
    if not relu:
        return rescale_name

    relu_name = fresh(f"relu{idx}_")
    out_lines.append(
        f'  {relu_name} = "tosa.clamp"({rescale_name}) '
        "{min_val = 0 : i8, max_val = 127 : i8} : "
        f"(tensor<1x{rows}x{out_dim}xi8>) -> tensor<1x{rows}x{out_dim}xi8>"
    )
    return relu_name


def emit_tosa(q, hidden, rows=1):
    lines = []
    lines.append("module {")
    lines.append(f"  func.func @mnist_mlp(%input: tensor<1x{rows}x784xi8>) {{")
    lines.append('    %aZp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>')
    lines.append('    %bZp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>')
    lines.append(
        '    %inZp32 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>'
    )

    ctr = [0]
    r1 = emit_layer(1, 784, hidden, q["w1_q"], q["b1_q"], q["mult1"], q["shift1"],
                    "%input", rows, lines, ctr, relu=True)
    r2 = emit_layer(2, hidden, 10, q["w2_q"], q["b2_q"], q["mult2"], q["shift2"],
                    r1, rows, lines, ctr, relu=False)

    lines.append("    return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--out", default="./mnist_out")
    ap.add_argument("--cache", default="./mnist_cache")
    ap.add_argument("--num-reference", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("loading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist(args.cache)

    print(f"training ({args.epochs} epochs, hidden={args.hidden})...")
    w1, b1, w2, b2 = train(
        x_train, y_train, x_test, y_test, args.hidden, args.epochs, args.batch_size, args.lr
    )

    print("quantizing...")
    q = quantize_model(w1, b1, w2, b2, x_train)
    print(
        f"  layer1: mult={q['mult1']} shift={q['shift1']}  "
        f"layer2: mult={q['mult2']} shift={q['shift2']}"
    )

    correct = 0
    n_eval = len(x_test)
    for i in range(n_eval):
        _, _, out_q = quantized_forward(x_test[i].reshape(-1), q)
        if int(np.argmax(out_q)) == int(y_test[i]):
            correct += 1
    print(f"deployed (int8, real ReLU) test accuracy: {correct / n_eval:.4f}")

    mlir = emit_tosa(q, args.hidden, rows=1)
    mlir_path = os.path.join(args.out, "mnist_mlp.mlir")
    with open(mlir_path, "w") as f:
        f.write(mlir)
    print(f"wrote {mlir_path} ({len(mlir)} bytes)")

    quant_images = np.zeros((n_eval, 784), dtype=np.int8)
    quant_labels = y_test.astype(np.int32)
    for i in range(n_eval):
        x_q, _, _ = quantized_forward(x_test[i].reshape(-1), q)
        quant_images[i] = x_q
    np.savez(os.path.join(args.out, "test_images.npz"), images=quant_images, labels=quant_labels)
    print(f"wrote {os.path.join(args.out, 'test_images.npz')}")

    reference = []
    for i in range(args.num_reference):
        x_q, a1_q, out_q = quantized_forward(x_test[i].reshape(-1), q)
        reference.append({
            "index": int(i),
            "label": int(y_test[i]),
            "input_i8": x_q.tolist(),
            "hidden_i8": a1_q.tolist(),
            "output_i8": out_q.tolist(),
            "predicted": int(np.argmax(out_q)),
        })
    ref_path = os.path.join(args.out, "reference.json")
    with open(ref_path, "w") as f:
        json.dump(reference, f)
    print(f"wrote {ref_path} ({args.num_reference} reference examples)")


if __name__ == "__main__":
    main()
