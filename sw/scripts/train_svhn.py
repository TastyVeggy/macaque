#!/usr/bin/env python3
import argparse
import json
import os
import urllib.request

import numpy as np
import scipy.io
from PIL import Image

from train_mnist import (
    dense_literal,
    emit_layer,
    init_layer,
    quantize_multiplier,
    quantize_symmetric,
    requantize,
)

SVHN_TRAIN_URL = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
SVHN_TEST_URL = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"


def load_svhn(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, "train_32x32.mat")
    test_path = os.path.join(cache_dir, "test_32x32.mat")
    if not os.path.exists(train_path):
        print(f"downloading SVHN train to {train_path} ...")
        urllib.request.urlretrieve(SVHN_TRAIN_URL, train_path)
    if not os.path.exists(test_path):
        print(f"downloading SVHN test to {test_path} ...")
        urllib.request.urlretrieve(SVHN_TEST_URL, test_path)
        print("Why")

    train_mat = scipy.io.loadmat(train_path)
    print("HI")
    test_mat = scipy.io.loadmat(test_path)
    print("O")
    # SVHN's X is (32, 32, 3, N) - move N to the front to match the (N, H, W,
    # C) convention the rest of this script expects.
    x_train = np.transpose(train_mat["X"], (3, 0, 1, 2))
    x_test = np.transpose(test_mat["X"], (3, 0, 1, 2))
    # SVHN labels digit '0' as 10 (matlab has no 0 index) - remap to 0-9 to
    # match MNIST's convention (and this pipeline's argmax-over-10 decoder).
    y_train = (train_mat["y"].flatten() % 10).astype(np.int64)
    y_test = (test_mat["y"].flatten() % 10).astype(np.int64)
    return x_train, y_train, x_test, y_test


def to_grayscale(x_rgb):
    """ITU-R BT.601 luma - collapses (N, 32, 32, 3) uint8 RGB crops to
    (N, 32, 32) grayscale, matching MNIST's single-channel pixel convention
    (quantize_symmetric, input_scale, etc. below all assume one channel).
    """
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (x_rgb.astype(np.float32) @ weights).clip(0, 255).astype(np.uint8)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train(x_train, y_train, x_test, y_test, n_in, hidden, epochs, batch_size, lr, seed=0):
    rng = np.random.default_rng(seed)
    n_hidden, n_out = hidden, 10

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


def quantize_model(w1, b1, w2, b2, x_train, n_in):
    input_scale = 1.0 / 127.0  # inputs are normalized to [0, 1], never negative

    w1_q, w1_scale = quantize_symmetric(w1)
    w2_q, w2_scale = quantize_symmetric(w2)

    bias1_scale = input_scale * w1_scale
    b1_q = np.round(b1 / bias1_scale).astype(np.int32)

    # Calibrate activation scales from a sample of real training data, run
    # through the *exact* real-ReLU-quantized layer-1 math (matching what
    # gets deployed - see requantize's relu=True), so act1_scale reflects
    # reality rather than the idealized float activation range.
    sample = x_train[:2000].reshape(-1, n_in).astype(np.float32) / 255.0
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
    x_q = np.round(x_u8.astype(np.float32) / 255.0 / q["input_scale"]).astype(np.int32)
    x_q = np.clip(x_q, -127, 127)

    acc1 = x_q.astype(np.int64) @ q["w1_q"].astype(np.int64) + q["b1_q"]
    a1_q = requantize(acc1, q["mult1"], q["shift1"], relu=True)

    acc2 = a1_q.astype(np.int64) @ q["w2_q"].astype(np.int64) + q["b2_q"]
    out_q = requantize(acc2, q["mult2"], q["shift2"], relu=False)
    return x_q.astype(np.int8), a1_q, out_q


def write_pngs(x_test, y_test, out_dir, count):
    """Writes `count` real, *raw* (unquantized, 0-255) grayscale test crops
    as <label>/<index>.png - the same layout as
    sw/runtime/examples/mnist/mnist_pngs/<label>/*.png, so
    sw/runtime/examples/infer_digit.cpp (already dataset-agnostic - it just
    loads whatever PNG + program.macq it's given, quantizing at runtime via
    Device::infer) can run against SVHN with no code changes at all.
    """
    for i in range(min(count, len(x_test))):
        label_dir = os.path.join(out_dir, str(int(y_test[i])))
        os.makedirs(label_dir, exist_ok=True)
        Image.fromarray(x_test[i], mode="L").save(os.path.join(label_dir, f"{i}.png"))


def emit_tosa(q, n_in, hidden, rows=1):
    lines = []
    lines.append("module {")
    lines.append(f"  func.func @svhn_mlp(%input: tensor<1x{rows}x{n_in}xi8>) {{")
    lines.append('    %aZp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>')
    lines.append('    %bZp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>')
    lines.append(
        '    %inZp32 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>'
    )

    ctr = [0]
    r1 = emit_layer(1, n_in, hidden, q["w1_q"], q["b1_q"], q["mult1"], q["shift1"],
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
    ap.add_argument("--out", default="./svhn_out")
    ap.add_argument("--cache", default="./svhn_cache")
    ap.add_argument("--num-reference", type=int, default=5)
    ap.add_argument("--num-pngs", type=int, default=20,
                    help="real test crops to export as PNGs for infer_image (0 to skip)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("loading SVHN format 2...")
    x_train_rgb, y_train, x_test_rgb, y_test = load_svhn(args.cache)
    x_train = to_grayscale(x_train_rgb)
    x_test = to_grayscale(x_test_rgb)
    n_in = x_train.shape[1] * x_train.shape[2]  # 32*32 = 1024, not a multiple of 14
    print(f"  {x_train.shape[0]} train / {x_test.shape[0]} test, n_in={n_in} "
         f"({n_in} % 14 = {n_in % 14}, so K-tile padding is exercised)")

    print(f"training ({args.epochs} epochs, hidden={args.hidden})...")
    w1, b1, w2, b2 = train(
        x_train, y_train, x_test, y_test, n_in, args.hidden, args.epochs, args.batch_size, args.lr
    )

    print("quantizing...")
    q = quantize_model(w1, b1, w2, b2, x_train, n_in)
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

    mlir = emit_tosa(q, n_in, args.hidden, rows=1)
    mlir_path = os.path.join(args.out, "svhn_mlp.mlir")
    with open(mlir_path, "w") as f:
        f.write(mlir)
    print(f"wrote {mlir_path} ({len(mlir)} bytes)")

    quant_images = np.zeros((n_eval, n_in), dtype=np.int8)
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

    if args.num_pngs > 0:
        png_dir = os.path.join(args.out, "svhn_pngs")
        write_pngs(x_test, y_test, png_dir, args.num_pngs)
        print(f"wrote {args.num_pngs} PNGs under {png_dir}/<label>/ "
             "- feed one straight into infer_image")


if __name__ == "__main__":
    main()
