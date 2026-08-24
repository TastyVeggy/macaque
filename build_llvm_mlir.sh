#!/usr/bin/env bash
set -euo pipefail

LLVM_TAG=llvmorg-22.1.6

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM_DIR="$ROOT_DIR/third-party/llvm-project"

if [ -d "$LLVM_DIR" ]; then
  echo "error: $LLVM_DIR already exists. Remove it first if you want a clean re-clone" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/third-party"
git clone --depth 1 --branch "$LLVM_TAG" https://github.com/llvm/llvm-project.git "$LLVM_DIR"

cmake -G Ninja -S "$LLVM_DIR/llvm" -B "$LLVM_DIR/build" \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON

cmake --build "$LLVM_DIR/build"

echo "LLVM/MLIR $LLVM_TAG built at $LLVM_DIR/build"
