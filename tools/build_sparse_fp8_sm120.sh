#!/usr/bin/env bash
# Build the 2:4 sparse FP8 kernel with the nvcc and CUTLASS shipped in the venv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SITE="$("$ROOT/.venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
CUDA="$SITE/nvidia/cu13"
CUTLASS="$SITE/flashinfer/data/cutlass"
"$CUDA/bin/nvcc" -std=c++20 -O3 -DNDEBUG -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK \
  -gencode=arch=compute_120a,code=sm_120a -shared -Xcompiler -fPIC \
  -I"$CUDA/include" -I"$CUTLASS/include" -I"$CUTLASS/tools/util/include" -I"$ROOT/csrc" \
  "$ROOT/csrc/quantization/w8a8/cutlass/c3x/sparse_fp8_mm_sm120.cu" \
  -o "$ROOT/vllm/_sparse_fp8_sm120.so" \
  -Xlinker "$CUDA/lib/libcudart.so.13"
