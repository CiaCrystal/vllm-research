"""Achievable roofs on this GPU: bf16 / INT8 / sparse INT8 GEMM and memory bandwidth."""
import json, statistics, sys
import torch
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.scaled_mm import cutlass_sparse as cs

dev = "cuda"
res = {}


def bench(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(3):  # 3 trials, each the mean of `iters` back-to-back launches
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); e.synchronize()
        out.append(s.elapsed_time(e) / iters * 1e-3)
    return out


def gemm(tag, M, N, K):
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    t = bench(lambda: torch.mm(a, w.t()), 10)
    res[f"bf16_{tag}"] = dict(M=M, N=N, K=K, s=t, tflops=[2 * M * N * K / x / 1e12 for x in t])
    del a, w
    a = torch.randint(-127, 127, (M, K), device=dev, dtype=torch.int8)
    w = torch.randint(-127, 127, (N, K), device=dev, dtype=torch.int8)
    sa = torch.rand(M, 1, device=dev); sb = torch.rand(1, N, device=dev)
    t = bench(lambda: ops.cutlass_scaled_mm(a, w.t(), sa, sb, torch.bfloat16), 10)
    res[f"int8_cutlass_{tag}"] = dict(M=M, N=N, K=K, s=t, tops=[2 * M * N * K / x / 1e12 for x in t])
    t = bench(lambda: torch._int_mm(a, w.t()), 10)
    res[f"int8_torch_int_mm_{tag}"] = dict(M=M, N=N, K=K, s=t, tops=[2 * M * N * K / x / 1e12 for x in t])
    # sparse: weight [N_out, K] is operand A (2:4), tokens [M, K] operand B
    w24 = w.clone().view(N, K // 4, 4)
    w24[:, :, 2:] = 0
    w24 = w24.view(N, K)
    vals, meta = cs.compress_24_int8(w24)
    xs = torch.rand(M, device=dev)
    ws = torch.rand(N, device=dev)
    fn = lambda: torch.ops.vllm.sparse_int8_mm_func(a, vals, meta, xs, ws)
    t = bench(fn, 10)
    res[f"int8_sparse24_{tag}"] = dict(M=M, N=N, K=K, s=t, dense_equiv_tops=[2 * M * N * K / x / 1e12 for x in t])
    del a, w, w24, vals, meta
    torch.cuda.empty_cache()


gemm("8192cube", 8192, 8192, 8192)
gemm("gate_up_8192", 8192, 24576, 4096)
gemm("down_8192", 8192, 4096, 12288)

n = 1 << 29  # 1 GiB of bf16
src = torch.empty(n, device=dev, dtype=torch.bfloat16).normal_()
dst = torch.empty_like(src)
t = bench(lambda: dst.copy_(src), 20)
res["copy_1GiB"] = dict(bytes=2 * src.nbytes, s=t, gbps=[2 * src.nbytes / x / 1e9 for x in t])
t = bench(lambda: dst.fill_(1.0), 20)
res["fill_1GiB"] = dict(bytes=dst.nbytes, s=t, gbps=[dst.nbytes / x / 1e9 for x in t])
f32 = src.view(torch.float32)
t = bench(lambda: f32.sum(), 20)
res["sum_read_1GiB"] = dict(bytes=src.nbytes, s=t, gbps=[src.nbytes / x / 1e9 for x in t])
# weight-sized read through a GEMV (decode-like): bf16 4096 -> 12288*2
w = torch.randn(24576, 4096, device=dev, dtype=torch.bfloat16)
x = torch.randn(1, 4096, device=dev, dtype=torch.bfloat16)
t = bench(lambda: torch.mm(x, w.t()), 50)
res["gemv_bf16_gate_up"] = dict(bytes=w.nbytes, s=t, gbps=[w.nbytes / x_ / 1e9 for x_ in t])

p = torch.cuda.get_device_properties(0)
res["device"] = dict(name=p.name, sms=p.multi_processor_count, mem_gib=p.total_memory / 2**30, l2_mb=getattr(p, "L2_cache_size", 0) / 2**20)
for k, v in res.items():
    if "s" in v:
        key = [x for x in v if x in ("tflops", "tops", "dense_equiv_tops", "gbps")][0]
        print(f"{k:30s} {key:17s} median {statistics.median(v[key]):8.1f}  max {max(v[key]):8.1f}")
json.dump(res, open(sys.argv[1], "w"), indent=1)
