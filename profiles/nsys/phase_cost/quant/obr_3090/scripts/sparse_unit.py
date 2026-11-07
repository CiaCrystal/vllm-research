"""Sparse INT8 kernel against a float reference, on the real vLLM op."""
import sys, torch
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.scaled_mm.cutlass_sparse import compress_24_int8
torch.manual_seed(0)
def sparse_weight(M, K):
    w = torch.randint(-127, 128, (M, K), dtype=torch.int8, device="cuda")
    w[w == 0] = 1
    drop = torch.rand(M, K // 4, 4, device="cuda").argsort(dim=2)[:, :, :2]
    mask = torch.ones(M, K // 4, 4, device="cuda", dtype=torch.bool).scatter(2, drop, False)
    return w * mask.reshape(M, K)
NS = [int(a) for a in sys.argv[1:]]
(lambda a: (a @ a).sum().item())(torch.ones(4, 4, device="cuda"))
bad = 0
for M, K in [(256, 256), (4096, 4096), (24576, 4096), (4096, 12288), (6144, 4096)]:
    w = sparse_weight(M, K)
    v, e = compress_24_int8(w)
    ws = torch.rand(M, device="cuda") * 0.01 + 0.001
    for N in NS:
        if M * N > 3e8: continue
        x = torch.randint(-127, 128, (N, K), dtype=torch.int8, device="cuda")
        xs = torch.rand(N, device="cuda") * 0.01 + 0.001
        out = torch.ops.vllm.sparse_int8_mm_func(x, v, e, xs, ws)
        ref = (x.float() * xs[:, None]) @ (w.float() * ws[:, None]).t()
        torch.cuda.synchronize()
        err = ((out.float() - ref).norm() / ref.norm()).item()
        ok = err < 1e-2 and out.shape == (N, M)
        bad += not ok
        print(f"M={M} K={K} N={N} shape={tuple(out.shape)} relerr={err:.5f} {'OK' if ok else 'FAIL'}", flush=True)
    xs1 = torch.tensor([0.01], device="cuda")
    out = torch.ops.vllm.sparse_int8_mm_func(x, v, e, xs1, ws)
    ref = (x.float() * 0.01) @ (w.float() * ws[:, None]).t()
    err = ((out.float() - ref).norm() / ref.norm()).item()
    bad += err >= 1e-2
    print(f"M={M} K={K} scalar-scale relerr={err:.5f}", flush=True)
print("FAILED", bad)
