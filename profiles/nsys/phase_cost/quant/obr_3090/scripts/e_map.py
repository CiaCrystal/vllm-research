"""Map each metadata nibble in memory to the (row, group) of A it controls."""
import sys, torch, json
from vllm.model_executor.kernels.linear.scaled_mm import cutlass_sparse as cs
M, K = int(sys.argv[1]), int(sys.argv[2])
G = K // 4; N = G if G % 16 == 0 else (G + 15) // 16 * 16
d = "cuda"
v = torch.ones(M, K // 2, dtype=torch.int8, device=d)
# token g is nonzero only at positions 2,3 of group g
x = torch.zeros(N, K, dtype=torch.int8, device=d)
for g in range(G):
    x[g, g * 4 + 2:g * 4 + 4] = 1
ws = torch.ones(M, device=d); xs = torch.ones(1, device=d)
base = torch.full((M * K // 32,), 0x44444444, dtype=torch.uint32, device=d)
out = {}
for w in range(base.numel()):
    for j in range(8):
        e = base.clone(); e[w] = int(0x44444444 ^ ((0x4 ^ 0xE) << (j * 4)))
        r = torch.ops.vllm.sparse_int8_mm_func(x, v, e, xs, ws)
        nz = r.float().nonzero()
        out[f"{w},{j}"] = nz.tolist()
bad = [k for k, v in out.items() if len(v) != 1]
print("nonsingle", len(bad), bad[:5])
rows = {}
tab = {}
for k, vv in out.items():
    w, j = map(int, k.split(","))
    g, m = vv[0]
    tab[(m, g)] = (w, j)
print("covered", len(tab), "of", M * G)
for m in range(min(M, 32)):
    print("row", m, "group0 at word,nib", tab[(m, 0)], "group8", tab.get((m, 8)), "group16", tab.get((m, 16)), "group24", tab.get((m, 24)))
json.dump({f"{m},{g}": list(t) for (m, g), t in tab.items()}, open(f"emap_{M}_{K}.json", "w"))
