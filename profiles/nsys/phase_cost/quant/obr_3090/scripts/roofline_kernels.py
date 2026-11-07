"""Kernel-level roofline for Qwen3-8B from `vllm bench forward --profile-dir` traces.

Usage: roofline_kernels.py <prof_dir> <out_dir>
Reads <prof_dir>/<route>/<case>.rank0.trace.json.gz, classifies every kernel by
layer position (pit 19), and writes kernel_roofline.csv / kernel_names.csv.
"""
import collections, csv, gzip, json, re, statistics, sys
from pathlib import Path

H, HKV, D, HID, INTER, LAYERS = 32, 8, 128, 4096, 12288, 36
LIN = {"qkv": (HID, (H + 2 * HKV) * D), "o": (H * D, HID), "gate_up": (HID, 2 * INTER), "down": (INTER, HID)}
CASES = {"t1": (1, 1, 1), "t8": (8, 1, 8), "t64": (64, 1, 64), "t256": (256, 1, 256), "t1024": (1024, 1, 1024),
         "t2048": (2048, 1, 2048), "t8192": (8192, 1, 8192), "d1": (1, 1, 1024), "d8": (8, 8, 1024)}  # tokens, reqs, kv_len
ROUTES = ["bf16", "int8", "int8_24", "sparse", "sparse_col"]
# measured roofs (profiling/roof.json): bf16 cuBLAS 8192^3, INT8 torch._int_mm 8192^3, sparse dense-equivalent, read BW
ROOF = {"bf16": 77.8e12, "int8": 235.2e12, "sparse": 252.5e12, "bw": 887.0e9}


def category(name):
    n = name
    if "reshape_and_cache" in n:
        return "kv_write"
    if "flash" in n:
        return "attention"
    if "rotary_embedding" in n:
        return "rotary"
    if "fused_add_rms_norm" in n:
        return "norm"
    if "rms_norm_kernel" in n:
        return "rms_norm"  # input norm of layer 0, or q/k norm; resolved below
    if "act_and_mul" in n:
        return "activation"
    if "dynamic_scaled_int8_quant" in n:
        return "act_quant"
    if re.search(r"gemm|gemv|Kernel2|cutlass|splitKreduce|Sparse", n) or n == "kernel" or n.startswith("void gemv"):
        return "linear"
    return "other"


def parse(path):
    ev = json.load(gzip.open(path))["traceEvents"]
    ks = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])
    replays, cur, names = [], None, collections.Counter()
    proj, layer, in_fwd = "qkv", 0, False
    for e in ks:
        c = category(e["name"]) if e["cat"] == "kernel" else "other"
        if not in_fwd:
            if e["cat"] != "kernel":
                continue  # state reset copies before the forward (pre_forward_us)
            in_fwd, cur = True, collections.defaultdict(float)
            cur["_t0"] = e["ts"]
            replays.append(cur)
        if c == "rms_norm":
            c = "qk_norm" if proj == "qkv" and cur.get("linear:qkv") else "norm"
        key = f"{c}:{proj}" if c in ("linear", "act_quant", "other") else c
        cur[key] += e["dur"]
        cur["_t1"] = e["ts"] + e["dur"]
        names[(key, re.split(r"[<(]", re.sub(r"^(void |std::enable_if<.*?>::type )", "", e["name"]))[0][:70])] += 1
        if c == "attention" and "combine" not in e["name"]:
            proj = "o"
        elif c == "activation":
            proj = "down"
        elif c == "norm" and e["name"].find("fused_add") >= 0:
            if proj == "o":
                proj = "gate_up"
            elif proj == "down":
                proj, layer = "qkv", layer + 1
                if layer % LAYERS == 0:
                    in_fwd = False
    return replays, names


def workload(route, case):
    T, B, L = CASES[case]
    wb = {"bf16": 2.0}.get(route, 1.0)
    if route.startswith("sparse"):
        wb = 0.5 + 0.125  # 2:4 values + 4-bit-per-pair metadata
    ab = 2.0 if route == "bf16" else 1.0
    w = {}
    for p, (K, N) in LIN.items():
        w[f"linear:{p}"] = (2 * T * K * N, K * N * wb + T * K * ab + T * N * 2, K * N * wb, ("bf16" if route == "bf16" else "sparse" if route.startswith("sparse") else "int8"))
        if route != "bf16":
            w[f"act_quant:{p}"] = (0, T * K * 2 + T * K + T * 4, 0, "bw")
    if case.startswith("d"):
        af, ab_kv = 4 * H * D * L * B, B * L * HKV * D * 2 * 2 + 2 * T * H * D * 2
    else:
        af, ab_kv = 4 * H * D * T * (T + 1) // 2, T * (2 * H * D + 2 * HKV * D) * 2
    w["attention"] = (af, ab_kv, 0, "bf16")
    w["kv_write"] = (0, T * 2 * HKV * D * 2 * 2, 0, "bw")
    w["rotary"] = (0, T * (H + HKV) * D * 2 * 2, 0, "bw")
    w["qk_norm"] = (0, T * (H + HKV) * D * 2 * 2, 0, "bw")
    w["norm"] = (0, 2 * T * HID * 2 * 2, 0, "bw")  # fused add+norm x2 per layer, approx (reads x,res; writes x,res)
    w["activation"] = (0, T * 2 * INTER * 2 + T * INTER * 2, 0, "bw")
    return {k: (f * LAYERS, b * LAYERS, wt * LAYERS, roof) for k, (f, b, wt, roof) in w.items()}


def main():
    prof, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows, name_rows = [], collections.Counter()
    for route in ROUTES:
        for case in CASES:
            p = prof / route / f"{case}.rank0.trace.json.gz"
            if not p.exists():
                continue
            replays, names = parse(p)
            replays = [r for r in replays if r.get("attention")]
            for (k, n), cnt in names.items():
                name_rows[(route, k.split(":")[0], n)] += cnt
            wl = workload(route, case)
            keys = sorted({k for r in replays for k in r if not k.startswith("_")})
            span = statistics.median(r["_t1"] - r["_t0"] for r in replays)
            for k in keys + ["TOTAL"]:
                if k == "TOTAL":
                    us = statistics.median(sum(v for kk, v in r.items() if not kk.startswith("_")) for r in replays)
                    f = sum(v[0] for v in wl.values()); b = sum(v[1] for v in wl.values()); roof = "mixed"
                else:
                    us = statistics.median(r.get(k, 0.0) for r in replays)
                    f, b, _, roof = wl.get(k, (0, 0, 0, "bw"))
                s = us * 1e-6
                tf = f / s / 1e12 if s and f else 0.0
                gbs = b / s / 1e9 if s and b else 0.0
                proof = ROOF.get(roof, 0)
                t_roof = max(f / proof if proof else 0.0, b / ROOF["bw"]) if (f or b) else 0.0
                rows.append(dict(route=route, case=case, category=k, replays=len(replays), us=round(us, 1),
                                 span_us=round(span, 1), gflop=round(f / 1e9, 3), mbytes=round(b / 1e6, 3),
                                 ai=round(f / b, 2) if b else 0.0, tflops=round(tf, 2), gbps=round(gbs, 1),
                                 pct_compute_roof=round(100 * f / s / proof, 1) if proof and s and f else "",
                                 pct_bw_roof=round(100 * b / s / ROOF["bw"], 1) if s and b else "",
                                 roof_us=round(t_roof * 1e6, 1), roof_eff_pct=round(100 * t_roof / s, 1) if s and t_roof else ""))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "kernel_roofline.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    with open(out / "kernel_names.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["route", "category", "kernel", "count_3_replays"])
        for (r, c, n), cnt in sorted(name_rows.items()):
            wr.writerow([r, c, n, cnt])
    print("rows", len(rows))


if __name__ == "__main__":
    main()
