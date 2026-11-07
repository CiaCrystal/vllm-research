"""Per-decoder-layer kernel time from `vllm bench forward` traces.

Usage: layer_breakdown.py <prof_dir> <out_dir>
Writes layer_breakdown.csv (median over the 3 replays): block totals for every
layer, per-category rows for layer 0 and for the median of layers 1-35.
Kernels are assigned to layers by in-layer order (pit 19); the fused add+norm
after `down` is counted in the layer it closes.
"""
import collections, csv, gzip, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roofline_kernels import CASES, LAYERS, category  # noqa: E402

ROUTES = ["bf16", "int8", "sparse_col"]


def parse_layers(path):
    ev = json.load(gzip.open(path))["traceEvents"]
    ks = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])
    out = collections.defaultdict(lambda: collections.defaultdict(float))  # (replay, layer) -> cat -> us
    proj, layer, in_fwd, seen_qkv = "qkv", 0, False, False
    for e in ks:
        c = category(e["name"]) if e["cat"] == "kernel" else "other"
        if not in_fwd:
            if e["cat"] != "kernel":
                continue
            in_fwd, seen_qkv = True, False
        if c == "rms_norm":
            c = "qk_norm" if proj == "qkv" and seen_qkv else "norm"
        if c == "linear" and proj == "qkv":
            seen_qkv = True
        key = f"{c}:{proj}" if c in ("linear", "act_quant", "other") else c
        out[(layer // LAYERS, layer % LAYERS)][key] += e["dur"]
        if c == "attention" and "combine" not in e["name"]:
            proj = "o"
        elif c == "activation":
            proj = "down"
        elif c == "norm" and "fused_add" in e["name"]:
            if proj == "o":
                proj = "gate_up"
            elif proj == "down":
                proj, layer, seen_qkv = "qkv", layer + 1, False
                if layer % LAYERS == 0:
                    in_fwd = False
    return out


def block(cat):
    if cat.startswith(("linear:qkv", "linear:o", "act_quant:qkv", "act_quant:o", "other:qkv", "other:o")) or cat in (
            "qk_norm", "rotary", "kv_write", "attention"):
        return "attn_block"
    if cat.startswith(("linear:gate_up", "linear:down", "act_quant:gate_up", "act_quant:down", "other:gate_up", "other:down")) or cat == "activation":
        return "ffn_block"
    return "norm"


def main():
    prof, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for route in ROUTES:
        for case in CASES:
            p = prof / route / f"{case}.rank0.trace.json.gz"
            if not p.exists():
                continue
            d = parse_layers(p)
            replays = sorted({r for r, _ in d})
            all_cats = sorted({k for (r, _), m in d.items() for k in m})
            per_cat = collections.defaultdict(list)
            for layer in range(LAYERS):
                # per-category rows only for layer 0; other layers keep blocks + TOTAL
                cats = all_cats if layer == 0 else []
                for k in all_cats:
                    if layer:
                        per_cat[k].append(statistics.median(d[(r, layer)].get(k, 0.0) for r in replays))
                for k in cats + ["attn_block", "ffn_block", "norm_block", "TOTAL"]:
                    def val(r):
                        m = d[(r, layer)]
                        if k == "TOTAL":
                            return sum(m.values())
                        if k in ("attn_block", "ffn_block", "norm_block"):
                            b = k.replace("_block", "") if k == "norm_block" else k
                            return sum(v for kk, v in m.items() if block(kk) == b)
                        return m.get(k, 0.0)
                    rows.append(dict(route=route, case=case, layer=layer, category=k,
                                     us=round(statistics.median(val(r) for r in replays), 2)))
            for k, v in per_cat.items():
                rows.append(dict(route=route, case=case, layer="median_1_35", category=k,
                                 us=round(statistics.median(v), 2)))
    with open(out / "layer_breakdown.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("rows", len(rows))


if __name__ == "__main__":
    main()
