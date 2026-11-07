"""Summarize stress_bf16.py output. Usage: stress_summary.py <stress_dir> <out_csv>

Throughput: median of the 3 runs. Latency percentiles and SLO attainment: pooled
over the 3 runs (per-request data from --save-detailed). Server metrics and CPU
come from the sampler (0.5 s period).
"""
import collections, csv, json, statistics, sys
from pathlib import Path

import numpy as np

SLOS = {"slo_a": (1000.0, 50.0), "slo_b": (500.0, 30.0)}  # (TTFT ms, TPOT ms)


def cpu_stats(samples, key, tick=100):
    pts = [(s["t"], s["cpu"].get(key)) for s in samples if s["cpu"].get(key) is not None]
    if len(pts) < 2:
        return None, None
    avg = (pts[-1][1] - pts[0][1]) / tick / (pts[-1][0] - pts[0][0]) * 100
    peak = max(((b[1] - a[1]) / tick / (b[0] - a[0]) * 100 for a, b in zip(pts, pts[4:]) if b[0] > a[0]), default=None)
    return avg, peak


def run_stats(raw, smp):
    ttft = np.array(raw["ttfts"]) * 1e3
    lat = np.array(raw["latencies"]) * 1e3
    olen = np.array(raw["output_lens"])
    tpot = (lat - ttft) / np.maximum(olen - 1, 1)
    itl = np.concatenate([np.array(x) * 1e3 for x in raw["itls"] if x]) if raw["itls"] else np.array([])
    m0, m1 = smp["metrics_before"], smp["metrics_after"]
    ms = [s["m"] for s in smp["samples"] if "m" in s]
    hist = {}
    for h in ("request_queue_time", "time_to_first_token", "inter_token_latency", "request_prefill_time", "request_decode_time"):
        k = f"vllm:{h}_seconds"
        dc = m1.get(k + "_count", 0) - m0.get(k + "_count", 0)
        hist[h] = (m1.get(k + "_sum", 0) - m0.get(k + "_sum", 0)) / dc * 1e3 if dc else None
    return dict(
        ttft=ttft, tpot=tpot, itl=itl, e2el=lat, failed=raw.get("failed", 0),
        req_tput=raw["request_throughput"], out_tput=raw["output_throughput"], tok_tput=raw["total_token_throughput"],
        duration=raw["duration"], max_conc=raw.get("max_concurrent_requests"),
        preempt=m1.get("vllm:num_preemptions_total", 0) - m0.get("vllm:num_preemptions_total", 0),
        kv_max=max((m.get("vllm:kv_cache_usage_perc", 0) for m in ms), default=0),
        kv_mean=statistics.mean(m.get("vllm:kv_cache_usage_perc", 0) for m in ms) if ms else 0,
        run_max=max((m.get("vllm:num_requests_running", 0) for m in ms), default=0),
        run_mean=statistics.mean(m.get("vllm:num_requests_running", 0) for m in ms) if ms else 0,
        wait_max=max((m.get("vllm:num_requests_waiting", 0) for m in ms), default=0),
        wait_mean=statistics.mean(m.get("vllm:num_requests_waiting", 0) for m in ms) if ms else 0,
        hist=hist, cpu={k: cpu_stats(smp["samples"], k) for k in ("api", "engine", "client")},
        foreign=smp["foreign_gpu_apps"],
    )


def main():
    d, out = Path(sys.argv[1]), sys.argv[2]
    groups = collections.defaultdict(list)
    for sp in sorted((d / "samples").glob("*.json")):
        smp = json.load(open(sp))
        rp = d / "raw" / f"{smp['name']}.json"
        if smp["rc"] != 0 or not rp.exists():
            print("skip", smp["name"], smp["rc"]); continue
        groups[(smp["phase"], smp["mnbt"], smp["label"])].append((smp, run_stats(json.load(open(rp)), smp)))
    rows = []
    med = statistics.median
    for (ph, mnbt, label), runs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], float(kv[0][2][1:]))):
        st = [r for _, r in runs]
        s0 = runs[0][0]
        pool = lambda k: np.concatenate([r[k] for r in st])
        row = dict(phase=ph, mnbt=mnbt, label=label, input=s0["input"], output=s0["output"],
                   max_concurrency=s0["max_concurrency"] or "", request_rate=s0["request_rate"], runs=len(st),
                   requests=sum(len(r["ttft"]) for r in st), failed=sum(r["failed"] for r in st),
                   req_per_s=round(med(r["req_tput"] for r in st), 3),
                   out_tok_per_s=round(med(r["out_tput"] for r in st), 1),
                   out_tok_per_s_min=round(min(r["out_tput"] for r in st), 1),
                   out_tok_per_s_max=round(max(r["out_tput"] for r in st), 1),
                   total_tok_per_s=round(med(r["tok_tput"] for r in st), 1))
        for k in ("ttft", "tpot", "itl", "e2el"):
            v = pool(k)
            for q in (50, 95, 99):
                row[f"{k}_p{q}_ms"] = round(float(np.percentile(v, q)), 1) if len(v) else ""
        tt, tp = pool("ttft"), pool("tpot")
        for name, (a, b) in SLOS.items():
            ok = (tt <= a) & (tp <= b)
            row[f"{name}_attain_pct"] = round(float(100 * ok.mean()), 1)
            row[f"{name}_goodput_req_s"] = round(med(float(((r["ttft"] <= a) & (r["tpot"] <= b)).sum()) / r["duration"] for r in st), 3)
        itl = pool("itl")
        row["ttft_le_1000_pct"] = round(float(100 * (tt <= 1000).mean()), 1)
        row["tpot_le_50_pct"] = round(float(100 * (tp <= 50).mean()), 1)
        row["tpot_le_30_pct"] = round(float(100 * (tp <= 30).mean()), 1)
        row["itl_gt_100_pct"] = round(float(100 * (itl > 100).mean()), 2)
        row["itl_gt_200_pct"] = round(float(100 * (itl > 200).mean()), 2)
        row["itl_max_ms"] = round(float(itl.max()), 0)
        for k in ("preempt", "kv_max", "kv_mean", "run_max", "run_mean", "wait_max", "wait_mean"):
            row[k] = round(med(r[k] for r in st), 3)
        row["preempt_total"] = sum(r["preempt"] for r in st)
        for h in st[0]["hist"]:
            vals = [r["hist"][h] for r in st if r["hist"][h] is not None]
            row[f"engine_mean_{h}_ms"] = round(med(vals), 1) if vals else ""
        for k in ("api", "engine", "client"):
            avg = [r["cpu"][k][0] for r in st if r["cpu"][k][0] is not None]
            pk = [r["cpu"][k][1] for r in st if r["cpu"][k][1] is not None]
            row[f"cpu_{k}_avg_pct"] = round(med(avg), 1) if avg else ""
            row[f"cpu_{k}_peak2s_pct"] = round(max(pk), 1) if pk else ""
        row["foreign_gpu_apps"] = ";".join(sorted({f for r in st for f in r["foreign"]}))
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
