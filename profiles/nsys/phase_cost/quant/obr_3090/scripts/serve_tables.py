"""Median-of-3 tables from serve_breakdown.json. Usage: serve_tables.py <serve_breakdown.json> <out_csv>"""
import csv, json, statistics, sys

r = json.load(open(sys.argv[1]))
med = statistics.median
rows = []
for key, reqs in r.items():
    route, cfg = key.split("/")
    m = lambda f: round(med(f(q) for q in reqs), 3)
    row = dict(route=route, cfg=cfg,
               client_ttft_ms=m(lambda q: q["client"]["ttft"]), client_tpot_ms=m(lambda q: q["client"]["tpot"]),
               client_e2e_ms=m(lambda q: q["client"]["e2e"]),
               gpu_busy_pct_of_request=m(lambda q: 100 * q["gpu_busy_us"] / q["window_us"]),
               gpu_idle_ms_request=m(lambda q: q["gpu_idle_us"] / 1e3))
    for k in ("queue_to_schedule", "cpu_until_gpu_start", "gpu_prefill_span", "gpu_prefill_busy", "sample_after_fwd",
              "output_to_engine_ready", "frontend_to_client", "prefill_cpu_sched", "prefill_cpu_exec", "prefill_cpu_prep"):
        row[f"ttft_{k}_ms"] = m(lambda q: q["ttft"][k] / 1e3)
    row["ttft_sum_check_ms"] = round(sum(row[f"ttft_{k}_ms"] for k in ("queue_to_schedule", "cpu_until_gpu_start", "gpu_prefill_span", "sample_after_fwd", "output_to_engine_ready", "frontend_to_client")), 3)
    for k in ("prefill_cats",):
        cats = set().union(*[q[k] for q in reqs])
        for c in sorted(cats):
            row[f"prefill_gpu_{c}_ms"] = m(lambda q: q[k].get(c, 0.0) / 1e3)
    for k in ("cycle_median", "gpu_busy_median", "gpu_idle_median", "fwd_gpu_median", "lm_head_median", "sampler_median",
              "cpu_sched", "cpu_exec", "cpu_prep", "cpu_sample", "cpu_wait", "cpu_update", "cpu_nonwait"):
        row[f"decode_{k}_ms"] = m(lambda q: q["tpot"][k] / 1e3)
    row["decode_steps"] = reqs[0]["tpot"]["decode_steps"]; row["decode_graph_steps"] = reqs[0]["tpot"]["graph_steps"]
    cats = set().union(*[q["tpot"]["cats"] for q in reqs])
    for c in sorted(cats):
        row[f"decode_gpu_{c}_ms"] = m(lambda q: q["tpot"]["cats"].get(c, 0.0) / 1e3)
    rows.append(row)
keys = []
for row in rows:
    keys += [k for k in row if k not in keys]
with open(sys.argv[2], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
for row in rows:
    print(row["route"], row["cfg"], {k: v for k, v in row.items() if k not in ("route", "cfg")})
