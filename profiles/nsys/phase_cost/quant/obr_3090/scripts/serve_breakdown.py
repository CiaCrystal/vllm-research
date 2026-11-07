"""Break down torch-profiler serve traces (prof_serve.py) into per-step CPU/GPU time.

Usage: serve_breakdown.py <trace_root> <out_dir>
<trace_root>/<route>/{<route>_<cfg>_rank0.*.pt.trace.json.gz, <route>.client.json}
Needs VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 scopes ("core: ...", "runner: ...").
Trace ts (us) + baseTimeNanoseconds = wall clock, same clock as the client.
"""
import bisect, collections, csv, glob, gzip, json, re, statistics, sys
from pathlib import Path

CFG = {"A": (256, 33, 1), "D": (1024, 33, 8), "F": (256, 129, 8)}
med = statistics.median


def kcat(name, where):
    n = name
    if where == "sample":
        return "lm_head" if re.search(r"gemm|gemv|Kernel2|cutlass|splitKreduce", n) else "sampler"
    if where == "prep":
        return "input_prep"
    if "reshape_and_cache" in n:
        return "kv_write"
    if "flash" in n:
        return "attention"
    # Inductor names fused norm/silu kernels after every op they touch, e.g.
    # triton_red_fused_dynamic_scaled_int8_quant_fused_add_rms_norm; the real
    # quantization is still the standalone vllm kernel (4 per layer).
    if "vllm::dynamic_scaled_int8_quant" in n:
        return "act_quant"
    if re.search(r"gemm|gemv|Kernel2|cutlass::Kernel|splitKreduce|SparseGemm", n) or n == "kernel":
        return "linear"
    if "rms_norm" in n and ("index_select" in n or "cat" in n):
        return "qk_norm_rotary"
    if "rms_norm" in n:
        return "norm"
    if "silu" in n or "act_and_mul" in n:
        return "activation"
    if "rotary" in n:
        return "rotary"
    return "other"


def union(iv):
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return tot + (cur_e - cur_s if cur_e is not None else 0.0)


def analyze(trace, client, cfg):
    t = json.load(gzip.open(trace))
    base_us = t["baseTimeNanoseconds"] / 1e3
    ev = t["traceEvents"]
    ann = sorted((e for e in ev if e.get("cat") == "user_annotation"), key=lambda e: e["ts"])
    rt = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})]
    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    by_corr = collections.defaultdict(list)
    for k in gpu:
        by_corr[k["args"].get("correlation")].append(k)
    rt.sort(key=lambda e: e["ts"]); rts = [e["ts"] for e in rt]

    def kernels_in(a):
        i, j = bisect.bisect_left(rts, a["ts"]), bisect.bisect_right(rts, a["ts"] + a["dur"])
        return [k for r in rt[i:j] for k in by_corr.get(r["args"]["correlation"], ())]

    named = lambda p: [a for a in ann if a["name"].startswith(p)]
    sched, execs, samples = named("core: schedule"), named("core: execute_model"), named("core: sample_tokens")
    waits, updates, preps, ctxs = named("core: wait_output"), named("core: update_from_output"), named("runner: prepare_inputs"), named("execute_context_")
    steps = []
    for i, ex in enumerate(execs):
        ctx = next((c for c in ctxs if ex["ts"] <= c["ts"] <= ex["ts"] + ex["dur"]), None)
        m = re.match(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)", ctx["name"]) if ctx else None
        nctx, ntok_ctx, ngen = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (0, 0, 0)
        if nctx == 0 and ngen == 0:
            continue
        sc = max((s for s in sched if s["ts"] <= ex["ts"]), key=lambda s: s["ts"])
        sm = next((s for s in samples if s["ts"] >= ex["ts"] + ex["dur"] - 1), None)
        pr = next((p for p in preps if ex["ts"] <= p["ts"] <= ex["ts"] + ex["dur"]), None)
        prep_k = {id(k) for k in kernels_in(pr)} if pr else set()
        fwd = [k for k in kernels_in(ex)]
        smp = kernels_in(sm) if sm else []
        cats = collections.defaultdict(float)
        for k in fwd:
            cats[kcat(k["name"], "prep" if id(k) in prep_k else "fwd")] += k["dur"]
        for k in smp:
            cats[kcat(k["name"], "sample")] += k["dur"]
        fwd_main = [k for k in fwd if id(k) not in prep_k]
        steps.append(dict(phase="prefill" if nctx else "decode", nctx=nctx, ntok_ctx=ntok_ctx, ngen=ngen,
                          sched_ts=sc["ts"], sched_us=sc["dur"], exec_ts=ex["ts"], exec_us=ex["dur"],
                          prep_us=pr["dur"] if pr else 0.0, sample_cpu_us=sm["dur"] if sm else 0.0,
                          graph=any(r["name"] == "cudaGraphLaunch" for r in rt[bisect.bisect_left(rts, ex["ts"]):bisect.bisect_right(rts, ex["ts"] + ex["dur"])]),
                          fwd_start=min((k["ts"] for k in fwd_main), default=None), fwd_end=max((k["ts"] + k["dur"] for k in fwd_main), default=None),
                          smp_end=max((k["ts"] + k["dur"] for k in smp), default=None),
                          gpu_iv=[(k["ts"], k["ts"] + k["dur"]) for k in fwd + smp], cats=dict(cats)))
    # wait/update pair with steps in FIFO order (only non-empty steps pop real outputs; match by time)
    for s in steps:
        w = next((w for w in waits if w["ts"] >= s["exec_ts"] and w["ts"] + w["dur"] >= (s["smp_end"] or 0)), None)
        u = next((u for u in updates if w and u["ts"] >= w["ts"]), None)
        s["wait_us"] = w["dur"] if w else 0.0
        s["update_us"] = u["dur"] if u else 0.0
        s["out_ready"] = (u["ts"] + u["dur"]) if u else None
    # assign steps to the 3 client requests
    reqs = []
    for r in client[cfg]:
        t0, tf, tl = r["t0_ns"] / 1e3 - base_us, r["first_ns"] / 1e3 - base_us, r["last_ns"] / 1e3 - base_us
        ss = [s for s in steps if t0 - 1e3 <= s["sched_ts"] <= tl]
        pre = [s for s in ss if s["phase"] == "prefill"]
        dec = [s for s in ss if s["phase"] == "decode"]
        p0, pl = pre[0], pre[-1]
        win_iv = [iv for s in ss for iv in s["gpu_iv"] if iv[0] >= t0]
        busy = union([(max(a, t0), min(b, tl)) for a, b in win_iv if b > t0 and a < tl])
        pre_iv = [iv for s in pre for iv in s["gpu_iv"]]
        # first decode step's kernels can start before prefill outputs are processed (async scheduling)
        ttft = dict(
            queue_to_schedule=p0["sched_ts"] - t0,
            cpu_until_gpu_start=p0["fwd_start"] - p0["sched_ts"],
            gpu_prefill_span=pl["fwd_end"] - p0["fwd_start"],
            gpu_prefill_busy=union([iv for s in pre for iv in s["gpu_iv"] if iv[1] <= pl["fwd_end"] + 0.01]),
            sample_after_fwd=(pl["smp_end"] - pl["fwd_end"]) if pl["smp_end"] else 0.0,
            output_to_engine_ready=(pl["out_ready"] - pl["smp_end"]) if pl["out_ready"] and pl["smp_end"] else 0.0,
            frontend_to_client=tf - (pl["out_ready"] or tf),
            total_client=tf - t0,
            prefill_cpu_sched=sum(s["sched_us"] for s in pre), prefill_cpu_exec=sum(s["exec_us"] for s in pre),
            prefill_cpu_prep=sum(s["prep_us"] for s in pre), prefill_steps=len(pre),
        )
        cyc = []
        for a, b in zip(dec, dec[1:]):
            if a["out_ready"] and b["out_ready"]:
                cyc.append(b["out_ready"] - a["out_ready"])
        dmed = lambda k: med(s[k] for s in dec) if dec else 0.0
        dcats = collections.defaultdict(list)
        for s in dec:
            for k in set(s["cats"]) | {"lm_head", "sampler"}:
                dcats[k].append(s["cats"].get(k, 0.0))
        gpu_per_step = [union(s["gpu_iv"]) for s in dec]
        tpot = dict(
            client_tpot=(tl - tf) / (CFG[cfg][1] - 1), decode_steps=len(dec), graph_steps=sum(s["graph"] for s in dec),
            cycle_median=med(cyc) if cyc else 0.0, gpu_busy_median=med(gpu_per_step) if gpu_per_step else 0.0,
            fwd_gpu_median=med(sum(v for k, v in s["cats"].items() if k not in ("lm_head", "sampler")) for s in dec),
            lm_head_median=med(dcats["lm_head"]), sampler_median=med(dcats["sampler"]),
            cpu_sched=dmed("sched_us"), cpu_exec=dmed("exec_us"), cpu_prep=dmed("prep_us"), cpu_sample=dmed("sample_cpu_us"),
            cpu_wait=dmed("wait_us"), cpu_update=dmed("update_us"),
            cats={k: med(v) for k, v in dcats.items()},
        )
        tpot["cpu_nonwait"] = tpot["cpu_sched"] + tpot["cpu_exec"] + tpot["cpu_sample"] + tpot["cpu_update"]
        tpot["gpu_idle_median"] = tpot["cycle_median"] - tpot["gpu_busy_median"]
        reqs.append(dict(client=dict(ttft=r["ttft"], tpot=r["tpot"], e2e=r["e2e"]), window_us=tl - t0,
                         gpu_busy_us=busy, gpu_idle_us=(tl - t0) - busy, ttft=ttft, tpot=tpot,
                         prefill_cats={k: sum(s["cats"].get(k, 0.0) for s in pre) for k in set().union(*[s["cats"] for s in pre])}))
    return reqs


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for rdir in sorted(p for p in root.iterdir() if p.is_dir()):
        route = rdir.name
        cj = rdir / f"{route}.client.json"
        if not cj.exists():
            continue
        client = json.load(open(cj))
        for cfg in CFG:
            tr = sorted(glob.glob(str(rdir / f"{route}_{cfg}_rank0.*.pt.trace.json.gz")))
            if tr:
                res[f"{route}/{cfg}"] = analyze(tr[-1], client, cfg)
                print(route, cfg, "ok", flush=True)
    json.dump(res, open(out / "serve_breakdown.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
