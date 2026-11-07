"""bf16 load test on one GPU: concurrency sweep, KV saturation, Poisson arrivals.

Usage: stress_bf16.py <out_dir> [--phases E1,E2,E3,E4] [--no-wait]
Waits until the GPU has no compute processes, then for each server config
starts `vllm serve`, runs `vllm bench serve` (3 runs per combination) and
samples /metrics plus CPU of the API server, EngineCore and bench client.
"""
import argparse, json, os, re, subprocess, sys, threading, time, urllib.request
from pathlib import Path

W = "/root/rivermind-data"
VLLM = f"{W}/vllm/.venv/bin/vllm"
MODEL = "Qwen/Qwen3-8B"
PORT = 8031
URL = f"http://127.0.0.1:{PORT}"
BASE = ["--gpu-memory-utilization", "0.85", "--no-enable-prefix-caching", "--max-model-len", "2048"]
SLO = ["--goodput", "ttft:1000", "tpot:50"]
TICK = os.sysconf("SC_CLK_TCK")
METRICS = ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
           "vllm:num_preemptions_total"]
HISTS = ["vllm:request_queue_time_seconds", "vllm:time_to_first_token_seconds", "vllm:inter_token_latency_seconds",
         "vllm:request_prefill_time_seconds", "vllm:request_decode_time_seconds"]

# (phase, server mnbt, label, input, output, num_prompts, max_concurrency, request_rate)
E1 = [("E1", 8192, f"c{c}", 512, 128, n, c, "inf") for c, n in
      [(1, 8), (8, 32), (16, 64), (32, 128), (48, 192), (64, 256), (128, 384)]]
E2 = [("E2", 8192, f"c{c}", 1536, 384, n, c, "inf") for c, n in [(16, 48), (32, 64)]]
E3 = [("E3", 8192, f"r{r}", 1024, 128, 40, None, str(r)) for r in (1, 2, 3, 4)]
E4 = [("E4", 2048, f"r{r}", 1024, 128, 40, None, str(r)) for r in (2, 3, 4)]
PHASES = {"E1": E1, "E2": E2, "E3": E3, "E4": E4}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def gpu_apps():
    return [l for l in sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader").splitlines() if l.strip()]


def wait_gpu_idle(log):
    ok = 0
    while ok < 3:
        apps, mem = gpu_apps(), sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").strip()
        ok = ok + 1 if not apps and int(mem or 1) < 100 else 0
        if ok == 0:
            log(f"GPU busy: {apps} mem={mem} MiB, waiting")
            time.sleep(60)
        else:
            time.sleep(10)
    log("GPU idle 3 checks in a row")


def cpu_ticks(pid):
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return int(f[11]) + int(f[12])
    except Exception:
        return None


def scrape():
    txt = urllib.request.urlopen(URL + "/metrics", timeout=5).read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"([a-zA-Z_:]+)(\{[^}]*\})? ([0-9.eE+-]+)$", line)
        if not m:
            continue
        name, val = m.group(1), float(m.group(3))
        if name in METRICS or any(name in (h + "_sum", h + "_count") for h in HISTS):
            out[name] = out.get(name, 0.0) + val
    return out


class Sampler(threading.Thread):
    def __init__(self, pids):
        super().__init__(daemon=True)
        self.pids, self.stop, self.samples = pids, threading.Event(), []

    def run(self):
        last_gpu = 0.0
        while not self.stop.is_set():
            t = time.time()
            s = {"t": t, "cpu": {k: cpu_ticks(p) for k, p in self.pids.items() if p}}
            try:
                s["m"] = scrape()
            except Exception as e:
                s["err"] = str(e)[:100]
            if t - last_gpu > 2:
                s["gpu_apps"] = gpu_apps(); last_gpu = t
            self.samples.append(s)
            self.stop.wait(0.5)


def find_engine_core(parent):
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                st = open(f"/proc/{d}/stat").read()
                # comm is truncated to 15 chars: "VLLM::EngineCor"
                if "EngineCor" in st and int(st.rsplit(")", 1)[1].split()[1]) == parent:
                    return int(d)
            except Exception:
                pass
    return None


def start_server(mnbt, out, log):
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_NVTX_ENABLE="0")
    cmd = [VLLM, "serve", MODEL, "--host", "127.0.0.1", "--port", str(PORT), *BASE, "--max-num-batched-tokens", str(mnbt)]
    log("SERVE " + " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=W, env=env, stdout=open(out / f"serve_mnbt{mnbt}.log", "w"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    for i in range(600):
        try:
            urllib.request.urlopen(URL + "/health", timeout=2); break
        except Exception:
            time.sleep(1)
            if p.poll() is not None:
                raise RuntimeError(f"server died rc={p.returncode}")
    log(f"ready after {i}s")
    return p


def bench(case, rep, out, log, pids, warm=False):
    ph, mnbt, label, il, ol, n, c, rate = case
    name = f"{ph}_{label}_rep{rep}" if not warm else f"warm_{ph}"
    cmd = [VLLM, "bench", "serve", "--backend", "vllm", "--base-url", URL, "--endpoint", "/v1/completions",
           "--model", MODEL, "--dataset-name", "random", "--random-input-len", str(il), "--random-output-len", str(ol),
           "--random-range-ratio", "0", "--num-prompts", str(n), "--request-rate", rate, "--ignore-eos",
           "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99", *SLO,
           "--seed", str(rep), "--disable-tqdm", "--save-result", "--save-detailed",
           "--result-dir", str(out / "raw"), "--result-filename", f"{name}.json"]
    if c:
        cmd += ["--max-concurrency", str(c)]
    m0 = scrape()
    p = subprocess.Popen(cmd, cwd=W, env=dict(os.environ, HF_HUB_OFFLINE="1"),
                         stdout=open(out / "raw" / f"{name}.log", "w"), stderr=subprocess.STDOUT)
    s = Sampler({**pids, "client": p.pid}); s.start()
    t0 = time.time(); rc = p.wait(); t1 = time.time()
    s.stop.set(); s.join()
    m1 = scrape()
    foreign = sorted({a for x in s.samples for a in x.get("gpu_apps", []) if str(pids["engine"]) not in a})
    rec = dict(name=name, phase=ph, mnbt=mnbt, label=label, input=il, output=ol, num_prompts=n,
               max_concurrency=c, request_rate=rate, rep=rep, rc=rc, wall_s=t1 - t0,
               metrics_before=m0, metrics_after=m1, samples=s.samples, foreign_gpu_apps=foreign)
    if not warm:
        (out / "samples" / f"{name}.json").write_text(json.dumps(rec))
    log(f"{name} rc={rc} {t1 - t0:.1f}s foreign={foreign}")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--phases", default="E1,E2,E3,E4"); ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); (out / "raw").mkdir(parents=True, exist_ok=True); (out / "samples").mkdir(exist_ok=True)
    logf = open(out / "driver.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    cases = [c for ph in a.phases.split(",") for c in PHASES[ph]]
    for mnbt in sorted({c[1] for c in cases}, reverse=True):
        group = [c for c in cases if c[1] == mnbt]
        if not a.no_wait:
            wait_gpu_idle(log)
        p = start_server(mnbt, out, log)
        try:
            pids = {"api": p.pid, "engine": find_engine_core(p.pid)}
            log(f"pids {pids}")
            bench(("warm", mnbt, "w", 512, 128, 16, 8, "inf"), 0, out, log, pids, warm=True)
            for rep in (1, 2, 3):
                for c in group:
                    bench(c, rep, out, log, pids)
        finally:
            os.killpg(p.pid, 15)
            try:
                p.wait(timeout=60)
            except Exception:
                os.killpg(p.pid, 9)
            time.sleep(5)
    log("ALL_DONE")


if __name__ == "__main__":
    sys.exit(main())
