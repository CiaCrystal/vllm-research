"""Serve one route with the torch profiler; profile 3 streamed requests per config.

Usage: prof_serve.py <route> <trace_dir>
Each config gets its own /start_profile ... /stop_profile window (prefix
<route>_<cfg>). Client timings (wall-clock ns) are saved for alignment.
"""
import json, os, random, subprocess, sys, threading, time, urllib.request
from pathlib import Path

W = "/root/rivermind-data"
SP = ["--linear-backend", "cutlass_sparse"]
ROUTES = {"bf16": ("Qwen/Qwen3-8B", []), "int8": (f"{W}/ckpt/int8_dense", []),
          "sparse": ("/root/obr_ckpt/int8_24_obr", SP)}
CONFIGS = {"A": (256, 33, 1), "D": (1024, 33, 8), "F": (256, 129, 8)}
route, tdir = sys.argv[1], Path(sys.argv[2]).resolve()
tdir.mkdir(parents=True, exist_ok=True)
model, extra = ROUTES[route]
port = 8021
url = f"http://127.0.0.1:{port}"
rng = random.Random(0)
pcfg = {"profiler": "torch", "torch_profiler_dir": str(tdir), "torch_profiler_with_stack": False}
env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_SERVER_DEV_MODE="1", VLLM_NVTX_ENABLE="0",
           VLLM_CUSTOM_SCOPES_FOR_PROFILING="1")
cmd = [f"{W}/vllm/.venv/bin/vllm", "serve", model, "--host", "127.0.0.1", "--port", str(port),
       "--gpu-memory-utilization", "0.85", "--no-enable-prefix-caching", "--max-model-len", "2048",
       "--max-num-batched-tokens", "8192", "--profiler-config", json.dumps(pcfg)] + extra
print("CMD:", " ".join(cmd), flush=True)


def post(path, body=None, timeout=600):
    req = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                 method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def stream(cfg):
    plen, mtok, bs = CONFIGS[cfg]
    p = [[rng.randint(1000, 20000) for _ in range(plen)] for _ in range(bs)]
    body = {"model": model, "prompt": p[0] if bs == 1 else p, "max_tokens": mtok, "temperature": 0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    post("/pause?mode=keep&clear_cache=false", timeout=60)
    chunks, usage, err = [], {}, []

    def send():
        try:
            req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                for line in r:
                    now = time.time_ns()
                    if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                        continue
                    d = json.loads(line[6:])
                    if d.get("usage"):
                        usage.update(d["usage"])
                    for ch in d.get("choices") or []:
                        chunks.append((ch["index"], now))
        except Exception as e:
            err.append(e)

    th = threading.Thread(target=send); th.start(); time.sleep(0.3)
    t0 = time.time_ns(); post("/resume", timeout=60); th.join()
    if err:
        raise err[0]
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (plen * bs, mtok * bs), usage
    first = {}
    for i, t in chunks:
        first.setdefault(i, t)
    tf, tl = max(first.values()), max(t for _, t in chunks)
    return dict(t0_ns=t0, first_ns=tf, last_ns=tl, ttft=(tf - t0) / 1e6, tpot=(tl - tf) / 1e6 / (mtok - 1),
                e2e=(tl - t0) / 1e6, n_chunks=len(chunks), chunk_ns=[t for _, t in chunks])


log = open(tdir / f"{route}.serve.log", "w")
proc = subprocess.Popen(cmd, cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
res, rc = {"route": route, "cmd": cmd}, 0
try:
    for i in range(600):
        try:
            urllib.request.urlopen(url + "/health", timeout=2); res["ready_s"] = i; break
        except Exception:
            time.sleep(1)
            if proc.poll() is not None:
                sys.exit(f"server died rc={proc.returncode}")
    else:
        sys.exit("NOT READY")
    print("ready", res["ready_s"], flush=True)
    for c in CONFIGS:
        stream(c)  # warmup, not profiled
        t = time.time_ns(); post("/start_profile", {"profile_prefix": f"{route}_{c}"}); res[f"{c}_start_ns"] = t
        try:
            res[c] = [stream(c) for _ in range(3)]
        finally:
            t = time.time(); post("/stop_profile", timeout=1800); res[f"{c}_stop_s"] = round(time.time() - t, 1)
        print(c, [{k: round(v[k], 2) for k in ("ttft", "tpot", "e2e")} for v in res[c]], "stop", res[f"{c}_stop_s"], "s", flush=True)
    (tdir / f"{route}.client.json").write_text(json.dumps(res))
except Exception as e:
    print("FAILED", type(e).__name__, str(e)[:500], flush=True); rc = 3
finally:
    try:
        os.killpg(proc.pid, 15); proc.wait(timeout=60)
    except Exception:
        os.killpg(proc.pid, 9)
sys.exit(rc)
