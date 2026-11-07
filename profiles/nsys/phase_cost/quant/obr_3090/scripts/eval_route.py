"""Serve one route, then measure A/D/F streaming, GSM8K and perplexity in that run."""
import argparse, json, math, os, pathlib, random, statistics, subprocess, sys, threading, time, urllib.request
W = "/root/rivermind-data"
OUT = pathlib.Path(f"{W}/eval"); OUT.mkdir(exist_ok=True)
S = pathlib.Path(__file__).resolve().parent
HF = "Qwen/Qwen3-8B"
SP = ["--linear-backend", "cutlass_sparse"]
ROUTES = {"bf16": (HF, []), "int8": (f"{W}/ckpt/int8_dense", []), "int8_nocomp": (f"{W}/ckpt/int8_24_nocomp", []),
          "int8_obr": ("/root/obr_ckpt/int8_24_obr", []), "int4_obr": ("/root/obr_ckpt/int4_24_obr", []),
          "int8_obr_sparse": ("/root/obr_ckpt/int8_24_obr", SP),
          "int8_obr_selfgen": (f"{W}/ckpt/int8_24_obr_selfgen", []), "int4_obr_selfgen": (f"{W}/ckpt/int4_24_obr_selfgen", []),
          "int8_obr_mix": (f"{W}/ckpt/int8_24_obr_mix", []), "int4_obr_mix": ("/root/obr_ckpt/int4_24_obr_mix", []),
          "float_obr": (f"{W}/ckpt/float_obr", []), "float_nocomp": (f"{W}/ckpt/float_nocomp", [])}
CONFIGS = {"A": (256, 33, 1), "D": (1024, 33, 8), "F": (256, 129, 8)}
ap = argparse.ArgumentParser()
ap.add_argument("route"); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--port", type=int, default=8021)
ap.add_argument("--skip-gsm", action="store_true"); ap.add_argument("--skip-stream", action="store_true")
args = ap.parse_args()
prefix = OUT / args.route
model, extra = ROUTES[args.route]
url = f"http://127.0.0.1:{args.port}"
rng = random.Random(0)
env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_SERVER_DEV_MODE="1", VLLM_NVTX_ENABLE="0")
cmd = [f"{W}/vllm/.venv/bin/vllm", "serve", model, "--host", "127.0.0.1", "--port", str(args.port), "--gpu-memory-utilization", "0.85",
       "--no-enable-prefix-caching", "--max-model-len", "2048", "--max-num-batched-tokens", "8192"] + extra
print("CMD:", " ".join(cmd), flush=True)

def post(path, body=None, timeout=600):
    req = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def stream(cfg):
    plen, mtok, bs = CONFIGS[cfg]
    p = [[rng.randint(1000, 20000) for _ in range(plen)] for _ in range(bs)]
    body = {"model": model, "prompt": p[0] if bs == 1 else p, "max_tokens": mtok, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    post("/pause?mode=keep&clear_cache=false", timeout=60)
    seen, last, usage, err = {}, [0.0], {}, []
    def send():
        try:
            req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                for line in r:
                    now = time.perf_counter()
                    if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                        continue
                    d = json.loads(line[6:])
                    if d.get("usage"):
                        usage.update(d["usage"])
                    for ch in d.get("choices") or []:
                        seen.setdefault(ch["index"], now); last[0] = now
        except Exception as e:
            err.append(e)
    th = threading.Thread(target=send); th.start(); time.sleep(0.3)
    t0 = time.perf_counter(); post("/resume", timeout=60); th.join()
    if err:
        raise err[0]
    assert len(seen) == bs and (usage["prompt_tokens"], usage["completion_tokens"]) == (plen * bs, mtok * bs), usage
    first = max(seen.values())
    return dict(ttft=(first - t0) * 1e3, tpot=(last[0] - first) * 1e3 / (mtok - 1), e2e=(last[0] - t0) * 1e3)

def perplexity():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF)
    ids = tok(open(f"{W}/data/wiki_test.txt").read(), add_special_tokens=False).input_ids
    T, n = 2047, min(len(ids) // 2047, 64)
    nll, cnt = 0.0, 0
    for i in range(n):
        r = json.loads(post("/v1/completions", {"model": model, "prompt": ids[i * T:(i + 1) * T], "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}))
        for e in r["choices"][0]["prompt_logprobs"][1:]:
            nll -= next(iter(e.values()))["logprob"]; cnt += 1
    return math.exp(nll / cnt), cnt

log = open(str(prefix) + ".serve.log", "w")
proc = subprocess.Popen(cmd, cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
rc, t0 = 0, time.time()
res = {}
try:
    for i in range(420):
        try:
            urllib.request.urlopen(url + "/health", timeout=2); res["ready_s"] = i; print("ready", i, flush=True); break
        except Exception:
            time.sleep(1)
            if proc.poll() is not None:
                sys.exit(f"server died rc={proc.returncode}")
    else:
        sys.exit("NOT READY")
    for c in ([] if args.skip_stream else CONFIGS):
        stream(c)
        x = [stream(c) for _ in range(args.reps)]
        res[c] = x
        print(c, {k: round(statistics.median(v[k] for v in x), 2) for k in x[0]}, flush=True)
    res["ppl"], res["ppl_tokens"] = perplexity()
    print("ppl", round(res["ppl"], 3), res["ppl_tokens"], flush=True)
    if not args.skip_gsm:
        t = time.time()
        r = subprocess.run([f"{W}/vllm/.venv/bin/python", "-I", str(S / "gsm_8b.py"), str(prefix) + ".answers.json", "--port", str(args.port),
                            "--model", model, "--save-results", str(prefix) + ".result.json"], cwd=W, env=env,
                           stdout=open(str(prefix) + ".eval.log", "w"), stderr=subprocess.STDOUT)
        res["gsm_rc"] = r.returncode; res["gsm_s"] = round(time.time() - t)
        if r.returncode == 0:
            res["gsm"] = json.load(open(str(prefix) + ".result.json"))["accuracy"]
        print("gsm", res.get("gsm"), "rc", r.returncode, res["gsm_s"], "s", flush=True)
        rc = r.returncode
    (OUT / f"{args.route}.json").write_text(json.dumps(res))
except Exception as e:
    print("FAILED", type(e).__name__, str(e)[:500], flush=True); rc = 3
finally:
    try:
        os.killpg(proc.pid, 15); proc.wait(timeout=30)
    except Exception:
        os.killpg(proc.pid, 9)
sys.exit(rc)
