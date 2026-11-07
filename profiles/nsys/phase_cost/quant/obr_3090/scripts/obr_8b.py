"""Sequential 2:4 OBR for Qwen3-8B: one block on the GPU at a time, batched PCG.

Writes INT8 and INT4 (stored as s8) checkpoints as one shard per block, so a run
can resume. The calibration chain follows the INT8 weights.
"""
import argparse, glob, json, pathlib, shutil, time
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=36)
ap.add_argument("--check", action="store_true")
ap.add_argument("--tol", type=float, default=1e-3)
ap.add_argument("--maxit", type=int, default=40)
ap.add_argument("--calib", default="", help="Tensor file of [128, 2048] calibration token ids.")
ap.add_argument("--root", default="/root/obr_ckpt")
ap.add_argument("--tag", default="")
ap.add_argument("--root4", default="", help="Write the INT4 checkpoint here instead of --root.")
args = ap.parse_args()

W = pathlib.Path("/root/rivermind-data")
SRC = pathlib.Path(glob.glob(str(W / "hf/hub/models--Qwen--Qwen3-8B/snapshots/*/config.json"))[0]).parent
ROOT = pathlib.Path(args.root)
LEVELS = {"int8": 127, "int4": 7}
LINEAR = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
SHARED = {"self_attn.k_proj": "self_attn.q_proj", "self_attn.v_proj": "self_attn.q_proj", "mlp.up_proj": "mlp.gate_proj"}
DAMP, BS = 0.01, 4
dev = "cuda"
torch.set_num_threads(16)


def pcg(H, lam, mask, err):
    """Minimize (err + d)^T H (err + d) over rows of d supported on mask."""
    inv = (1 / (H.diagonal() + lam))[None, :]
    mv = lambda v: (v @ H) * mask + lam * v
    b = -(err @ H) * mask
    x = torch.zeros_like(b)
    r = b.clone()
    z = r * inv * mask
    p = z.clone()
    rz = (r * z).sum(1, keepdim=True)
    b2 = (b * b).sum(1, keepdim=True).clamp(min=1e-30)
    for it in range(args.maxit):
        q = mv(p)
        a = rz / (p * q).sum(1, keepdim=True).clamp(min=1e-30)
        x += a * p
        r -= a * q
        if ((r * r).sum(1, keepdim=True) / b2).max().sqrt() < args.tol:
            break
        z = r * inv * mask
        rz2 = (r * z).sum(1, keepdim=True)
        p = z + (rz2 / rz.clamp(min=1e-30)) * p
        rz = rz2
    return x, it + 1


def mask_24(w, d):
    M, K = w.shape
    top = (w.abs() * d.sqrt()).reshape(M, K // 4, 4).topk(2, dim=2).indices.sort(dim=2).values
    keep = (top + (torch.arange(K // 4, device=dev) * 4)[None, :, None]).reshape(M, -1)
    return torch.zeros_like(w).scatter_(1, keep, 1.0), keep


def quant(v, s, qmax):
    return (v / s).round().clamp(-qmax, qmax) * s


def rel(H, d, w):
    return ((d @ H * d).sum() / (w @ H * w).sum()).item()


def exact_check(H, lam, mask, err, x, name):
    """Compare the objective reached by PCG with the direct solve on a few rows."""
    obj = lambda d: ((d @ H) * d).sum().item()
    idx = torch.linspace(0, mask.shape[0] - 1, 16).long().to(dev)
    e, m, xs = err[idx], mask[idx].bool(), x[idx]
    ex = torch.zeros_like(xs)
    t = time.time()
    for i in range(16):
        r = m[i].nonzero()[:, 0]
        Hrr = H[r[:, None], r[None, :]].clone()
        Hrr.diagonal().add_(lam)
        ex[i, r] = torch.linalg.solve(Hrr, -(e[i] @ H)[r])
    print("  check", name, "objective pcg/none %.5f exact/none %.5f, 16 rows direct %.1fs" % (obj(e + xs) / obj(e), obj(e + ex) / obj(e), time.time() - t), flush=True)


def compress(w, H, name):
    lam = DAMP * H.diagonal().mean()
    mask, keep = mask_24(w, H.diagonal())
    err = -w * (1 - mask)
    d1, it1 = pcg(H, lam, mask, err)
    if args.check:
        exact_check(H, lam, mask, err, d1, name)
    comp = (w + d1) * mask
    if args.check:
        drop = rel(H, err, w)
        kept = rel(H, comp - w, w)
        print("  true relative error", name, "pruned %.4f compensated %.4f" % (drop, kept), flush=True)
        assert kept < drop
    n2 = keep.shape[1] // 2
    me = torch.zeros_like(w).scatter_(1, keep[:, :n2], 1.0)
    mr = mask - me
    res, its = {}, [it1]
    for lv, qmax in LEVELS.items():
        s = comp.abs().amax(1, keepdim=True).clamp(min=1e-12) / qmax
        qe = quant(comp, s, qmax) * me
        d2, it = pcg(H, lam, mr, qe - comp * me)
        its.append(it)
        final = qe + quant(comp * mr + d2, s, qmax) * mr
        res[lv] = (final, s)
    return res, its


from transformers import AutoModelForCausalLM, AutoTokenizer
t0 = time.time()
tok = AutoTokenizer.from_pretrained(SRC)
ids = tok((W / "data/wiki_train.txt").read_text(), return_tensors="pt", add_special_tokens=False).input_ids[0]
g = torch.Generator().manual_seed(0)
starts = torch.randint(0, ids.numel() - 2049, (128,), generator=g)
data = torch.stack([ids[s:s + 2048] for s in starts.tolist()])
if args.calib:
    data = torch.load(args.calib)
    assert data.shape == (128, 2048), data.shape
    print("calibration from", args.calib, flush=True)
print("tokens", ids.numel(), round(time.time() - t0), flush=True)
model = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.bfloat16).eval()
m = model.model
T = 2048
pos = torch.arange(T, device=dev)[None]
amask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))[None, None]
print("loaded", round(time.time() - t0), flush=True)

with safe_open(SRC / "model-00001-of-00005.safetensors", "pt") as _:
    pass
index = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
get = lambda k: safe_open(SRC / index[k], "pt").get_tensor(k)
dirs = {lv: ROOT / f"{lv}_24_obr{args.tag}" for lv in LEVELS}
if args.root4:
    dirs["int4"] = pathlib.Path(args.root4) / f"int4_24_obr{args.tag}"
if args.limit == 36:
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)


def run(layer, h, pe):
    r = layer(h.to(dev), attention_mask=amask, position_embeddings=pe)
    return r[0] if isinstance(r, tuple) else r


with torch.inference_mode():
    m.embed_tokens.to(dev)
    hs = [m.embed_tokens(data[i:i + BS].to(dev)).cpu() for i in range(0, 128, BS)]
    m.embed_tokens.cpu()
    m.rotary_emb.to(dev)
    pe = m.rotary_emb(hs[0].to(dev), pos)
    if args.check:
        model.to(dev)
        x = hs[0].to(dev)
        for l in m.layers[: m.config.num_hidden_layers]:
            x = run(l, x, pe)
        ref = m(input_ids=data[:BS].to(dev), use_cache=False).last_hidden_state
        print("sanity rel err", ((m.norm(x) - ref).float().norm() / ref.float().norm()).item(), flush=True)
        model.cpu(); m.rotary_emb.to(dev)
        torch.cuda.empty_cache()

    for li in range(args.limit):
        layer = m.layers[li].to(dev)
        tb = time.time()
        shard = {lv: dirs[lv] / f"block-{li:02d}.safetensors" for lv in LEVELS}
        if all(p.exists() for p in shard.values()):
            sd = load_file(shard["int8"])
            for n in LINEAR:
                k = f"model.layers.{li}.{n}.weight"
                mod = layer.get_submodule(n)
                mod.weight.data.copy_((sd[k].to(dev).float() * sd[k[:-6] + "weight_scale"].to(dev)).to(mod.weight.dtype))
            hs = [run(layer, h, pe).cpu() for h in hs]
            layer.cpu()
            print(li, "resumed", flush=True)
            continue
        Hs, cnt, hooks = {}, {}, []

        def mk(name):
            def pre(mod, inp):
                if name in SHARED:
                    return
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                Hs[name] = Hs.get(name, 0) + x.T @ x
                cnt[name] = cnt.get(name, 0) + x.shape[0]
            return pre

        for n in LINEAR:
            hooks.append(layer.get_submodule(n).register_forward_pre_hook(mk(n)))
        for h in hs:
            run(layer, h, pe)
        for h in hooks:
            h.remove()
        Hs = {k: v / cnt[k] for k, v in Hs.items()}
        tc = time.time() - tb
        out = {lv: {} for lv in LEVELS}
        errs = {lv: [] for lv in LEVELS}
        its = []
        for n in LINEAR:
            H = Hs[SHARED.get(n, n)]
            mod = layer.get_submodule(n)
            w = mod.weight.data.float()
            res, it = compress(w, H, n)
            its.append(it)
            k = f"model.layers.{li}.{n}.weight"
            for lv, (final, s) in res.items():
                errs[lv].append(rel(H, final - w, w))
                out[lv][k] = (final / s).round().to(torch.int8).cpu()
                out[lv][k[:-6] + "weight_scale"] = s.cpu()
            mod.weight.data.copy_(res["int8"][0].to(mod.weight.dtype))
        ts = time.time() - tb - tc
        del Hs
        hs = [run(layer, h, pe).cpu() for h in hs]
        layer.cpu()
        extra = {k: get(k) for k in index if k.startswith(f"model.layers.{li}.") and not any(k == f"model.layers.{li}.{n}.weight" for n in LINEAR)}
        if args.limit == 36:
            for lv in LEVELS:
                tmp = shard[lv].with_suffix(".tmp")
                save_file({**out[lv], **extra}, tmp, metadata={"format": "pt"})
                tmp.rename(shard[lv])
        print(li, "t", round(time.time() - t0), "calib", round(tc, 1), "solve", round(ts, 1), "block", round(time.time() - tb, 1),
              "err8", round(sum(errs["int8"]) / 7, 5), "err4", round(sum(errs["int4"]) / 7, 5), "its", its,
              "peakGB", round(torch.cuda.max_memory_allocated() / 2**30, 1), flush=True)

if args.limit == 36:
    rest = {k: get(k) for k in index if not k.startswith("model.layers.")}
    cfg = json.loads((SRC / "config.json").read_text())
    cfg["quantization_config"] = {"config_groups": {"group_0": {"targets": ["Linear"],
        "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel", "dynamic": False},
        "input_activations": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "token", "dynamic": True}}},
        "quant_method": "compressed-tensors", "format": "int-quantized", "quantization_status": "compressed", "ignore": ["lm_head"]}
    for lv, d in dirs.items():
        save_file(rest, d / "rest.safetensors", metadata={"format": "pt"})
        for n in ("generation_config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
            shutil.copy2(SRC / n, d / n)
        (d / "config.json").write_text(json.dumps(cfg, indent=2))
    print("OBRDONE", round(time.time() - t0), flush=True)
