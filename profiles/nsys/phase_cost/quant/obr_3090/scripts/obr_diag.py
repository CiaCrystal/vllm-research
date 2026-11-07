"""Layer output error of OBR vs magnitude pruning on calibration and held-out text."""
import glob, pathlib, torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
W = pathlib.Path("/root/rivermind-data"); dev = "cuda"
SRC = pathlib.Path(glob.glob(str(W / "hf/hub/models--Qwen--Qwen3-8B/snapshots/*/config.json"))[0]).parent
tok = AutoTokenizer.from_pretrained(SRC)
enc = lambda f: tok((W / f"data/{f}").read_text(), return_tensors="pt", add_special_tokens=False).input_ids[0]
ids = enc("wiki_train.txt")
g = torch.Generator().manual_seed(0)
starts = torch.randint(0, ids.numel() - 2049, (128,), generator=g)[:16]
train = torch.stack([ids[s:s + 2048] for s in starts.tolist()])
ti = enc("wiki_test.txt"); test = ti[:16 * 2048].reshape(16, 2048)
model = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.bfloat16).to(dev).eval()
m = model.model
pos = torch.arange(2048, device=dev)[None]
am = torch.tril(torch.ones(2048, 2048, dtype=torch.bool, device=dev))[None, None]
nc = safe_open(W / "ckpt/int8_24_nocomp/model.safetensors", "pt")
BLOCKS = [0, 11, 23, 35]
NAMES = ["self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"]
inputs = {"train": {}, "test": {}}
def collect(tag, data):
    hooks = []
    for li in BLOCKS:
        for n in NAMES:
            def pre(mod, inp, key=(li, n)):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                inputs[tag].setdefault(key, []).append(x[::8].clone())
            hooks.append(m.layers[li].get_submodule(n).register_forward_pre_hook(pre))
    with torch.inference_mode():
        for i in range(0, 16, 4):
            m(input_ids=data[i:i + 4].to(dev), use_cache=False)
    for h in hooks:
        h.remove()
collect("train", train); collect("test", test)
print("%-5s %-10s %-6s %8s %8s %8s %8s %8s" % ("blk", "layer", "data", "nocomp", "obr", "normR", "maxRow", "clip%"))
with torch.inference_mode():
    for li in BLOCKS:
        ob = safe_open(f"/root/obr_ckpt/int8_24_obr/block-{li:02d}.safetensors", "pt")
        for n in NAMES:
            k = f"model.layers.{li}.{n}.weight"
            w = m.layers[li].get_submodule(n).weight.float()
            dec = lambda f: f.get_tensor(k).to(dev).float() * f.get_tensor(k[:-6] + "weight_scale").to(dev)
            wn, wo = dec(nc), dec(ob)
            q = ob.get_tensor(k).to(dev)
            nr = (wo.norm(dim=1) / w.norm(dim=1).clamp(min=1e-9))
            clip = (q.abs() == 127).float().mean().item() * 100
            for tag in ("train", "test"):
                x = torch.cat(inputs[tag][(li, n)])
                base = (x @ w.T).norm()
                e = lambda v: ((x @ (v - w).T).norm() / base).item() ** 2
                print("%-5d %-10s %-6s %8.4f %8.4f %8.2f %8.2f %8.3f" % (li, n.split(".")[1][:10], tag, e(wn), e(wo), nr.median().item(), nr.max().item(), clip), flush=True)
