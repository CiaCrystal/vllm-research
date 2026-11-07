"""Stream Qwen3-8B into a W8A8 checkpoint, optionally with magnitude 2:4 pruning."""
import glob, json, pathlib, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = pathlib.Path(glob.glob("/root/rivermind-data/hf/hub/models--Qwen--Qwen3-8B/snapshots/*/config.json")[0]).parent
DST, PRUNE = pathlib.Path(sys.argv[1]), sys.argv[2] == "24"
LINEAR = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
torch.set_num_threads(16)
DST.mkdir(parents=True, exist_ok=True)

def compress(w):
    M, K = w.shape
    if PRUNE:
        top = w.abs().reshape(M, K // 4, 4).topk(2, dim=2).indices
        keep = (top + (torch.arange(K // 4) * 4)[None, :, None]).reshape(M, -1)
        w = torch.zeros_like(w).scatter_(1, keep, w.gather(1, keep))
    s = w.abs().amax(1, keepdim=True).clamp(min=1e-12) / 127
    return (w / s).round().clamp(-127, 127).to(torch.int8), s

t0, out, n = time.time(), {}, 0
for f in sorted(SRC.glob("model-*.safetensors")):
    with safe_open(f, "pt") as sf:
        for k in sf.keys():
            t = sf.get_tensor(k)
            if k.endswith(".weight") and any(x in k for x in LINEAR):
                q, s = compress(t.float())
                out[k] = q; out[k[:-len("weight")] + "weight_scale"] = s; n += 1
            else:
                out[k] = t
print("tensors", n, round(time.time() - t0), flush=True)
save_file(out, DST / "model.safetensors", metadata={"format": "pt"})
for name in ("generation_config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
    shutil.copy2(SRC / name, DST / name)
cfg = json.loads((SRC / "config.json").read_text())
cfg["quantization_config"] = {"config_groups": {"group_0": {"targets": ["Linear"],
    "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel", "dynamic": False},
    "input_activations": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "token", "dynamic": True}}},
    "quant_method": "compressed-tensors", "format": "int-quantized", "quantization_status": "compressed", "ignore": ["lm_head"]}
(DST / "config.json").write_text(json.dumps(cfg, indent=2))
print("DONE", DST, round(time.time() - t0), flush=True)
