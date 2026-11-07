"""Decode a W8A8 checkpoint into plain bf16 weights, so activations stay float."""
import glob, json, pathlib, shutil, sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
out = {}
for f in sorted(glob.glob(str(src / "*.safetensors"))):
    with safe_open(f, "pt") as sf:
        keys = set(sf.keys())
        for k in keys:
            if k.endswith(".weight_scale"):
                continue
            t = sf.get_tensor(k)
            sk = k[:-6] + "weight_scale"
            out[k] = (t.float() * sf.get_tensor(sk)).to(torch.bfloat16) if sk in keys else t
save_file(out, dst / "model.safetensors", metadata={"format": "pt"})
cfg = json.loads((src / "config.json").read_text()); cfg.pop("quantization_config")
(dst / "config.json").write_text(json.dumps(cfg, indent=2))
for n in ("generation_config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
    shutil.copy2(src / n, dst / n)
print("DONE", dst, len(out))
