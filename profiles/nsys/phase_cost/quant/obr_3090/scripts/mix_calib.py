"""Mix calibration: the first 64 WikiText windows of obr_8b.py plus 64 self-generated GSM8K windows."""
import glob, pathlib, torch
from transformers import AutoTokenizer
W = pathlib.Path("/root/rivermind-data")
SRC = pathlib.Path(glob.glob(str(W / "hf/hub/models--Qwen--Qwen3-8B/snapshots/*/config.json"))[0]).parent
tok = AutoTokenizer.from_pretrained(SRC)
ids = tok((W / "data/wiki_train.txt").read_text(), return_tensors="pt", add_special_tokens=False).input_ids[0]
g = torch.Generator().manual_seed(0)
starts = torch.randint(0, ids.numel() - 2049, (128,), generator=g)
wiki = torch.stack([ids[s:s + 2048] for s in starts.tolist()])[:64]
self_gen = torch.load(W / "data/calib_selfgen.pt")[:64]
mix = torch.cat([wiki, self_gen])
torch.save(mix, W / "data/calib_mix.pt")
print("MIX", mix.shape)
