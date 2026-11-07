"""Self-generated GSM8K calibration: bf16 Qwen3-8B answers train questions in the eval format."""
import json, os, pathlib, torch
from vllm import LLM, SamplingParams
W = pathlib.Path("/root/rivermind-data")
train = [json.loads(l) for l in open(os.path.join(os.environ["TMPDIR"], "train.jsonl"))]
shots = "".join(f"Question: {t[chr(113)+chr(117)+chr(101)+chr(115)+chr(116)+chr(105)+chr(111)+chr(110)]}\nAnswer: {t[chr(97)+chr(110)+chr(115)+chr(119)+chr(101)+chr(114)]}\n\n" for t in train[:5])
qs = [t["question"] for t in train[5:1205]]
llm = LLM("Qwen/Qwen3-8B", max_model_len=2048, gpu_memory_utilization=0.85, enable_prefix_caching=False)
outs = llm.generate([shots + f"Question: {q}\nAnswer:" for q in qs], SamplingParams(temperature=0, max_tokens=256, stop=["Question"]))
pairs = [f"Question: {q}\nAnswer:{o.outputs[0].text.rstrip()}\n\n" for q, o in zip(qs, outs)]
tok = llm.get_tokenizer()
head = tok(shots, add_special_tokens=False).input_ids
P = [tok(p, add_special_tokens=False).input_ids for p in pairs]
wins, cur, i = [], list(head), 0
while len(wins) < 128:
    cur += P[i % len(P)]; i += 1
    if len(cur) >= 2048:
        wins.append(cur[:2048]); cur = list(head)
data = torch.tensor(wins)
torch.save(data, W / "data/calib_selfgen.pt")
json.dump(dict(pairs_used=i, pairs=len(P), head_tokens=len(head), mean_answer_tokens=sum(len(o.outputs[0].token_ids) for o in outs) / len(outs)), open(W / "data/calib_selfgen.json", "w"))
print("CALIB", data.shape, "pairs used", i, "of", len(P), "head", len(head), flush=True)
print(repr(tok.decode(wins[0][len(head):len(head) + 160])))
