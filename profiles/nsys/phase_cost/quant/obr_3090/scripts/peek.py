import json, sys, collections
d = json.load(open(sys.argv[1]))
s, o = d["states"], d["output_tokens"]
print("mean tokens", sum(o)/len(o), "at cap", sum(x >= 256 for x in o), "correct", sum(p == l for p, l in zip(d["preds"], d["labels"])))
for i in (0, 1, 2):
    print(i, "label", d["labels"][i], "pred", d["preds"][i], "|", repr(s[i][:220]))
