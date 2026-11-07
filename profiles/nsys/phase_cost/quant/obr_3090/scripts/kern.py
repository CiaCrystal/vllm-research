import json
R = ["bf16","int8","int8_24","sparse"]
for r in R:
    k = json.load(open(f"/root/rivermind-data/prof/{r}/t8192.rank0.kernels.json"))
    print("##", r, "span", round(k["span_us"]), "busy", round(k["busy_us"]))
    for e in k["kernels"][:9]:
        print("  %10.1f %4d %s | %s" % (e["us"], e["calls"], e["name"][:50], e["example"][:60]))
