import json
W = "/root/rivermind-data"
def lin(r, c):
    k = json.load(open(f"{W}/prof/{r}/{c}.rank0.kernels.json"))["kernels"]
    g = sum(e["us"] for e in k if "Sparse" in e["name"] or "Kernel2" in e["name"])
    t = sum(e["us"] for e in k if "elementwise_kernel" in e["name"])
    return g, t
med = lambda r: {c["case"]: c["median_us"] for c in json.load(open(f"{W}/res/{r}.json"))}
a, b = med("int8_24"), med("sparse_col")
print("case dense sparse_col extra ratio_linear ratio_forward")
for c in ["t1","t8","t64","t256","t1024","t2048","t8192","d1","d8"]:
    ga, _ = lin("int8_24", c); gb, tb = lin("sparse_col", c)
    print("%-6s %8.2f %8.2f %7.2f %6.2f %6.2f" % (c, ga/1e3, gb/1e3, tb/1e3, ga/(gb+tb), a[c]/b[c]))
