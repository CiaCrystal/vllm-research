import json, sys
W = "/root/rivermind-data"
R = sys.argv[1:] or ["bf16", "int8", "int8_24", "sparse"]
C = ["t1","t8","t64","t256","t1024","t2048","t8192","d1","d8"]
med = {r: {c["case"]: c["median_us"] for c in json.load(open(f"{W}/res/{r}.json"))} for r in R}
spread = {r: {c["case"]: c["ranks"][0]["latency_us"]["sample_spread_pct"] for c in json.load(open(f"{W}/res/{r}.json"))} for r in R}
def lin(r, c):
    k = json.load(open(f"{W}/prof/{r}/{c}.rank0.kernels.json"))["kernels"]
    g = sum(e["us"] for e in k if "Sparse" in e["name"] or "Kernel2" in e["name"] or "s16816gemm" in e["name"] or e["name"] in ("kernel",) or "gemm" in e["name"].lower())
    t = sum(e["us"] for e in k if "elementwise_kernel" in e["name"])
    q = sum(e["us"] for e in k if "dynamic_scaled_int8" in e["name"])
    return g, t, q
print("median forward ms, spread%")
print("case  " + "".join("%18s" % r for r in R))
for c in C:
    print("%-6s" % c + "".join("%11.2f (%4.1f%%)" % (med[r][c]/1e3, spread[r][c]) for r in R))
print("\nkernel ms: gemm / transpose / quant")
print("case  " + "".join("%24s" % r for r in R))
for c in C:
    print("%-6s" % c + "".join("  %8.2f /%7.2f /%5.2f" % tuple(x/1e3 for x in lin(r, c)) for r in R))
if "sparse" in R and "int8_24" in R:
    print("\nratio dense/sparse: linear-only(gemm+transpose), gemm-only, whole forward")
    for c in C:
        a = lin("int8_24", c); b = lin("sparse", c)
        print("%-6s %6.2f %6.2f %6.2f" % (c, a[0]/(b[0]+b[1]), a[0]/b[0], med["int8_24"][c]/med["sparse"][c]))
