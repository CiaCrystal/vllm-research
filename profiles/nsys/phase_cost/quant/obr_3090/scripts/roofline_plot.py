"""Roofline SVG + CSV from kernel_roofline.csv (no plotting library needed).

Usage: roofline_plot.py <profiling_dir>
"""
import csv, collections, math, sys
from pathlib import Path

d = Path(sys.argv[1])
rows = list(csv.DictReader(open(d / "kernel_roofline.csv")))
ROOF = {"bf16": 77.8, "int8": 235.2, "sparse": 252.5}
BW = 887.0
ROUTES = {"bf16": ("bf16", "#2563eb"), "int8": ("INT8 dense", "#16a34a"), "sparse_col": ("INT8 2:4 sparse", "#dc2626")}
PREFILL = ["t64", "t256", "t1024", "t2048", "t8192"]
DECODE = ["d1", "d8"]

agg = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
for r in rows:
    if r["route"] not in ROUTES:
        continue
    cat = r["category"]
    group = "linear" if cat.startswith("linear:") else "attention" if cat == "attention" else None
    if group is None:
        continue
    a = agg[(r["route"], r["case"], group)]
    a[0] += float(r["us"]); a[1] += float(r["gflop"]); a[2] += float(r["mbytes"])

pts = []
for (route, case, group), (us, gf, mb) in sorted(agg.items()):
    if case not in PREFILL + DECODE or gf == 0:
        continue
    ai = gf * 1e9 / (mb * 1e6)
    tf = gf * 1e9 / (us * 1e-6) / 1e12
    roof = ROOF["bf16" if group == "attention" or route == "bf16" else "sparse" if route.startswith("sparse") else "int8"]
    attainable = min(roof, ai * BW / 1e3)
    pts.append(dict(route=route, case=case, phase="decode" if case in DECODE else "prefill", group=group,
                    us=round(us, 1), gflop=round(gf, 2), mbytes=round(mb, 1), ai=round(ai, 2), tflops=round(tf, 2),
                    attainable_tflops=round(attainable, 2), pct_of_attainable=round(100 * tf / attainable, 1)))
with open(d / "roofline_points.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(pts[0])); w.writeheader(); w.writerows(pts)

# SVG, log-log
Wd, Ht, L, R, T, B = 900, 560, 70, 200, 30, 50
xmin, xmax, ymin, ymax = 0.5, 10000, 0.1, 400


def X(v): return L + (math.log10(v) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * (Wd - L - R)
def Y(v): return Ht - B - (math.log10(v) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)) * (Ht - T - B)


s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wd}" height="{Ht}" font-family="sans-serif" font-size="12">',
     f'<rect width="{Wd}" height="{Ht}" fill="white"/>']
for e in range(0, 5):
    for m in (1, 2, 5):
        v = m * 10 ** e
        if xmin <= v <= xmax:
            s.append(f'<line x1="{X(v):.1f}" y1="{T}" x2="{X(v):.1f}" y2="{Ht-B}" stroke="#eee"/>')
            if m == 1: s.append(f'<text x="{X(v):.1f}" y="{Ht-B+16}" text-anchor="middle">{v:g}</text>')
for e in range(-1, 3):
    for m in (1, 2, 5):
        v = m * 10 ** e
        if ymin <= v <= ymax:
            s.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{Wd-R}" y2="{Y(v):.1f}" stroke="#eee"/>')
            if m == 1: s.append(f'<text x="{L-6}" y="{Y(v)+4:.1f}" text-anchor="end">{v:g}</text>')
s.append(f'<rect x="{L}" y="{T}" width="{Wd-L-R}" height="{Ht-T-B}" fill="none" stroke="#333"/>')
s.append(f'<text x="{(Wd-R+L)/2}" y="{Ht-12}" text-anchor="middle">Arithmetic intensity (FLOP/byte, dense-equivalent FLOPs)</text>')
s.append(f'<text transform="translate(18,{(Ht-B+T)/2}) rotate(-90)" text-anchor="middle">TFLOPS / TOPS</text>')
for name, roof, col in [("bf16 77.8", ROOF["bf16"], "#2563eb"), ("INT8 235", ROOF["int8"], "#16a34a"), ("sparse 252", ROOF["sparse"], "#dc2626")]:
    knee = roof * 1e3 / BW
    pts_line = [(xmin, xmin * BW / 1e3), (knee, roof), (xmax, roof)]
    s.append('<polyline fill="none" stroke="%s" stroke-dasharray="5,3" points="%s"/>' % (col, " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in pts_line)))
    s.append(f'<text x="{Wd-R+4}" y="{Y(roof)+4:.1f}" fill="{col}">{name}</text>')
s.append(f'<text x="{X(1.2):.1f}" y="{Y(1.2*BW/1e3)-8:.1f}" fill="#555" transform="rotate(-27 {X(1.2):.1f} {Y(1.2*BW/1e3)-8:.1f})">887 GB/s</text>')
for p in pts:
    col = ROUTES[p["route"]][1]
    x, y = X(max(p["ai"], xmin)), Y(max(p["tflops"], ymin))
    if p["group"] == "linear":
        shape = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{col}" fill-opacity="{0.9 if p["phase"]=="prefill" else 0.35}" stroke="{col}"/>'
    else:
        shape = f'<rect x="{x-4:.1f}" y="{y-4:.1f}" width="8" height="8" fill="none" stroke="{col}" stroke-width="1.5"/>'
    s.append(shape)
    s.append(f'<text x="{x+7:.1f}" y="{y+3:.1f}" font-size="9" fill="{col}">{p["case"]}</text>')
ly = T + 10
for route, (label, col) in ROUTES.items():
    s.append(f'<circle cx="{Wd-R+12}" cy="{ly+40}" r="5" fill="{col}"/><text x="{Wd-R+22}" y="{ly+44}">{label}</text>'); ly += 18
s.append(f'<text x="{Wd-R+6}" y="{ly+60}" font-size="11">filled dot: linear, prefill</text>')
s.append(f'<text x="{Wd-R+6}" y="{ly+76}" font-size="11">pale dot: linear, decode</text>')
s.append(f'<text x="{Wd-R+6}" y="{ly+92}" font-size="11">square: attention</text>')
s.append("</svg>")
(d / "roofline.svg").write_text("\n".join(s))
print(len(pts), "points")
