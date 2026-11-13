"""
Combines adr/0012_dp_scan_tables.py's and adr/0012_dp_scan_fit.py's CSVs into
one plot: the achieved Gauss-Legendre relative residual vs. dp = p10 - p0 for
all three PErecoil representations of the same real detector (402_NR_Cf.h5),
backing adr/0012-perecoil-handles-register-a-fit-not-a-table.md's Context.

    python adr/0012_dp_scan_plot.py adr/0012_dp_scan_tables.csv adr/0012_dp_scan_fit.csv
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PNG = os.path.join(REPO_ROOT, "test", "figures", "adr0012_dp_scan.png")

STYLE = {"PCHIP": dict(color="#d62728", marker="o"),
        "SMOOTH": dict(color="#9467bd", marker="s"),
        "FITTED": dict(color="#2ca02c", marker="^")}


def load(*paths):
    series = {}
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                series.setdefault(row["label"], []).append((float(row["dp"]), float(row["rel_residual"])))
    return series


series = load(*sys.argv[1:])

fig, ax = plt.subplots(figsize=(7, 5))
for label in ("PCHIP", "SMOOTH", "FITTED"):
    pts = sorted(series[label])
    dp, rel = zip(*pts)
    ax.plot(dp, rel, label=label, **STYLE[label])
ax.set_yscale("log")
ax.set_xlabel(r"$dp = p_{10} - p_0$")
ax.set_ylabel("achieved Gauss-Legendre relative residual")
ax.set_title("Region-integral precision vs. PErecoil representation\n(detector 402, NR, Cf-252)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
fig.savefig(OUT_PNG, dpi=150)
print(f"wrote {OUT_PNG}")

for label in ("PCHIP", "SMOOTH", "FITTED"):
    vals = [r for _, r in series[label]]
    print(f"{label}: worst={max(vals):.3e}  best={min(vals):.3e}  max/min={max(vals)/min(vals):.1f}x")
