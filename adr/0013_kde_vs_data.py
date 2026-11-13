"""
Builds adr/0013's comparison figure: a Silverman-reflected Gaussian KDE
(correct at the Erecoil=0 boundary -- an ordinary KDE leaks density past
a hard boundary when real data sits right at it, understating the true
one-sided density there) against the real, cleaned calibration data for
all three cases this record considers -- NR-Cf and ER-Cf (detector 402,
both well served by adr/0012's fitted family already) and ER-Ba (detector
402, the one case with a real, non-monotonic feature -- leftover Ge-75m
neutron-activation contamination, documented in nrFanoII_paper2022's
scripts/extract_erecoil_tables_ba.py -- that family structurally cannot
represent at any bandwidth).

Reads the real, unbinned, already-cleaned Erecoil events from
adr/0013_recoil_events_402.npz (extracted once from nrFanoII_paper2022's
raw calibration data via its own extract_erecoil_tables.py/
extract_erecoil_tables_ba.py, exactly as erecoil_tables/*.h5 were) --
self-contained, no cross-repo checkout needed to reproduce the figure.

    python adr/0013_kde_vs_data.py
"""
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def reflected_density(xs, points, bandwidth_keV):
    """Silverman's reflection method: each real event contributes a direct
    Gaussian kernel and a mirror-image kernel across Erecoil=0, each
    normalized by the total event count -- the density this returns
    integrates to exactly len(points) over Erecoil >= 0, with no leakage
    past the boundary. bandwidth_keV is the literal kernel standard
    deviation, not a Scott/Silverman auto-scaled factor."""
    direct = np.exp(-0.5 * ((xs[:, None] - points[None, :]) / bandwidth_keV) ** 2)
    mirror = np.exp(-0.5 * ((xs[:, None] + points[None, :]) / bandwidth_keV) ** 2)
    kernel_norm = 1.0 / (bandwidth_keV * np.sqrt(2 * np.pi) * len(points))
    return kernel_norm * (direct + mirror).sum(axis=1) * len(points)


events = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "0013_recoil_events_402.npz"))

with h5py.File(os.path.join(REPO_ROOT, "erecoil_tables", "402_NR_Cf.h5")) as f:
    nr_raw_x, nr_raw_y = f["erecoil"][:], f["counts"][:]
with h5py.File(os.path.join(REPO_ROOT, "erecoil_tables", "402_ER_Cf.h5")) as f:
    ercf_raw_x, ercf_raw_y = f["erecoil"][:], f["counts"][:]
with h5py.File(os.path.join(REPO_ROOT, "erecoil_tables", "402_ER_Ba.h5")) as f:
    ba_raw_x, ba_raw_y = f["erecoil"][:], f["counts"][:]

# NR-Cf's own near-threshold rise decays on a ~6.4 keV scale (its own
# closed-form biexponential's short decay length) -- far too sharp for the
# bandwidth margin a safe degenerate-edge table needs (see this ADR's
# Context); shown here at the best bandwidth that still tracks it
# reasonably, specifically to make that limitation visible, not to propose
# using it. ER-Cf and ER-Ba's own native features are broad enough (tens
# to ~100 keV) that bandwidth=20 keV -- the value this record's dp/dq scan
# validates as a safe margin over this project's own smallest realistic
# p0/q0 -- costs them nothing.
nr_xs = np.linspace(0.001, 90, 1600)
ercf_xs = np.linspace(0.001, 320, 1600)
ba_xs = np.linspace(0.001, 400.0, 1600)

nr_kde = reflected_density(nr_xs, events["nr_cf"], 5.0)
ercf_kde = reflected_density(ercf_xs, events["er_cf"], 20.0)
ba_kde = reflected_density(ba_xs, events["er_ba"], 20.0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.plot(nr_raw_x, nr_raw_y, ".", ms=3, color="0.5", label="raw density")
ax.plot(nr_xs, nr_kde, "-", color="C0", lw=1.5, label="KDE, bw=5 keV")
ax.set_yscale("log"); ax.set_xlim(0, 80)
ax.set_title("NR-Cf (detector 402)\nfitted family already sufficient", fontsize=10)
ax.set_xlabel("Erecoil (keV)"); ax.set_ylabel("density"); ax.legend(fontsize=8)

ax = axes[1]
ax.plot(ercf_raw_x, ercf_raw_y, ".", ms=3, color="0.5", label="raw density")
ax.plot(ercf_xs, ercf_kde, "-", color="C0", lw=1.5, label="KDE, bw=20 keV")
ax.set_yscale("log"); ax.set_xlim(0, 260)
ax.set_title("ER-Cf (detector 402)", fontsize=10)
ax.set_xlabel("Erecoil (keV)"); ax.legend(fontsize=8)

ax = axes[2]
ax.plot(ba_raw_x, ba_raw_y, ".", ms=3, color="0.5", label="raw density")
ax.plot(ba_xs, ba_kde, "-", color="C0", lw=1.5, label="KDE, bw=20 keV")
ax.axvspan(70, 200, color="red", alpha=0.08, label="Ge-75m contamination")
ax.set_yscale("log"); ax.set_xlim(0, 300)
ax.set_title("ER-Ba (detector 402)\nonly a table/KDE can represent this", fontsize=10)
ax.set_xlabel("Erecoil (keV)"); ax.legend(fontsize=8)

fig.tight_layout()
out_path = os.path.join(REPO_ROOT, "test", "figures", "adr0013_kde_vs_data.png")
fig.savefig(out_path, dpi=150)
print(f"wrote {out_path}")
