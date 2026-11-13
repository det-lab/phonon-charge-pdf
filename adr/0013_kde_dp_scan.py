"""
Reproduces adr/0013's central measurement: the Gauss-Legendre doubling
residual for Ppq_region_NR/ER's region integral, scanned across both of
the phonon resolution model's degenerate edges (dp = p10 - p0 -> 0 and
dq = q10 - q0 -> 0), for a Silverman-reflected Gaussian KDE table
(bandwidth=20 keV) built from each of this record's three real detector
402 datasets -- registered via PErecoil_table_register, which this
commit (the one just before adr/0012's implementation) still has; HEAD no
longer does (see adr/0012_dp_scan_tables.py for the same convention).

    git worktree add /tmp/pq-pdf-table-era ab45bb3
    cd /tmp/pq-pdf-table-era && fpm build   # or the usual Dockerfile_gfortran build
    LD_LIBRARY_PATH=lib python adr/0013_kde_dp_scan.py erecoil_tables/ adr/0013_recoil_events_402.npz
"""
import sys
import os

import h5py
import numpy as np
from scipy.interpolate import PPoly, splrep

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python", "internal"))

from _Ppq_fort_bindings import (GLConvergenceError, PErecoil_table_register,
                                Ppq_region_NR_with_error, Ppq_region_ER_with_error)

ERECOIL_TABLES_DIR, EVENTS_NPZ = sys.argv[1], sys.argv[2]

BANDWIDTH_KEV = 20.0
P0, Q0 = 0.06421907, 0.23718488
EPSREL_LEVELS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
ER_BASELINE = dict(F0=0.2, eps=3.0e-3, V=3.0, q0=Q0, q10=0.27093151)
NR_BASELINE = dict(k=0.18, Z=32.0, F0=0.122, eps=3.0e-3, V=3.0, q0=Q0, q10=0.27093151)
# NR's own established analysis region (adr/0012); ER's own real calibration
# data reaches up to ~300-400 keV, so a correspondingly wider region.
NR_REGION = ((3.0, 80.0), (0.5, 30.0))
ER_REGION = ((3.0, 500.0), (0.75, 300.0))


def reflected_fortran_table(points, bandwidth_keV, hi):
    """Same reflection as adr/0013_kde_vs_data.py's reflected_density, but
    densely sampled and exactly re-interpolated (not re-smoothed -- the
    KDE's own bandwidth already set the smoothness scale) into the
    power-basis cubic form PErecoil_table_register needs."""
    xs = np.linspace(0.0, hi, 1600)
    direct = np.exp(-0.5 * ((xs[:, None] - points[None, :]) / bandwidth_keV) ** 2)
    mirror = np.exp(-0.5 * ((xs[:, None] + points[None, :]) / bandwidth_keV) ** 2)
    kernel_norm = 1.0 / (bandwidth_keV * np.sqrt(2 * np.pi) * len(points))
    density = kernel_norm * (direct + mirror).sum(axis=1) * len(points)
    tck = splrep(xs, density, s=0, k=3)
    pp = PPoly.from_spline(tck)
    norm = pp.integrate(pp.x[0], pp.x[-1])
    c = pp.c / norm
    return pp.x, c[0], c[1], c[2], c[3]


def converged_at(region_fn, region, handle, params):
    for epsrel in EPSREL_LEVELS:
        try:
            value, err = region_fn(*region[0], *region[1], epsrel=epsrel, epsabs=1e-12,
                                   table_handle=handle, **params)
            return epsrel, value, err
        except GLConvergenceError:
            continue
    return None, None, None


def scan(label, region_fn, region, handle, baseline):
    worst, best = 0.0, 1.0
    for dp in np.linspace(0.0, 0.6, 21):
        params = dict(baseline, p0=P0, p10=float(P0 + dp))
        epsrel, value, err = converged_at(region_fn, region, handle, params)
        if value is None:
            continue
        rel = err / value
        worst, best = max(worst, rel), min(best, rel)
    print(f"{label} dp-scan: worst={worst:.3e} best={best:.3e}")
    worst_q, best_q = 0.0, 1.0
    for dq in np.linspace(0.0, 0.3, 11):
        params = dict(baseline, p0=P0, p10=float(P0 + 0.3), q0=Q0, q10=float(Q0 + dq))
        epsrel, value, err = converged_at(region_fn, region, handle, params)
        if value is None:
            continue
        rel = err / value
        worst_q, best_q = max(worst_q, rel), min(best_q, rel)
    print(f"{label} dq-scan: worst={worst_q:.3e} best={best_q:.3e}")


events = np.load(EVENTS_NPZ)
with h5py.File(os.path.join(ERECOIL_TABLES_DIR, "402_NR_Cf.h5")) as f:
    nr_hi = float(f["erecoil"][-1]) * 1.1 + 10
with h5py.File(os.path.join(ERECOIL_TABLES_DIR, "402_ER_Cf.h5")) as f:
    ercf_hi = float(f["erecoil"][-1]) * 1.1 + 10

nr_handle = PErecoil_table_register(*reflected_fortran_table(events["nr_cf"], BANDWIDTH_KEV, 90.0))
ercf_handle = PErecoil_table_register(*reflected_fortran_table(events["er_cf"], BANDWIDTH_KEV, 320.0))
ba_handle = PErecoil_table_register(*reflected_fortran_table(events["er_ba"], BANDWIDTH_KEV, 400.0))

scan("NR-Cf  ", Ppq_region_NR_with_error, NR_REGION, nr_handle, NR_BASELINE)
scan("ER-Cf  ", Ppq_region_ER_with_error, ER_REGION, ercf_handle, ER_BASELINE)
scan("ER-Ba  ", Ppq_region_ER_with_error, ER_REGION, ba_handle, ER_BASELINE)
