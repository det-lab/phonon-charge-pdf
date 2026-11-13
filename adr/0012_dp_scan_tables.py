"""
PCHIP and Poisson-matched-smoothing-spline legs of
adr/0012-perecoil-handles-register-a-fit-not-a-table.md's dp-scan
reproduction. Both need PErecoil_table_register, which was removed from the
quadrature path by this ADR's own implementation (82cd798) -- run this
against a checkout of the commit just before it (ab45bb3, or any ancestor
that still has PErecoil_table_register), e.g. via a throwaway worktree:

    git worktree add /tmp/pq-pdf-table-era ab45bb3
    cd /tmp/pq-pdf-table-era && fpm build   # or the usual Dockerfile_gfortran build
    LD_LIBRARY_PATH=lib python adr/0012_dp_scan_tables.py erecoil_tables/402_NR_Cf.h5 /tmp/phaseA.csv

See adr/0012_dp_scan_fit.py for the FITTED leg, which runs against current
HEAD instead, and adr/0012_dp_scan_plot.py to combine both legs' CSVs into
test/figures/adr0012_dp_scan.png.
"""
import csv
import os
import sys

import h5py
import numpy as np
from scipy.interpolate import splrep, PPoly

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python", "internal"))

from erecoil_table import PErecoilTable
from _Ppq_fort_bindings import GLConvergenceError, PErecoil_table_register, Ppq_region_NR_with_error

TABLE_PATH, OUT_CSV = sys.argv[1], sys.argv[2]

EP_RANGE, EQ_RANGE = (3.0, 80.0), (0.5, 30.0)
BASELINE = dict(k=0.18, Z=32.0, F0=0.122, eps=3.0e-3, V=3.0,
                q0=0.23718488, q10=0.27093151)
P0 = 0.06421907
EPSREL_LEVELS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
DP_VALUES = np.linspace(0.0, 0.6, 21)


RAW_N_PER_BIN = 72.5  # equal-count calibration bins; see adr/0012's Context


def smoothed_fortran_table(erecoil, counts):
    """A cubic smoothing spline through the same raw (erecoil, counts),
    weighted for the real Poisson counting error: counts here is a
    *density* from equal-count bins (RAW_N_PER_BIN raw events each, not
    itself a Poisson count -- see adr/0012's Context), so its absolute
    error is counts/sqrt(RAW_N_PER_BIN) and the correctly calibrated
    weight is sqrt(RAW_N_PER_BIN)/counts, with splrep's target residual
    s = n matching a standard chi2 ~ n_dof convention under that
    calibration. Normalized to integrate to 1 over [erecoil[0],
    erecoil[-1]], then handed to PErecoil_table_register in the exact
    power-basis convention PErecoilTable.fortran_table uses."""
    w = np.sqrt(RAW_N_PER_BIN) / counts
    tck = splrep(erecoil, counts, w=w, s=len(erecoil), k=3)
    pp = PPoly.from_spline(tck)
    lo, hi = erecoil[0], erecoil[-1]
    norm = pp.integrate(lo, hi)
    c = pp.c / norm
    return pp.x, c[0], c[1], c[2], c[3]


def converged_at(handle, params):
    for epsrel in EPSREL_LEVELS:
        try:
            value, err = Ppq_region_NR_with_error(*EP_RANGE, *EQ_RANGE, epsrel=epsrel, epsabs=1e-12,
                                                  table_handle=handle, **params)
            return epsrel, value, err
        except GLConvergenceError:
            continue
    return None, None, None


def run_scan(handle, label, rows):
    for dp in DP_VALUES:
        params = dict(BASELINE, p0=P0, p10=float(P0 + dp))
        epsrel, value, err = converged_at(handle, params)
        if value is None:
            print(f"[{label}] dp={dp:.4f}  DID NOT CONVERGE")
            continue
        rel = err / value
        rows.append((label, float(dp), epsrel, rel))
        print(f"[{label}] dp={dp:.4f}   converged@epsrel={epsrel:.0e}   rel={rel:.3e}")


table = PErecoilTable.from_hdf5(TABLE_PATH)
pchip_handle = PErecoil_table_register(*table.fortran_table)

with h5py.File(TABLE_PATH) as f:
    erecoil, counts = f["erecoil"][:], f["counts"][:]
smooth_handle = PErecoil_table_register(*smoothed_fortran_table(erecoil, counts))

rows = []
run_scan(pchip_handle, "PCHIP", rows)
run_scan(smooth_handle, "SMOOTH", rows)

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["label", "dp", "epsrel", "rel_residual"])
    w.writerows(rows)
print(f"wrote {OUT_CSV}")
