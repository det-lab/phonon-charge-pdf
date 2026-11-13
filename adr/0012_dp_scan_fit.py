"""
Phase B of adr/0012-perecoil-handles-register-a-fit-not-a-table.md's dp-scan
reproduction: the FITTED representation, against this branch's current
PErecoil_fit_register (table registration no longer exists here -- see
adr0012_phaseA_tables.py, run separately against the commit right before
this ADR's implementation, for the PCHIP/SMOOTH representations it needs).

    LD_LIBRARY_PATH=lib python adr/0012_dp_scan_fit.py erecoil_tables/402_NR_Cf.h5 /out/phaseB.csv
"""
import csv
import os
import sys

import h5py

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(REPO_ROOT, "python", "internal"))

from erecoil_fit import PErecoilFit
from _Ppq_fort_bindings import GLConvergenceError, PErecoil_fit_register, Ppq_region_NR_with_error

import numpy as np

TABLE_PATH, OUT_CSV = sys.argv[1], sys.argv[2]

EP_RANGE, EQ_RANGE = (3.0, 80.0), (0.5, 30.0)
BASELINE = dict(k=0.18, Z=32.0, F0=0.122, eps=3.0e-3, V=3.0,
                q0=0.23718488, q10=0.27093151)
P0 = 0.06421907
EPSREL_LEVELS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
DP_VALUES = np.linspace(0.0, 0.6, 21)

# Detector 402's own 2-exponential-plus-constant fit to the real,
# threshold-cleaned NR calibration data (chi2/dof=0.884, 122 points above
# the 7.215 keV turnon boundary) -- see the ADR's own Context section.
FIT_402_A, FIT_402_B, FIT_402_C = [910.328613, 208.695469], [10.0900835, 26.6685288], 0.315653769

with h5py.File(TABLE_PATH) as f:
    erecoil_max = float(f["erecoil"][-1])

fit = PErecoilFit(FIT_402_A, FIT_402_B, FIT_402_C, erecoil_max=erecoil_max,
                  detector="402", band="NR", source=TABLE_PATH)
fit_handle = PErecoil_fit_register(*fit.fortran_params)


def converged_at(handle, params):
    for epsrel in EPSREL_LEVELS:
        try:
            value, err = Ppq_region_NR_with_error(*EP_RANGE, *EQ_RANGE, epsrel=epsrel, epsabs=1e-12,
                                                  table_handle=handle, **params)
            return epsrel, value, err
        except GLConvergenceError:
            continue
    return None, None, None


rows = []
for dp in DP_VALUES:
    params = dict(BASELINE, p0=P0, p10=float(P0 + dp))
    epsrel, value, err = converged_at(fit_handle, params)
    if value is None:
        print(f"[FITTED] dp={dp:.4f}  DID NOT CONVERGE")
        continue
    rel = err / value
    rows.append(("FITTED", float(dp), epsrel, rel))
    print(f"[FITTED] dp={dp:.4f}   converged@epsrel={epsrel:.0e}   rel={rel:.3e}")

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["label", "dp", "epsrel", "rel_residual"])
    w.writerows(rows)
print(f"wrote {OUT_CSV}")
