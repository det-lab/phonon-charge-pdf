"""
Tests for erecoil_table.PErecoilTable wired into Ppq_py.py's integration path
(pq_given_Erecoil_NR/ER's p_erecoil argument, Ppq_inspect_NR/ER's own
p_erecoil, and integrate_inspect's erecoil_bounds): a table covering the full
range a point needs should reproduce the closed-form PErecoil_NR/PErecoil_ER
result closely, and a table too narrow for the point being evaluated should
raise normgrid.OutOfBoxError rather than silently return a truncated value.

Needs h5py. No Fortran needed.

Run from the repository root:  python test/python/test_erecoil_table_integration.py
"""

import os
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(REPO_ROOT, "python", "cli"))
import Ppq_py as ppq
from erecoil_table import PErecoilTable, save_erecoil_table
from normgrid import OutOfBoxError

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail and not cond else ''}")
    if not cond:
        failures.append(name)


def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def write_exponential_like_fixture(path, pdf_func, erecoil_max, *, band, n=3000):
    """See test_erecoil_table.py's identical helper for why log-spaced."""
    erecoil = np.concatenate(([0.0], np.geomspace(1.0e-3, erecoil_max, n - 1)))
    counts = pdf_func(erecoil)
    save_erecoil_table(path, erecoil, counts, detector="TEST", band=band, provenance="integration test fixture")


tmp_dir = tempfile.mkdtemp()

# Typical CDMS-like parameters, matching test_PpqFort.py's values.
PARAMS = dict(k=0.18, Z=32.0, F0=0.122, eps=3e-3, V=3.0,
              p0=0.06421907, p10=0.48998486, q0=0.06421907, q10=0.48998486)
Ep, Eq = 30.0, 10.0
ERECOIL_STEP = 0.05

# ---- NR: a table covering the band's real support reproduces the closed form ----

nr_full_path = os.path.join(tmp_dir, "erecoil_table_NR_full.h5")
write_exponential_like_fixture(nr_full_path, ppq.PErecoil_NR, erecoil_max=200.0, band="NR")
p_nr_full = PErecoilTable.from_hdf5(nr_full_path)

baseline = ppq.Ppq_inspect_NR(Ep, Eq, erecoil_step=ERECOIL_STEP, **PARAMS)["value"]
tabulated = ppq.Ppq_inspect_NR(Ep, Eq, erecoil_step=ERECOIL_STEP, p_erecoil=p_nr_full, **PARAMS)["value"]
check("Ppq_inspect_NR with a full-range table matches the closed form",
      np.isclose(tabulated, baseline, rtol=1e-2),
      f"baseline={baseline!r} tabulated={tabulated!r}")

# ---- NR: a table too narrow for this (Ep, Eq) raises, instead of truncating silently ----

nr_narrow_path = os.path.join(tmp_dir, "erecoil_table_NR_narrow.h5")
write_exponential_like_fixture(nr_narrow_path, ppq.PErecoil_NR, erecoil_max=2.0, band="NR")
p_nr_narrow = PErecoilTable.from_hdf5(nr_narrow_path)

check("Ppq_inspect_NR with a too-narrow table raises OutOfBoxError",
      raises(OutOfBoxError, lambda: ppq.Ppq_inspect_NR(
          Ep, Eq, erecoil_step=ERECOIL_STEP, p_erecoil=p_nr_narrow, **PARAMS)))

# ---- ER: same two checks ----

ER_PARAMS = dict(F0=0.122, eps=3e-3, V=3.0,
                  p0=0.06421907, p10=0.48998486, q0=0.06421907, q10=0.48998486)

er_full_path = os.path.join(tmp_dir, "erecoil_table_ER_full.h5")
write_exponential_like_fixture(er_full_path, ppq.PErecoil_ER, erecoil_max=2000.0, band="ER")
p_er_full = PErecoilTable.from_hdf5(er_full_path)

baseline_er = ppq.Ppq_inspect_ER(Ep, Eq, erecoil_step=ERECOIL_STEP, **ER_PARAMS)["value"]
tabulated_er = ppq.Ppq_inspect_ER(Ep, Eq, erecoil_step=ERECOIL_STEP, p_erecoil=p_er_full, **ER_PARAMS)["value"]
check("Ppq_inspect_ER with a full-range table matches the closed form",
      np.isclose(tabulated_er, baseline_er, rtol=1e-2),
      f"baseline={baseline_er!r} tabulated={tabulated_er!r}")

er_narrow_path = os.path.join(tmp_dir, "erecoil_table_ER_narrow.h5")
write_exponential_like_fixture(er_narrow_path, ppq.PErecoil_ER, erecoil_max=2.0, band="ER")
p_er_narrow = PErecoilTable.from_hdf5(er_narrow_path)

check("Ppq_inspect_ER with a too-narrow table raises OutOfBoxError",
      raises(OutOfBoxError, lambda: ppq.Ppq_inspect_ER(
          Ep, Eq, erecoil_step=ERECOIL_STEP, p_erecoil=p_er_narrow, **ER_PARAMS)))

if failures:
    print(f"\n{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("\nAll tests passed.")
