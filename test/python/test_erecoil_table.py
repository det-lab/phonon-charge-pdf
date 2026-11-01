"""
Tests for python/erecoil_table.py: build a PErecoilTable from an HDF5
fixture table sampled from Ppq_py.PErecoil_NR's own closed-form
biexponential (so a correct PCHIP interpolant + normalization should
reproduce it closely), then check the edge behavior that formula can't
exercise: a query past the table's range raises normgrid.OutOfBoxError
(never extrapolates), while a slightly negative Erecoil -- the one case the
integration/peak-scan machinery is allowed to probe -- still returns 0,
matching PErecoil_NR/PErecoil_ER's own convention. Also checks the HDF5
round trip (save_erecoil_table -> from_hdf5) preserves its detector/band/
provenance metadata.

Needs h5py. No Fortran needed.

Run from the repository root:  python test/python/test_erecoil_table.py
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


def write_exponential_like_fixture(path, pdf_func, erecoil_max, *, detector, band, n=3000):
    """Write an HDF5 fixture sampling pdf_func (PErecoil_NR or PErecoil_ER)
    on [0, erecoil_max] -- a stand-in for a real tabulated spectrum, shaped
    like the current closed form so a correct table pipeline should
    reproduce it. Log-spaced (plus an explicit point at 0): PErecoil_ER's
    fast component decays on a ~0.17 keV scale, orders of magnitude finer
    than erecoil_max, so a linear grid badly under-resolves it near 0 --
    this is a property of the fixture's sampling, not of PErecoilTable
    itself."""
    erecoil = np.concatenate(([0.0], np.geomspace(1.0e-3, erecoil_max, n - 1)))
    counts = pdf_func(erecoil)
    save_erecoil_table(path, erecoil, counts, detector=detector, band=band,
                        provenance="synthetic fixture, sampled from Ppq_py's own formula")


tmp_dir = tempfile.mkdtemp()
nr_path = os.path.join(tmp_dir, "TEST_NR.h5")
er_path = os.path.join(tmp_dir, "TEST_ER.h5")
write_exponential_like_fixture(nr_path, ppq.PErecoil_NR, erecoil_max=200.0, detector="TEST", band="NR")
write_exponential_like_fixture(er_path, ppq.PErecoil_ER, erecoil_max=2000.0, detector="TEST", band="ER")

# ---- HDF5 round trip: metadata preserved ----

p_nr = PErecoilTable.from_hdf5(nr_path)
p_er = PErecoilTable.from_hdf5(er_path)

check("from_hdf5: detector round-trips", p_nr.detector == "TEST")
check("from_hdf5: band round-trips", p_nr.band == "NR")
check("from_hdf5: provenance round-trips", "synthetic fixture" in p_nr.provenance)
check("from_hdf5: source is the file path", p_nr.source == nr_path)

# ---- PErecoilTable reproduces PErecoil_NR/PErecoil_ER ----

lo, hi = p_nr.erecoil_range
check("erecoil_range: lo is 0", lo == 0.0, f"lo={lo!r}")
check("erecoil_range: hi matches fixture", np.isclose(hi, 200.0), f"hi={hi!r}")

# PErecoil_NR/ER are already normalized closed forms; the table, built from
# samples of that same closed form, should come out close to it pointwise
# once its own normalization is applied.
probe_points = [0.0, 0.5, 3.0, 10.0, 50.0, 150.0]
for e in probe_points:
    want = float(ppq.PErecoil_NR(e))
    got = p_nr(e)
    check(f"PErecoilTable(NR) matches PErecoil_NR at Erecoil={e}",
          np.isclose(got, want, rtol=5e-3), f"got={got!r} want={want!r}")

probe_points_er = [0.0, 0.1, 1.0, 10.0, 100.0, 1500.0]
for e in probe_points_er:
    want = float(ppq.PErecoil_ER(e))
    got = p_er(e)
    check(f"PErecoilTable(ER) matches PErecoil_ER at Erecoil={e}",
          np.isclose(got, want, rtol=5e-3), f"got={got!r} want={want!r}")

# ---- normalization ----

from scipy.integrate import quad
integral, _ = quad(p_nr, 0.0, hi, limit=200)
check("PErecoilTable(NR) integrates to ~1 over its own range", np.isclose(integral, 1.0, atol=1e-3),
      f"integral={integral!r}")

# ---- edge behavior: negative Erecoil is 0 (not an error), past the table raises ----

check("Erecoil < 0 returns 0.0, not an error", p_nr(-0.5) == 0.0)
check("Erecoil just past the high edge raises OutOfBoxError", raises(OutOfBoxError, lambda: p_nr(hi + 1.0)))
check("Erecoil far past the high edge raises OutOfBoxError", raises(OutOfBoxError, lambda: p_nr(1.0e6)))
check("Erecoil exactly at the edges does not raise",
      not raises(Exception, lambda: (p_nr(lo), p_nr(hi))))

# ---- construction-time validation ----

check("non-increasing erecoil is rejected", raises(ValueError, lambda: PErecoilTable([1.0, 1.0, 2.0], [1.0, 1.0, 1.0])))
check("negative counts are rejected", raises(ValueError, lambda: PErecoilTable([0.0, 1.0, 2.0], [1.0, -1.0, 1.0])))
check("too few points are rejected", raises(ValueError, lambda: PErecoilTable([0.0], [1.0])))
check("save_erecoil_table rejects an unknown band",
      raises(ValueError, lambda: save_erecoil_table(
          os.path.join(tmp_dir, "bad.h5"), [0.0, 1.0], [1.0, 1.0], detector="TEST", band="XX")))

if failures:
    print(f"\n{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("\nAll tests passed.")
