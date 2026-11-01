"""
Tests for the tabulated PErecoil's Fortran port: PErecoil_set_table_NR/ER
and PErecoil_clear_table_NR/ER (python/internal/_Ppq_fort_bindings.py ->
src/Ppq_fort_s.f90's eval_cubic_table), and Ppq_compiled's construction-time
erecoil_range check (python/Ppq_fort.py's _load_erecoil_table).

Needs the compiled Fortran library -- run where it is built (e.g. in the
container):

    LD_LIBRARY_PATH=lib python test/python/test_erecoil_table_fortran.py
"""

import ctypes
import os
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(REPO_ROOT, "python", "internal"))
import Ppq_py as ppq
from erecoil_table import PErecoilTable
from Ppq_fort import Ppq_compiled
import _Ppq_fort_bindings as bindings

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


def write_exponential_like_fixture(path, pdf_func, erecoil_max, n=3000):
    """See test_erecoil_table.py's identical helper for why log-spaced."""
    erecoil = np.concatenate(([0.0], np.geomspace(1.0e-3, erecoil_max, n - 1)))
    counts = pdf_func(erecoil)
    with open(path, "w") as f:
        for e, c in zip(erecoil, counts):
            f.write(f"{float(e)!r} {float(c)!r}\n")


# Raw ctypes handles on PErecoil_NR/ER themselves, mirroring test_PpqFort.py's
# own style -- these are the two bind(c) entry points eval_cubic_table sits
# behind once a table is loaded.
api = bindings._load_library()
api.PErecoil_NR.argtypes = [ctypes.c_double]
api.PErecoil_NR.restype = ctypes.c_double
api.PErecoil_ER.argtypes = [ctypes.c_double]
api.PErecoil_ER.restype = ctypes.c_double

tmp_dir = tempfile.mkdtemp()

# ---- closed-form default, before any table is ever set ----

for e in (0.0, 1.0, 10.0, 100.0):
    check(f"PErecoil_NR (no table) matches Ppq_py at Erecoil={e}",
          np.isclose(api.PErecoil_NR(e), ppq.PErecoil_NR(e)))
    check(f"PErecoil_ER (no table) matches Ppq_py at Erecoil={e}",
          np.isclose(api.PErecoil_ER(e), ppq.PErecoil_ER(e)))

# ---- set a table: Fortran's eval_cubic_table should reproduce the exact
# same piecewise cubic erecoil_table.PErecoilTable evaluates in Python ----

nr_path = os.path.join(tmp_dir, "erecoil_table_NR.dat")
write_exponential_like_fixture(nr_path, ppq.PErecoil_NR, erecoil_max=200.0)
table_nr = PErecoilTable.from_file(nr_path)
bindings.PErecoil_set_table_NR(*table_nr.fortran_table)

for e in (0.0, 0.5, 3.0, 10.0, 50.0, 150.0, 200.0):
    want = table_nr(e)
    got = api.PErecoil_NR(e)
    check(f"PErecoil_NR (NR table set) matches PErecoilTable at Erecoil={e}",
          np.isclose(got, want, rtol=1e-9), f"got={got!r} want={want!r}")

check("PErecoil_NR (NR table set) still 0 for negative Erecoil", api.PErecoil_NR(-1.0) == 0.0)
# ER untouched by setting the NR table
check("PErecoil_ER unaffected by PErecoil_set_table_NR",
      np.isclose(api.PErecoil_ER(10.0), ppq.PErecoil_ER(10.0)))

# ---- clear it: back to the closed form ----

bindings.PErecoil_clear_table_NR()
for e in (0.0, 1.0, 10.0, 100.0):
    check(f"PErecoil_NR (cleared) matches Ppq_py again at Erecoil={e}",
          np.isclose(api.PErecoil_NR(e), ppq.PErecoil_NR(e)))

# ---- same round trip for ER ----

er_path = os.path.join(tmp_dir, "erecoil_table_ER.dat")
write_exponential_like_fixture(er_path, ppq.PErecoil_ER, erecoil_max=2000.0)
table_er = PErecoilTable.from_file(er_path)
bindings.PErecoil_set_table_ER(*table_er.fortran_table)

for e in (0.0, 0.1, 1.0, 10.0, 100.0, 1500.0, 2000.0):
    want = table_er(e)
    got = api.PErecoil_ER(e)
    check(f"PErecoil_ER (ER table set) matches PErecoilTable at Erecoil={e}",
          np.isclose(got, want, rtol=1e-9), f"got={got!r} want={want!r}")

bindings.PErecoil_clear_table_ER()
check("PErecoil_ER (cleared) matches Ppq_py again", np.isclose(api.PErecoil_ER(10.0), ppq.PErecoil_ER(10.0)))

# ---- Ppq_compiled's construction-time erecoil_range check ----

EP_RANGE, EQ_RANGE = (2.5, 350.0), (0.75, 200.0)
ep_data, eq_data = np.array([30.0]), np.array([10.0])

# A table narrower than max(ep_max, eq_max) + 5 must raise at construction,
# not be silently accepted (see _load_erecoil_table's docstring).
narrow_path = os.path.join(tmp_dir, "erecoil_table_NR_narrow.dat")
write_exponential_like_fixture(narrow_path, ppq.PErecoil_NR, erecoil_max=10.0)
check("Ppq_compiled rejects a PErecoil table narrower than the region needs",
      raises(ValueError, lambda: Ppq_compiled(
          EP_RANGE, EQ_RANGE, ep_data, eq_data,
          norm_epsrel=1e-4, norm_epsabs=1e-10, PErecoil_table_NR=narrow_path)))

# A table that doesn't start at/below 0 must also raise.
positive_lo_path = os.path.join(tmp_dir, "erecoil_table_NR_positive_lo.dat")
erecoil = np.linspace(1.0, 400.0, 2000)
counts = ppq.PErecoil_NR(erecoil)
with open(positive_lo_path, "w") as f:
    for e, c in zip(erecoil, counts):
        f.write(f"{float(e)!r} {float(c)!r}\n")
check("Ppq_compiled rejects a PErecoil table that does not start at/below 0",
      raises(ValueError, lambda: Ppq_compiled(
          EP_RANGE, EQ_RANGE, ep_data, eq_data,
          norm_epsrel=1e-4, norm_epsabs=1e-10, PErecoil_table_NR=positive_lo_path)))

# A table wide enough for this region is accepted, and the compiled
# library's PErecoil_NR now reads from it (set_table was actually called).
wide_path = os.path.join(tmp_dir, "erecoil_table_NR_wide.dat")
write_exponential_like_fixture(wide_path, ppq.PErecoil_NR, erecoil_max=400.0)
fit_pdf = Ppq_compiled(EP_RANGE, EQ_RANGE, ep_data, eq_data,
                        norm_epsrel=1e-4, norm_epsabs=1e-10, PErecoil_table_NR=wide_path)
check("Ppq_compiled accepts a PErecoil table wide enough for the region",
      fit_pdf.PErecoil_table_NR is not None)
check("Compiled PErecoil_NR reads from the table Ppq_compiled just loaded",
      np.isclose(api.PErecoil_NR(50.0), fit_pdf.PErecoil_table_NR(50.0), rtol=1e-9))

# Clean up the global Fortran state this test set, so it doesn't leak into
# whatever test runs after this one in the same process.
bindings.PErecoil_clear_table_NR()

if failures:
    print(f"\n{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("\nAll tests passed.")
