"""
Tests for the tabulated PErecoil's Fortran port: PErecoil_table_register
(python/internal/_Ppq_fort_bindings.py -> src/Ppq_fort_s.f90's
eval_cubic_table, via the handle-based registry -- see
adr/0003-per-instance-perecoil-tables.md), and Ppq_compiled's
construction-time erecoil_range check (python/internal/
_Ppq_fort_bindings.py's _load_erecoil_table, shared with that module's
Ppq_densities).

The table is an explicit table_handle argument of every real evaluation
call now, not ambient global state: PErecoil_NR/ER themselves stay
closed-form-only (see table_hi_bound's own comment in Ppq_fort_s.f90), so
this cross-checks the real compute path (Ppq_NR/ER, via
make_Ppq_densities_NR/ER) against Ppq_py.py's slow, independent
Ppq_inspect_NR/ER(p_erecoil=...) reference instead of probing PErecoil_NR
directly.

Needs the compiled Fortran library and h5py -- run where the library is
built (e.g. in the container):

    LD_LIBRARY_PATH=lib python test/python/test_erecoil_table_fortran.py
"""

import os
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(REPO_ROOT, "python", "internal"))
import Ppq_py as ppq
from erecoil_table import PErecoilTable, save_erecoil_table
from Ppq_fort import Ppq_compiled
from _Ppq_fort_bindings import PErecoil_table_register, make_Ppq_densities_NR, make_Ppq_densities_ER

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
    save_erecoil_table(path, erecoil, counts, detector="TEST", band=band, provenance="Fortran cross-check fixture")


def density_NR_at(ep, eq, *, table_handle, **params):
    return float(make_Ppq_densities_NR(**params, table_handle=table_handle, n_workers=1)(
        np.array([ep]), np.array([eq]))[0])


def density_ER_at(ep, eq, *, table_handle, **params):
    return float(make_Ppq_densities_ER(**params, table_handle=table_handle, n_workers=1)(
        np.array([ep]), np.array([eq]))[0])


tmp_dir = tempfile.mkdtemp()
PARAMS = dict(k=0.18, Z=32.0, F0=0.122, eps=3e-3, V=3.0,
              p0=0.06421907, p10=0.48998486, q0=0.06421907, q10=0.48998486)
ER_PARAMS = dict(F0=0.122, eps=3e-3, V=3.0,
                  p0=0.06421907, p10=0.48998486, q0=0.06421907, q10=0.48998486)
Ep, Eq = 30.0, 10.0
ERECOIL_STEP = 0.05

# ---- table_handle=0 (closed form) matches Ppq_py's own closed form ----

for e_p, e_q in [(30.0, 10.0), (100.0, 40.0)]:
    want = ppq.Ppq_inspect_NR(e_p, e_q, erecoil_step=ERECOIL_STEP, **PARAMS)["value"]
    got = density_NR_at(e_p, e_q, table_handle=0, **PARAMS)
    check(f"Ppq_NR (table_handle=0) matches Ppq_py at (Ep,Eq)=({e_p},{e_q})",
          np.isclose(got, want, rtol=1e-2), f"got={got!r} want={want!r}")

want_er = ppq.Ppq_inspect_ER(Ep, Eq, erecoil_step=ERECOIL_STEP, **ER_PARAMS)["value"]
got_er = density_ER_at(Ep, Eq, table_handle=0, **ER_PARAMS)
check("Ppq_ER (table_handle=0) matches Ppq_py",
      np.isclose(got_er, want_er, rtol=1e-2), f"got={got_er!r} want={want_er!r}")

# ---- registering a table and passing its handle reproduces Ppq_py's own
# Ppq_inspect_NR/ER(p_erecoil=<that table>) reference ----

nr_path = os.path.join(tmp_dir, "erecoil_table_NR.h5")
write_exponential_like_fixture(nr_path, ppq.PErecoil_NR, erecoil_max=200.0, band="NR")
table_nr = PErecoilTable.from_hdf5(nr_path)
handle_nr = PErecoil_table_register(*table_nr.fortran_table)
check("PErecoil_table_register returns a positive handle", handle_nr > 0)

want_tab = ppq.Ppq_inspect_NR(Ep, Eq, erecoil_step=ERECOIL_STEP, p_erecoil=table_nr, **PARAMS)["value"]
got_tab = density_NR_at(Ep, Eq, table_handle=handle_nr, **PARAMS)
check("Ppq_NR with a registered NR table matches Ppq_py's Ppq_inspect_NR(p_erecoil=...)",
      np.isclose(got_tab, want_tab, rtol=1e-2), f"got={got_tab!r} want={want_tab!r}")

# table_handle=0 at the same (Ep, Eq) is unaffected by the registration above
# (registration is append-only, never ambient -- see adr/0003)
got_closed_again = density_NR_at(Ep, Eq, table_handle=0, **PARAMS)
check("table_handle=0 still gives the closed form after a table is registered",
      np.isclose(got_closed_again, ppq.Ppq_inspect_NR(Ep, Eq, erecoil_step=ERECOIL_STEP, **PARAMS)["value"],
                 rtol=1e-2))

er_path = os.path.join(tmp_dir, "erecoil_table_ER.h5")
write_exponential_like_fixture(er_path, ppq.PErecoil_ER, erecoil_max=2000.0, band="ER")
table_er = PErecoilTable.from_hdf5(er_path)
handle_er = PErecoil_table_register(*table_er.fortran_table)
check("PErecoil_table_register (ER) returns a positive, distinct handle", handle_er > 0 and handle_er != handle_nr)

want_tab_er = ppq.Ppq_inspect_ER(Ep, Eq, erecoil_step=ERECOIL_STEP, p_erecoil=table_er, **ER_PARAMS)["value"]
got_tab_er = density_ER_at(Ep, Eq, table_handle=handle_er, **ER_PARAMS)
check("Ppq_ER with a registered ER table matches Ppq_py's Ppq_inspect_ER(p_erecoil=...)",
      np.isclose(got_tab_er, want_tab_er, rtol=1e-2), f"got={got_tab_er!r} want={want_tab_er!r}")

# ---- two different NR tables, evaluated interleaved, never interfere --
# the actual problem adr/0003 fixes (e.g. a joint Cf + Ba fit) ----

nr_path_b = os.path.join(tmp_dir, "erecoil_table_NR_b.h5")
write_exponential_like_fixture(nr_path_b, ppq.PErecoil_NR, erecoil_max=350.0, band="NR")
table_nr_b = PErecoilTable.from_hdf5(nr_path_b)
handle_nr_b = PErecoil_table_register(*table_nr_b.fortran_table)

want_a = ppq.Ppq_inspect_NR(Ep, Eq, erecoil_step=ERECOIL_STEP, p_erecoil=table_nr, **PARAMS)["value"]
want_b = ppq.Ppq_inspect_NR(Ep, Eq, erecoil_step=ERECOIL_STEP, p_erecoil=table_nr_b, **PARAMS)["value"]
results = [
    density_NR_at(Ep, Eq, table_handle=handle_nr, **PARAMS),
    density_NR_at(Ep, Eq, table_handle=handle_nr_b, **PARAMS),
    density_NR_at(Ep, Eq, table_handle=handle_nr, **PARAMS),
    density_NR_at(Ep, Eq, table_handle=handle_nr_b, **PARAMS),
]
check("interleaved calls with table A's handle all match table A's reference",
      np.isclose(results[0], want_a, rtol=1e-2) and np.isclose(results[2], want_a, rtol=1e-2))
check("interleaved calls with table B's handle all match table B's reference",
      np.isclose(results[1], want_b, rtol=1e-2) and np.isclose(results[3], want_b, rtol=1e-2))
check("table A's two interleaved evaluations are identical to each other",
      results[0] == results[2])
check("table B's two interleaved evaluations are identical to each other",
      results[1] == results[3])

# ---- Ppq_compiled's construction-time erecoil_range check (unchanged by
# the handle redesign -- _load_erecoil_table's validation logic is the
# same, it now also returns a handle alongside the table) ----

EP_RANGE, EQ_RANGE = (2.5, 350.0), (0.75, 200.0)
ep_data, eq_data = np.array([30.0]), np.array([10.0])

# A table narrower than max(ep_max, eq_max) + 5 must raise at construction,
# not be silently accepted (see _load_erecoil_table's docstring).
narrow_path = os.path.join(tmp_dir, "erecoil_table_NR_narrow.h5")
write_exponential_like_fixture(narrow_path, ppq.PErecoil_NR, erecoil_max=10.0, band="NR")
check("Ppq_compiled rejects a PErecoil table narrower than the region needs",
      raises(ValueError, lambda: Ppq_compiled(
          EP_RANGE, EQ_RANGE, ep_data, eq_data,
          norm_epsrel=1e-4, norm_epsabs=1e-10, PErecoil_table_NR=narrow_path)))

# A table that doesn't start at/below 0 must also raise.
positive_lo_path = os.path.join(tmp_dir, "erecoil_table_NR_positive_lo.h5")
erecoil = np.linspace(1.0, 400.0, 2000)
counts = ppq.PErecoil_NR(erecoil)
save_erecoil_table(positive_lo_path, erecoil, counts, detector="TEST", band="NR",
                    provenance="deliberately starts above 0, for the construction-time check")
check("Ppq_compiled rejects a PErecoil table that does not start at/below 0",
      raises(ValueError, lambda: Ppq_compiled(
          EP_RANGE, EQ_RANGE, ep_data, eq_data,
          norm_epsrel=1e-4, norm_epsabs=1e-10, PErecoil_table_NR=positive_lo_path)))

# A table whose own metadata says the wrong band must raise too.
wrong_band_path = os.path.join(tmp_dir, "erecoil_table_mislabeled.h5")
write_exponential_like_fixture(wrong_band_path, ppq.PErecoil_NR, erecoil_max=400.0, band="ER")
check("Ppq_compiled rejects a PErecoil table whose metadata says the wrong band",
      raises(ValueError, lambda: Ppq_compiled(
          EP_RANGE, EQ_RANGE, ep_data, eq_data,
          norm_epsrel=1e-4, norm_epsabs=1e-10, PErecoil_table_NR=wrong_band_path)))

# A table wide enough for this region is accepted, registers a real handle,
# and Ppq_compiled's own density call reads from it correctly.
wide_path = os.path.join(tmp_dir, "erecoil_table_NR_wide.h5")
write_exponential_like_fixture(wide_path, ppq.PErecoil_NR, erecoil_max=400.0, band="NR")
fit_pdf = Ppq_compiled(EP_RANGE, EQ_RANGE, ep_data, eq_data,
                        norm_epsrel=1e-4, norm_epsabs=1e-10, PErecoil_table_NR=wide_path)
check("Ppq_compiled accepts a PErecoil table wide enough for the region",
      fit_pdf.PErecoil_table_NR is not None and fit_pdf.nr_table_handle > 0)
want_fit = ppq.Ppq_inspect_NR(float(ep_data[0]), float(eq_data[0]), erecoil_step=ERECOIL_STEP,
                              p_erecoil=fit_pdf.PErecoil_table_NR, **PARAMS)["value"]
got_fit = float(fit_pdf._Ppq_densities_NR(**PARAMS)[0])
check("Ppq_compiled's own density call reads from the table it just registered",
      np.isclose(got_fit, want_fit, rtol=1e-2), f"got={got_fit!r} want={want_fit!r}")

if failures:
    print(f"\n{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("\nAll tests passed.")
