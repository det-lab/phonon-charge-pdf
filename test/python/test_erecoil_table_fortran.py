"""
Tests for the fitted PErecoil's Fortran port: PErecoil_fit_register
(python/internal/_Ppq_fort_bindings.py -> src/Ppq_fort_s.f90's eval_fit,
via the handle-based registry -- see adr/0003-per-instance-perecoil-
tables.md, adr/0012-perecoil-handles-register-a-fit-not-a-table.md), and
Ppq_compiled's construction-time band-metadata check (python/internal/
_Ppq_fort_bindings.py's _load_erecoil_fit, shared with that module's
Ppq_densities).

The fit is an explicit table_handle argument of every real evaluation
call, not ambient global state: PErecoil_NR/ER themselves stay
closed-form-only, so this cross-checks the real compute path (Ppq_NR/ER, via
make_Ppq_densities_NR/ER) against Ppq_py.py's slow, independent
Ppq_inspect_NR/ER(p_erecoil=...) reference instead of probing PErecoil_NR
directly.

A fit has no erecoil_range to validate (unlike the table this used to be:
a sum of exponentials plus a constant is smooth and defined for every
Erecoil >= 0, nothing for a Fortran search window to run off the edge of)
-- so there is no "too narrow"/"doesn't start at 0" construction-time
check to test here the way there used to be for a table.

Needs the compiled Fortran library -- run where the library is built
(e.g. in the container):

    LD_LIBRARY_PATH=lib python test/python/test_erecoil_table_fortran.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(REPO_ROOT, "python", "internal"))
import Ppq_py as ppq
from erecoil_fit import PErecoilFit, CF252_BIEXPONENTIAL_ER
from Ppq_fort import Ppq_compiled
from _Ppq_fort_bindings import PErecoil_fit_register, make_Ppq_densities_NR, make_Ppq_densities_ER

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


def density_NR_at(ep, eq, *, table_handle, **params):
    return float(make_Ppq_densities_NR(**params, table_handle=table_handle, n_workers=1)(
        np.array([ep]), np.array([eq]))[0])


def density_ER_at(ep, eq, *, table_handle, **params):
    return float(make_Ppq_densities_ER(**params, table_handle=table_handle, n_workers=1)(
        np.array([ep]), np.array([eq]))[0])


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

# ---- registering a fit and passing its handle reproduces Ppq_py's own
# Ppq_inspect_NR/ER(p_erecoil=<that fit>) reference ----
# Deliberately different decay constants from the closed form's own
# (PNb=6.42, PNd=23.72 / PGb=0.17, PGd=279.55), so this genuinely
# exercises a distinct registered fit, not a coincidentally-identical one.

fit_nr = PErecoilFit([0.5, 0.5], [8.0, 25.0], 0.0, erecoil_max=400.0, detector="TEST", band="NR")
handle_nr = PErecoil_fit_register(*fit_nr.fortran_params)
check("PErecoil_fit_register returns a positive handle", handle_nr > 0)

want_fit = ppq.Ppq_inspect_NR(Ep, Eq, erecoil_step=ERECOIL_STEP, p_erecoil=fit_nr, **PARAMS)["value"]
got_fit = density_NR_at(Ep, Eq, table_handle=handle_nr, **PARAMS)
check("Ppq_NR with a registered NR fit matches Ppq_py's Ppq_inspect_NR(p_erecoil=...)",
      np.isclose(got_fit, want_fit, rtol=1e-2), f"got={got_fit!r} want={want_fit!r}")

# table_handle=0 at the same (Ep, Eq) is unaffected by the registration above
# (registration is append-only, never ambient -- see adr/0003)
got_closed_again = density_NR_at(Ep, Eq, table_handle=0, **PARAMS)
check("table_handle=0 still gives the closed form after a fit is registered",
      np.isclose(got_closed_again, ppq.Ppq_inspect_NR(Ep, Eq, erecoil_step=ERECOIL_STEP, **PARAMS)["value"],
                 rtol=1e-2))

fit_er = PErecoilFit([0.6, 0.4], [0.3, 400.0], 0.0, erecoil_max=2000.0, detector="TEST", band="ER")
handle_er = PErecoil_fit_register(*fit_er.fortran_params)
check("PErecoil_fit_register (ER) returns a positive, distinct handle", handle_er > 0 and handle_er != handle_nr)

want_fit_er = ppq.Ppq_inspect_ER(Ep, Eq, erecoil_step=ERECOIL_STEP, p_erecoil=fit_er, **ER_PARAMS)["value"]
got_fit_er = density_ER_at(Ep, Eq, table_handle=handle_er, **ER_PARAMS)
check("Ppq_ER with a registered ER fit matches Ppq_py's Ppq_inspect_ER(p_erecoil=...)",
      np.isclose(got_fit_er, want_fit_er, rtol=1e-2), f"got={got_fit_er!r} want={want_fit_er!r}")

# ---- two different NR fits, evaluated interleaved, never interfere --
# the actual problem adr/0003 fixes (e.g. a joint Cf + Ba fit) ----

fit_nr_b = PErecoilFit([0.3, 0.7], [15.0, 60.0], 0.0, erecoil_max=350.0, detector="TEST", band="NR")
handle_nr_b = PErecoil_fit_register(*fit_nr_b.fortran_params)

want_a = ppq.Ppq_inspect_NR(Ep, Eq, erecoil_step=ERECOIL_STEP, p_erecoil=fit_nr, **PARAMS)["value"]
want_b = ppq.Ppq_inspect_NR(Ep, Eq, erecoil_step=ERECOIL_STEP, p_erecoil=fit_nr_b, **PARAMS)["value"]
results = [
    density_NR_at(Ep, Eq, table_handle=handle_nr, **PARAMS),
    density_NR_at(Ep, Eq, table_handle=handle_nr_b, **PARAMS),
    density_NR_at(Ep, Eq, table_handle=handle_nr, **PARAMS),
    density_NR_at(Ep, Eq, table_handle=handle_nr_b, **PARAMS),
]
check("interleaved calls with fit A's handle all match fit A's reference",
      np.isclose(results[0], want_a, rtol=1e-2) and np.isclose(results[2], want_a, rtol=1e-2))
check("interleaved calls with fit B's handle all match fit B's reference",
      np.isclose(results[1], want_b, rtol=1e-2) and np.isclose(results[3], want_b, rtol=1e-2))
check("fit A's two interleaved evaluations are identical to each other",
      results[0] == results[2])
check("fit B's two interleaved evaluations are identical to each other",
      results[1] == results[3])

# ---- Ppq_compiled's construction-time band-metadata check (unchanged by
# the table -> fit redesign -- _load_erecoil_fit's validation logic is the
# same, it now registers a fit instead of a table) ----

EP_RANGE, EQ_RANGE = (2.5, 350.0), (0.75, 200.0)
ep_data, eq_data = np.array([30.0]), np.array([10.0])

# A fit whose own metadata says the wrong band must raise.
wrong_band_fit = PErecoilFit([1.0], [20.0], 0.0, erecoil_max=400.0, detector="TEST", band="ER")
check("Ppq_compiled rejects a PErecoil fit whose metadata says the wrong band",
      raises(ValueError, lambda: Ppq_compiled(
          EP_RANGE, EQ_RANGE, ep_data, eq_data,
          norm_epsrel=1e-4, norm_epsabs=1e-10,
          PErecoil_fit_NR=wrong_band_fit, PErecoil_fit_ER=CF252_BIEXPONENTIAL_ER)))

# A correctly-labeled fit is accepted, registers a real handle, and
# Ppq_compiled's own density call reads from it correctly.
good_fit = PErecoilFit([0.5, 0.5], [8.0, 25.0], 0.0, erecoil_max=400.0, detector="TEST", band="NR")
fit_pdf = Ppq_compiled(EP_RANGE, EQ_RANGE, ep_data, eq_data,
                       norm_epsrel=1e-4, norm_epsabs=1e-10,
                       PErecoil_fit_NR=good_fit, PErecoil_fit_ER=CF252_BIEXPONENTIAL_ER)
check("Ppq_compiled accepts a PErecoil fit",
      fit_pdf.PErecoil_fit_NR is not None and fit_pdf.nr_table_handle > 0)
want_compiled = ppq.Ppq_inspect_NR(float(ep_data[0]), float(eq_data[0]), erecoil_step=ERECOIL_STEP,
                                   p_erecoil=fit_pdf.PErecoil_fit_NR, **PARAMS)["value"]
got_compiled = float(fit_pdf._Ppq_densities_NR(**PARAMS)[0])
check("Ppq_compiled's own density call reads from the fit it just registered",
      np.isclose(got_compiled, want_compiled, rtol=1e-2), f"got={got_compiled!r} want={want_compiled!r}")

if failures:
    print(f"\n{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("\nAll tests passed.")
