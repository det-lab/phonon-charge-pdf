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
every family here is smooth and defined for every Erecoil >= 0, nothing
for a Fortran search window to run off the edge of) -- so there is no
"too narrow"/"doesn't start at 0" construction-time check to test here
the way there used to be for a table. PErecoilFit's own valid_min/
valid_max (adr/0016) DOES cross into this registration path (two more
PErecoil_fit_register/PErecoil_turnon_fit_register arguments), with two
different behaviors depending on how a bad Erecoil was reached:
- A transient probe outside the valid region (Newton's own internal
  search, which can legitimately overshoot while looking for the true
  peak) returns a quiet NaN directly from PErecoil_NR_impl/ER_impl --
  the same protection PErecoilFit.__call__ already gives a Python caller.
- A genuinely *accepted* Erecoil value (a located mode, or the last-
  resort fallback's own range) landing outside the valid region is not
  transient -- it means the real physics this analysis needs lies where
  this fit was never checked against data. That's `error stop`
  (integrate_Ppq's check_valid_region), not a NaN: nothing retries its
  way out of a fit that doesn't cover what's needed, so this kills the
  whole process rather than let an MCMC or table-building run complete
  on a result nobody should trust. Checked below via a subprocess (an
  error stop can't be caught in-process the way a Python exception can,
  so testing it means checking that the child process actually died).

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
from _Ppq_fort_bindings import (PErecoil_fit_register, PErecoil_turnon_fit_register,
                                make_Ppq_densities_NR, make_Ppq_densities_ER)

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

# ---- the turnon_times_sum_of_exponentials_plus_const family (adr/0016):
# PErecoil_turnon_fit_register, and eval_fit's erf-windowed branch, cross-
# checked the same way as the plain family above against Ppq_py's own
# reference (which needs no changes -- it already takes any callable
# p_erecoil, PErecoilFit included) ----

fit_turnon = PErecoilFit([0.5, 0.5], [8.0, 25.0], 0.0, erecoil_max=400.0, detector="TEST", band="NR",
                         family="turnon_times_sum_of_exponentials_plus_const", x_on=6.0, w_on=3.0)
handle_turnon = PErecoil_turnon_fit_register(fit_turnon.a, fit_turnon.b, fit_turnon.C,
                                             fit_turnon.x_on, fit_turnon.w_on,
                                             fit_turnon.valid_min, fit_turnon.valid_max)
check("PErecoil_turnon_fit_register returns a positive, distinct handle",
      handle_turnon > 0 and handle_turnon not in (handle_nr, handle_er))

want_turnon = ppq.Ppq_inspect_NR(Ep, Eq, erecoil_step=ERECOIL_STEP, p_erecoil=fit_turnon, **PARAMS)["value"]
got_turnon = density_NR_at(Ep, Eq, table_handle=handle_turnon, **PARAMS)
check("Ppq_NR with a registered turnon fit matches Ppq_py's Ppq_inspect_NR(p_erecoil=...)",
      np.isclose(got_turnon, want_turnon, rtol=1e-2), f"got={got_turnon!r} want={want_turnon!r}")

# the plain family's own handle (registered earlier) is unaffected by a
# turnon registration landing in a later slot -- same append-only-registry
# guarantee adr/0003 already covers, now exercised across both families.
got_nr_again = density_NR_at(Ep, Eq, table_handle=handle_nr, **PARAMS)
check("the plain fit's handle still matches its own reference after a turnon fit is registered",
      np.isclose(got_nr_again, want_fit, rtol=1e-2), f"got={got_nr_again!r} want={want_fit!r}")

# a second turnon handle at a different (x_on, w_on), checked the same
# interleaved way the two plain NR fits were above -- confirms eval_fit
# reads x_on/w_on from the *correct* registry slot, not just that one
# registered turnon fit happens to work.
fit_turnon_b = PErecoilFit([0.3, 0.7], [15.0, 60.0], 0.0, erecoil_max=350.0, detector="TEST", band="NR",
                           family="turnon_times_sum_of_exponentials_plus_const", x_on=10.0, w_on=5.0)
handle_turnon_b = PErecoil_turnon_fit_register(fit_turnon_b.a, fit_turnon_b.b, fit_turnon_b.C,
                                               fit_turnon_b.x_on, fit_turnon_b.w_on,
                                               fit_turnon_b.valid_min, fit_turnon_b.valid_max)
want_turnon_b = ppq.Ppq_inspect_NR(Ep, Eq, erecoil_step=ERECOIL_STEP, p_erecoil=fit_turnon_b,
                                   **PARAMS)["value"]
results_turnon = [
    density_NR_at(Ep, Eq, table_handle=handle_turnon, **PARAMS),
    density_NR_at(Ep, Eq, table_handle=handle_turnon_b, **PARAMS),
    density_NR_at(Ep, Eq, table_handle=handle_turnon, **PARAMS),
]
check("interleaved turnon handles each match their own reference, not each other's",
      np.isclose(results_turnon[0], want_turnon, rtol=1e-2) and
      np.isclose(results_turnon[1], want_turnon_b, rtol=1e-2) and
      results_turnon[0] == results_turnon[2])

# ---- a direct Fortran caller gets the same out-of-range guard a Python
# caller already gets from PErecoilFit.__call__ (adr/0016): a fit
# registered with a narrow valid_max returns NaN when queried at an
# (Ep, Eq) whose own true Erecoil clearly falls outside it, and still
# matches its reference normally well inside it ----

fit_narrow = PErecoilFit([0.5, 0.5], [8.0, 25.0], 0.0, erecoil_max=400.0,
                         detector="TEST", band="NR", valid_min=0.0, valid_max=50.0)
handle_narrow = PErecoil_fit_register(*fit_narrow.fortran_params)

# Well inside [0, 50]: matches Ppq_py's reference exactly as any other fit would.
# (Ppq_py's own reference scans a window around the resolution-driven peak, not
# just the single nominal point, so the margin needs real room -- not just
# Ep_in/Eq_in themselves clear of valid_max, but that whole scan window too.)
Ep_in, Eq_in = 5.0, 2.0
want_in = ppq.Ppq_inspect_NR(Ep_in, Eq_in, erecoil_step=ERECOIL_STEP, p_erecoil=fit_narrow, **PARAMS)["value"]
got_in = density_NR_at(Ep_in, Eq_in, table_handle=handle_narrow, **PARAMS)
check("narrow valid_max: a query whose true Erecoil is well inside it matches the reference",
      np.isclose(got_in, want_in, rtol=1e-2), f"got={got_in!r} want={want_in!r}")

# Far outside [0, 50]: (Ep, Eq)=(100, 40) is the same point already used above
# to confirm the closed form matches Ppq_py -- its own resolution-driven true
# Erecoil peaks near 100, nowhere close to valid_max=50, so the mode-finder
# ACCEPTS a peak well outside this fit's valid region, not just a transient
# Newton overshoot. That's integrate_Ppq's check_valid_region (adr/0016): an
# error stop, not a NaN -- it can't be caught in-process the way a Python
# exception can, so testing it means running it in a subprocess and checking
# that the child actually died, not catching anything here.
import subprocess


def _run_in_subprocess(code, timeout=60):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)


_CRASH_SCRIPT = f"""
import sys
sys.path.insert(0, {os.path.join(REPO_ROOT, "python")!r})
sys.path.insert(0, {os.path.join(REPO_ROOT, "python", "internal")!r})
import numpy as np
from erecoil_fit import PErecoilFit
from _Ppq_fort_bindings import PErecoil_fit_register, make_Ppq_densities_NR

fit = PErecoilFit([0.5, 0.5], [8.0, 25.0], 0.0, erecoil_max=400.0,
                  detector="TEST", band="NR", valid_min=0.0, valid_max=50.0)
handle = PErecoil_fit_register(*fit.fortran_params)
pdf = make_Ppq_densities_NR(table_handle=handle, n_workers=1, **{PARAMS!r})
pdf(np.array([100.0]), np.array([40.0]))
print("SHOULD_NOT_REACH_HERE")
"""

proc_crash = _run_in_subprocess(_CRASH_SCRIPT)
check("narrow valid_max: a query whose true Erecoil falls outside it kills the process (error stop)",
      proc_crash.returncode != 0 and "SHOULD_NOT_REACH_HERE" not in proc_crash.stdout,
      f"returncode={proc_crash.returncode!r} stdout={proc_crash.stdout!r} stderr={proc_crash.stderr[-400:]!r}")
check("the error stop message says this fit doesn't cover what the analysis needs",
      "does not cover the Erecoil range" in (proc_crash.stdout + proc_crash.stderr),
      f"stderr={proc_crash.stderr[-400:]!r}")

# A query whose true Erecoil is well inside [0, 50] must NOT crash, in the
# same subprocess style -- confirms the guard is selective, not a blanket
# crash-on-any-call-with-this-handle bug.
_OK_SCRIPT = f"""
import sys
sys.path.insert(0, {os.path.join(REPO_ROOT, "python")!r})
sys.path.insert(0, {os.path.join(REPO_ROOT, "python", "internal")!r})
import numpy as np
from erecoil_fit import PErecoilFit
from _Ppq_fort_bindings import PErecoil_fit_register, make_Ppq_densities_NR

fit = PErecoilFit([0.5, 0.5], [8.0, 25.0], 0.0, erecoil_max=400.0,
                  detector="TEST", band="NR", valid_min=0.0, valid_max=50.0)
handle = PErecoil_fit_register(*fit.fortran_params)
pdf = make_Ppq_densities_NR(table_handle=handle, n_workers=1, **{PARAMS!r})
pdf(np.array([5.0]), np.array([2.0]))
print("REACHED_THE_END")
"""

proc_ok = _run_in_subprocess(_OK_SCRIPT)
check("narrow valid_max: a query whose true Erecoil is well inside it does not crash",
      proc_ok.returncode == 0 and "REACHED_THE_END" in proc_ok.stdout,
      f"returncode={proc_ok.returncode!r} stdout={proc_ok.stdout!r} stderr={proc_ok.stderr[-400:]!r}")

# The same a/b/C, registered with a generously wide valid_max instead, is
# unaffected at the same (Ep, Eq)=(100, 40) point -- confirms the NaN above
# is really about THIS handle's own narrower valid_max, not something wrong
# with (100, 40) or this fit's shape in general.
fit_wide = PErecoilFit([0.5, 0.5], [8.0, 25.0], 0.0, erecoil_max=2000.0,
                       detector="TEST", band="NR")
handle_wide = PErecoil_fit_register(*fit_wide.fortran_params)
want_unrestricted = ppq.Ppq_inspect_NR(100.0, 40.0, erecoil_step=ERECOIL_STEP, p_erecoil=fit_wide,
                                       **PARAMS)["value"]
got_unrestricted = density_NR_at(100.0, 40.0, table_handle=handle_wide, **PARAMS)
check("the same (Ep, Eq) point is fine on a handle with no valid_max restriction",
      np.isclose(got_unrestricted, want_unrestricted, rtol=1e-2),
      f"got={got_unrestricted!r} want={want_unrestricted!r}")

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
