"""
Tests for python/erecoil_fit.py's fit_sum_of_exponentials: no Fortran needed,
pure Python/numpy/scipy. Specifically regression-tests the positivity bug
found while checking this function against Cf-252's own real ER-band data
(2026-11): plain curve_fit, checked only at the bin centers, could land on
two failure modes invisible there but visible on a dense grid between them
-- two large, nearly-canceling same-decay-length terms, and a decay length
so long it's a disguised constant canceling against a negative C. Both are
now structurally impossible (a/C carried as squared values, b as exp(log_b)
-- see fit_sum_of_exponentials' own docstring).

Run from the repository root:  python test/python/test_erecoil_fit.py
"""

import os
import sys
import warnings

import numpy as np

# A multi-restart fit deliberately explores parameter combinations that
# overflow/divide-by-zero in intermediate math -- rejected downstream by the
# dense positivity check (fit_sum_of_exponentials/fit_turnon_...'s own
# docstrings), not a sign anything here is wrong.
warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(REPO_ROOT, "python", "cli"))
from erecoil_fit import (fit_sum_of_exponentials, fit_turnon_times_sum_of_exponentials,
                         check_fit_against_events, PErecoilFit, _density)
from normgrid import OutOfBoxError

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail and not cond else ''}")
    if not cond:
        failures.append(name)


def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def synthetic_density(xs, a, b, C, rng, rel_noise=0.05):
    true = sum(ai * np.exp(-xs / bi) for ai, bi in zip(a, b)) + C
    noisy = true * (1 + rng.normal(0, rel_noise, size=xs.shape))
    return np.clip(noisy, 1e-9, None)


# A clean, well-separated two-decay-length case: the fit should recover
# parameters close to the truth, not just *a* low chi2.
rng = np.random.default_rng(1)
xs = np.linspace(0.5, 300, 150)
a_true, b_true, C_true = [50.0, 20.0], [15.0, 80.0], 0.5
density = synthetic_density(xs, a_true, b_true, C_true, rng)
raw_n = np.full_like(xs, 200.0)

fit = fit_sum_of_exponentials(xs, density, raw_n, n_terms=2, rng=np.random.default_rng(7), n_restarts=40)
check("recovers decay lengths close to the truth (within 15%)",
     all(abs(sorted(fit.b)[i] - sorted(b_true)[i]) / sorted(b_true)[i] < 0.15 for i in range(2)),
     detail=f"fit.b={fit.b}")
check("fit amplitudes are non-negative", np.all(fit.a >= 0), detail=f"fit.a={fit.a}")
check("fit floor is non-negative", fit.C >= 0, detail=f"fit.C={fit.C}")
check("chi2_dof is a reasonable value for noisy synthetic data", 0.2 < fit.chi2_dof < 3.0,
     detail=f"chi2_dof={fit.chi2_dof}")

# The actual regression test: density must stay positive on a dense grid,
# including well past the fit's own data range -- not just at the sparse
# bin centers the old (buggy) acceptance check looked at. This checks the
# raw family function directly (_density, bypassing __call__'s valid-region
# check below) -- it's a statement about the underlying closed form's
# unconditional smoothness (what the compiled Fortran quadrature needs,
# adr/0012), independent of where PErecoilFit.__call__ considers this fit
# *trustworthy* to evaluate (adr/0014).
xs_dense = np.linspace(0, 10 * xs.max(), 20000)
vals = _density(xs_dense, fit.a, fit.b, fit.C)
check("density is positive everywhere on a dense grid, including past erecoil_max",
     np.all(vals >= 0), detail=f"min={vals.min()}")

# A harder case, closer to what actually triggered the bug: two decay
# lengths close enough together that an unconstrained fit has an easy
# near-cancellation escape hatch available (a1 ~= -a2, b1 ~= b2).
rng2 = np.random.default_rng(3)
a_true2, b_true2, C_true2 = [30.0, 25.0], [60.0, 90.0], 0.1
density2 = synthetic_density(xs, a_true2, b_true2, C_true2, rng2)
fit2 = fit_sum_of_exponentials(xs, density2, raw_n, n_terms=2, rng=np.random.default_rng(11), n_restarts=40)
check("close-decay-length case: amplitudes still non-negative", np.all(fit2.a >= 0), detail=f"fit2.a={fit2.a}")
xs_dense2 = np.linspace(0, 10 * xs.max(), 20000)
vals2 = _density(xs_dense2, fit2.a, fit2.b, fit2.C)
check("close-decay-length case: stays positive densely", np.all(vals2 >= 0), detail=f"min={vals2.min()}")

# check_fit_against_events: a fit checked against events drawn from its
# own shape should look statistically consistent (large p-value), not
# flagged as a bad description of itself.
rng3 = np.random.default_rng(5)
n_events = 20000
# Inverse-CDF sample from the fit's own normalized density via rejection
# sampling (simple and exact enough for a self-consistency smoke test).
lo, hi = 0.0, fit.erecoil_max
xs_grid = np.linspace(lo, hi, 20000)
pdf_grid = fit(xs_grid)
cdf_grid = np.cumsum(pdf_grid)
cdf_grid /= cdf_grid[-1]
u = rng3.random(n_events)
events = np.interp(u, cdf_grid, xs_grid)
ks = check_fit_against_events(fit, events)
check("check_fit_against_events: self-consistent sample passes (p > 0.01)", ks.pvalue > 0.01,
     detail=f"p={ks.pvalue}")

# --- valid-region hard failure (adr/0014) ---

# Default valid region is [0, erecoil_max] -- unchanged historical behavior:
# evaluating within it works exactly as before.
try:
    fit(fit.erecoil_max * 0.5)
    check("default valid region: evaluating well inside [0, erecoil_max] doesn't raise", True)
except OutOfBoxError:
    check("default valid region: evaluating well inside [0, erecoil_max] doesn't raise", False)

try:
    fit(fit.erecoil_max + 1.0)
    check("default valid region: evaluating past erecoil_max raises OutOfBoxError", False)
except OutOfBoxError:
    check("default valid region: evaluating past erecoil_max raises OutOfBoxError", True)

check("negative Erecoil still returns 0.0, not a raise", float(fit(-5.0)) == 0.0)

# An explicit, narrower valid region (the Ba-436/445-style case: a fit
# trained on a truncated subset of its own data).
narrow_fit = PErecoilFit(fit.a * (fit.erecoil_max / 100.0), fit.b, fit.C * (fit.erecoil_max / 100.0),
                         erecoil_max=100.0, valid_min=10.0, valid_max=90.0)
try:
    narrow_fit(50.0)
    check("narrow valid region: evaluating inside [valid_min, valid_max] doesn't raise", True)
except OutOfBoxError:
    check("narrow valid region: evaluating inside [valid_min, valid_max] doesn't raise", False)

try:
    narrow_fit(95.0)
    check("narrow valid region: evaluating just outside valid_max raises", False)
except OutOfBoxError as exc:
    check("narrow valid region: evaluating just outside valid_max raises", True)
    check("OutOfBoxError carries axis/value/box/source", exc.axis == "Erecoil" and exc.box == (10.0, 90.0),
         detail=f"axis={exc.axis} box={exc.box}")

try:
    narrow_fit(5.0)
    check("narrow valid region: evaluating just outside valid_min raises", False)
except OutOfBoxError:
    check("narrow valid region: evaluating just outside valid_min raises", True)

check("valid_max must be > valid_min at construction",
     _raises(lambda: PErecoilFit([1.0], [10.0], 0.0, erecoil_max=100.0, valid_min=50.0, valid_max=50.0)))

# --- the turnon_times_sum_of_exponentials_plus_const family (adr/0014) ---

rng4 = np.random.default_rng(13)
x_on_true, w_on_true = 6.0, 3.0


def synthetic_turnon_density(xs, a, b, C, x_on, w_on, rng, rel_noise=0.05):
    from scipy.special import erf
    turnon = 0.5 * (1.0 + erf((xs - x_on) / w_on))
    true = turnon * (sum(ai * np.exp(-xs / bi) for ai, bi in zip(a, b)) + C)
    noisy = true * (1 + rng.normal(0, rel_noise, size=xs.shape))
    return np.clip(noisy, 1e-9, None)


xs_turnon = np.linspace(0.1, 250.0, 200)
a_on_true, b_on_true, C_on_true = [40.0, 15.0], [20.0, 90.0], 0.3
density_turnon = synthetic_turnon_density(xs_turnon, a_on_true, b_on_true, C_on_true,
                                          x_on_true, w_on_true, rng4)
raw_n_turnon = np.full_like(xs_turnon, 300.0)

# events drawn via rejection sampling from the true turnon shape, for the KS check below
rng5 = np.random.default_rng(17)
n_events_on = 20000
xs_grid_on = np.linspace(0.0, 250.0, 20000)
pdf_grid_on = synthetic_turnon_density(xs_grid_on, a_on_true, b_on_true, C_on_true,
                                       x_on_true, w_on_true, np.random.default_rng(0), rel_noise=0.0)
cdf_grid_on = np.cumsum(pdf_grid_on)
cdf_grid_on /= cdf_grid_on[-1]
u_on = rng5.random(n_events_on)
events_on = np.interp(u_on, cdf_grid_on, xs_grid_on)

fit_on = fit_turnon_times_sum_of_exponentials(xs_turnon, density_turnon, raw_n_turnon, events_on,
                                              n_restarts=40, n_seeds=3,
                                              rng=np.random.default_rng(23))
check("turnon fit recovers x_on close to the truth (within 50%)",
     abs(fit_on.x_on - x_on_true) / x_on_true < 0.5, detail=f"x_on={fit_on.x_on}")
check("turnon fit recovers w_on close to the truth (within 50%)",
     abs(fit_on.w_on - w_on_true) / w_on_true < 0.5, detail=f"w_on={fit_on.w_on}")
check("turnon fit amplitudes are non-negative", np.all(fit_on.a >= 0), detail=f"a={fit_on.a}")
check("turnon fit's default valid_min is 0.0 (modeled from threshold, not discarded)",
     fit_on.valid_min == 0.0)

ks_on = check_fit_against_events(fit_on, events_on)
check("turnon family: check_fit_against_events on a self-consistent sample passes (p > 0.01)",
     ks_on.pvalue > 0.01, detail=f"p={ks_on.pvalue}")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
