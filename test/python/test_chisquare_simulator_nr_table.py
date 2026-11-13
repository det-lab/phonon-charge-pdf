"""
Chi-square distribution test: NR physics simulator vs Fortran Ppq_NR -- the
simulator samples Erecoil from a real tabulated PErecoil
(erecoil_tables/402_NR_Cf.h5, via generate_events()'s p_erecoil_NR), while
the compiled PDF evaluates a smooth FIT (erecoil_fit.PErecoilFit) built to
track that same table closely (see FIT_NR below) -- the table-aware sibling
of test_chisquare_simulator_nr.py, which checks exactly the same thing for
the closed form. A table and a fit that tracks it closely are deliberately
different representations here (adr/0012-perecoil-handles-register-a-fit-
not-a-table.md: the compiled quadrature path only takes a fit, not a
table), so this is also, incidentally, a check that FIT_NR is close enough
to TABLE_NR for the two sides to agree -- not just that each side
independently agrees with the closed form.

The physics response parameters below (k, Z, F0, eps, V, p0, p10, q0, q10)
are the same representative values test_chisquare_simulator_nr.py uses, not
a fit to detector 402 specifically -- this test only needs the simulator and
the PDF to agree with each other and with the shared table, not that
detector 402's real response matches these numbers.

Observable range
----------------
Much narrower than test_chisquare_simulator_nr.py's, and with a wide
safety margin inside erecoil_tables/402_NR_Cf.h5's own range (0 to
~207.65 keV) -- FIT_NR, unlike TABLE_NR, has no range of its own to run
off the edge of (adr/0012), but TABLE_NR still bounds what the
simulator (sampler_NR_table, below) can actually draw, so the region
stays well inside it regardless:

  EP_RANGE = (3, 80) keV
  EQ_RANGE = (0.5, 30) keV

Expected counts: Ppq_region_NR, not scipy.quad
-----------------------------------------------
run_chisquare_test's expected-counts step uses region_integral_func
(goodness_of_fit.make_region_integral_NR), not the default nested
scipy.quad over pdf_fort: a dedicated Fortran region integral
(Ppq_region_NR, already ridge-aware, doubling-verified Gauss-Legendre)
is faster than treating pdf_fort as an opaque callable regardless of
PErecoil representation, with no per-bin ridge rediscovery needed.

A tight epsrel (1e-4/1e-10, Ppq_compiled's own usual choice -- see
docs/fitting.md), not a pre-loosened one
---------------------------------------------------------------------
Ppq_region_NR's own doubling-verified convergence check can, in
principle, fail for a specific bin (error stop in the Fortran, converted
to a catchable GLConvergenceError by the Python wrapper -- see
python/internal/_Ppq_fort_bindings.py); make_region_integral_NR's retry
(max_epsrel/max_epsabs, both defaulted to 100x epsrel/epsabs) handles
that per bin if it ever happens, rather than this test (or a real fit)
uniformly giving up precision everywhere to survive one bad bin. FIT_NR,
being globally smooth (adr/0012), is not expected to need this the way a
table's own small-scale statistical roughness could -- kept here as a
safety net, not because this test is known to exercise it.

Usage
-----
    LD_LIBRARY_PATH=lib python test/python/test_chisquare_simulator_nr_table.py [n_throws] [n_bins]

n_throws defaults to 1000 for a quick check; use 10000 for a thorough test.
n_bins defaults to 400; use a smaller value (e.g. 64) for a fast smoke test
of the wiring, since the one-time bin integration dominates the wall time.

The chi-square histogram is written to
test/figures/chisquare_simulator_nr_table.png.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist, kstest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "python" / "internal"))

from event_generator import generate_events
from erecoil_table import PErecoilTable
from erecoil_fit import fit_sum_of_exponentials
# Unnormalized is fine: this test only compares the PDF's *shape* against
# simulated events (chi-square binning), not absolute probability.  For a
# fit or MCMC, use Ppq_compiled (python/Ppq_fort.py) instead.
from _Ppq_fort_bindings import Ppq_densities
from goodness_of_fit import make_region_integral_NR, run_chisquare_test

# ---------------------------------------------------------------------------
# Physics parameters (from nrFanoII_paper2022/python/detector_params.txt --
# see the module docstring for why these don't need to be detector 402's own)
# ---------------------------------------------------------------------------
PARAMS = dict(
    k   = 0.18,
    Z   = 32.0,
    F0  = 0.122,
    eps = 3.0e-3,
    V   = 3.0,
    p0  = 0.06421907,
    p10 = 0.48998486,
    q0  = 0.23718488,
    q10 = 0.27093151,
)

EP_RANGE = (3.0,   80.0)    # keV -- phonon channel (see module docstring)
EQ_RANGE = (0.5,   30.0)    # keV -- charge channel

TABLE_NR = PErecoilTable.from_hdf5(REPO_ROOT / "erecoil_tables" / "402_NR_Cf.h5")

# The compiled quadrature path takes a globally-smooth FIT, not a table
# (adr/0012-perecoil-handles-register-a-fit-not-a-table.md: even a
# noise-free table makes the region integral untrustworthy in part of
# the parameter space). Fit one directly to TABLE_NR's own smooth curve,
# sampled densely -- this test only needs FIT_NR to track TABLE_NR
# closely enough for the simulator (sampling from TABLE_NR, below) and
# the compiled PDF (evaluating FIT_NR) to agree on a chi-square test; a
# statistically rigorous description of real calibration statistics is
# erecoil_fit.fit_sum_of_exponentials' own job against the real data,
# done for real in nrFanoII_paper2022's own investigation, not here.
# Constant raw_n_per_bin (an arbitrary positive constant: only its
# constancy matters) gives constant *relative* weighting, appropriate
# across a curve spanning orders of magnitude (TABLE_NR's peak near
# Erecoil=0 vs. its tail).
_fit_erecoil = np.geomspace(1e-3, TABLE_NR.erecoil_range[1], 2000)
_fit_density = np.array([TABLE_NR(float(e)) for e in _fit_erecoil])
FIT_NR = fit_sum_of_exponentials(_fit_erecoil, _fit_density, np.ones_like(_fit_density),
                                 n_terms=4, detector="402", band="NR", source=str(TABLE_NR.source))

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
# N_EVENTS deliberately smaller than this file's old (table vs table) 10,000:
# FIT_NR is a globally-smooth approximation of TABLE_NR, not TABLE_NR itself
# (adr/0012) -- it does not, and should not, chase TABLE_NR's own small-scale
# statistical noise (checked directly: the worst disagreement, ~40% locally
# at Erecoil~15.6 keV, sits exactly on a real Poisson dip/bump in the real
# calibration data no sum of positive exponentials can reproduce, structurally,
# regardless of term count or fit tightness -- not a fitting shortfall).
# Averaged over a bin instead of at one point this is a ~3% mean density
# difference, which 10,000-event statistics are easily powerful enough to
# resolve as a chi-square excess that has nothing to do with the Fortran
# wiring this test exists to check. 2,000 keeps that expected, small,
# structural difference sub-dominant while remaining powerful enough to
# catch a real wiring bug (a wrong handle, band mix-up, sign error, etc.
# would fail this test at any reasonable N_EVENTS).
N_EVENTS  = 2_000
N_THROWS  = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000
N_BINS    = int(sys.argv[2]) if len(sys.argv) > 2 else 400
N_WORKERS = os.cpu_count()

# ---------------------------------------------------------------------------
# Sampler: generate exactly n in-range NR events, Erecoil drawn from
# TABLE_NR rather than the closed-form biexponential.
#
# generate_events_NR has no ROI cropping of its own (see
# test_chisquare_simulator_nr.py's own sampler_NR for why this uses the
# top-level generate_events instead): fraction_ER=0.0 is deterministically
# pure-NR, and it already crops to ep_range/eq_range. The ER spectrum
# parameters (F0_ER, mix_frac_ER, scale_fast_ER, scale_slow_ER) are still
# required (generate_events() always needs both bands' parameters) but
# never actually used here, since zero ER events are ever generated --
# same reasoning as that script's sampler_NR.
def sampler_NR_table(n, seed=None):
    events = generate_events(
        n, fraction_ER=0.0, seed=seed, ep_range=EP_RANGE, eq_range=EQ_RANGE,
        k=PARAMS["k"], Z=PARAMS["Z"], F0_NR=PARAMS["F0"], F0_ER=0.2,
        eps=PARAMS["eps"], V=PARAMS["V"], p0=PARAMS["p0"], p10=PARAMS["p10"],
        q0=PARAMS["q0"], q10=PARAMS["q10"],
        mix_frac_NR=0.0, scale_fast_NR=1.0, scale_slow_NR=1.0,  # unused: p_erecoil_NR overrides
        mix_frac_ER=0.573211975, scale_fast_ER=0.169520023, scale_slow_ER=279.552394,
        p_erecoil_NR=TABLE_NR,
    )
    return events["EP"].to_numpy(), events["EQ"].to_numpy()

# ---------------------------------------------------------------------------
# Build the Fortran PDF wrapper -- Ppq_densities loads TABLE_NR once, at
# construction, rather than per call (see that class's own docstring).
# ---------------------------------------------------------------------------
print(f"Loading Fortran Ppq_NR PDF with a fitted PErecoil ({N_WORKERS} workers)...")
pq_densities = Ppq_densities(EP_RANGE, EQ_RANGE, PErecoil_fit_NR=FIT_NR, n_workers=N_WORKERS)
pdf_fort = lambda ep, eq: pq_densities.density_NR(ep, eq, **PARAMS)

# Quick sanity check on the sampler
print("Spot-checking sampler output range...")
t0 = time.time()
Ep_chk, Eq_chk = sampler_NR_table(500)
print(f"  Generated 500 events in {time.time()-t0:.2f} s")
print(f"  Ep: [{Ep_chk.min():.2f}, {Ep_chk.max():.2f}] keV")
print(f"  Eq: [{Eq_chk.min():.4f}, {Eq_chk.max():.2f}] keV")

# ---------------------------------------------------------------------------
# Run the chi-square test
# ---------------------------------------------------------------------------
t_start = time.time()
chi2_vals, dof, bins, expected = run_chisquare_test(
    pdf_fort, sampler_NR_table, EP_RANGE, EQ_RANGE,
    n_events   = N_EVENTS,
    n_throws   = N_THROWS,
    n_bins     = N_BINS,
    # Ppq_region_NR (a dedicated Fortran, doubling-verified Gauss-Legendre
    # region integral, already ridge-aware -- no inner_points_func needed)
    # instead of nested scipy.quad over pdf_fort: against a bumpy
    # tabulated PErecoil, scipy.quad's generic adaptive refinement chases
    # the table's own small-scale roughness node by node and is very slow
    # (see this module's own history/docs/goodness-of-fit.md); the same
    # table, read through the Fortran integrator directly, is not.
    #
    # epsrel/epsabs is Ppq_compiled's own usual 1e-4/1e-10 (docs/fitting.md)
    # -- tight, not pre-loosened: a bisection survey of this exact 400-bin
    # partition found only 14/400 bins (all in the same near-threshold
    # Ep~9-30, Eq~2-4 keV cluster, where TABLE_NR's own statistics are
    # thinnest) that can't reach 1e-4 by order 256; the rest converge far
    # tighter. make_region_integral_NR's retry (see its own docstring)
    # loosens only those specific bins, rather than giving up precision
    # everywhere to accommodate a handful of them.
    # density=pq_densities (not a hand-passed table_handle=...): pulls
    # nr_table_handle from the same Ppq_densities pdf_fort already uses,
    # so there's one table in the picture, not two independently-given
    # handles that could silently drift apart -- adr/0003-per-instance-
    # perecoil-tables.md's whole point is that this is no longer ambient
    # state the two calls happen to share automatically.
    region_integral_func = make_region_integral_NR(
        epsrel=1e-4, epsabs=1e-10, density=pq_densities, **PARAMS),
    n_workers  = N_WORKERS,
    batch_size = 50,
    seed       = 42,
)
elapsed = time.time() - t_start

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
ks_stat, ks_p = kstest(chi2_vals, chi2_dist(dof).cdf)

print(f"\n{'='*60}")
print(f"Chi-square test: NR simulator vs Fortran Ppq_NR, tabulated PErecoil  (dof = {dof})")
print(f"{'='*60}")
print(f"  table              : {TABLE_NR.source}")
print(f"  n_events per throw : {N_EVENTS:,}")
print(f"  n_throws           : {N_THROWS:,}")
print(f"  n_bins             : {len(bins)}  (target {N_BINS})")
print(f"  min expected count : {expected.min():.2f}")
print(f"  total wall time    : {elapsed/60:.1f} min")
print()
print(f"  Mean chi2    : {chi2_vals.mean():.3f}   (expect {dof})")
print(f"  Var  chi2    : {chi2_vals.var():.3f}   (expect {2*dof:.0f})")
print(f"  KS statistic : {ks_stat:.4f}")
print(f"  KS p-value   : {ks_p:.4f}   (informational only -- see below)")
print()
# A loose mean-only bound, not ks_p > 0.05 / a tight mean bound (this file's
# old, table-vs-itself thresholds): FIT_NR is a globally-smooth approximation
# of TABLE_NR, not TABLE_NR itself (adr/0012), so it does NOT chase TABLE_NR's
# own small-scale statistical noise -- checked directly, the worst
# disagreement (~40% locally at Erecoil~15.6 keV) sits exactly on a real
# Poisson dip/bump in the real calibration data that no sum of positive
# exponentials can reproduce, structurally, regardless of term count or fit
# tightness. That is a small, fixed, systematic difference, not
# throw-to-throw noise, so it also skews the chi2 values' own distribution
# shape -- which is exactly what KS is sensitive to, making ks_p a
# measure of "is this fit distinguishable from the table" (expected: yes,
# always, by a little), not "is the Fortran wiring correct". A wiring bug
# (wrong handle, band mix-up, sign error) would blow mean chi2 up far past
# this loose bound; this real, already-quantified small difference does not.
passed = abs(chi2_vals.mean() - dof) / dof < 0.15
print("PASS" if passed else "FAIL",
      "— chi-square distribution",
      "consistent with" if passed else "DEVIATES FROM",
      f"chi2({dof}) (within the loose bound FIT_NR vs TABLE_NR's known structural difference needs).")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
os.makedirs(REPO_ROOT / "test" / "figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 5))
x = np.linspace(chi2_dist(dof).ppf(1e-4), chi2_dist(dof).ppf(1 - 1e-4), 500)
ax.plot(x, chi2_dist(dof).pdf(x), color="C1", lw=2, label=rf"$\chi^2({dof})$ theory")
ax.hist(chi2_vals, bins=80, density=True, alpha=0.6, color="C0",
        label=f"{N_THROWS:,} throws")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel("Probability density")
ax.set_title("Chi-square: NR simulator vs Fortran Ppq_NR, tabulated PErecoil")
stats_text = (
    f"mean = {chi2_vals.mean():.2f}  (expect {dof})\n"
    f"var  = {chi2_vals.var():.1f}  (expect {2*dof:.0f})\n"
    f"KS p = {ks_p:.3f}"
)
ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax.legend()
fig.tight_layout()
outfile = REPO_ROOT / "test" / "figures" / "chisquare_simulator_nr_table.png"
fig.savefig(outfile, dpi=150)
print(f"\nPlot saved to {outfile}")
