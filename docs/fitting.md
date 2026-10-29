# Fitting

**If you want normalized PDF values or a likelihood** — a fit, an MCMC, anything beyond
generating or describing simulated events — `Ppq_compiled` is the one entrypoint. It
always needs the compiled Fortran library; see [Two ways to use this](two-ways.md) for
how to get it (pip install, a published container, or
[building from source](building-from-source.md)). See the
[API reference](api/index.md#phonon_charge_pdf.Ppq_compiled) for the full method list.

The script below (1) defines the data range, (2) builds the probability distribution for a
dataset, (3) defines the likelihood function, and finally (4) finds the optimal parameters.
[MCMC](mcmc.md) builds on the exact same setup (region, table, box, prior), just sampled
with emcee instead of optimized.

```python
import numpy as np
import scipy.optimize
from scipy.stats import norm
import phonon_charge_pdf as pdf

# 1. Define your region: your analysis ROI in keV -- the same (Ep, Eq) window
# your data is already cut to. 2.5-350 / 0.75-200 is this project's region;
# use YOUR actual cut, not this one, unless it happens to match.
ep_range = (2.5, 350.0)
eq_range = (0.75, 200.0)

# 2. Build fit_pdf: the probability distribution bound to your region and data,
# from a precomputed table (see "Computing new normalization tables") -- an
# optimizer loop or MCMC needs one; quadrature alone (norm_epsrel=/
# norm_epsabs=, no table) costs ~1 second/call, fine for a handful of
# evaluations but not a loop. A table needs prior_bounds too -- see "The
# parameter box" below.
BOUNDS = {"V": (2.7, 3.3), "p0": (0.0129, 0.1155), "p10": (0.3, 0.6),
          "q0": (0.0475, 0.4), "q10": (0.2, 0.4),
          "k": (0.13, 0.22), "F0_NR": (1e-5, 1.0)}

fit_pdf = pdf.Ppq_compiled(ep_range, eq_range, ep_data, eq_data,  # ep_data/eq_data: your measured events, keV
                      Ppq_table_NR="NR_ep2.5-350_eq0.75-200",
                      prior_bounds=BOUNDS)                  # fails now, not partway into the fit

# 3. The likelihood function: the log of fit_pdf's normalized densities, summed
# over all events, plus a Gaussian prior on p0/q0 (often needed for
# convergence -- see below). p10 >= p0/q10 >= q0 is enforced before fit_pdf ever
# evaluates a violating point.

Z, EPS = 32.0, 3.0e-3           # fixed detector/material constants, not fit parameters here

# Gaussian prior on p0/q0, e.g. centered on a previous best fit -- see
# "A note about terminology" for why this makes it a MAP fit, not a plain MLE
# fit. Replace with YOUR own prior, not these specific numbers.
P0_MEAN, P0_SIGMA = 0.0642, 0.01
Q0_MEAN, Q0_SIGMA = 0.2372, 0.02

# theta's one and only ordering, defined once -- everything below (unpack,
# theta0, bounds) reads from this instead of each repeating "k, F0, V, p0,
# p10, q0, q10" by hand and risking one getting out of sync with the rest.
PARAM_NAMES = ("k", "F0_NR", "V", "p0", "p10", "q0", "q10")

def unpack(theta):
    p = dict(zip(PARAM_NAMES, theta))
    p["Z"], p["eps"] = Z, EPS
    return p

def loglike(p):
    # Ppq_normalized_densities_NR's own parameter is just "F0" (it's already
    # NR-specific, nothing to disambiguate) -- p's F0_NR becomes that one
    # call's F0 right here, the only place the rename has to happen.
    normalized = fit_pdf.Ppq_normalized_densities_NR(
        k=p["k"], Z=p["Z"], F0=p["F0_NR"], eps=p["eps"], V=p["V"],
        p0=p["p0"], p10=p["p10"], q0=p["q0"], q10=p["q10"])
    return np.sum(np.log(normalized))

def neg_log_posterior(theta):
    p = unpack(theta)
    if p["p10"] < p["p0"] or p["q10"] < p["q0"]:
        return np.inf                              # never evaluates fit_pdf
    log_prior = (norm.logpdf(p["p0"], loc=P0_MEAN, scale=P0_SIGMA)
                 + norm.logpdf(p["q0"], loc=Q0_MEAN, scale=Q0_SIGMA))
    return -(log_prior + loglike(p))

# 4. Check the starting guess is inside the box, then optimize -- bounds
# from BOUNDS (or the dp/dq substitution below). Powell (derivative-free)
# tolerates the hard np.inf above; see "The likelihood function" below if
# you want a gradient-based method instead.
theta0 = [0.18, 0.122, 3.0, 0.06421907, 0.48998486, 0.23718488, 0.27093151]  # PARAM_NAMES order

fit_pdf.check_points(unpack(theta0))         # a clear error for a bad start, instead of
                                              # whatever exception the optimizer's first
                                              # call happens to surface

bounds = [BOUNDS[name] for name in PARAM_NAMES]
result = scipy.optimize.minimize(neg_log_posterior, theta0, method="Powell", bounds=bounds)
```

If you also want `Ppq_normalized_densities_ER` (no `k`/`Z` — `Y=1` there — also
`Ppq_integral_ER` if you want the normalization on its own, e.g. to check it's finite and
nonzero before trusting a fit) on the same `fit_pdf`, pass `Ppq_table_ER=` too and add
`F0_ER` to `BOUNDS` — same shape, just the other band's table/bound. Build `fit_pdf` once,
outside the fit loop; call its methods once per step, inside.

For a joint fit across both bands (independent NR/ER datasets, each with its own
likelihood), sum their log-likelihoods: `loglike_NR(p) + loglike_ER(p)`, each built the
same way `loglike` is above from its own `Ppq_normalized_densities_NR`/`_ER` call. A
mixture model (one dataset, per-event fractional band membership) needs more than this —
that's specific to your own analysis, not something this page can show generically.

## The parameter box (step 2)

The region is the box the normalization integral is computed over — it has to be your real
analysis ROI, not an arbitrary wide range, or the normalization (and so the likelihood) is
wrong.

!!! warning "A new ROI or a wider parameter range means rebuilding your tables"
    A table only covers the exact region and parameter box it was built for — this trips
    people up often enough to call out up front:

    - **New ROI:** if your region doesn't match the one a table was built for exactly,
      `Ppq_compiled` raises at construction rather than silently normalizing wrong.
    - **Parameter range:** a fit that keeps landing on a `BOUNDS` edge is being limited by
      the box, not the data (see "The likelihood function" below) — widen it with `BOX=`.

    Either way, see [Computing new normalization tables](normalization-tables.md) for
    building a new one.

A table covers a fixed box of parameters and never extrapolates, so every fit — with a
table or without — has to **keep every parameter inside the box, with `p10 >= p0` and
`q10 >= q0`**. Without a table, the Fortran side's own check for this is a fatal
`error stop` — it kills the whole process, not a catchable Python exception — so never let
the optimizer evaluate a violating point in the first place; with a table, the same
violation raises a catchable `OutOfBoxError` instead. [MCMC](mcmc.md) enforces the
identical rules, just through `log_prior` instead of inline in the objective.

`BOUNDS` is `{name: (lo, hi)}` in physical units, flat — the same convention
`generate_events()` already uses for its `F0_NR`/`F0_ER` keyword arguments: `F0` is the one
parameter that genuinely differs per band, so it's two separate keys, `F0_NR`/`F0_ER`
(`k` needs no suffix; it's unambiguously NR-only). `Z` and `eps` are fixed inside a table;
pass them as their fixed values or leave them out.

- `Ppq_compiled(..., prior_bounds=BOUNDS)` checks `BOUNDS` against every table given, at
  construction, and raises `PriorNotCoveredError` if it reaches outside one — listing each
  short axis and the `BOX=` rebuild command that would cover it.
- `check_points(points)` checks a set of points — a fit's starting guess, or MCMC's
  starting walkers — against the box the same way, before a slow run gets the chance to
  hit one outside it.
- At run time, a query outside the box raises `OutOfBoxError` naming the axis, the value,
  the box and the table file. Once the setup checks above pass, that means a bug or a NaN,
  not a parameter choice.
- `table_bounds()` reports the widest bounds every given table actually covers, in this
  same flat shape (`F0_NR`/`F0_ER` per band; shared parameters intersected if more than one
  table is given) — useful for sanity-checking `BOUNDS` you chose for other reasons, not
  for choosing `BOUNDS` in the first place: pick your parameter ranges from what's
  physically reasonable for your analysis, not by copying a table's box.

`check_points` is more clearly worth it for a **multi-start fit** (several candidate
starting points, to avoid a local minimum): it validates all of them in one call, before
any optimizer run begins.

```python
candidate_thetas = [                          # PARAM_NAMES order, as always
    [0.18, 0.122, 3.0, 0.02, 0.35, 0.10, 0.25],
    [0.18, 0.122, 3.0, 0.05, 0.40, 0.15, 0.30],
    [0.18, 0.122, 3.0, 0.08, 0.45, 0.20, 0.30],
]
# {param: array of 3}, built from candidate_thetas directly instead of
# retyping the same 3 values under each name a second time
starts = {name: np.array(vals) for name, vals in zip(PARAM_NAMES, zip(*candidate_thetas))}
fit_pdf.check_points(starts)                 # all 3 candidates checked in one call

results = [scipy.optimize.minimize(neg_log_posterior, theta0, method="Powell", bounds=bounds)
           for theta0 in candidate_thetas]
best = min(results, key=lambda r: r.fun)
```

## The likelihood function (step 3)

Box bounds alone cannot express `p10 >= p0`/`q10 >= q0`: `p10` and `q10` each have their own
independent range in `BOUNDS` above, so a corner like `p10=0.3, p0=0.1` is reachable even
though it violates the ordering. `neg_log_posterior` above checks the ordering directly with
a hard `np.inf`, which works for a derivative-free method (Powell) but gives a gradient-based
one (L-BFGS-B) nothing to climb away from once it lands there. Fit `dp = p10 - p0` and
`dq = q10 - q0` instead of `p10`/`q10` directly — bounding `dp`/`dq` at `>= 0` makes the
ordering structurally impossible to violate, so `neg_log_posterior`'s `np.inf` branch is no
longer needed:

```python
PARAM_NAMES_DP = ("k", "F0_NR", "V", "p0", "dp", "q0", "dq")

def unpack_dp(theta):
    p = dict(zip(PARAM_NAMES_DP, theta))
    p["p10"] = p["p0"] + p.pop("dp")
    p["q10"] = p["q0"] + p.pop("dq")
    p["Z"], p["eps"] = Z, EPS
    return p

def neg_log_posterior_dp(theta):
    p = unpack_dp(theta)
    log_prior = (norm.logpdf(p["p0"], loc=P0_MEAN, scale=P0_SIGMA)
                 + norm.logpdf(p["q0"], loc=Q0_MEAN, scale=Q0_SIGMA))
    return -(log_prior + loglike(p))

dp_lo, dp_hi = fit_pdf.Ppq_table_NR.box["dp"]     # 0.18-0.59 by default
dq_lo, dq_hi = fit_pdf.Ppq_table_NR.box["dq"]     # 0-0.353 by default
bounds_for_dp = dict(BOUNDS, dp=(dp_lo, dp_hi), dq=(dq_lo, dq_hi))
bounds_dp = [bounds_for_dp[name] for name in PARAM_NAMES_DP]

# dp0, dq0: p10 - p0 and q10 - q0 at theta0 above
theta0_dp = [0.18, 0.122, 3.0, 0.06421907, 0.42576579, 0.23718488, 0.03374663]

result = scipy.optimize.minimize(neg_log_posterior_dp, theta0_dp, bounds=bounds_dp)
```

A fit that ends on a `dp`/`dq`/`BOUNDS` edge is being limited by the box, not the data:
widen it with `BOX=` and rebuild the tables.

A plain log-likelihood objective (drop `log_prior` above, keep just `-loglike(p)`) frequently
fails to converge on this model, so this project's fits always include the Gaussian term.
It has to be consistent with `BOUNDS`: one centered near, or wide enough to put real weight
past, the table's edge is fighting the box instead of regularizing the fit.

## A note about terminology

By Bayes' rule, P(θ | data) ∝ P(data | θ) × P(θ) — posterior ∝ likelihood × prior.
Maximizing `log_prior + loglike(p)` together, as `neg_log_posterior` does, maximizes
that posterior instead of the likelihood alone: that's exactly what "maximum a
posteriori" (MAP) means. There's no mathematical difference between "a penalty term
added to help convergence" and "a prior".
