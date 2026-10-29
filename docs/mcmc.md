# Using MCMC to get errors

MCMC needs the same `fit_pdf` (a `Ppq_compiled` instance, see [Fitting](fitting.md)) as a fit
does, called many more times -- see [Fitting](fitting.md) for the basic construction/call
pattern and performance notes (vectorized calls, shuffling events). The one thing that
changes for MCMC specifically: at ~1 second per call, computing the normalization integral
at every step is out of the question, so an MCMC run always needs
[precomputed normalization tables](normalization-tables.md) (`Ppq_table_NR=`/
`Ppq_table_ER=`), not just the compiled library.

MCMC samples the full posterior (so you get errors, not just a point); a likelihood fit
with the same Gaussian prior (see [Fitting](fitting.md#the-likelihood-function-step-3))
instead finds the single point that maximizes it ("MAP", maximum a posteriori), using an
optimizer rather than emcee. The parameter box and prior below apply identically to both --
it's the same `BOUNDS`/table setup either way.

## Walkthrough

This uses [emcee](https://emcee.readthedocs.io/) (`pip install emcee` -- not a
`phonon_charge_pdf` dependency; any ensemble sampler works the same way). See
[Fitting](fitting.md#the-parameter-box-step-2) for what the parameter box, `BOUNDS`,
`prior_bounds`, `check_points()`, and `OutOfBoxError`/`PriorNotCoveredError` mean — the
rules are identical for MCMC, just enforced through `log_prior` below instead of inline in
the objective (as [Fitting](fitting.md) does for a point estimate).

```python
import numpy as np
import emcee
from scipy.stats import norm
import phonon_charge_pdf as pdf

# define the data region: must match the named table below exactly
ep_range = (2.5, 350.0)
eq_range = (0.75, 200.0)

BOUNDS = {"V": (2.7, 3.3), "p0": (0.0129, 0.1155), "p10": (0.3, 0.6),
          "q0": (0.0475, 0.4), "q10": (0.2, 0.4),
          "k": (0.13, 0.22), "F0_NR": (1e-5, 1.0)}

fit_pdf = pdf.Ppq_compiled(ep_range, eq_range, ep_data, eq_data,  # ep_data/eq_data: your measured events, keV
                      Ppq_table_NR="NR_ep2.5-350_eq0.75-200",
                      prior_bounds=BOUNDS)                  # 1. fails now, not hours into the run

Z, EPS = 32.0, 3.0e-3                       # fixed detector/material constants, not fit parameters
P0_MEAN, P0_SIGMA = 0.0642, 0.01            # Gaussian prior on p0/q0 -- replace with YOUR own
Q0_MEAN, Q0_SIGMA = 0.2372, 0.02            # prior, not these specific numbers
PARAM_NAMES = ("k", "F0_NR", "V", "p0", "p10", "q0", "q10")  # theta's one and only ordering

def unpack(theta):
    p = dict(zip(PARAM_NAMES, theta))
    p["Z"], p["eps"] = Z, EPS
    return p

def log_prior(theta):                             # 2. -inf outside the hard bounds and the ordering
    p = unpack(theta)
    inside_box = all(BOUNDS[name][0] <= p[name] <= BOUNDS[name][1] for name in PARAM_NAMES)
    if not inside_box or p["p10"] < p["p0"] or p["q10"] < p["q0"]:
        return -np.inf
    return (norm.logpdf(p["p0"], loc=P0_MEAN, scale=P0_SIGMA)             # soft terms,
            + norm.logpdf(p["q0"], loc=Q0_MEAN, scale=Q0_SIGMA))          # truncated by BOUNDS above

def log_likelihood(theta):
    p = unpack(theta)
    normalized = fit_pdf.Ppq_normalized_densities_NR(
        k=p["k"], Z=p["Z"], F0=p["F0_NR"], eps=p["eps"], V=p["V"],
        p0=p["p0"], p10=p["p10"], q0=p["q0"], q10=p["q10"])
    return np.sum(np.log(normalized))

def log_prob(theta):                              # 3. the prior first: fit_pdf never sees
    lp = log_prior(theta)                         #    a point outside it
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# 4. Starting walkers: a small ball around a best-fit point, inside the box
# (a real best-fit point -- see Fitting -- not literally theta0 below).
theta0 = np.array([0.18, 0.122, 3.0, 0.06421907, 0.48998486, 0.23718488, 0.27093151])
n_walkers, n_dim = 32, len(PARAM_NAMES)
rng = np.random.default_rng(0)
start = theta0 + 1e-4 * rng.standard_normal((n_walkers, n_dim))

fit_pdf.check_points({name: start[:, i] for i, name in enumerate(PARAM_NAMES)})

sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob)
sampler.run_mcmc(start, 5000, progress=True)

# burn-in/thinning below are illustrative -- tune both from
# sampler.get_autocorr_time() on your own run, not these numbers
chain = sampler.get_chain(discard=1000, thin=15, flat=True)
```

A Gaussian prior (on `p0` or `q0`, say) must be truncated by the hard bounds:
`prior_bounds` rejects a parameter without finite bounds, since no table can cover it.
Walkers started in a small ball around a best fit can still land outside near an edge,
which is what step 4's `check_points` catches.

For a joint fit across both bands, sum their log-likelihoods in `log_likelihood` the same
way [Fitting](fitting.md) does: `log_likelihood_NR(theta) + log_likelihood_ER(theta)`, each
from its own `fit_pdf.Ppq_normalized_densities_NR`/`_ER` call. A mixture model (one
dataset, per-event fractional band membership) needs more than this — that's specific to
your own analysis.
