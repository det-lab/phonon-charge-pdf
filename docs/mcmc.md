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

See [Fitting](fitting.md#the-parameter-box-step-2) for what the parameter box, `BOUNDS`,
`prior_bounds`, `check_points()`, and `OutOfBoxError`/`PriorNotCoveredError` mean — the
rules are identical for MCMC, just enforced through `log_prior` below instead of inline in
the objective.

```python
BOUNDS = {"V": (2.7, 3.3), "p0": (0.0129, 0.1155), "p10": (0.3, 0.6),
          "q0": (0.0475, 0.4), "q10": (0.2, 0.4),
          "k": (0.13, 0.22), "F0_NR": (1e-5, 1.0), "F0_ER": (0.1, 0.35)}

# define the data region: must match the named tables below exactly
ep_min, ep_max = 2.5, 350.0
eq_min, eq_max = 0.75, 200.0

fit_pdf = pdf.Ppq_compiled(ep_min, ep_max, eq_min, eq_max, ep_data, eq_data,
                      Ppq_table_NR="NR_ep2.5-350_eq0.75-200", Ppq_table_ER="ER_ep2.5-350_eq0.75-200",
                      prior_bounds=BOUNDS)                  # 1. fails now, not hours into the run

def log_prior(theta):                             # 2. -inf outside the hard bounds and the ordering
    p = unpack(theta)
    if not within(p, BOUNDS) or p["p10"] < p["p0"] or p["q10"] < p["q0"]:
        return -np.inf
    return soft_terms(p)                          # e.g. Gaussians on p0, q0 -- truncated by BOUNDS

def log_prob(theta):                              # 3. the prior first: fit_pdf never sees
    lp = log_prior(theta)                         #    a point outside it
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, fit_pdf)

fit_pdf.check_points(as_params(start))               # 4. starting walkers inside, before sampling
                                                  #    ({param: array}, laid out like BOUNDS)
```

A Gaussian prior (on `p0` or `q0`, say) must be truncated by the hard bounds:
`prior_bounds` rejects a parameter without finite bounds, since no table can cover it.
Walkers started in a small ball around a best fit can still land outside near an edge,
which is what step 4 catches.
