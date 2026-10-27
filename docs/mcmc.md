# Using MCMC to get errors

MCMC needs the same `Ppq_compiled` likelihood as a [fit](fitting.md) does, called many
more times -- see [Fitting](fitting.md) for the basic construction/call pattern and
performance notes (vectorized calls, shuffling events). The one thing that changes for
MCMC specifically: at ~1 second per call, computing the normalization integral at every
step is out of the question, so an MCMC run always needs
[precomputed normalization tables](normalization-tables.md) (`Ppq_table_NR=`/
`Ppq_table_ER=`), not just the compiled library.

## What a table enforces

A table covers a fixed box of parameters and never extrapolates, so the rule for MCMC is:
**keep every parameter inside the tables' box, with `p10 >= p0` and `q10 >= q0`** (the PDF
is undefined otherwise). The library checks this and says what to change; the MCMC code
has to enforce it.

- `Ppq_compiled(..., prior_bounds=BOUNDS)` checks hard bounds against every table given, at
  construction. If they reach outside a table it raises `PriorNotCoveredError`, listing
  each short axis and the `BOX=` rebuild command that would cover them.
- `likelihood.table_bounds()` gives, per band, the widest bounds each table covers: a
  ready-made set of hard bounds. Take bounds from it, or round inward: the box edges are
  not round numbers (`p0` starts at 0.0128438), and bounds rounded outward fail the check.
- `likelihood.check_points(points)` checks a set of points, e.g. an MCMC's starting walkers.
- At run time, a query outside a table raises `OutOfBoxError` naming the axis, the value,
  the box and the table file. After the setup checks pass, that means a bug or a NaN.

Bounds are physical parameters, `{name: (lo, hi)}`. Parameters the two bands share go at
the top level; each band's own (NR's `k`, and the Fano factor `F0`, which differs between
the bands) go under `"NR"` and `"ER"`. `Z` and `eps` are fixed inside a table; pass them as
their fixed values or leave them out.

## Walkthrough

```python
BOUNDS = {"V": (2.7, 3.3), "p0": (0.0129, 0.1155), "p10": (0.3, 0.6),
          "q0": (0.0475, 0.4), "q10": (0.2, 0.4),
          "NR": {"k": (0.13, 0.22), "F0": (1e-5, 1.0)},
          "ER": {"F0": (0.1, 0.35)}}

# define the data region: must match the named tables below exactly
ep_min, ep_max = 2.5, 350.0
eq_min, eq_max = 0.75, 200.0

likelihood = pdf.Ppq_compiled(ep_min, ep_max, eq_min, eq_max, ep_data, eq_data,
                      Ppq_table_NR="NR_ep2.5-350_eq0.75-200", Ppq_table_ER="ER_ep2.5-350_eq0.75-200",
                      prior_bounds=BOUNDS)                  # 1. fails now, not hours into the run

def log_prior(theta):                             # 2. -inf outside the hard bounds and the ordering
    p = unpack(theta)
    if not within(p, BOUNDS) or p["p10"] < p["p0"] or p["q10"] < p["q0"]:
        return -np.inf
    return soft_terms(p)                          # e.g. Gaussians on p0, q0 -- truncated by BOUNDS

def log_prob(theta):                              # 3. the prior first: the table never sees
    lp = log_prior(theta)                         #    a point outside it
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, likelihood)

likelihood.check_points(as_params(start))        # 4. starting walkers inside, before sampling
                                                  #    ({param: array}, laid out like BOUNDS)
```

A Gaussian prior (on `p0` or `q0`, say) must be truncated by the hard bounds:
`prior_bounds` rejects a parameter without finite bounds, since no table can cover it.
Walkers started in a small ball around a best fit can still land outside near an edge,
which is what step 4 catches.
