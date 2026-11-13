# Using MCMC to get errors

MCMC needs the same `fit_pdf` (a `Ppq_compiled` instance, see [Fitting](fitting.md)) as a fit
does, called many more times -- see [Fitting](fitting.md) for the basic construction/call
pattern. The one thing that changes for MCMC specifically: at ~1 second per call, computing
the normalization integral at every step is out of the question, so an MCMC run always needs
[precomputed normalization tables](normalization-tables.md) (`Ppq_table_NR=`/
`Ppq_table_ER=`), not just the compiled library -- a table turns that into tens of
microseconds per step instead.

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
from phonon_charge_pdf.erecoil_fit import CF252_BIEXPONENTIAL_NR, CF252_BIEXPONENTIAL_ER

# define the data region: must match the named table below exactly
ep_range = (2.5, 350.0)
eq_range = (0.75, 200.0)

BOUNDS = {"V": (2.7, 3.3), "p0": (0.0129, 0.1155), "p10": (0.3, 0.6),
          "q0": (0.0475, 0.4), "q10": (0.2, 0.4),
          "k": (0.13, 0.22), "F0_NR": (1e-5, 1.0)}

fit_pdf = pdf.Ppq_compiled(ep_range, eq_range, ep_data, eq_data,  # ep_data/eq_data: your measured events, keV
                      PErecoil_fit_NR=CF252_BIEXPONENTIAL_NR, PErecoil_fit_ER=CF252_BIEXPONENTIAL_ER,
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

## Running in parallel

Each emcee step evaluates `log_prob` once per walker, and those evaluations can run at the
same time on different cores: pass emcee a `pool`. emcee evaluates the walkers in two
halves (each half's proposals depend on the other half's positions), so at most
`n_walkers / 2` evaluations run at once. More workers than that sit idle.

How you hand the work to a pool matters more than anything else in the parallel setup.
**A pool sends the function and its argument to a worker for every evaluation**, by
pickling them. Anything the function carries goes along every time. So:

- **Never pass the data or `fit_pdf` along with `log_prob`.** Each of these ships the
  dataset (and for `fit_pdf`, both normalization tables, about 1 MB) with every walker
  evaluation: emcee's `args=(ep_data, eq_data)` or `args=(fit_pdf,)`,
  `functools.partial(log_prob, fit_pdf=fit_pdf)`, and a method of `fit_pdf`.
- **Instead, each worker builds its own `fit_pdf` once, when it starts**, and the pooled
  function takes only `theta`. Each evaluation then ships a handful of numbers.
- **Guard the script's setup with `if __name__ == "__main__":`.** Worker processes import
  your script to find `log_prob`. On macOS and Windows, and on Linux since Python 3.14,
  that re-runs everything at the script's top level in every worker, including reading
  the data and loading the tables.

### On one machine: a process pool

```python
# mcmc_parallel.py
import os
from multiprocessing import Pool

import numpy as np
import emcee
import phonon_charge_pdf as pdf
from phonon_charge_pdf.erecoil_fit import CF252_BIEXPONENTIAL_NR, CF252_BIEXPONENTIAL_ER

# Module level: cheap definitions only -- every worker imports this file.
# ep_range, eq_range, BOUNDS, Z, EPS, P0_*/Q0_*, PARAM_NAMES, unpack() and
# log_prior() exactly as in the walkthrough above.

_fit_pdf = None                     # this process's own copy, set by init_worker


def init_worker(ep_data, eq_data):
    """Runs once in each worker process, when the pool starts it."""
    global _fit_pdf
    _fit_pdf = pdf.Ppq_compiled(ep_range, eq_range, ep_data, eq_data,
                                PErecoil_fit_NR=CF252_BIEXPONENTIAL_NR, PErecoil_fit_ER=CF252_BIEXPONENTIAL_ER,
                                Ppq_table_NR="NR_ep2.5-350_eq0.75-200",
                                prior_bounds=BOUNDS,
                                n_workers=1)       # threads: OpenMP only, see below


def log_prob(theta):                # theta only: all a worker needs per evaluation
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    p = unpack(theta)
    normalized = _fit_pdf.Ppq_normalized_densities_NR(
        k=p["k"], Z=p["Z"], F0=p["F0_NR"], eps=p["eps"], V=p["V"],
        p0=p["p0"], p10=p["p10"], q0=p["q0"], q10=p["q10"])
    return lp + np.sum(np.log(normalized))


if __name__ == "__main__":
    ep_data, eq_data = ...          # your events, cut to the region (keV)
    n_procs, threads_per_proc = 8, 4            # n_procs x threads_per_proc = your cores

    # Each worker inherits this when the pool starts it, so set it first.
    os.environ["OMP_NUM_THREADS"] = str(threads_per_proc)

    init_worker(ep_data, eq_data)               # this process's copy, for setup checks
    n_walkers, n_dim = 2 * n_procs, len(PARAM_NAMES)
    start = theta0 + 1e-4 * np.random.default_rng(0).standard_normal((n_walkers, n_dim))
    _fit_pdf.check_points({name: start[:, i] for i, name in enumerate(PARAM_NAMES)})

    # initargs is sent once per worker, at startup -- not per evaluation.
    with Pool(n_procs, initializer=init_worker, initargs=(ep_data, eq_data)) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob, pool=pool)
        sampler.run_mcmc(start, 5000, progress=True)
```

The library parallelizes each evaluation over the events itself (OpenMP,
`OMP_NUM_THREADS`), so a process pool combines two kinds of parallelism. Keep
`processes × OMP_NUM_THREADS` equal to the number of cores, and keep `n_workers=1`:
Python threads on top of OpenMP only add overhead, and by default each one starts a full
set of OpenMP threads of its own. OpenMP and walker-parallel processes are about equally
efficient, but the number of steps a chain needs barely depends on the walker count, so
**fewer walkers with more threads each finishes sooner**. Use enough walkers to sample
the posterior well (a few per parameter), not as many as you have cores.

### Across nodes: MPI

On a cluster, [schwimmbad](https://schwimmbad.readthedocs.io/)'s `MPIPool` spreads the
walkers over MPI processes ("ranks") on several nodes. It needs `mpi4py` built against
the cluster's MPI (`MPICC=mpicc pip install --no-binary mpi4py mpi4py`). Same module-level
definitions as above, but every rank runs the whole script, so every rank builds its own
`fit_pdf` *before* the pool exists:

```python
import sys
from schwimmbad import MPIPool

if __name__ == "__main__":
    ep_data, eq_data = ...          # every rank reads the events
    init_worker(ep_data, eq_data)   # every rank: its own fit_pdf, once

    with MPIPool() as pool:
        if not pool.is_master():    # every rank but 0: evaluate log_prob until told to stop
            pool.wait()
            sys.exit(0)
        # rank 0 only, from here: it runs emcee and sends each worker a theta
        n_walkers = 2 * pool.size   # pool.size = worker ranks; at most n_walkers/2 are busy
        ...                         # start, check_points, EnsembleSampler(..., pool=pool), run_mcmc
```

Launch it with one rank per walker evaluation plus rank 0, each with its share of a
node's cores for OpenMP, e.g. under Slurm:

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4         # 32 ranks x 4 cores = 128 cores per node
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --mpi=pmi2 python mcmc_parallel_mpi.py
```
