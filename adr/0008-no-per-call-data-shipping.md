# 0008. Parallel workers build their own Ppq_compiled; it refuses to be pickled

Status: Proposed
Date: 2026-11-09

## Context

An MCMC spends nearly all its time in posterior evaluations, so anything paid
*per evaluation* outside the Fortran is multiplied by walkers × steps
(e.g. 62 × 5,000 ≈ 300,000). A scan of the fit/MCMC path while building
`MCMC_demo/` found the per-call Python work small (library load and ctypes
signatures cached once; table lookup is a ~µs Chebyshev contraction;
contiguous data isn't copied), **except** for how work gets to parallel workers:

- **Pools pickle the function with every task.** schwimmbad's `MPIPool.map` sends
  `(func, arg)` per walker evaluation, and `multiprocessing.Pool.map` pickles it
  per chunk. emcee wraps the user's log-probability, *including* its `args=`, in
  that function (`_FunctionWrapper`).
- **So anything reachable from the log-probability is shipped every call.**
  `args=(ep, eq)` ships the dataset. That was Joseph's original Expanse script
  (his "optimized" version replaced it with globals for this reason).
  `args=(fit_pdf,)`, `functools.partial(log_prob, fit_pdf=fit_pdf)` or a bound
  method ship the whole `Ppq_compiled`: both normalization tables' Chebyshev
  coefficients (~0.8 MB for the published pair) plus the data. That's per walker
  evaluation: serialized on rank 0, sent, and unpickled on the worker, every
  time.
- **It's silent.** `Ppq_compiled` has no pickling hooks, so this just works,
  slowly. On a fast likelihood (small dataset), the transfer can exceed the
  evaluation itself.
- **The docs don't show a pool at all**, so users invent one. And on Python
  3.14, multiprocessing's default start method on Linux is now `forkserver`, so a
  script without an `if __name__ == "__main__":` guard re-runs its top level
  (table loading included) in every worker.

A smaller, related cost: with `n_workers > 1`, `_make_threaded_pdf` re-sorts the
events by cost (`argsort`) and re-gathers every chunk on *every* call, although
the data are fixed for the lifetime of a `Ppq_compiled`.

## Decision

1. **Fail loudly, not slowly.** `Ppq_compiled.__reduce__` (and `Ppq_densities`'s)
   raises `TypeError`: "Ppq_compiled holds the dataset and tables; don't pass it
   to a pool. Build one per worker process (initializer / every MPI rank), and
   give the pool a module-level function of theta only. See docs/mcmc.md,
   'Running in parallel'."
2. **Document the one pattern that's cheap.** A `docs/mcmc.md` section "Running in
   parallel" shows it for both a `multiprocessing`/schwimmbad `MultiPool`
   (`initializer=`) and `MPIPool` (build before the pool, on every rank): each
   worker constructs its own `Ppq_compiled` once, and the pooled function takes
   only θ, so each task ships ~10 floats. Include the `__main__` guard and a
   warning against emcee's `args=` with any pool. `MCMC_demo/model.py`
   (`worker_init` / `worker_log_probability`) is the worked example.
3. **Do fixed-data work once.** When `n_workers > 1`, `Ppq_compiled` computes the
   cost ordering and chunk index arrays at construction (data are fixed), and
   reuses one `ThreadPoolExecutor` rather than creating one per call.

## Consequences

- The expensive mistake becomes an immediate error with the fix in the message.
  The cost: users who *want* to pickle an instance (e.g. to save a fit object)
  can't. That's acceptable, because what should be saved is the parameters and
  the region/table names, not the evaluator.
- Item 3 matters only for `n_workers > 1`, which ADR 0005 makes opt-in. It's
  cheap to do and removes per-call allocation for gfortran users.
- Verification: `pickle.dumps(fit_pdf)` raises with the message; the documented
  pool pattern runs under both `fork` and `forkserver` start methods; for
  `n_workers > 1`, results are identical before and after item 3.
