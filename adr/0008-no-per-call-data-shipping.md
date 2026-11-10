# 0008. Parallel workers build their own Ppq_compiled; it refuses to be pickled and flags unguarded scripts

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
  method ship the whole `Ppq_compiled`: the normalization tables' Chebyshev
  coefficients plus the data. That's per walker evaluation: serialized on rank
  0, sent, and unpickled on the worker, every time. Measured (Expanse,
  2,000 Cf events, NR table only, bytes pickled per task): θ only **279**;
  `args=(ep, eq)` 32,323 (16 bytes/event, so ~265 KB for a full 16.5k-event Cf
  set); `args=(fit_pdf,)` **623,621**, about 2,200× the θ-only pattern.
- **It's silent.** `Ppq_compiled` has no pickling hooks, so this just works,
  slowly. On a fast likelihood (small dataset), the transfer can exceed the
  evaluation itself.
- **The docs don't show a pool at all**, so users invent one.
- **Unguarded scripts rebuild everything in every worker.** Since Python 3.14,
  multiprocessing's default start method on Linux is `forkserver` (already
  `spawn` on macOS/Windows). Either way, each worker imports the user's script
  as a module named `__mp_main__` to find the pooled function, so a script
  without an `if __name__ == "__main__":` guard re-runs its whole top level in
  every worker: reading the data, loading and `chebfit`-ing the tables,
  sometimes the sampler setup. multiprocessing does this, not this library, so
  no library change can prevent it. Forcing the `fork` start method would hide
  it, but forking after Intel/LLVM OpenMP has started its threads can deadlock
  the child, which is why Python moved off `fork`.

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
4. **Detect the unguarded script and say so.** `Ppq_compiled.__init__` (and
   `Ppq_densities.__init__`) walk the call stack, and if any frame is
   module-level code (`f_code.co_name == "<module>"`) of a module whose
   `__name__` is `"__mp_main__"`, issue a `RuntimeWarning`: "Ppq_compiled is
   being built by your script's top level inside a multiprocessing worker, so
   every worker re-runs your whole setup. Put it under
   `if __name__ == "__main__":` and build Ppq_compiled in the pool's
   initializer. See docs/mcmc.md, 'Running in parallel'." This fires only for
   the mistake: construction inside an initializer is a function frame, and
   MPI ranks run the script as `__main__`. It's a warning, not an error,
   because results are still correct, just slow. The library never changes the
   start method.

## Consequences

- The expensive mistake becomes an immediate error with the fix in the message.
  The cost: users who *want* to pickle an instance (e.g. to save a fit object)
  can't. That's acceptable, because what should be saved is the parameters and
  the region/table names, not the evaluator.
- Item 3 matters only for `n_workers > 1`, which ADR 0005 makes opt-in. It's
  cheap to do and removes per-call allocation for gfortran users.
- Item 4 costs one stack walk per construction (not per call). It relies on
  CPython's `__mp_main__` naming, which has been stable since Python 3.4; if
  that ever changes, the warning stops firing, but nothing breaks.
- Verification: `pickle.dumps(fit_pdf)` raises with the message; the documented
  pool pattern runs under both `fork` and `forkserver` start methods; for
  `n_workers > 1`, results are identical before and after item 3; an unguarded
  script run with a `forkserver` pool emits item 4's warning once per worker,
  and the documented (guarded, initializer) pattern emits none.
