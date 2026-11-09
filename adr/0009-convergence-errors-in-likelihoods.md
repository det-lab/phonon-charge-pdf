# 0009. What a fit or MCMC does when the region integral can't converge

Status: Proposed
Date: 2026-11-09

## Context

`feature/erecoil-table-pdf` replaces the Fortran `error stop` in the region
quadrature (`Ppq_region_NR/ER`) with a catchable `GLConvergenceError`. That's
good: an `error stop` killed the whole process, including an MPI job's other
ranks, with no chance to recover. But it raises a question the docs don't answer
yet: what should a likelihood do when it's raised?

The tempting answer, `except GLConvergenceError: return -np.inf`, is wrong in
both places it would be used:

- **In an MCMC**, `-inf` means "zero posterior here". The sampler then rejects
  every proposal into that region, so the chain silently avoids part of
  parameter space for numerical, not physical, reasons. The posterior comes out
  biased, with no record of where.
- **In an optimizer** (Powell), an `inf` objective near the optimum can stop it
  early or steer it away, giving a "best fit" that's an artifact.

With precomputed tables (`Ppq_table_*`), the quadrature isn't called at all
during a fit, so this can only arise without tables, or while *building* a table
(`normgrid.py run`, which already records the point as failed and moves on;
that's correct there).

## Decision

1. **Likelihoods let it propagate.** `docs/fitting.md` and `docs/mcmc.md` say so
   explicitly: don't catch `GLConvergenceError` in a log-probability. A
   convergence failure stops the fit or chain with the parameters in the message,
   and the fix is a looser `norm_epsrel`/`norm_epsabs` or a table, never a
   silent `-inf`. The emcee HDF5 backend keeps every step before the failure, so
   resuming after the fix loses nothing.
2. **`-inf` stays reserved for the prior.** It means "outside the prior box or
   the ordering `p10 >= p0`, `q10 >= q0`", checked before the likelihood is ever
   called, as the docs' examples already do.
3. `GLConvergenceError`'s message includes the full parameter set and the
   tolerances, so the failing point is reproducible from the log alone.

## Consequences

- A numerical failure is visible instead of quietly reshaping the posterior.
- A long chain can stop partway. That's the intended trade: it's resumable, and a
  chain that silently avoided part of parameter space would need a full rerun
  once noticed.
- Under MPI, an exception on a worker rank must reach rank 0 and stop the job,
  not hang it: verify schwimmbad's `MPIPool` behavior when a worker raises, and
  document `mpirun`/`srun` abort-on-error if it doesn't.
- Verification: a test that a forced non-convergence (tiny `epsabs`) raises
  through `Ppq_compiled.Ppq_normalized_densities_NR` with the parameters in the
  message.
