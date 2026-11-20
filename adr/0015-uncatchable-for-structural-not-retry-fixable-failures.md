# 0015. Structural, not-retry-fixable failures should be uncatchable, not a Python exception

Status: Proposed
Date: 2026-11-20

## Context

`adr/0014-perecoil-valid-region-and-turnon-family.md` added a hard failure for a
PErecoil fit evaluated outside its own valid region -- `normgrid.OutOfBoxError`
(a catchable `ValueError` subclass) on the Python side, Fortran `error stop` on
the compiled side once a mode is *accepted* there (not merely probed; see that
record for the transient-vs-accepted distinction). Picking `error stop` for the
Fortran side, instead of mirroring the Python side with something equally
catchable, came from a direct question: a Python exception in this codebase is
already an established "sometimes caught and treated as reject-and-retry"
idiom -- `goodness_of_fit`'s own retry logic deliberately catches
`GLConvergenceError` and tries a looser tolerance. A new exception of similar
shape risks someone, now or later, writing a handler that treats "this fit
doesn't cover what's needed" the same forgiving way. Nothing retries a fit's
way out of not covering the right range; the person fitting needs new data or
different parameters, and nobody should be able to get a trusted-looking
result without that happening first.

That reasoning doesn't stop at the new guard. Two existing, already-shipped
mechanisms arguably belong in the same category:

- `normgrid.OutOfBoxError` -- not just at MCMC setup (`check_prior`/
  `check_points`, called once), but also from `NormInterpolator.__call__`,
  which `Ppq_compiled.Ppq_integral_NR/ER` calls on *every* likelihood
  evaluation. A walker proposing a point whose normalization the precomputed
  table doesn't cover can raise this deep inside a long run, not just at
  setup -- an earlier framing of this as "setup-time, cheap to fail early"
  was wrong once actually checked against the code, not assumed.
- `normgrid.PriorNotCoveredError` -- raised at setup (`check_prior`/
  `check_points`), but for the same underlying reason as the other two: the
  prior reaches outside the table's box, and nothing about retrying a sample
  fixes that; the table needs rebuilding over a wider box.

**This is not the first time this project has weighed `error stop` against a
catchable exception for a quadrature failure, and the precedent cuts the other
way for a different reason.** `adr/0009-convergence-errors-in-likelihoods.md`
(still Proposed) describes replacing `Ppq_region_NR/ER`'s own `error stop` (on
a failed Gauss-Legendre doubling) with the catchable `GLConvergenceError`,
specifically because an `error stop` "killed the whole process, including an
MPI job's other ranks, with no chance to recover" -- the emcee HDF5 backend
already has every step before the failure, so a catchable exception that's
*allowed to propagate uncaught* (the docs explicitly say not to catch it) lets
a long chain resume after a tolerance fix, rather than losing hours of
otherwise-good progress to a numerical hiccup unrelated to whether those
earlier steps were sound.

That resumability argument does **not** carry over to this record's failures
-- though not for identical reasons in both cases, worth being precise about
rather than lumping them together. A fit evaluated outside its own Erecoil
valid region is wrong from the first step: fixing it means refitting (new
data, different parameters), which changes the *physical model* every step of
the chain used, start to finish. There is no sense in which steps 1-49,999
and steps 50,000-on could both be "the same analysis" once the fit itself is
different -- resuming isn't just inconvenient here, it's incoherent.

A table too narrow for the box a prior or a normalization lookup actually
needs is a softer case: fixing it means rebuilding the *same* table over a
*wider* box for the *same* underlying physics (same PErecoil fit, same
resolution model, same everything that defines the analysis) -- so, in
principle, a chain could resume against the rebuilt table with nothing about
the physical model having changed underneath it. That's a real, narrower
exception, not a reason to treat this case like a GL-convergence retry.

The reason this record still treats it the same as the Erecoil case: this
project wants a stronger, simpler invariant than "resuming is theoretically
sound" -- the same configuration should run for the *entire* MCMC process, not
part of it under one table and the rest under a rebuilt one stitched
together. A GL-tolerance retry doesn't touch that invariant at all (it's a
purely numerical precision dial around computing the same well-defined
quantity; nothing about the analysis itself was ever in question). Rebuilding
a table mid-run does touch it, even though the physics model it computes
against didn't change -- and that's enough to put it on the uncatchable side
of the line, by choice, not because resuming would be factually wrong.

What ADR 0009 does **not** resolve, and this record inherits unverified rather
than quietly assuming settled: whether this project's MPI layer (schwimmbad's
`MPIPool`, per that record's own open "Consequences" item) actually terminates
the whole job cleanly when one worker dies, by any mechanism -- `error stop`,
an uncaught Python exception reaching the top, a segfault. If a dead worker
can leave the rest of a job hanging rather than aborting, that is a gap in the
MPI orchestration layer, not something either failure mechanism fixes or
worsens on its own.

## Decision

1. **The general principle, stated explicitly rather than left as inferable
   precedent:** the test is not "could this in principle be resumed" -- it's
   whether fixing the failure touches the analysis's own configuration (data,
   priors, tables, fits) at all, even just by rebuilding one wider, versus
   being a purely numerical precision dial around computing an already
   well-defined quantity more carefully.
   - **A purely numerical retry, nothing about the analysis itself in
     question** (GL non-convergence, fixable with a looser tolerance) ->
     stays a catchable Python exception (or, in Fortran, a NaN a Python
     wrapper turns into one), allowed to propagate but never silently
     caught-and-continued-past. This is `GLConvergenceError`'s own category
     (`adr/0009`), unchanged by this record.
   - **Anything that requires touching the configuration** -- refitting (the
     physical model itself changes, `check_valid_region`'s own case) or
     rebuilding a table/box over a wider range (the model doesn't change, but
     the configuration still does) -- **-> uncatchable**, regardless of
     whether a narrower argument for "this one could in principle resume"
     exists. The project wants one invariant stronger than case-by-case
     resumability: the same configuration runs for the entire MCMC process,
     never part under one setup and the rest stitched on after a mid-run
     fix. This is `check_valid_region`'s category (`adr/0014`), and --
     reclassified by this record -- `OutOfBoxError`'s and
     `PriorNotCoveredError`'s own, even though the latter two could, in a
     narrower sense, be argued resumable (see Context).

2. **`normgrid.OutOfBoxError` and `normgrid.PriorNotCoveredError` become
   uncatchable**, matching `check_valid_region`'s own Fortran `error stop` in
   spirit. Python has no exact `error stop` equivalent:
   - `sys.exit()`/`raise SystemExit` is the conventional "fatal" signal, but
     it is still a real exception -- a bare `except:` (or `except
     BaseException:`) can swallow it, exactly the failure mode this record
     is trying to close off.
   - `os._exit()` is genuinely uncatchable -- no Python-level exception
     machinery runs at all, so nothing downstream can intercept it. The real
     cost: it skips `finally`/context-manager `__exit__`/`atexit` entirely,
     so nothing gets a chance to flush or clean up. **Checked directly, not
     assumed** (forking a child process through `emcee.backends.hdf.HDFBackend`'s
     real `save_step`, killing it at four points spanning a step's own
     open-write-close sequence, both via `os._exit()` and a normal uncaught
     exception -- see Consequences): the file was fully openable in every
     case tested, because `save_step` opens, writes, and closes the HDF5
     file fresh on every single step rather than holding it open across the
     run, so nothing is durably committed until that per-step `close()`
     actually happens. The only real difference `os._exit()` makes is losing
     a step whose data was fully written but not yet flushed when the kill
     landed (a normal exception still saves it, since stack-unwinding runs
     `close()` on the way out) -- and that window never applies to any of
     this record's failures: `compute_log_prob` (where they fire) always
     finishes, success or exception, strictly before emcee's own sampler
     calls `save_step` for that iteration (confirmed directly in
     `emcee.ensemble.EnsembleSampler.sample`'s source, not assumed from the
     docs) -- so the write window and the failure window structurally never
     overlap.

3. **Not a free pass to add process-killing failures casually.** Both
   reclassified errors, and `check_valid_region`, share the same test: would
   retrying with different numerical settings alone ever produce a correct
   result? If yes, it stays catchable. If the honest answer is "no, the
   setup/fit/box itself has to change first," it's uncatchable. This is a
   narrow category, not a default.

## Consequences

- This is a breaking change to already-shipped, public behavior: any existing
  code (inside or outside this repo) that catches `OutOfBoxError` or
  `PriorNotCoveredError` today -- for any reason, including a reasonable one
  this record didn't anticipate -- stops working the way it used to. Tracked
  as a GitHub issue, not implemented directly by this record: an audit of
  every existing raise site and every existing catch site (in this repo and,
  as far as discoverable, downstream), confirming none of them are quietly
  relying on catching these two for something legitimate that isn't actually
  "retry with different settings." `GLConvergenceError` must NOT be swept into
  this conversion -- `goodness_of_fit`'s own retry logic depends on it staying
  catchable (`adr/0009`), and the whole point of this record is narrowing
  which category a given failure belongs to, not collapsing the distinction.
- The `os._exit()` vs. on-disk-consistency question is answered, not left
  open: `adr/0015_hdf5_crash_experiment.py` forks a child process through
  `emcee.backends.hdf.HDFBackend.save_step` (emcee 3.1.6), killing it at four
  points spanning a step's own open/write/close sequence via both
  `os._exit()` and a normal uncaught exception, and found the file fully
  openable in all eight cases --
  `save_step` opens, writes, and closes the file fresh every step, so
  nothing is durably committed until that close actually happens, and a kill
  before it cleanly reverts to the last fully-completed step rather than
  corrupting anything. The one real difference `os._exit()` makes (losing a
  step that was fully written but not yet flushed when killed) cannot
  coincide with any of this record's failures: `emcee.ensemble.
  EnsembleSampler.sample`'s own source confirms `compute_log_prob` (where
  they fire) always completes, success or exception, strictly before
  `save_step` is called for that iteration -- the write window and the
  failure window never overlap. This closes the precondition this record
  originally required before the conversion could ship.
- `adr/0009`'s own open item -- verifying schwimmbad's `MPIPool` behavior when
  a worker dies (by any mechanism) -- remains open, inherited here rather than
  resolved. Both records now depend on it; it should be settled once, not
  investigated twice.
- Once converted, `check_prior`/`check_points`'s own docs (and any place that
  currently documents "catch `OutOfBoxError` and rebuild your box") need a
  pass to stop describing a catch-and-continue workflow that no longer works
  -- the new workflow is "the process ends, read the message, rebuild the box
  or refit, start over," the same shape `check_valid_region`'s own message
  already describes.
- Verification for this record's own claims (the no-resumability argument,
  the MPI-termination gap being pre-existing and not novel): none needed
  beyond what's written here and in `adr/0009` -- this is a design record, not
  an implementation; the actual conversion's own verification is the GitHub
  issue's job.
