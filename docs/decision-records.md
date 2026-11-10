# Decision records (ADRs)

This is a maintainer workflow, like [Profiling](PROFILING.md) -- not needed to use
`phonon_charge_pdf` for a fit or MCMC. See [Two ways to use this](two-ways.md) for that.

Significant technical decisions this project has made -- the kind where a later
maintainer could reasonably ask "why didn't we just do the obvious thing instead?" --
are recorded in `adr/` at the repo root, not under `docs/`. That's deliberate: `adr/`
sits outside mkdocs' `docs_dir`, so it's never part of this rendered site. These records
are for whoever is reading the source later, not for a user of the library.

Current records:

| record | status | subject |
|---|---|---|
| `0001-batched-fortran-quadrature.md` | Accepted | batched Fortran quadrature instead of per-point `scipy.integrate.quad` |
| `0002-normalization-tables-record-perecoil.md` | Accepted | normalization tables embed the PErecoil spectrum (closed form, or a table's own (erecoil, counts) data) they were built with, checked against `Ppq_compiled`'s `PErecoil_table_NR`/`ER` at construction |
| `0003-per-instance-perecoil-tables.md` | Accepted | PErecoil tables per instance, stateless handle argument, not process-global (joint Cf + Ba fits) |
| `0004-reject-data-outside-region.md` | Proposed | `Ppq_compiled` rejects events outside its region |
| `0005-openmp-is-the-thread-layer.md` | Proposed | `n_workers` defaults to 1; `set_threads()` for OpenMP |
| `0006-one-flat-parameter-dict.md` | Proposed | one flat parameter dict through every method; mixture/joint docs |
| `0007-runpath-origin.md` | Proposed | the shared library finds `libassert`/`libjulienne` itself |
| `0008-no-per-call-data-shipping.md` | Proposed | workers build their own `Ppq_compiled`; it refuses to be pickled and warns on unguarded scripts |
| `0009-convergence-errors-in-likelihoods.md` | Proposed | likelihoods let `GLConvergenceError` propagate, never `-inf` |
| `0010-region-integral-precision-vs-dataset-size.md` | Proposed | a converged region integral's precision floor isn't checked against dataset size; the per-event term has no analog at all |
| `0011-recoil-spectrum-always-explicit.md` | Proposed | the recoil spectrum is a required argument, the same object for `generate_events` and `Ppq_compiled`; no silent Cf-252 default |

0002–0009 came out of building `MCMC_demo/` (a Cf fit + MCMC on Expanse) on 2026-11-09.
Wherever a record mentions `MCMC_demo/`, it means
[`nrFanoII_paper2022/MCMC_demo/`](https://github.com/villano-lab/nrFanoII_paper2022/tree/master/MCMC_demo),
the analysis repository that uses this library; it isn't part of this repository.
0010 came out of validating `Ppq_densities`/`generate_events` against a real tabulated
`PErecoil` on `feature/erecoil-table-pdf`.

## Writing one

Read `adr/README.md` first -- it has the format and the one rule that matters: a record
is immutable once accepted. A later change of mind gets a new, numbered record that
supersedes the old one; it doesn't rewrite history by editing it in place.

Not every decision needs one -- see `adr/README.md`'s "When to write one" for where the
line is.
