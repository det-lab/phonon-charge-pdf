# Decision records (ADRs)

This is a maintainer workflow, like [Profiling](PROFILING.md) -- not needed to use
`phonon_charge_pdf` for a fit or MCMC. See [Two ways to use this](two-ways.md) for that.

Significant technical decisions this project has made -- the kind where a later
maintainer could reasonably ask "why didn't we just do the obvious thing instead?" --
are recorded in `adr/` at the repo root, not under `docs/`. That's deliberate: `adr/`
sits outside mkdocs' `docs_dir`, so it's never part of this rendered site. These records
are for whoever is reading the source later, not for a user of the library.

`adr/0001-batched-fortran-quadrature.md` is the first one, covering why `Ppq_compiled`
uses a batched Fortran quadrature instead of calling `scipy.integrate.quad` per point.

## Writing one

Read `adr/README.md` first -- it has the format and the one rule that matters: a record
is immutable once accepted. A later change of mind gets a new, numbered record that
supersedes the old one; it doesn't rewrite history by editing it in place.

Not every decision needs one -- see `adr/README.md`'s "When to write one" for where the
line is.
