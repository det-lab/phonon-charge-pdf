# 0011. The recoil spectrum is always explicit, and the same object on both sides

Status: Proposed
Date: 2026-11-09

## Context

Every density and every simulated event depends on a true-recoil-energy
spectrum, `PErecoil_NR`/`PErecoil_ER`. Today the two halves of the library ask
for it differently, so they can disagree without anything noticing.

- **Simulation** (`generate_events`): six required floats
  (`mix_frac_*`, `scale_fast_*`, `scale_slow_*`), plus an optional
  `p_erecoil_NR/ER` `PErecoilTable` that *overrides* them. With a table you must
  still pass six numbers that are then ignored. With floats you can pass any
  values, including ones the likelihood can't represent.
- **Likelihood** (`Ppq_compiled`, `Ppq_densities`): the Fortran's hardcoded
  biexponential (`PNa/PNb/PNd`, `PGa/PGb/PGd` in `src/Ppq_fort_s.f90`, fitted to
  **Cf-252**), unless a `PErecoil_table_*` is given. Nobody chooses the closed
  form; it's what you get by saying nothing.

Two ways this goes wrong silently:

1. **Simulated data from a different PDF than the one being fit.** Building
   `MCMC_demo/simulate.py` meant finding the Fortran's constants and passing
   them to `generate_events` by hand (from `Ppq_py._PNa` etc., private names
   that only happen to match the Fortran). Any other values simulate a different
   PDF, and a fit then "fails" for a reason unrelated to what's being tested.
2. **The wrong spectrum by default.** A Ba-133 likelihood built without a
   `PErecoil_table_ER` evaluates the ER band with Cf-252's gamma spectrum. The
   joint Cf + Ba fit is exactly this case.

Choosing a spectrum is a physics decision about the source. The library
shouldn't make it implicitly on either side.

## Decision

1. **Required, everywhere, no default.** `generate_events`, `Ppq_compiled` and
   `Ppq_densities` each take keyword arguments `PErecoil_NR=` and `PErecoil_ER=`
   with no default value. Omitting one is a `TypeError`. A band the caller
   doesn't use is declared explicitly with `None` (e.g. `PErecoil_NR=None` for an
   ER-only Ba likelihood or `fraction_ER=1.0` simulation); any use of that band
   then raises, naming it.
2. **One type on both sides.** Both accept the same two things:
   - a `PErecoilTable` (as now), or
   - the library's built-in closed form, exported by name as
     `erecoil_table.CF252_BIEXPONENTIAL_NR` / `_ER`: a small object carrying
     the Fortran's constants (the single Python copy, replacing `Ppq_py._PN*`/
     `_PG*`), with `sample(n, rng)` and `pdf(erecoil)` like `PErecoilTable`, and
     `perecoil_identity()` returning ADR 0002's `{"kind": "closed_form"}`.

   So the natural code passes *the same object* to the simulation and to the
   likelihood, and they match by construction:

   ```python
   nr, er = erecoil_table.CF252_BIEXPONENTIAL_NR, erecoil_table.CF252_BIEXPONENTIAL_ER
   events = pdf.generate_events(..., PErecoil_NR=nr, PErecoil_ER=er)
   fit_pdf = pdf.Ppq_compiled(..., PErecoil_NR=nr, PErecoil_ER=er)
   ```
3. **Remove the other spellings.** `generate_events` loses the six
   biexponential floats and `p_erecoil_*`. `Ppq_compiled`/`Ppq_densities`'
   `PErecoil_table_*` becomes `PErecoil_*` (it can now be the closed form, not
   only a table). A different biexponential, e.g. for a simulation study, is
   built as a `PErecoilTable` (a `PErecoilTable.from_function(pdf, erecoil_grid)`
   helper), so it's usable on both sides too. No spectrum exists that only one
   side can use.
4. **Simulated events carry their spectrum's identity.** `generate_events`
   records `perecoil_identity()` for each band in the returned DataFrame's
   `attrs`, so a fit to simulated data can check it against its likelihood's
   spectrum, the same check ADR 0002 makes against normalization tables.

## Consequences

- No silent spectrum choice remains. Simulation and likelihood agree whenever
  they're given the same object, and a joint Cf + Ba fit has to state each
  dataset's ER spectrum, which is the point.
- Breaking API change: every `generate_events` and `Ppq_compiled` call site
  (docs, tests, notebooks, `nrFanoII_paper2022/python/simulate_events.py`) adds
  two arguments. The error message for a missing one should name
  `CF252_BIEXPONENTIAL_*` as the Cf-252 choice, so the fix is one line.
- More verbose for the common Cf case. That's the trade the decision makes on
  purpose: the common case is still two named constants, and the uncommon case
  (anything not Cf-252) can't be reached by forgetting.
- The closed form's identity stays ADR 0002's `{"kind": "closed_form"}`, so
  existing normalization tables still match. The Python name says Cf-252; the
  stored identity doesn't. If a second built-in closed form is ever added, the
  identity needs a name field and a table-format bump.
- Verification: omitting `PErecoil_NR`/`_ER` raises `TypeError` on all three
  entry points; `None` plus use of that band raises naming it; simulated events
  with `CF252_BIEXPONENTIAL_*` fitted by a likelihood with the same objects
  recover their truth (`MCMC_demo/`'s simulated-data check); a mismatch between
  the events' recorded identity and the likelihood's is reported.
