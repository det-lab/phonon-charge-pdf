# 0003. PErecoil tables are per-instance, not process-global

Status: Proposed
Date: 2026-11-09

## Context

On `feature/erecoil-table-pdf`, a `PErecoil_table_NR`/`_ER` is loaded into
module-level state in the compiled library (`nr_table_x` and friends in
`src/Ppq_fort_s.f90`). `Ppq_compiled` and `Ppq_densities` document the
consequence: constructing a second instance in the same process, with or without
its own tables, replaces the first one's. The first instance keeps evaluating,
now with the wrong spectrum, and nothing raises.

That's exactly the shape of the next planned analysis, a **joint Cf-252 + Ba-133
fit**. Both datasets have an ER band, but their true recoil spectra differ (Cf's
gammas are not Ba-133's lines), so the joint likelihood needs two `Ppq_compiled`
instances with different ER PErecoil tables, evaluated alternately in the same
process at every step. Under the current design the second construction
silently overwrites the first, and the Cf term is then computed with Ba's
spectrum.

MPI and process pools don't help: each rank holds both datasets' terms.

## Decision

Make the table an argument of the evaluation, not ambient state.

- The Fortran entry points that evaluate densities or region integrals
  (`Ppq_vector_NR/ER`, `Ppq_region_NR/ER`, `Ppq_NR/ER`) take the PErecoil table
  explicitly: either a small integer handle returned by a
  `PErecoil_table_register(x, y, n) -> handle` routine (the library keeps a list
  of registered tables, never overwriting one), or the arrays themselves
  (pointer + length). Handle 0 / null means closed-form.
- `Ppq_compiled` and `Ppq_densities` register their tables at construction and
  pass their own handle on every call.
- Registration is the only mutation of library state, and it's append-only, so
  instances can't interfere, and nothing is shared across threads except
  read-only table data.

If the Fortran change has to wait, an interim guard is cheap: a global generation
counter bumped on every load, recorded by each instance at construction and
checked before each evaluation. The superseded instance then raises instead of
computing with another instance's table. This guard is the minimum to ship
before PErecoil tables are used in any fit.

## Consequences

- Joint fits with per-dataset spectra work, which is the purpose of this change.
- The `bind(c)` signatures change, so this touches `_Ppq_fort_bindings.py`'s
  `argtypes`, the Fortran interfaces, and the ABI version
  (`PpqFort_version`, so a stale library is detected).
- A handle registry holds tables for the process lifetime. They're small (a few
  thousand points), so not freeing them is acceptable; a `PErecoil_table_free`
  can come later if needed.
- Verification: a test with two instances holding different ER tables,
  evaluated alternately, each matching what it gives when constructed alone in
  a fresh process.
