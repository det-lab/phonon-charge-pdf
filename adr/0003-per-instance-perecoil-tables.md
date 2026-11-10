# 0003. PErecoil tables are per-instance, not process-global

Status: Accepted
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

Make the table an explicit argument of every evaluation call, not ambient
state of any kind -- fully stateless at the Fortran boundary. Two other
shapes were considered and rejected; both are worth recording since they're
likely to come up again:

- **A raw pointer + length instead of a handle.** C-interoperable derived
  types can't hold Fortran allocatable arrays, only `c_ptr`s, which would mean
  Fortran holding raw pointers into memory numpy/Python owns rather than its
  own copy. If that array is ever garbage-collected or reallocated while
  Fortran still holds a pointer to it, that's silent memory corruption, not a
  catchable error -- a real regression versus today's design, which already
  copies the data in (`PErecoil_set_table_NR`'s current behavior). Rejected on
  that basis alone.
- **An "activate" call that sets which table is current, leaving the real
  compute functions argument-free.** Appealing (fewer arguments on the
  functions that do real work), and close to a prior sketch of this record.
  Rejected because it reintroduces exactly the bug this record exists to fix,
  just moved somewhere harder to catch: `Ppq_densities`/`make_Ppq_densities_NR`
  already evaluates one call's points concurrently across a `ThreadPoolExecutor`
  (`_make_threaded_pdf`). If a second instance's "activate" runs on another
  Python thread while the first instance's pool is still mid-flight -- the
  exact joint-Cf+Ba-fit scenario motivating this record -- it can overwrite the
  active table underneath the first instance's still-running computation. A
  thread-local "active handle" would dodge that specific race, but collides
  with this library's own internal parallelism (`Ppq_vector_NR`'s
  `do concurrent`, genuinely parallel under ifx/flang, serial under gfortran):
  a freshly spawned worker thread doesn't automatically inherit the calling
  thread's thread-local value without more OpenMP machinery, and behavior
  would then differ by compiler. Passing the handle as a plain argument has
  none of these hazards, for every concurrency pattern, by construction --
  worth the extra argument.

Concretely:

- A new, non-pure entry point, `PErecoil_table_register(x, c0, c1, c2, c3, n)
  -> handle`: copies the table into a fixed-capacity, process-wide registry
  (`type(perecoil_table_t)` array -- fixed size, not grown by reallocation, so
  registering a new table can never invalidate another thread's in-progress
  read of an already-registered one) and returns its slot index. Handle `0`
  always means "use the closed-form spectrum", for both bands. Registration is
  append-only: it never overwrites a slot another instance already holds a
  handle to, so two instances with different tables coexist in one process,
  including evaluated alternately.
- Exactly the entry points that do real per-event or per-region work gain one
  trailing `table_handle` argument: `Ppq_NR`/`Ppq_ER`, `Ppq_vector_NR`/`ER`,
  `Ppq_region_NR`/`ER` (and `_with_error`). `Ppq_NR`/`ER` need it only because
  `Ppq_vector_NR`/`ER`'s internal loop calls them per point and must forward
  it, not because anything calls them directly with a table in mind.
- `PErecoil_NR`/`ER` and `pq_given_Erecoil_NR`/`ER` are **not** touched. Their
  own docstrings already say they're not a stable public API, used only by
  `test_PpqFort.py` to cross-check the closed form against `Ppq_py.py`'s
  reference implementation, and the real compute path (`Ppq_integrand`, inside
  `integrate_Ppq`) never goes through them -- it calls the elemental
  `PErecoil_NR_impl(Erecoil, handle)`/`ER_impl` helpers directly. Keeping
  these two functions closed-form-only keeps the bind(c) surface that changes
  as small as it can be.
- `Ppq_compiled`/`Ppq_densities` register their table(s) once at construction
  and store the returned handle(s); every method call on that instance
  forwards its own stored handle automatically. Nothing about what a Python
  caller writes changes -- the handle never appears in `Ppq_compiled`'s own
  public methods, only in the ctypes plumbing underneath them.

## Consequences

- Joint fits with per-dataset spectra work, under any concurrency pattern,
  which is the purpose of this change.
- The `bind(c)` signatures of the functions listed above change, so this
  touches `_Ppq_fort_bindings.py`'s `argtypes`, the Fortran interfaces in
  `Ppq_fort_m.f90`, and the ABI version (`PpqFort_version`, so a stale library
  is detected) -- `PErecoil_set_table_NR/ER`/`PErecoil_clear_table_NR/ER` are
  removed outright (no deprecation shim), since this is a pre-release feature
  branch and nothing published depends on them.
- A fixed-capacity registry (256 slots) holds tables for the process
  lifetime; registering past capacity is a hard error rather than silently
  growing or wrapping. They're small (a few thousand points each), so not
  freeing them is acceptable; a `PErecoil_table_free` can come later if
  needed, though it would have to contend with the same "don't invalidate a
  concurrent reader" constraint that ruled out a growable array here.
- Verification: a test with two instances holding different ER tables,
  evaluated alternately (and, ideally, genuinely concurrently from two
  threads), each matching what it gives when constructed alone in a fresh
  process.
