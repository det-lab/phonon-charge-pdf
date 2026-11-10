# 0002. Normalization tables record the PErecoil they were built with

Status: Accepted
Date: 2026-11-09

## Context

`Ppq_compiled` normalizes each band's density over the analysis region by
dividing by its region integral, and for any fit or MCMC that integral comes from
a precomputed table (`Ppq_table_NR`/`Ppq_table_ER`, built by
`python/cli/normgrid.py`).

On `feature/erecoil-table-pdf`, `PErecoil_table_NR`/`PErecoil_table_ER` replace the
closed-form biexponential recoil spectrum inside every Erecoil integral. That
changes the density itself, and so it changes the region integral too. But the
published normalization tables were all computed with the closed-form PErecoil,
and nothing records which spectrum a table assumed:

- `normgrid.py` has no PErecoil metadata (no mention of it in the spec, the HDF5
  attributes or `NormInterpolator`);
- `Ppq_compiled.__init__` accepts a `Ppq_table_*` and a `PErecoil_table_*` for the
  same band without comparing them.

Passing both gives densities computed with the tabulated spectrum divided by an
integral computed with the closed-form one. The likelihood is wrong, with no
error, and the bias shifts with every parameter the normalization depends on.
This is easy to hit: a user adopting PErecoil tables for a fit will naturally
keep the `Ppq_table_*` names they were already passing.

## Decision

1. **Record it at build time.** `normgrid.py`'s spec and the table's HDF5
   attributes gain a `perecoil` entry for the band the table was built with:
   `{"kind": "closed_form"}`, or `{"kind": "table", "sha256": ..., "detector": ...,
   "calibration_source": ..., "source": ..., "erecoil": [...], "counts": [...]}`
   (`erecoil_table.PErecoilTable.perecoil_identity()`). The raw `(erecoil, counts)`
   arrays are embedded directly, not just referenced by path: a spec is
   distributed to Slurm/OSG workers on other machines, which have no access to
   whatever local path the table was loaded from, and a published table should
   be able to say what spectrum it assumed on its own, indefinitely, without a
   separate file that can move or disappear. `sha256` is still carried for the
   cheap equality check at fit time (comparing it is just as valid as comparing
   the arrays, and doesn't need the arrays decoded). A worker registers the
   table (`PErecoil_table_register`) from the embedded arrays once per process
   before evaluating, so the table is computed with the spectrum it says it used.
2. **Check it at fit time.** `Ppq_compiled.__init__` compares each band's
   `PErecoil_table_*` (or its absence) with the normalization table's recorded
   `perecoil` -- `kind` and, for `"table"`, `sha256` only, never the embedded
   arrays themselves -- and raises `ValueError` naming both on any mismatch.
3. **Old tables.** Bump `TABLE_FORMAT_VERSION`; `NormInterpolator.from_hdf5` then
   refuses any table built before this outright, with no migration path --
   rebuild it (`make-spec`/`run`/`merge`) instead. A table rebuilt with the
   normal closed-form flow gets `perecoil={"kind": "closed_form"}` for free
   (`make_grid_spec`/`make_random_spec` default to it), so only a table that
   actually used a PErecoil table needs `--perecoil-table` on the rebuild to
   keep using it.

## Consequences

- A mismatched normalization becomes an error at construction, not a silently
  biased fit.
- Each PErecoil table needs its own normalization tables, built with
  `normgrid.py` like any other region or box. That's a real compute cost per
  detector, and the registry (`table_registry.json`) will grow per detector.
  That's inherent: the integral genuinely differs.
- The check is a hash comparison, so a PErecoil table that's regenerated with
  identical content still matches, and one regenerated with different cuts
  (the branch has done this several times) correctly doesn't.
- Verification: a test that builds a tiny normalization table with a PErecoil
  table and checks that `Ppq_compiled` (a) accepts it with the same PErecoil
  table, (b) raises with none, (c) raises with a different one, and (d) still
  accepts every existing published table with no PErecoil table.
- The `TABLE_FORMAT_VERSION` bump breaks the two tables already published to
  Zenodo (`table_registry.json`) outright: they must be rebuilt from scratch
  (not migrated -- there is no migration path) and re-published, with
  `table_registry.json`'s sha256 updated to match, before anyone can fetch
  them again.
