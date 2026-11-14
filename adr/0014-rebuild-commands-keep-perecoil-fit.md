# 0014. Normalization tables record their PErecoil fit JSON, so rebuild commands keep the spectrum

Status: Proposed
Date: 2026-11-13

## Context

`NormInterpolator.rebuild_command` (`python/cli/normgrid.py`) prints the
`build_table.job` command for a table that would cover what a fit needs.
`check_prior` puts it in every `PriorNotCoveredError`, so it's the line a user
copies and submits. It is built from the table's box, band and window only:

```
BOX="q0=0.04:0.45" sbatch table_building/slurm/build_table.job ER 3 85 2 40
```

Since `adr/0011`/`adr/0012`, a table can be built from a PErecoil fit
(`PERECOIL_FIT_JSON=<fit>.json` → `build_table.sh` → `--perecoil-fit-json` →
`normgrid._perecoil_from_fit_json`). `rebuild_command` never emits
`PERECOIL_FIT_JSON=`, so for such a table the printed command rebuilds it with
the closed-form spectrum. The rebuilt table loads, but `Ppq_compiled._check_perecoil`
(`adr/0002`) rejects it against the user's `PErecoil_fit_*`. The user loses a
full cluster build and gets a mismatch error instead of a table. This affects
42 of the 44 planned per-detector tables (every one but the two closed-form ones).

`build_table.sh`'s own retry hint, printed when the held-out check fails,
already appends `PERECOIL_FIT_JSON=` (around line 160). It can, because it still
holds the path. `rebuild_command` runs later, from the table alone, and the
table doesn't record enough to reproduce the build:

- The `perecoil` block is `PErecoilFit.perecoil_identity()`: `kind`, `family`,
  `formula`, `a`, `b`, `C`, `sha256`, `detector`, `calibration_source`, `source`.
- `source` is the JSON's own free-text field, not a path. The test fixture's
  reads "erecoil_fit.CF252_BIEXPONENTIAL_ER's own raw (pre-normalization)
  coefficients".
- `a`/`b`/`C` are the normalized coefficients, and `erecoil_max` isn't kept.
  The fit JSON can't be rebuilt from them: `_perecoil_from_fit_json` expects
  raw coefficients, and its docstring says re-normalizing normalized ones
  would be wrong.

So neither the path nor the file's contents can be recovered from a built
table.

## Decision

1. **Record the build input.** `_perecoil_from_fit_json(path)` adds two
   keys to the identity it returns, and so to the table's `perecoil_json`:
   - `fit_json_path`: the path exactly as given to the build
   - `fit_json`: the JSON file's parsed contents, verbatim (raw `a`/`b`/`C`,
     `erecoil_max`, metadata)

   They're added on the build side only, not in `PErecoilFit.perecoil_identity()`:
   at fit time a `PErecoilFit` has no file, and the build side is the only
   place that has the JSON.
2. **No format bump.** Both keys are optional. `_check_perecoil` and
   `validate` compare only `kind` and `sha256`, so nothing that compares
   identities changes, and tables built before this still load.
3. **`rebuild_command` reproduces the spectrum.** For `kind == "fit"` it
   emits `PERECOIL_FIT_JSON=<fit_json_path>` after `BOX=`, the same order as
   `build_table.sh`'s retry hint, with paths `shlex.quote`d. Before printing
   it checks the file, relative to the repository root, the directory
   `build_table.job` is submitted from:
   - present, and `_perecoil_from_fit_json(path)` gives the recorded `sha256`:
     the command alone
   - missing, unreadable, or a different `sha256`: the command plus a warning
     saying which, and the `export-fit-json` line below
   - no `fit_json_path` (a table built before this change): a placeholder
     naming `detector`, `calibration_source` and `sha256`, with a note to
     substitute a matching fit JSON

   It warns and never raises: it runs while another error is being raised, and
   a second exception would hide the first. `closed_form` tables, including
   ones built with `CF252_BIEXPONENTIAL_NR/ER` passed explicitly (`adr/0011`),
   get exactly today's command.
4. **Recovery from the table.** A new `normgrid.py export-fit-json NORM_TABLE
   OUT.json` writes `fit_json` back out, and the result builds a
   `PErecoilFit` with the recorded `sha256`. It's part of this decision rather
   than a follow-up because it's the only fix the missing-file warning can
   offer.
5. **The rule, for any later kind.** A non-closed-form `perecoil` kind must
   record its build input, verbatim, alongside its identity, and
   `rebuild_command` must emit the argument that reproduces it. This applies to
   the `"kde"` kind `adr/0013` proposes.

## Consequences

- The command in every `PriorNotCoveredError` rebuilds a table that
  `_check_perecoil` accepts against the same fit. The planned window-mismatch
  error reuses `rebuild_command`, so it gets this too.
- A table is self-contained for rebuilding, as well as for identification,
  which is `adr/0002`'s intent for a published table: a Zenodo table can be
  rebuilt with nothing but itself.
- Each table carries a second copy of its fit: a few dozen numbers, small next
  to the table's values.
- `perecoil` now holds two kinds of data: identity (`kind`, `sha256`, compared)
  and provenance (`fit_json_path`, `fit_json`, carried along). Code that
  compares identities must keep comparing only the identity fields. This is
  already true of `_check_perecoil` and `validate`, and needs to stay true of
  anything new.
- Tables built on this branch before this change can't be rebuilt
  automatically. They print the placeholder, and should be rebuilt once the
  44-table run uses the new code.
- Printing a command now reads a file, but only on the error path.
