# 0011. PErecoil handles register a fit, not a table

Status: Accepted
Date: 2026-11-13

## Context

`adr/0003-per-instance-perecoil-tables.md` made `table_handle` an explicit,
per-instance argument threaded through every real evaluation call, backed by
`PErecoil_table_register`: copy an arbitrary piecewise-cubic table (PCHIP-
interpolated from real calibration data, `erecoil_table.PErecoilTable`) into
a fixed registry, hand back a handle. The premise was that the Fortran
quadrature's doubling-verified convergence check would just handle whatever
table it was given, the same way it handles the closed form.

That premise doesn't hold. Investigating why a real tabulated NR spectrum
(`erecoil_tables/402_NR_Cf.h5`) made region integrals near the phonon
resolution model's degenerate edge (`p10 -> p0`, i.e. `dp -> 0`) converge far
more loosely than the closed form -- measured, not assumed:

- The closed-form biexponential shows **no** effect at `dp = 0`: flat
  ~1e-12 relative residual at every `p10` tested, including exactly `p10 =
  p0`. Swapping in the tabulated spectrum, same baseline, same region, same
  `dp = 0` point: relative residual jumps to ~8e-4 -- isolated cleanly to the
  PErecoil representation, nothing else.
- Mechanism: `sigp(Ep)^2 = p0^2 + (p10^2-p0^2)(Ep/scale)^2` is pinned at its
  floor `p0` uniformly across the whole Ep range exactly when `dp -> 0` (it
  only grows with Ep when `dp > 0`), and the per-mode integration window is
  sized from this resolution width (`sig_m = 1/sqrt(-H'')` in
  `locate_modes`). A uniformly narrow window makes the quadrature sensitive
  to the integrand's *own* local shape rather than letting a wide,
  smooth resolution kernel average it out. The closed form is globally
  smooth at every scale, so a narrow window costs it nothing; a real
  spectrum's own curvature (not just noise) becomes numerically significant
  exactly there.
- A Poisson-noise-matched smoothing spline (replacing PCHIP's exact,
  wiggle-preserving interpolation) only partly closed this gap -- it fixed
  noise-driven spikes elsewhere in parameter space (10-1300x improvement at
  specific points) but barely touched the `dp -> 0` floor itself (~1.5-2x),
  because that floor isn't primarily a noise artifact: it's the integrand's
  genuine local curvature mattering more once the window narrows, and
  smoothing removes counting noise, not real shape.
- A **global, analytic fit** (no local/piecewise structure at any scale)
  closes it completely: fit to the same real data, registered the same way,
  the relative residual is flat at ~1e-12 to 1e-13 across the *entire*
  `dp` range -- parity with the closed form, ~8 orders of magnitude better
  than the raw table at the worst point.

The first fit family tried (the closed form's own 2-exponential family,
fit per-detector) failed on real NR/Cf data: chi2/dof ~3-4, a real
localized mismatch. Two independent bugs turned out to be responsible, not
the family: (1) `Erecoil` below the detector's own trigger-threshold-turnon
boundary isn't real data at all (the extraction pipeline,
`nrFanoII_paper2022/scripts/extract_erecoil_tables.py`'s
`correct_nr_threshold_turnon`, already replaces it with a placeholder fit);
(2) the table's `counts` field is a *density* (`raw_count / bin_width` for
equal-count bins), so the true per-bin uncertainty is a constant *relative*
~1/sqrt(raw_count), not `1/sqrt(density)` -- the error model, not the
family, was wrong. Fixed, a 2-exponential-plus-constant family
(`a[0]*exp(-x/b[0]) + a[1]*exp(-x/b[1]) + C`) fits the real, trustworthy
data (above each detector's own threshold boundary) to chi2/dof in
[0.82, 1.16] across all 14 surveyed Cf NR detectors, and passes a
Kolmogorov-Smirnov test against the real unbinned events (p in
[0.13, 0.82], nothing near the 0.05 significance line) for every one.

## Decision

1. **A PErecoil handle registers a fit, not a table.** Fortran:
   `PErecoil_table_register`/`table_registry`/`eval_cubic_table` are removed
   from the quadrature path entirely, replaced by `PErecoil_fit_register(a,
   b, n, C) -> handle` / a `fit_registry` of `perecoil_fit_t` (allocatable
   `a(:)`, `b(:)`, scalar `C`) / `eval_fit`. `table_handle`'s role --
   explicit, per-instance, append-only, thread-safe (`adr/0003`) -- is
   unchanged; only what it points to changes. `table_hi_bound` becomes an
   unconditional `huge(1.0d0)`: a sum of exponentials plus a constant is
   defined, and smooth, for every `Erecoil >= 0`, with no edge for a Fortran
   search window to run off -- the whole reason that bound existed.

2. **A curated family, not an arbitrary user-specified function.**
   `density(Erecoil) = sum(a[i] * exp(-Erecoil/b[i])) + C`, variable-length
   (any number of terms) so a future detector needing a different term
   count isn't blocked, but still one of a small, explicitly-supported
   set (`erecoil_fit.FAMILY_FORMULAS`) -- never an arbitrary expression a
   caller supplies. Genericity is exactly what caused the original problem
   (a table can represent anything, including something numerically
   hostile); an open-ended expression language has the identical property.
   A per-fit `"formula"` string, fixed per family and reused verbatim
   (`FAMILY_FORMULAS[family]`), is recorded in `perecoil` provenance for a
   human reading the HDF5 file directly -- documentation, never read back
   or evaluated.

3. **`erecoil_table.py` narrows to simulation and fit-checking; it no
   longer feeds the compiled quadrature path.** `erecoil_table.PErecoilTable`
   (PCHIP, HDF5 I/O, `.sample()`) stays exactly as it was for
   `event_generator.generate_events()`'s simulated events -- nothing about
   sampling needed the compiled library in the first place, and a real
   table (with its real small-scale structure) is arguably the *more*
   honest choice there, not less. The new `erecoil_fit.py` is what
   `Ppq_compiled`/`Ppq_densities`/`normgrid.py` take instead
   (`PErecoil_fit_NR/ER`), and provides
   `check_fit_against_events(fit, erecoil_events)`: a Kolmogorov-Smirnov
   test of a fit's own shape against real (or table-simulated) events,
   so that gap is a routine, checkable thing, not something only this
   investigation looked at once.

4. **`normgrid`'s `perecoil` provenance (`adr/0002`) gains a `"fit"` kind**
   alongside `"closed_form"`: `{"kind": "fit", "family", "formula", "a",
   "b", "C", "sha256", "detector", "calibration_source", "source"}`,
   embedded directly (no file format of its own -- `a`/`b`/`C` are a
   handful of numbers, not worth one), checked at `Ppq_compiled`
   construction the same way the old `"table"` kind was (`kind` and
   `sha256` only). The `"table"` kind itself is retired from that
   mechanism along with the Fortran path it described.

## Consequences

- ABI bump: `fpm.toml` `3.0.0 -> 4.0.0` (another breaking `bind(c)` change).
- `python/cli/normgrid.py`'s `--perecoil-table PATH` CLI flag is replaced by
  `--perecoil-fit-json PATH` (a small JSON file: `a`, `b`, `C`,
  `erecoil_max`, optional `family`/`detector`/`calibration_source`/`source`)
  -- a fit has no dedicated binary file format, deliberately (point 2).
- `erecoil_table.PErecoilTable.perecoil_identity()`/`.sha256`/`.fortran_table`
  are removed: nothing registers a table with Fortran any more, so nothing
  needs a table's own provenance identity or its piecewise-cubic form.
- Fitting is now a real, separate step between "real calibration data" and
  "something the integrator can use" -- whoever builds a table has to also
  fit it (`erecoil_fit.fit_sum_of_exponentials`) and check that fit
  (`check_fit_against_events`) before using it for a region integral or a
  normalization table, rather than registering the table directly. That is
  real, additional work per detector, in exchange for the quadrature
  guarantee the table path never actually had.
- Verification: `test_erecoil_table_fortran.py` rewritten for
  `PErecoil_fit_register` (same interleaved-multi-handle, closed-form-
  unaffected-by-registration checks `adr/0003` calls for, on fits instead
  of tables -- a fit has no narrow/wrong-range construction-time check to
  test, since it has no range at all); `test_normgrid.py`'s `adr/0002`
  section rewritten for the `"fit"` kind. `test_chisquare_simulator_nr_table.py`
  now simulates from a real table while the compiled PDF evaluates a fit to
  that same table -- deliberately different representations, so its
  pass criterion was loosened from exact `chi2(dof)`-distribution matching
  to a bound wide enough for the fit-vs-table difference this record
  measured, not tight enough to miss an actual wiring bug.
