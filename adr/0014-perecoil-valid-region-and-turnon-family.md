# 0014. PErecoilFit gets its own valid region, and a turn-on family for Cf NR

Status: Accepted
Date: 2026-11-20

## Context

Companion work in `nrFanoII_paper2022`'s two fit-validation notebooks surfaced two
real gaps in `erecoil_fit.PErecoilFit`, and a third, separate opportunity for Cf's
NR band:

1. **Silent trust in a bad extrapolation.** Ba-133 detectors 436/445 fit better at
   low Erecoil when trained only on data `<=290 keV` (a real 330-390 keV feature the
   baseline+bump family can't represent otherwise distorts the shared baseline), but
   that narrower fit's own extrapolation past 290 keV diverges sharply from real
   data. Adopting the narrower fit was rejected for exactly that reason last
   session -- but the real problem was never the narrower fit, it was that nothing
   stopped anyone from evaluating it past where it's good and trusting the result.
2. **A step-floor anchor that isn't physical.** Cf-252's ER band needed its flat
   constant floor replaced with a smoothed step (`adr/0013`'s own follow-on work,
   in the Cf ER notebook) to stay normalizable; the step's edge (`x0 = data_max`) is
   a deliberate, conservative modeling choice, not a measured feature of the
   spectrum. Past it, the model is still perfectly well-defined -- just not
   something to read as physical truth.
3. Both of these are instances of the same missing concept: `PErecoilFit` has no
   notion of a *trustworthy* range distinct from `erecoil_max` (the normalization
   bound). Its own docstring said evaluation is deliberately unbounded ("defined,
   and smooth, for every Erecoil >= 0 ... that absence is the point"), and
   `_load_erecoil_fit` (`python/internal/_Ppq_fort_bindings.py`) said even more
   directly: "a fit has no erecoil_range to validate against the region ... so
   there is no analogous construction-time check to make here." Both statements are
   still true about *smoothness* (the property the compiled Fortran quadrature
   needs, per `adr/0012`) -- they conflated that with *trustworthiness*, which is a
   separate, Python-facing question this record addresses.
4. **Real precedent already exists and didn't need reinventing**:
   `erecoil_table.PErecoilTable.__call__` already does exactly this for a table's
   own range -- a call-time check against `[lo, hi]` (small relative tolerance for
   float noise), raising `normgrid.OutOfBoxError` on failure, which already carries
   `axis`/`value`/`box`/`source` attributes generic enough for more than one kind of
   box (it's also what `normgrid.py`'s own interpolator range checks use).
5. **Separately**: Cf's NR band currently handles its own near-threshold behavior
   by discarding every bin at/below the observed peak
   (`extract_erecoil_tables.correct_nr_threshold_turnon`) and replacing it with a
   backward extrapolation of the declining exponential fit from just past the peak
   -- i.e. silently assuming perfect trigger/ionization efficiency all the way to
   Erecoil=0, which is itself an unexamined choice. The real near-threshold bins
   were always being plotted, just never fit.

## Decision

1. **`PErecoilFit` gains `valid_min`/`valid_max`** (default `0.0`/`erecoil_max`,
   preserving every existing fit's historical behavior exactly). `__call__` raises
   `normgrid.OutOfBoxError` for any `Erecoil >= 0` outside `[valid_min, valid_max]`
   (same message shape/mechanism as `PErecoilTable.__call__`'s own check); `Erecoil
   < 0` keeps returning `0.0`, matching that same convention. A narrower region is
   for a caller who has separately established that only part of a fit's smooth
   range is actually good -- e.g. a fit trained on a truncated subset of its own
   data, which should also narrow `erecoil_max` to match (there's no reason to
   extrapolate the *normalization* integral past where real data ends either, for a
   family without a vanishing floor).

2. **A new curated family**, `turnon_times_sum_of_exponentials_plus_const`:

   ```
   density(Erecoil) = (0.5*(1 + erf((Erecoil-x_on)/w_on))) *
                      (sum(a[i]*exp(-Erecoil/b[i])) + C)
   ```

   A smooth rising step (mirroring the Cf ER step-floor's falling one) times the
   existing plain family, modeling a detector's own near-threshold efficiency
   ramping up instead of discarding that data. No closed-form antiderivative exists
   (erf times a sum of exponentials), so normalization and `check_fit_against_
   events`'s CDF both go through numeric quadrature for this family (`_raw_integral`
   dispatches by family; closed form for the plain family, unchanged). `fit_turnon_
   times_sum_of_exponentials` mirrors `fit_sum_of_exponentials`'s own fitting
   conventions (squared amplitudes, log decay-lengths) plus the two robustness
   fixes `adr/0012`'s own investigation and the Cf ER notebook's step-floor work
   both needed: many independent rng seeds, and selecting among the best few
   *distinct* local optima by a KS check against real unbinned events, not by chi2
   alone.

   Measured across all 14 surveyed Cf-252 NR detectors
   (`nrFanoII_paper2022/analysis_notebooks/PErecoil_Fit_vs_Data.ipynb`): `x_on`
   3.4-8.8 keV, `w_on` 1.7-4.7 keV -- consistent in scale with the turn-on peak
   locations (5.3-14.1 keV) already documented for this detector set, not a wildly
   varying parameter with no shared meaning. Full-range KS p-values land at
   0.33-0.90 for every detector, with no cost to the fit quality in the region
   already trusted above the old discard boundary (recomputed there under the
   turn-on model, every detector still passes comfortably). Full numbers and the
   overlay plot are in that notebook, not reproduced here.

3. **Fortran gets a new, separate registration entry point**,
   `PErecoil_turnon_fit_register(a, b, n, C, x_on, w_on)` -- not extra optional
   arguments on `PErecoil_fit_register`, so that function's existing `bind(c)`
   signature never changes for the plain family's own callers. `perecoil_fit_t`
   gains `is_turnon`/`x_on`/`w_on` (defaulted so a plain fit's registration is
   unaffected); `eval_fit` gains optional `x_on`/`w_on` dummy arguments (present
   only for a turnon-family lookup); `PErecoil_NR_impl`/`ER_impl` branch on
   `f%is_turnon` to pass them or not. `fpm.toml` version `4.0.0 -> 5.0.0` (another
   breaking `bind(c)` change, same pattern as `adr/0012`'s own bump).
   `Ppq_py.py`, the pure-Python reference, needs **no change** -- it already takes
   an arbitrary callable `p_erecoil` override rather than a handle/registry, so any
   `PErecoilFit` (turnon family included) already works there as-is.

4. **The same guard also reaches a direct Fortran caller, not just Python --
   with two different outcomes depending on whether a bad Erecoil was merely
   *probed* or genuinely *accepted*.** `perecoil_fit_t` gains `valid_min`/
   `valid_max` (default `0.0`/`huge(1.0d0)`, "valid everywhere" -- every fit
   registered before this record existed behaves this way);
   `PErecoil_fit_register`/`PErecoil_turnon_fit_register` both gain two more
   arguments for them.

   This was almost scoped out as "no existing hook, needs new physics-dependent
   work" -- tracing `integrate_Ppq`/`locate_modes` directly showed that isn't
   actually true for *this* question. `ep_range`/`eq_range` genuinely have no
   Erecoil-region concept, but that was never the right thing to check against
   in the first place: `integrate_Ppq`'s own integration window for the dominant
   (usually boundary) mode extends `erecoil_m + 35*sig_m`, with up to four more
   `+4*sig_m` edge-guard expansions on top -- governed entirely by **the
   registered fit's own decay shape**, not by Ep/Eq at all. A fit that doesn't
   actually decay the way its own tail implies (a truncated fit like Ba 436/445's,
   or a slowly-decaying floor before Cf ER's step-floor fix) can already have
   that window pushed well past where it's trustworthy, silently, today. The
   fix that question actually needed was simpler than deriving a safe bound from
   the box: store the fit's own already-known valid region in the registry and
   check the real Erecoil being evaluated against it, directly, the same way
   `PErecoilFit.__call__` already does in Python.

   Tracing `locate_modes` further (prompted by the user explicitly asking for
   loud failure specifically for a fit that doesn't cover the analysis) surfaced
   a real distinction this record's first pass over-simplified: Newton's own
   internal search for a mode's peak can legitimately, harmlessly overshoot
   outside any fixed range while iterating (bounded only loosely, up to
   `1.5x` the scan range) -- that is normal numerical housekeeping, not a sign
   of anything wrong, and would make a hard failure there a false-positive risk
   (crashing on a perfectly fine fit/analysis pair purely because the optimizer's
   search passed through untested territory on its way to a fine answer). A
   **genuinely accepted** mode -- or the rare "every candidate failed to refine"
   last-resort grid's own range -- landing outside the valid region is different
   in kind: it means the real physics this analysis needs lies where the fit was
   never checked against data, a structural mismatch nothing about retrying
   fixes. So, two tiers, not one:
   - `PErecoil_NR_impl`/`ER_impl` still return a quiet NaN
     (`ieee_arithmetic`'s `ieee_value(..., ieee_quiet_nan)`, factored into a
     small `quiet_nan()` helper) for any *transient* out-of-range probe -- the
     same "invalid input surfaces as NaN, not a crash" convention
     `region_integral_gl` already uses for a failed Gauss-Legendre doubling.
   - `integrate_Ppq` gains `check_valid_region`, called exactly twice: once per
     accepted mode in `erecoil_m(1:n_modes)` (covering all three ways
     `locate_modes` can arrive at one -- Newton-refined, finite-difference
     fallback, or its own "nothing refined cleanly" last resort), and once on
     the separate, rarer "`locate_modes` found zero candidates at all" full-
     range-grid fallback's own upper bound (closing a gap this record's first
     pass had flagged but left open). Either one raises `error stop` --
     deliberately **not** a Python exception a future `except:` could learn to
     catch-and-continue past the way `GLConvergenceError` already is caught and
     retried today (see Consequences) -- because nothing retries a fit's way
     out of not covering the right range; whoever is fitting needs new data or
     different parameters before any result (MCMC, table-building, or
     otherwise) that used it can be trusted.

   Traced by hand first (NaN propagates correctly through the scan interior and
   the edge-guard's own decision logic for a transient probe; a genuinely
   accepted mode or fallback-grid bound reaches `check_valid_region` regardless
   of which branch produced it -- with one narrow, documented exception: a NaN
   neighbor during the scan can make the candidate-detection loop miss a mode
   sitting exactly at the last valid scan point, vanishingly unlikely to matter
   since the real near-threshold mode is never that close to `valid_max` in
   practice), then verified empirically (see Consequences): a subprocess
   registering a narrow-valid-region fit and querying it at a point whose true
   Erecoil falls outside that region is confirmed to die (nonzero exit, the
   explanatory message, no further output) end-to-end through the real
   `Ppq_NR` compute path, while the same handle queried well inside its valid
   region does not crash.

   What this does **not** do: prevent an *analysis* from ever being configured
   with a fit too narrow for the region it's actually using by *clipping* the
   compiled quadrature's own adaptive integration window at `valid_max` instead
   of erroring -- that would risk the opposite failure (silently under-
   integrating a fit's real tail mass, a wrong-but-quiet normalization) rather
   than loudly flagging a genuine problem, so it isn't done here.

5. **Ba's bump+line family and Cf ER's step-floor family are not ported into this
   module by this record** -- `erecoil_fit.py` still supports only the plain and
   turnon families. Those two remain ad hoc functions local to their own
   `nrFanoII_paper2022` notebooks, which is separately-tracked pending work. Their
   valid-region enforcement for now lives in the notebooks themselves (a local
   `ValidRegionError` class mirroring this exact mechanism/message shape), so
   nothing has to be re-derived once they're eventually ported here for real.

## Consequences

- `perecoil_identity()` gains `valid_min`/`valid_max` (both families) and
  `x_on`/`w_on` (turnon family only) as informational provenance fields.
  `valid_min`/`valid_max` are deliberately **not** part of `sha256` -- they're a
  usage bound, not the function's own mathematical identity, so two fits with
  identical `a`/`b`/`C` but different valid regions still hash identically (same
  reasoning the existing docstring already gives for detector/source).
  `x_on`/`w_on` **are** part of `sha256` for the turnon family -- two different
  `x_on`/`w_on` are genuinely different functions.
- `test/python/test_erecoil_fit.py`'s old "density stays positive well past
  `erecoil_max`" regression check is retargeted at the raw family function
  (`_density`) directly rather than `PErecoilFit.__call__`: that check is a
  statement about the underlying closed form's unconditional smoothness (what
  Fortran's quadrature needs), which is a different, still-true guarantee from
  whether `__call__` considers a query *trustworthy* there.
- **Fortran compilation verified via a stable toolchain, not the local one.**
  Local `gfortran` (Homebrew GCC 16.2.0) hits an internal compiler error
  (`gfc_resolve_forall`, `resolve.cc:12853`) compiling `Ppq_fort_s.f90` --
  confirmed to reproduce identically on the unmodified file too, so it's a
  pre-existing local toolchain issue, not something this record introduced or
  a reason to doubt the change. Built and tested instead via `Dockerfile_gfortran`
  (Ubuntu 26.04's default gfortran 15.2, the project's own documented build path):
  `fpm test` passes, the Python package installs and reports `5.0.0`, and
  `test/python/test_erecoil_fit.py`, `test_erecoil_table_fortran.py` (including
  the turn-on family's Fortran registration/eval cross-check against
  `Ppq_py.py`'s independent reference, and the new valid-region guard's NaN
  behavior end-to-end through the real `Ppq_NR` compute path) and
  `test_normgrid.py` all pass unmodified in that environment.
- **ABI bump stays at `4.0.0 -> 5.0.0`, not a second bump.** Both
  `PErecoil_fit_register` and `PErecoil_turnon_fit_register`'s signatures (two
  more trailing arguments each, `valid_min`/`valid_max`) changed again after the
  version was already bumped once for the turnon family -- fine, since nothing
  at `5.0.0` had shipped yet; this is all one in-progress change, not two
  released, incompatible versions.
- `erecoil_fit.PErecoilFit.fortran_params` now returns a 5-tuple
  (`a, b, C, valid_min, valid_max`), not 3 -- every existing call site either
  already unpacked it generically (`PErecoil_fit_register(*fit.fortran_params)`)
  and needed no change, or (one case, `normgrid.py`'s `_erecoil_handle_for`,
  which called `PErecoil_fit_register(a, b, C)` directly with hand-picked
  arguments, bypassing `fortran_params` entirely) needed a real fix: it now also
  reads `valid_min`/`valid_max` (and, for the turnon family, `x_on`/`w_on`) out
  of the spec's embedded `perecoil_identity()` and folds them into both the
  sha256 recomputation and the registration call. `_perecoil_from_fit_json`
  similarly forwards those same optional JSON keys through to `PErecoilFit`
  now, where it previously silently dropped them.
- `test/python/test_erecoil_table_fortran.py` tests the `error stop` path via a
  subprocess (an `error stop` can't be caught in-process the way a Python
  exception can, so the test spawns a child process and asserts it actually
  died -- nonzero exit, the expected message, no output past the crash point --
  plus a companion subprocess confirming a well-inside query does not crash).
- **Raises a bigger question this record doesn't resolve on its own**: the
  user pointed out that `normgrid.OutOfBoxError` and `PriorNotCoveredError`
  (both pre-existing, already-shipped mechanisms, not introduced here) arguably
  belong in the same "nothing retries its way out of this, so why let it be
  catchable" category as `check_valid_region` above -- and checking rather than
  assuming confirmed at least one of them already reaches deep into a real run,
  not just setup: `NormInterpolator.__call__` (`normgrid.py`) raises its own
  `OutOfBoxError` on *every* normalization lookup `Ppq_compiled.Ppq_integral_NR/
  ER` makes, i.e. every single likelihood evaluation, not once at MCMC setup.
  That's a reclassification of existing, shipped behavior, not just new-feature
  documentation, so it's out of scope for this record specifically -- tracked
  instead as a follow-up ADR (the general principle: setup-time or genuinely
  retry-fixable mid-run failures stay catchable; everything else, including
  these two, becomes uncatchable) plus a GitHub issue for the actual audit/
  conversion work (every existing raise/catch site, careful not to sweep in
  `GLConvergenceError`, which `goodness_of_fit`'s own retry logic deliberately
  depends on staying catchable). Also noted there: Python has no exact
  `error stop` equivalent -- `sys.exit()`/`SystemExit` is still catchable by a
  bare `except:`, so a genuinely uncatchable version needs `os._exit()`
  (skips `finally`/`atexit`/cleanup entirely), a real tradeoff to decide
  deliberately rather than default into.
