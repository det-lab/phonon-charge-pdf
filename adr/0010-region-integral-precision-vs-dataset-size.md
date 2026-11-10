# 0010. Region-integral quadrature precision isn't verified against dataset size, for a tabulated PErecoil

Status: Proposed
Date: 2026-11-09

## Context

`feature/erecoil-table-pdf` added `Ppq_region_NR_with_error`/`ER_with_error`, reporting
the Gauss-Legendre doubling residual alongside the region integral -- a numerical
diagnostic for exactly the question this record raises: is a *converged* region integral
actually precise enough, or just inside the tolerance it happened to be asked for?

Measured, not assumed:

- For the closed-form `PErecoil`, that residual is utterly negligible (~4e-13 relative) at
  any realistic dataset size.
- For a real, tabulated `PErecoil`, the achieved relative precision has a genuine floor set
  by the table's own small-scale statistical roughness: order-256 Gauss-Legendre doesn't
  resolve past it no matter how tight `epsrel`/`epsabs` is asked for (measured ~6e-5
  relative floor on the roughest surveyed NR table, over a box smaller than a real analysis
  region -- asking for `1e-5`, `1e-6`, `1e-7` all failed to converge, plateaued at the same
  residual).
- That floor matters once `N * (relative error)` approaches order 1 -- the natural scale of
  a one-sigma change in profile likelihood/posterior width, since the region integral `Z`
  enters a likelihood as `N*log(Z)`. Measured crossover for that table/box: ~16,000 events.
  Ba calibration datasets reach ~100k counts -- past that threshold, not near it.
- This isn't only about the likelihood's absolute value: any predicted/expected count (a
  chi-square test's per-bin expectation, a predicted calibration-line count) is itself a
  region integral of the same density, so it inherits the identical floor.
- Precomputed normalization tables (`normgrid.py`, see
  `0002-normalization-tables-record-perecoil.md`) don't avoid this: each grid point in such
  a table is itself one `Ppq_region_NR`/`ER` call, inheriting the same floor, plus whatever
  additional error the Chebyshev interpolation between grid points adds on top. Nothing
  currently measures that combined effect.
- A second, likely more consequential source of imprecision is currently completely
  uninstrumented: the per-event density term (`Σᵢ log(densityᵢ)`, evaluated via
  `integrate_Ppq`'s fixed-window trapezoid rule inside `Ppq_vector_NR`/`ER`) has no
  convergence check or error estimate at all -- unlike the region integral, it was never
  doubling-verified in the first place. For a large dataset this sum has far more terms than
  the single normalization call, so it could dominate the picture this record is about.
- Whether any of this actually biases a real fit depends on something unmeasured: whether
  the residual is roughly constant across the parameter region a fit/MCMC explores (mostly
  harmless -- a near-constant shift cancels in the likelihood differences inference actually
  uses) or varies with the physics parameters (not harmless -- it would distort the *shape*
  of the likelihood surface, not just its overall scale, which can bias a best fit or
  manufacture spurious structure, not just widen a posterior).

## Decision

1. **Don't change the default return contract.** `Ppq_region_NR`/`ER` (and
   `Ppq_vector_NR`/`ER`, `make_Ppq_densities_NR`/`ER`) keep returning a plain value, not a
   `(value, error_estimate)` tuple. The opt-in pattern already shipped
   (`Ppq_region_NR_with_error`/`ER_with_error`) stays the way to get the diagnostic: the
   closed form's error is negligible for every existing caller (`Ppq_compiled`,
   `normgrid.py`, every script in `docs/`), so forcing all of them to unpack a tuple pays a
   real API cost for a case that, for the closed form, doesn't need paying.
2. **The per-event term's analog is explicitly out of scope here.** Instrumenting it would
   need `integrate_Ppq`'s fixed trapezoid window to become doubling-verified (or some other
   independent error estimator) -- a bigger change than this record makes. Recorded here so
   a future investigation doesn't assume that instrumentation already exists.
3. **No decision yet on how a measured error should feed back into a fit's reported
   uncertainty** (e.g. added in quadrature to the statistical error, or used to set a
   per-detector minimum `epsrel` below which a table is flagged unusable for that dataset
   size). That's a statistical question for whoever runs the investigation this record asks
   for, not a code change this record makes.

## Consequences

- Today's tooling is enough to check the normalization term's contribution for a specific
  detector/table/region before trusting a fit that uses a tabulated `PErecoil` with a large
  dataset (`Ppq_region_NR_with_error`/`ER_with_error` at the fit's actual region and
  tolerance) -- doing that is an explicit action for a Ba-scale analysis, not automatic, and
  nothing currently prompts anyone to do it.
- The per-event term's contribution cannot yet be checked at all; a thorough investigation
  needs that instrumentation first, and it may turn out to matter more than the
  normalization term measured here.
- This is an open question, not a resolved one: whether the residual is parameter-dependent
  (biasing, not just inflating uncertainty) is unmeasured.
- Suggested verification for the future work this record is scoping (none of it done yet):
  1. Instrument `integrate_Ppq` analogously to `region_integral_gl` (two trapezoid
     resolutions, or a real adaptive/doubling scheme, reporting the discrepancy).
  2. For a real table, scan `error_estimate / value` from `Ppq_region_NR_with_error`/`ER`
     across a grid of physics parameters spanning the fit's prior box, and check whether
     it's roughly flat (benign) or varies meaningfully (not).
  3. For a specific Ba dataset, propagate the measured residual at the actual best-fit point
     into the reported log-likelihood/expected counts and compare against the statistical
     uncertainty already being reported, to see whether it's actually sub-dominant there or
     not.
