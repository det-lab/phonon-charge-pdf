# 0001. Batched Fortran quadrature instead of per-point `scipy.integrate.quad`

Status: Accepted
Date: 2026-09-11 (recorded retroactively in this file on 2026-10-30; see
`4db8e33 feat: doubling-verified Gauss-Legendre quadrature for PpqN/PpqG_region_adaptive`)

## Context

A likelihood fit needs, at every step as the fit/MCMC explores parameter space: the
density at each data point (over the full physical plane, not yet renormalized to the fit
region), and the PDF's integral over that region, to renormalize onto it.

Computing that region integral with `scipy.integrate.quad` costs several seconds per call,
because it evaluates the PDF one point at a time at the ~3-25 ms/point scalar cost of the
underlying Erecoil integration. That's out of the question inside an optimizer or MCMC
loop that calls it every step.

## Decision

`Ppq_compiled` (`python/Ppq_fort.py`) computes the region integral with a doubling-verified
nested Gauss-Legendre quadrature, implemented in Fortran as `Ppq_region_NR`/`Ppq_region_ER`
(`src/Ppq_fort_m.f90`, `src/Ppq_fort_s.f90`) and evaluated in one batched, thread-parallel
call rather than point-by-point from Python:

- The ridge location/width is already known exactly (`ridge_eq_and_width` in
  `src/Ppq_fort_s.f90`), so the quadrature needs far fewer points than a naive fixed grid
  for the same accuracy.
- `epsrel`/`epsabs` (exposed as `Ppq_compiled`'s `norm_epsrel`/`norm_epsabs`) control how
  tightly two successive doubled quadrature orders must agree before the result is
  trusted; it error-stops rather than returning an unverified number if the highest order
  tried still hasn't converged. See the `Ppq_region_NR`/`ER` doc comments in
  `src/Ppq_fort_m.f90` (rendered in the Fortran API reference) for the exact convergence
  rule and the implementation-level detail this record doesn't repeat.
- `Ppq_compiled` bundles the things that don't change across a fit/MCMC run -- the region,
  the observed dataset, and the convergence tolerance -- at construction, so every
  subsequent call only needs the physics parameters the fit is actually varying.

## Consequences

- Lands well under a second per call even for a wide region under `ifx`/`flang`, instead
  of several seconds -- the actual thing this decision was for.
- Requires the compiled Fortran library; there is no pure-Python path to a normalized
  likelihood (see `docs/two-ways.md`).
- `norm_epsrel`/`norm_epsabs` = 1e-4/1e-10 agreed with `scipy.integrate.quad` to ~1e-12
  relative across the regions tested in `test/python/test_region_integral.py`, including a
  247 keV-wide region -- the quadrature's accuracy isn't just assumed, it's checked
  against the thing it replaced.
- Even at well under a second per call, that's still too slow for MCMC, which calls the
  normalization every step -- hence precomputed normalization tables
  (`docs/normalization-tables.md`) as the next optimization on top of this one, not an
  alternative to it.
