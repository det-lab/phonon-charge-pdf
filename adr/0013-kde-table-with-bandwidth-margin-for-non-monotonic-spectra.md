# 0013. A curated KDE table, with an enforced bandwidth margin, for recoil spectra a fit can't represent

Status: Proposed
Date: 2026-11-13

## Context

`adr/0012-perecoil-handles-register-a-fit-not-a-table.md` removed table
registration from the compiled quadrature path entirely, replacing it with
`PErecoil_fit_register`: a sum of exponentials plus a constant, the one
family `erecoil_fit.FAMILY_FORMULAS` supports. That family is, by
construction, monotonically non-increasing for `Erecoil >= 0` (every term
and the constant floor are non-negative). It cannot represent a spectrum
with a real local maximum away from `Erecoil = 0`, no matter how it's fit.

Ba-133's own ER calibration data has exactly that: a broad, genuine bump
around 120-160 keV in every one of the 14 surveyed detectors. It is
documented, not newly discovered here --
`nrFanoII_paper2022/scripts/extract_erecoil_tables_ba.py`'s own docstring
identifies it as leftover Ge-75m (139.7 keV) neutron-activation
contamination from a prior Cf-252 run on the same crystals, confirmed by
three independent lines of evidence, and **deliberately left unremoved**
after several real removal attempts (hard cuts, background+Gaussian
subtraction, an sPlot-style mixture fit, asymmetric peak shapes) each
introduced worse, less-understood artifacts than the contamination itself.
Since adr/0012, there has been no way to register this spectrum with the
compiled quadrature at all -- not "it fits badly," but no representation in
`FAMILY_FORMULAS` that could describe it over its full range regardless of
tuning.

**Is a table really unsafe, or was the real problem something narrower?**
adr/0010's and adr/0012's own measurements showed a raw PCHIP table's
relative residual reaching ~8e-4 at the phonon resolution model's
degenerate edge (`dp = p10 - p0 -> 0`), and a Poisson-noise-matched
smoothing spline still reaching ~1.7e-7 there. Both investigations
concluded "smoothing removes counting noise, not the integrand's real
local curvature, which is what the floor tracks once the window narrows."
That's correct, but it conflates two different things: *how much noise a
representation has* and *how wide its local features are relative to the
quadrature's own narrowest window*. The second is the one that actually
matters, and it's a concrete, checkable number: this project's own
realistic `p0`/`q0` floor (`normgrid.P0_MEAN`/`Q0_MEAN` and their prior box,
`(0.2, 1.8) x` mean -- roughly 0.013-0.12 keV for `p0` here) sets the
narrowest window `locate_modes` will ever actually use.

Measured directly, on real detector 402 data, registering a **Silverman-
reflected Gaussian KDE** (reflection corrects the well-known KDE boundary
bias: an ordinary kernel leaks density past `Erecoil = 0` when real events
sit right at the boundary, understating the true one-sided density there --
see `adr/0013_kde_vs_data.py`) with **bandwidth = 20 keV** (~170-1500x that
`p0`/`q0` floor) through `Ppq_region_NR/ER_with_error`, scanning both
degenerate edges (`dp -> 0` and `dq = q10 - q0 -> 0`):

| dataset | dp-scan worst / best | dq-scan worst / best |
|---|---|---|
| NR-Cf (402) | 3.0e-12 / 1.2e-13 | 6.7e-13 / 3.6e-13 |
| ER-Cf (402) | 2.4e-11 / 1.9e-13 | 4.3e-12 / 1.3e-13 |
| ER-Ba (402) | 2.3e-11 / 2.8e-13 | 4.1e-12 / 1.5e-13 |

All three land in the same 1e-11 to 1e-13 ballpark as the fitted family's
own measured floor (`adr/0012`: worst 1.3e-12, best 1.1e-14) -- orders of
magnitude past PCHIP's 8e-4 and the smoothing spline's 1.7e-7, with no
residual `dp`/`dq`-dependence left to speak of. (NR-Cf's own first pass at
this, using an oversized `Ep` range borrowed from ER's much wider physical
extent, showed a spurious 8e-7 floor at large `dp` -- not the degenerate-
edge mechanism at all, just the integration window at large `dp` running
past where the table was ever registered. It disappeared entirely once NR's
own established region, `adr/0012`'s `(3, 80) x (0.5, 30)`, was used
instead -- a reminder that any table's validation has to be checked against
the *actual* region an analysis uses, not a generic one.)

**Bandwidth is a real tradeoff, not a free parameter.** Pushing it wide
enough to be numerically safe also risks smoothing away the very feature
being preserved. Measured on NR-Cf specifically (`adr/0013_kde_vs_data.py`,
`test/figures/adr0013_kde_vs_data.png`): its own near-threshold rise decays
on a ~6.4 keV scale (its closed-form biexponential's own short decay
length), and even a bandwidth as small as 3-5 keV visibly under-represents
it -- a margin of only ~30-50x over `p0`'s floor, well short of the
~150-1500x this record's own measurements validate as safe. ER-Cf and
ER-Ba's real features are broad enough (tens of keV to the ~100 keV Ge-75m
bump) that 20 keV costs them nothing. **Conclusion: NR-Cf and ER-Cf's
existing fitted-family solution (`adr/0012`) remains strictly better there**
-- exact machine-precision parity *and* a faithful near-threshold shape,
with no bandwidth tradeoff to manage at all. A KDE table only earns its
keep for a spectrum the fitted family structurally cannot represent, which,
of the cases surveyed, is Ba's ER band alone.

**A second, independent gap this surfaced**: `Ppq_NR`/`Ppq_ER` (and their
vectorized `Ppq_vector_NR`/`Ppq_vector_ER`) have no error-reporting
mechanism at all. Unlike `Ppq_region_NR/ER_with_error`'s doubling-verified
Gauss-Legendre residual, the per-point Erecoil integration
(`integrate_Ppq` -> `locate_modes` + `integrate_piecewise`) uses a
fixed-resolution trapezoid rule with an edge-decay guard -- no internal
doubling, no convergence check, no residual to report. That was fine while
every representation was globally smooth with a floor always negligible
regardless of resolution parameters (the closed form, the fitted family);
it stops being fine the moment a representation's own local curvature is a
deliberately chosen, bandwidth-dependent knob that trades precision against
feature fidelity -- exactly what this record introduces. Today,
`make_Ppq_densities_NR/ER` and `Ppq_densities.density_NR/ER` (used directly
by `goodness_of_fit`'s chi-square and KS testing) have no way to flag
degraded per-point precision at all.

## Decision

1. **A PErecoil handle may register a KDE table, not only a fit** -- a new
   handle kind (Fortran: alongside `fit_registry`/`eval_fit`, a
   `kde_registry`/`eval_kde`, same append-only, per-instance, thread-safe
   pattern as `adr/0003`; Python: a new `erecoil_kde.py`, parallel to
   `erecoil_fit.py`), **registered only with an enforced minimum bandwidth
   margin over the caller's own `p0`/`q0` prior floor** -- construction
   raises, rather than silently registering an unsafe table, if the
   bandwidth doesn't clear that margin. The exact multiplier is this
   record's own measurement (100x as a starting point: comfortably above
   the ~20x margin that showed one isolated residual, well below the point
   -- ~30-50x -- where NR-Cf's own sharp features started visibly
   degrading), not a value to treat as settled without revisiting it
   against more detectors/bands. The KDE itself uses Silverman's reflection
   correction at `Erecoil = 0` unconditionally -- there is no case where
   the uncorrected boundary leakage this record measured is preferable.

2. **`Ppq_NR`/`Ppq_ER` and `Ppq_vector_NR`/`Ppq_vector_ER` gain
   `_with_error` variants**, matching `Ppq_region_NR/ER_with_error`'s
   existing pattern: a genuine convergence diagnostic for the per-point
   density path, not just the region integral. This needs
   `integrate_Ppq`/`integrate_piecewise`'s fixed trapezoid rule to actually
   verify its own convergence (e.g. doubling the per-window point count and
   comparing, the same spirit as `region_integral_gl`'s order doubling, not
   just the existing edge-decay guard, which only checks that a window's
   *boundary* has decayed enough, never that its *interior* resolution was
   adequate) -- a real numerical change to the core quadrature, not a thin
   wrapper. `make_Ppq_densities_NR/ER`/`Ppq_densities.density_NR/ER` should
   take an `return_error=` option that surfaces this once it exists, so
   goodness-of-fit testing can flag a degraded KDE (or any future
   representation) before it silently corrupts a chi-square or KS result.

Both are Proposed, not yet implemented: this record documents the
measurement and the design constraint it implies, not a finished change.

## Consequences

- Reopens, narrowly, the thing `adr/0012` closed: a handle can once again
  point at something other than the one curated fit family. The guardrail
  `adr/0012` didn't have (and the reason removing table support entirely
  was the right call *then*) is the enforced bandwidth-margin check in
  point 1 -- without it, this is exactly the "a table can represent
  anything, including something numerically hostile" problem `adr/0012`
  already rejected once.
- `erecoil_fit.py` remains the right, and simpler, choice for any
  genuinely monotonic spectrum (every NR band surveyed, and ER-Cf, which
  this record's own figure shows has no feature a KDE is needed for) --
  this decision does not suggest migrating those to a KDE table.
- Real follow-on work, not yet scoped in detail: the Fortran
  `kde_registry`/`eval_kde` path itself; `erecoil_kde.py`'s bandwidth-margin
  validation (needs the caller's actual `p0`/`q0` prior box, the same
  information `Ppq_compiled`'s own `prior_bounds` check already has);
  `normgrid`'s `perecoil` provenance gaining a `"kde"` kind alongside
  `"closed_form"`/`"fit"`; the `integrate_Ppq` doubling-verification
  itself (point 2), which touches the core per-point quadrature, not a
  peripheral path.
- Verification for this record's own measurement:
  `adr/0013_recoil_events_402.npz` (real, cleaned, unbinned Erecoil events
  for detector 402, NR-Cf/ER-Cf/ER-Ba, extracted once via
  `nrFanoII_paper2022`'s own extraction scripts -- self-contained, no
  cross-repo checkout needed to reproduce anything here), `adr/0013_kde_vs_data.py`
  (the reflected-KDE-vs-data figure,
  [`test/figures/adr0013_kde_vs_data.png`](../test/figures/adr0013_kde_vs_data.png)),
  `adr/0013_kde_dp_scan.py` (the dp/dq-scan table above -- needs a checkout
  with `PErecoil_table_register`, e.g. `ab45bb3`, since `adr/0012`'s own
  implementation removed it).
