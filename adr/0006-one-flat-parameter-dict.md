# 0006. One flat parameter dict through every Ppq_compiled method

Status: Proposed
Date: 2026-11-09

## Context

`Ppq_compiled` uses two parameter conventions:

- **Flat, band-suffixed:** `prior_bounds`, `check_points` and `table_bounds` take
  or return `{"k", "F0_NR", "F0_ER", "V", "p0", "p10", "q0", "q10", ...}`. This is
  the same convention as `generate_events`, and it's what a fit's θ naturally
  unpacks into.
- **Per band, bare `F0`:** `Ppq_normalized_densities_NR/ER`, `Ppq_integral_NR/ER`
  take keyword-only `F0`, plus `k`/`Z` for NR only, and reject any keyword
  they don't use.

So every fit has to translate between them, in every likelihood evaluation.
Writing `MCMC_demo/model.py`'s ER+NR mixture took a hand-built `shared` dict, a
renaming of `F0_NR`→`F0` and `F0_ER`→`F0`, and two call sites listing the same
six parameters. `docs/fitting.md` notes "the only place the rename has to
happen". A joint fit (Cf mixture + Ba ER) multiplies this per dataset, and the
translation is where mistakes hide: passing `F0_NR` to the ER band is a
plausible, silent physics error (Joseph's Expanse script did exactly that).

## Decision

The density and integral methods also accept the flat dict, as the primary form:

```python
p = {"k": ..., "Z": 32, "eps": 3e-3, "F0_NR": ..., "F0_ER": ..., "V": ...,
     "p0": ..., "p10": ..., "q0": ..., "q10": ..., "efrac": ...}
fit_pdf.Ppq_normalized_densities_NR(p)      # uses k, Z, F0_NR, eps, V, p0..q10
fit_pdf.Ppq_normalized_densities_ER(p)      # uses F0_ER, eps, V, p0..q10
```

- Each method picks the keys its band uses (`_band_view`, which already does this
  for `check_points`) and **ignores** the rest, so the same dict, including
  analysis-only parameters like a mixture fraction, goes everywhere.
- Required keys are still required: a missing `F0_ER` for the ER call raises
  `KeyError` naming it. A bare `F0` in the flat form is rejected as ambiguous.
- The existing keyword form keeps working unchanged (no deprecation needed now).

Alongside this, `docs/fitting.md` and `docs/mcmc.md` replace "a mixture model ...
is specific to your own analysis" with a short worked mixture and joint-fit
section (link `MCMC_demo/model.py`). Every Cf calibration fit is a mixture, and
Cf + Ba is a joint fit, so this is the common case, not an edge case.

## Consequences

- A fit's likelihood becomes `unpack(theta)` once, then plain calls. The F0
  cross-wiring mistake becomes impossible in the flat form.
- "Ignore unknown keys" means a typo (`F0_Nr`) in an *extra* key is ignored
  rather than caught, but a typo in a *required* key still raises. That's the
  usual trade-off for dict-style APIs, and the required keys are the ones that
  matter.
- Verification: the flat and keyword forms return identical arrays for both
  bands; the flat ER call with `F0_NR` only (no `F0_ER`) raises.
