# Generating simulated events

If you want simulated `(Ep, Eq)` detector events — to validate a fit
pipeline end to end, or to feed a downstream analysis — use `generate_events` (`fraction_ER=0.0`/`1.0`
for a pure-NR/pure-ER sample). It has **no Fortran dependency**: it's pure Python/numpy, so it
works in any environment this package is installed into, container or not.

```python
import phonon_charge_pdf as pdf

events = pdf.generate_events(
    target_size=20_000, fraction_ER=0.6, seed=8,
    ep_range=(2.5, 350.0), eq_range=(0.75, 200.0),
    k=0.18, Z=32.0, F0_NR=0.122, F0_ER=0.2, eps=3.0e-3, V=3.0,
    p0=0.06421907, p10=0.48998486, q0=0.23718488, q10=0.27093151,
    mix_frac_NR=0.53693208, scale_fast_NR=6.41515782, scale_slow_NR=23.71789286,
    mix_frac_ER=0.573211975, scale_fast_ER=0.169520023, scale_slow_ER=279.552394,
)
Ep, Eq = events["EP"].to_numpy(), events["EQ"].to_numpy()
```

`target_size` events are returned, cropped to `ep_range x eq_range` (your
real analysis ROI, the same box `Ppq_compiled` would be normalized over); the ER/NR split is
`Binomial(target_size, fraction_ER)`, so `fraction_ER=1.0`/`0.0` give deterministic pure-ER/pure-NR
samples and anything in between carries the same sampling fluctuation a real dataset would.
The returned `DataFrame` has columns `EP, EQ, ERECOIL, N, is_ER, seed, is_bkg` — `ERECOIL`
is the latent true recoil energy `Erecoil` (named to avoid reading as `is_ER`, a different
thing: "is this event an electron recoil?"). `is_bkg` is always `False` here; it exists
only so these rows can be concatenated directly with a separate zero-charge/surface-background
generator that produces the same columns (e.g. a KDE-based one kept in a downstream
analysis repo, out of scope for this library).

The same generative model this draws from — `Erecoil ~ PErecoil_NR`/`PErecoil_ER`, `N | Erecoil ~ TruncNormal`,
`(Ep, Eq) | Erecoil, N ~ Normal` — is what `Ppq_NR`/`Ppq_ER` are obtained by marginalizing
analytically over `N` and numerically over `Erecoil`; `test/python/test_chisquare_{nr,er}_simulator.py`
checks that events from this module land on the Fortran `Ppq_NR`/`Ppq_ER`, which is a real
physics check since this module never touches `Ppq_NR`/`Ppq_ER` itself. See
[Verification and validation](testing.md).

`generate_events_NR`/`generate_events_ER` (the single-species generators `generate_events`
itself is built on) are deliberately **not** exported: they have no ROI cropping of their
own, which is a real footgun if you reach for them directly (e.g. passing one as a
chi-square `sampler_func` silently produces inflated chi2 — see
[Testing goodness of fit](goodness-of-fit.md) and `run_chisquare_test`'s own docstring).
`generate_events(fraction_ER=0.0`/`1.0)` does everything they do, correctly cropped, with the
same columns. They're still importable directly from `phonon_charge_pdf.event_generator`
if you genuinely want the uncropped full physical model (e.g. studying the spectrum
itself, not a detector's cut of it) — see their docstrings there.

The resolution/yield model functions this is built on -- `Y`, `Nbar`, `sigp`, `sigq`,
`PErecoil_NR`, `PErecoil_ER` -- are also public, for analyzers who want one of them directly rather
than through `generate_events`; see the
[API reference](api/index.md#phonon_charge_pdf.Y) for the full list.
