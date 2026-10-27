# Generating simulated events

If you want simulated `(Ep, Eq)` detector events instead of PDF values — to validate a fit
pipeline end to end, or to feed a downstream analysis — `generate_events` (and the
lower-level `generate_NR_events`/`generate_ER_events`, implemented in `event_generator.py`)
is the entrypoint, independent of `Ppq_compiled`. Unlike `Ppq_compiled`, it has **no Fortran dependency**:
it's pure Python/numpy built on [`Ppq_py.py`](ppq-py.md), so it works in any environment
this package is installed into, container or not.

```python
import phonon_charge_pdf as pdf

events = pdf.generate_events(
    target_size=20_000, ER_frac=0.6, seed=8,
    ep_min=2.5, ep_max=350.0, eq_min=0.75, eq_max=200.0,
    k=0.18, Z=32.0, F0_ER=0.2, F0_NR=0.122, eps=3.0e-3, V=3.0,
    p0=0.06421907, p10=0.48998486, q0=0.23718488, q10=0.27093151,
    mix_frac_NR=0.53693208, scale_fast_NR=6.41515782, scale_slow_NR=23.71789286,
    mix_frac_ER=0.573211975, scale_fast_ER=0.169520023, scale_slow_ER=279.552394,
)
Ep, Eq = events["EP"].to_numpy(), events["EQ"].to_numpy()
```

`target_size` events are returned, cropped to `[ep_min, ep_max] x [eq_min, eq_max]` (your
real analysis ROI, the same box `Ppq_compiled` would be normalized over); the ER/NR split is
`Binomial(target_size, ER_frac)`, so `ER_frac=1.0`/`0.0` give deterministic pure-ER/pure-NR
samples and anything in between carries the same sampling fluctuation a real dataset would.
The returned `DataFrame` has columns `EP, EQ, N0, is_ER, seed, is_bkg` — `is_bkg` is
always `False` here; it exists only so these rows can be concatenated directly with a
separate zero-charge/surface-background generator that produces the same columns (e.g. a
KDE-based one kept in a downstream analysis repo, out of scope for this library).

The same generative model this draws from — `Er ~ PEr_NR`/`PEr_ER`, `N | Er ~ TruncNormal`,
`(Ep, Eq) | Er, N ~ Normal` — is what `Ppq_NR`/`Ppq_ER` are obtained by marginalizing
analytically over `N` and numerically over `Er`; `test/python/test_chisquare_{nr,er}_simulator.py`
checks that events from this module land on the Fortran `Ppq_NR`/`Ppq_ER`, which is a real
physics check since this module never touches `Ppq_NR`/`Ppq_ER` itself. See
[Verification and validation](testing.md).

For single-species sampling without the ROI retry/mixing logic (e.g. for a custom
pipeline), the lower-level `generate_NR_events`/`generate_ER_events` are also public — see
their docstrings, or the [API reference](api/index.md#phonon_charge_pdf.generate_events).
