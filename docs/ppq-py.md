# Resolution/yield model

The resolution/yield model functions (`sigp`, `sigq`, `Y`, `Nbar`, and friends below) are
implemented in `Ppq_py.py`, the pure-Python reference implementation that
[`event_generator.py`](python-events.md) and the compiled Fortran (`Ppq_NR`/`Ppq_ER`) are
both cross-checked against (`test/python/test_chisquare_{nr,er}_simulator.py`,
`test_chisquare_ppqn.py`, `test_PpqFort.py`). It has **no Fortran dependency** — only
numpy/scipy — so it works anywhere this package is installed.

## The resolution/yield model functions (most common case)

```python
import phonon_charge_pdf as pdf

# Phonon-energy resolution, sigma_p(Ep)
sp = pdf.sigp(Ep, eps=eps, V=V, p0=p0, p10=p10)

# Charge-energy resolution, sigma_q(Eq)
sq = pdf.sigq(Eq, q0=q0, q10=q10)

# Ionization yield Y(Er) and mean e/h pair count Nbar(Er)
y    = pdf.Y(Er, k=k, Z=Z)
nbar = pdf.Nbar(Er, k=k, Z=Z, eps=eps, is_ER=False)
```

The true-recoil-energy spectra (`PEr_NR`/`PEr_ER`) are public the same way — see the
[API reference](api/index.md#phonon_charge_pdf.Y) for the full list.

## The brute-force PDF scan (diagnostic only)

`Ppq_inspect_NR`/`Ppq_inspect_ER` are a slower, Python-only way to get `Ppq_NR`/`Ppq_ER`
values directly: instead of the Fortran side's analytic peak location and windowed
quadrature, they scan the *whole* plausible Er range on a uniform grid, locate the
integrand's peak(s) on that scan, and integrate each with `scipy.integrate.quad` — more
exhaustive, and much slower:

```python
import phonon_charge_pdf as pdf

(Ppq_NR, quad_error), (er_scan, integrand_scan), peaks = pdf.Ppq_inspect_NR(
    Ep=10.0, Eq=5.0,
    k=0.18, Z=32.0, F0=0.122, eps=3.0e-3, V=3.0,
    p0=0.06421907, p10=0.48998486, q0=0.23718488, q10=0.27093151,
    er_step=0.1,   # Er scan step, keV -- see "choosing er_step" below
)
```

!!! warning "Deliberately the slow path"
    Never use this in a fit/MCMC inner loop. Use [`Ppq_compiled`](ppq-fort.md) for a
    fit/MCMC; use this to independently verify `Ppq_compiled`/`event_generator.py` agree with
    the underlying physics, or when you want the scan grid and per-peak width/location
    diagnostics (`peaks` above — one dict per located peak) that `Ppq_compiled` doesn't expose.

### Choosing `er_step`

`er_step` (the Er scan step, in keV) has to be finer than the narrowest peak you expect: ER
peaks can be as narrow as the zero-energy ionization resolution `q0` (~0.06 keV for typical
CDMS parameters), so e.g. `er_step=0.01` for `Ppq_inspect_ER` is finer than the 0.01 keV
grid the Fortran `Ppq_ER` itself integrates on.
