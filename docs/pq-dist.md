# Resolution/yield model: `pq_dist.py`

`pq_dist.py` is the pure-Python reference implementation that [`event_generator.py`](python-events.md)
and the compiled Fortran `PpqN`/`PpqG` are both cross-checked against
(`test/python/test_chisquare_{nr,er}_simulator.py`, `test_chisquare_ppqn.py`,
`test_PpqFort.py`). It has **no Fortran dependency** — only numpy/scipy — so it works
anywhere this package is installed.

## The resolution/yield model functions (most common case)

As of version 2.1.0, the commonly-used pieces are re-exported directly on the top-level
package, with a stable public API — no need to import the `pq_dist` submodule yourself:

```python
import phonon_charge_pdf as pdf

# Phonon-energy resolution, sigma_p(Ep)
sp = pdf.sigp(Ep, eps=eps, V=V, p0=p0, p10=p10)

# Charge-energy resolution, sigma_q(Eq)
sq = pdf.sigq(Eq, q0=q0, q10=q10)

# Ionization yield Y(Er) and mean e/h pair count Nbar(Er)
y    = pdf.Y(Er, k=k, Z=Z)
nbar = pdf.Nbar(Er, k=k, Z=Z, eps=eps, is_gamma=False)
```

The true-recoil-energy spectra (`PErN`/`PErG`) and the per-Er unnormalized densities
(`PpqFullN`/`PpqFullG`) are public the same way — see the
[API reference](api/pq_dist.md) for the full list.

## The brute-force PDF scan (diagnostic only)

`PpqN_inspect`/`PpqG_inspect` are a slower, independent way to get `PpqN`/`PpqG`
values directly, by scanning a uniform grid to locate the Er integrand's peak(s) and
integrating each with `scipy.integrate.quad`, instead of the Fortran side's analytic peak
location:

```python
import phonon_charge_pdf as pdf

(PpqN, quad_error), (er_scan, integrand_scan), peaks = pdf.PpqN_inspect(
    Ep=10.0, Eq=5.0,
    k=0.18, Z=32.0, F0=0.122, eps=3.0e-3, V=3.0,
    p0=0.06421907, p10=0.48998486, q0=0.23718488, q10=0.27093151,
    er_step=0.1,   # Er scan step, keV -- see "choosing er_step" below
)
```

!!! warning "Deliberately the slow path"
    Never use this in a fit/MCMC inner loop. Use [`pqPDF`](fortran-pdf.md) for a
    fit/MCMC; use this to independently verify `pqPDF`/`event_generator.py` agree with
    the underlying physics, or when you want the scan grid and per-peak width/location
    diagnostics (`peaks` above — one dict per located peak) that `pqPDF` doesn't expose.

### Choosing `er_step`

`er_step` (the Er scan step, in keV) has to be finer than the narrowest peak you expect: ER
peaks can be as narrow as the zero-energy ionization resolution `q0` (~0.06 keV for typical
CDMS parameters), so e.g. `er_step=0.01` for `PpqG_inspect` is finer than the 0.01 keV
grid the Fortran `PpqG` itself integrates on.

Vectorized forms (`PpqN_inspect_vec`/`PpqG_inspect_vec`) evaluate paired
`(Ep, Eq)` arrays and return just the values, dropping the scan grid and per-peak
diagnostics.
