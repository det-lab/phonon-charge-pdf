# Fitting

**If you want normalized PDF values or a likelihood** — a fit, an MCMC, anything beyond
generating or describing simulated events — `Ppq_compiled` is the one entrypoint. It
always needs the compiled Fortran library; see [Two ways to use this](two-ways.md) for
how to get it (pip install, a published container, or
[building from source](building-from-source.md)). See the
[API reference](api/index.md#phonon_charge_pdf.Ppq_compiled) for the full method list.

```python
import phonon_charge_pdf as pdf

# define the data region: your analysis ROI in keV -- the same (Ep, Eq)
# window your data is already cut to. 2.5-350 / 0.75-200 is this project's
# region; use YOUR actual cut, not this one, unless it happens to match.
ep_min, ep_max = 2.5, 350.0
eq_min, eq_max = 0.75, 200.0

likelihood = pdf.Ppq_compiled(ep_min, ep_max, eq_min, eq_max, ep_data, eq_data,  # ep_data/eq_data: your measured events, keV
                      norm_epsrel=1e-4, norm_epsabs=1e-10)
normalized = likelihood.Ppq_normalized_values_NR(k=0.18, Z=32.0, F0=0.122, eps=3.0e-3, V=3.0,
                                                  p0=0.06421907, p10=0.48998486, q0=0.23718488, q10=0.27093151)
```

The region is the box the normalization integral is computed over — it has to be your real
analysis ROI, not an arbitrary wide range, or the normalization (and so the likelihood) is
wrong. **If you use precomputed tables (`Ppq_table_NR=`/`Ppq_table_ER=`, see
[Computing new normalization tables](normalization-tables.md)) and your region doesn't
match the one a table was built for exactly, `Ppq_compiled` raises at construction**
rather than silently normalizing wrong — the fix is to build a new table for your region.

Same shape for the ER band (`Ppq_normalized_values_ER`, no `k`/`Z` — `Y=1` there — also
`Ppq_values_ER`/`Ppq_integral_ER` if you want the unnormalized value and the normalization
separately). Build `likelihood` once, outside the fit loop; call its methods once per step,
inside.

## Why a batched Fortran call instead of `scipy.integrate.quad`

A likelihood fit needs the *un-normalized* PDF at each data point and the PDF's integral
over the fit region (to normalize it) — both re-evaluated at every step as the fit
explores parameter space. Computing that normalization with `scipy.integrate.quad` is
several seconds per call (it evaluates the PDF one point at a time, at the ~3-25 ms/point
*scalar* cost). `Ppq_compiled` instead uses a doubling-verified nested Gauss-Legendre quadrature
evaluated in one batched, thread-parallel call (`Ppq_region_NR`/`Ppq_region_ER` in Fortran —
the ridge location/width is already known exactly, so it needs far fewer points than a
naive grid for the same accuracy), landing well under a second per call even for a wide
region under `ifx`/`flang`.

`Ppq_compiled` bundles the things that *don't* change across a fit run — the region, the
observed dataset, and the normalization quadrature's convergence tolerance — at
construction, so every subsequent call only needs the physics parameters that the fit is
actually varying:

```python
import numpy as np
import phonon_charge_pdf as pdf

ep_min, ep_max = 2.5, 350.0
eq_min, eq_max = 0.75, 200.0

likelihood = pdf.Ppq_compiled(ep_min, ep_max, eq_min, eq_max, ep_data, eq_data,
                      norm_epsrel=1e-4, norm_epsabs=1e-10)
# norm_epsrel/norm_epsabs set how tightly two successive doubled quadrature
# orders must agree before the normalization integral is trusted -- NOT
# related to len(ep_data). 1e-4/1e-10 agreed with scipy.integrate.quad to
# ~1e-12 relative across the regions tested in test_region_integral.py,
# including a 247 keV-wide region.

def loglike(k, Z, F0, eps, V, p0, p10, q0, q10):
    normalized = likelihood.Ppq_normalized_values_NR(k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
    return np.sum(np.log(normalized))
```

## Performance notes

`Ppq_NR`/`Ppq_ER` integrate over a window placed around the located peak(s) of the Er
integrand rather than sampling the full physical range, evaluating at roughly 6
microseconds per (Ep, Eq) point in vector mode (18 threads under x86 emulation; measured
via `Ppq_vector_NR` over a representative grid). This number assumes an ifx or flang build;
gfortran builds run the band integrals serially and are ~25x slower (see the compiler
notes in [Building from source](building-from-source.md)). For likelihood loops (fitting
or [MCMC](mcmc.md)):

- Call the vectorized entry points (`Ppq_vector_NR`/`Ppq_vector_ER`) with all events in one
  call — the parallelism lives there, and per-event scalar calls pay OpenMP fork/join
  overhead instead.
- **Shuffle the event array once at load time if it is ordered.** The vector loops split
  the events into one contiguous chunk per thread (static scheduling), and the loop only
  finishes when the slowest chunk does. Per-event cost varies several-fold across the
  (Ep, Eq) plane — deep-tail events short-circuit in ~2 us while on-band events cost
  ~10-15 us — so an energy-ordered array hands some threads chunks of expensive events
  while others idle at the barrier. Shuffling measured 8-15% faster than energy-ordered
  input, with an identical result either way.

## Maximum likelihood / MAP fits

At ~1 second per call, computing the normalization integral at every optimizer step is
usually too slow for anything but a handful of evaluations -- see
[Computing new normalization tables](normalization-tables.md) for building and using
tables that make this tens of microseconds per step instead, the same way
[MCMC](mcmc.md) needs to.

- Give the optimizer bounds inside the tables (e.g. `bounds=` for `scipy.optimize.minimize`
  with L-BFGS-B, Powell or trust-constr), and check them once with `prior_bounds` or
  `table_bounds()` (see [Using MCMC to get errors](mcmc.md) for what these check).
- Box bounds cannot express `p10 >= p0`. Either fit `dp = p10 - p0` and `dq = q10 - q0`
  directly, bounded by the tables' own `dp`/`dq` axes (`likelihood.Ppq_table_NR.box["dp"]`,
  0.18–0.59 by default; `dq` 0–0.353), and convert back when calling the PDF, or add the
  inequalities as constraints (trust-constr, SLSQP).
- A fit that ends on a bound is being limited by the box, not the data: widen it with
  `BOX=` and rebuild the tables.
