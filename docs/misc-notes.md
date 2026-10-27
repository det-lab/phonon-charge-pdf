# Miscellaneous notes

Small facts that don't yet have a better home in these docs.

## Importing never touches the compiled library

`import phonon_charge_pdf` never touches the compiled Fortran library -- only
constructing a `Ppq_compiled` does, and that fails there, at construction, rather
than on the first likelihood call deep into a fit or MCMC run. `event_generator.py`
and `Ppq_py.py` have no Fortran dependency at all, so they work in any environment
this package installs into, container or not.

## Why a batched Fortran call instead of `scipy.integrate.quad`

A likelihood fit needs the density at each data point (over the full physical plane, not
yet renormalized to the fit region) and the PDF's integral over that region (to renormalize
onto it) — both re-evaluated at every step as the fit explores parameter space. Computing
that normalization with `scipy.integrate.quad` is several seconds per call (it evaluates
the PDF one point at a time, at the ~3-25 ms/point *scalar* cost). `Ppq_compiled` instead
uses a doubling-verified nested Gauss-Legendre quadrature evaluated in one batched,
thread-parallel call (`Ppq_region_NR`/`Ppq_region_ER` in Fortran — the ridge location/width
is already known exactly, so it needs far fewer points than a naive grid for the same
accuracy), landing well under a second per call even for a wide region under `ifx`/`flang`.

`Ppq_compiled` bundles the things that *don't* change across a fit run — the region, the
observed dataset, and the normalization quadrature's convergence tolerance — at
construction, so every subsequent call only needs the physics parameters that the fit is
actually varying. `norm_epsrel`/`norm_epsabs` (1e-4/1e-10 is a reasonable default) agreed
with `scipy.integrate.quad` to ~1e-12 relative across the regions tested in
`test_region_integral.py`, including a 247 keV-wide region.

## Performance notes

`Ppq_NR`/`Ppq_ER` integrate over a window placed around the located peak(s) of the Er
integrand rather than sampling the full physical range, evaluating at roughly 6
microseconds per (Ep, Eq) point in vector mode (18 threads under x86 emulation; measured
via `Ppq_vector_NR` over a representative grid). This number assumes an ifx or flang build;
gfortran builds run the band integrals serially and are ~25x slower (see the compiler
notes in [Building from source](building-from-source.md)). For likelihood loops ([fitting](fitting.md)
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
