# 0005. OpenMP is the thread layer: `n_workers` defaults to 1, plus `set_threads()`

Status: Proposed
Date: 2026-11-09

## Context

One posterior evaluation can be parallelized over the events in two layers:

1. **OpenMP inside the Fortran** (`do concurrent` compiled with `-qopenmp` /
   `-fdo-concurrent-to-openmp=host`), controlled by `OMP_NUM_THREADS`;
2. **Python threads** in `_make_threaded_pdf`, which sort the events by cost,
   deal them into chunks and call the library once per chunk from a
   `ThreadPoolExecutor` (`n_workers`).

Measured on one Expanse node (128 cores, ifx build, 10,000 Cf events, both
bands, `MCMC_demo/benchmark.py`):

| layer | 8 | 32 | 128 threads |
|---|---|---|---|
| OpenMP only (`n_workers=1`) | 7.8× | 30× | **102×** |
| Python threads only (`OMP_NUM_THREADS=1`) | 7.8× | 27× | 27× |

OpenMP scales almost linearly. The Python layer plateaus around 27×, and at
1,000 events it's *slower* at 128 threads than at 32.

The defaults combine them badly. `Ppq_compiled`, `make_Ppq_densities_NR/ER`, and
(on `feature/erecoil-table-pdf`) `Ppq_densities` default `n_workers=None`, which
becomes `os.cpu_count()`. Each of those threads calls into a library whose
OpenMP also defaults to every core. On a 128-core node that's 128 × 128
threads; at 1,000 events it measured 20× slower than OpenMP alone. Inside an
MCMC with one process per walker, it's 128 processes each starting 128² threads.

There's also no API for the OpenMP count. It's only read from
`OMP_NUM_THREADS` when OpenMP first starts its threads, so a script has to set an
environment variable before its first evaluation (`MCMC_demo/model.py`'s
`set_threads` does exactly this).

## Decision

1. `n_workers` defaults to **1** in `Ppq_compiled`, `Ppq_densities` and
   `make_Ppq_densities_NR/ER`. `None` still means `os.cpu_count()` for callers
   who opt in. Docstrings say when to use more: a gfortran build, which has no
   OpenMP parallelism (`docs/building-from-source.md`), with
   `OMP_NUM_THREADS=1`. Prepared for `develop` on branch
   `fix/n-workers-default-1` (`Ppq_compiled` and `make_Ppq_densities_NR/ER`);
   `Ppq_densities`, which exists only on this branch, needs the same change.
2. Add `phonon_charge_pdf.set_threads(n)` and `get_threads()`: thin wrappers over
   a `bind(c)` Fortran routine that calls `omp_set_num_threads` /
   `omp_get_max_threads` (no-ops returning 1 in a build without OpenMP). Docs
   show `set_threads` wherever they currently rely on the environment.

## Consequences

- A user who does nothing gets the efficient configuration (one Python thread,
  OpenMP across the cores), and nested oversubscription can't happen by default.
- gfortran users lose the Python-thread parallelism they got implicitly. This is
  acceptable because gfortran is documented as verification-only, and they can
  pass `n_workers` explicitly.
- `set_threads` adds two `bind(c)` symbols, so it's an ABI addition (bump
  `PpqFort_version`'s minor).
- Verification: the existing Python tests pass unchanged (they pass `n_workers`
  explicitly); a test that `set_threads(2); get_threads() == 2` in an OpenMP
  build.
