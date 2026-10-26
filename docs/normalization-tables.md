# Precomputed normalization tables for MCMC: `python/cli/normgrid.py`

Even at ~1-2 s, computing the normalization integral at every MCMC step is out of the
question. But the region is fixed for a whole run and the integral is a very smooth
function of the physics parameters the MCMC varies, so `normgrid.py` evaluates it once on
a small tensor grid (one independent integral per grid point — a good fit for the OSG) and
interpolates it at tens of microseconds per step. Separate tables for the NR band
(`k, F0, V, p0, p10, q0, q10`) and the ER band (same, no `k`); `Z` and `eps` are fixed
inside a table and checked on every call.

## Design points

- **Axes.** `k`, `F0` (used *linearly* — the integral is smooth in F0 but not in log F0, so
  3 nodes give ~3e-9 where log F0 needs 9+), `V`, `p0`, `dp = p10 - p0`, `q0`,
  `dq = q10 - q0`. The resolution model `sigp² = p0² + (p10² - p0²)(Ep/c)²` is only defined
  for `p10 >= p0` (likewise `q10 >= q0`), so a plain `(p0, p10)` or `(q0, q10)` box would
  contain unphysical corners (the `q0` and `q10` ranges overlap); `dp >= 0` and `dq >= 0`
  keep the box rectangular. Query the table with the physical `p10`/`q10`; it converts.
  `Ppq_region_NR`/`Ppq_region_ER` now error out in ~0.3 s on such inputs (and on
  `F0, eps, p0, q0, k, Z <= 0`, an empty region, or NaN) instead of grinding for minutes.
- **Interpolant.** Tensor-product Chebyshev series through Chebyshev–Lobatto nodes, built
  and evaluated with `numpy.polynomial.chebyshev` (`chebfit`/`chebval`; exact at the nodes,
  ~40 µs per lookup natively — about 0.1% of a 20,000-event likelihood evaluation). It
  **never extrapolates** — a query outside the table's box raises `OutOfBoxError`, so keep
  the MCMC prior inside the box (see [Fits and MCMC with the tables](#fits-and-mcmc-with-the-tables)
  below; default box: `k` 0.13–0.22, `F0` 1e-5–1 (ER: 0.1–0.35, bracketing the effective
  low-field electron-recoil Fano factor of ~0.2–0.3), `V` 2.7–3.3, `p0` 0.0128–0.1156 and
  `q0` 0.0474–0.4 (the ±4σ range of Gaussian priors centred on 0.0642 and 0.2372 with a 20%
  1σ width, `q0` capped at the maximum `q10`), `dp = p10-p0` 0.18–0.59 and `dq = q10-q0`
  0–0.353 (from `p10` 0.3–0.6, `q10` 0.2–0.4); for a different one, rebuild with `BOX=`,
  e.g. `BOX="V=2.5:4" sbatch slurm/build_table.job ...`, or change the defaults
  `DEFAULT_BOX`/`PDF_BOX_OVERRIDES`).
- **Node counts** (`RECOMMENDED_NODES`, from per-axis studies against directly computed
  held-out points; region-dependent): NR `k=8 F0=3 V=4 p0=4 dp=3 q0=16 dq=4` (73,728
  points); ER `F0=3 V=9 p0=4 dp=6 q0=11 dq=9` (64,152 points). The counts depend on the
  region: these are for the analysis ROI Ep 2.5–350 / Eq 0.75–200 (the earlier 2–200 /
  4–100 region needed far fewer, e.g. 5 nodes in `q0`). `q0` is the hard axis for NR — its
  error falls only ~4x per two nodes — so check it with `validate`. At roughly 1–3 s per
  point on one core that is tens of core-hours. These target ~1e-7 worst-case error per
  axis, i.e. ~0.02 in the log-likelihood at 20,000 events (the error is
  `N_events x δN/N`). Always confirm with `validate` — the per-axis studies cannot see
  cross terms.

## Building a table by hand

```bash
# describe the grid (region and epsrel are required; nodes default to RECOMMENDED_NODES
# and the box to the one above -- a wrong region makes a table that looks fine and isn't)
python python/cli/normgrid.py make-spec --band NR --region 2.5 350 0.75 200 --epsrel 1e-7 --out spec_NR.json
python python/cli/normgrid.py make-random-spec --band NR --n 200 --region 2.5 350 0.75 200 --epsrel 1e-7 --out held_NR.json

# run it: on one machine, or as batch jobs (osg/normgrid.sub + osg/normgrid_job.sh are an HTCondor template)
python python/cli/normgrid.py chunks spec_NR.json --size 100        # "start stop" ranges, one per job
python python/cli/normgrid.py run --spec spec_NR.json --start 0 --stop 100 --out res_0.txt
python python/cli/normgrid.py run --spec held_NR.json --out res_held.txt

# combine into one HDF5 file and check it against the held-out points
python python/cli/normgrid.py merge --spec spec_NR.json --results 'res_*.txt' --out norm_NR.h5
python python/cli/normgrid.py validate --table norm_NR.h5 --heldout-spec held_NR.json --results res_held.txt
```

**On a Slurm cluster** use the **ifx (or flang) container, not gfortran** — gfortran runs
the band integrals ~25x slower — via `slurm/normgrid.sbatch`, a job-array template (one
single-core task per chunk of grid points; `OMP_NUM_THREADS=1` so an ifx build doesn't
oversubscribe cores; a header comment gives the exact `sbatch` and `merge`/`validate`
commands, and resubmitting the same array only redoes unfinished chunks). It runs the
container `$PDF_SIF` (default `/scratch/$USER/containers/pdf.sif`, as pulled by
`slurm/pull_container.job`), a `pdf.sif` built from `Dockerfile_intel` or `Dockerfile_llvm`
(see [Docker/Apptainer images](building-from-source.md#dockerapptainer-images)), or set
`PDF_NATIVE=1` to use a library you built on the cluster. `osg/` holds a similar HTCondor
template.

## One command for a whole table

How many nodes an axis needs depends on the fit region (the same box needed 5 nodes in
`q0` for Eq 4–100 and about 16 for Eq 0.75–200), so `slurm/build_table.sh` measures instead
of assuming. For a region it (1) evaluates 33 Chebyshev–Lobatto nodes along each axis at
three baseline points (`python/cli/normplan.py points`, about 700 integrals), (2) reads the
Chebyshev coefficients of each axis to find the fewest nodes whose truncation error is
below `TOL` (`python/cli/normplan.py analyze`; it stops with an error if an axis is not
resolved by 33 nodes, rather than guessing), (3) evaluates the resulting grid and a set of
held-out points, and (4) merges and validates, exiting with status 1 if the worst held-out
error exceeds `ACCEPT`:

```bash
sbatch slurm/build_table.job NR 2.5 350 0.75 200              # -> tables/norm_NR_ep2.5-350_eq0.75-200.h5
TOL=2.5e-8 sbatch slurm/build_table.job NR 2.5 350 0.75 200   # retry with more nodes; earlier results are reused
```

Each Slurm stage is `sbatch --wait slurm/normgrid.sbatch ...`, so `slurm/build_table.sh`
runs until the whole chain is done; `slurm/build_table.job` runs it as a single-core driver
job that waits in the queue like any other (the environment at submission, e.g. `TOL`, is
passed on). Every array is checked with `sacct` when it ends and a failed task stops the
chain (`sbatch --wait` exits 0 on Alderaan even when tasks fail). The container is
`PDF_SIF`, by default `/scratch/$USER/containers/pdf.sif` where `slurm/pull_container.job`
puts it. Options (`TOL`, `EPSREL`, `ACCEPT`, chunk sizes, `ACCOUNT`/`PARTITION`/`TIME`) are
documented in the script header. The one-axis-at-a-time study cannot see cross terms
between axes, which is why the held-out check is the gate.
`test/slurm/test_build_table.sh` runs the whole chain locally against a mock `sbatch` and
an analytic stand-in for the integrals.

`run` is resumable and crash-tolerant: Fortran `error stop` (e.g. the quadrature not
certifying `epsrel=1e-7`) kills the process, so a supervisor records that point as `nan`
and restarts past it; re-run failures with a looser tolerance via
`run --retry-failed --epsrel 1e-6`. `merge` refuses to build a table with missing points.
The HDF5 file records the region, fixed parameters, library version and git commit it was
built with — rebuild if any of those change. (`h5py` is in `environment.yaml`.) Then hand
the tables to `Ppq`; nothing else in the fit changes:

```python
# define the data region: must match what the tables were built for exactly
ep_min, ep_max = 2.5, 350.0
eq_min, eq_max = 0.75, 200.0

likelihood = pdf.Ppq(ep_min, ep_max, eq_min, eq_max, ep_data, eq_data,
                      Ppq_table_NR="norm_NR.h5", Ppq_table_ER="norm_ER.h5")
# likelihood.Ppq_integral_NR(...) / likelihood.Ppq_integral_ER(...) now interpolate instead of integrating
```

**Published tables** are listed in `python/cli/table_registry.json` and can be passed by
name instead of a path: `python/cli/normtables.py` downloads them from
[Zenodo](https://doi.org/10.5281/zenodo.23048215) on first use, checks their SHA-256, and
caches them (`$PDF_TABLES_DIR`, else `~/.cache/phonon_charge_pdf`; the containers ship
with every registered table in `/app/tables`). `python python/cli/normtables.py list` shows
what is available.

## Fits and MCMC with the tables

A table covers a fixed box of parameters and never extrapolates, so the rule for a fit or
an MCMC is the same: **keep every parameter inside the tables' box, with `p10 >= p0` and
`q10 >= q0`** (the PDF is undefined otherwise). The library checks this and says what to
change; the fit or MCMC code has to enforce it.

- `Ppq(..., prior_bounds=BOUNDS)` checks hard bounds against every table given, at
  construction. If they reach outside a table it raises `PriorNotCoveredError`, listing
  each short axis and the `BOX=` rebuild command that would cover them.
- `likelihood.table_bounds()` gives, per band, the widest bounds each table covers: a
  ready-made set of hard bounds. Take bounds from it, or round inward: the box edges are
  not round numbers (`p0` starts at 0.0128438), and bounds rounded outward fail the check.
- `likelihood.check_points(points)` checks a set of points, e.g. an MCMC's starting walkers.
- At run time, a query outside a table raises `OutOfBoxError` naming the axis, the value,
  the box and the table file. After the setup checks pass, that means a bug or a NaN.

Bounds are physical parameters, `{name: (lo, hi)}`. Parameters the two bands share go at
the top level; each band's own (NR's `k`, and the Fano factor `F0`, which differs between
the bands) go under `"NR"` and `"ER"`. `Z` and `eps` are fixed inside a table; pass them as
their fixed values or leave them out.

### MCMC

```python
BOUNDS = {"V": (2.7, 3.3), "p0": (0.0129, 0.1155), "p10": (0.3, 0.6),
          "q0": (0.0475, 0.4), "q10": (0.2, 0.4),
          "NR": {"k": (0.13, 0.22), "F0": (1e-5, 1.0)},
          "ER": {"F0": (0.1, 0.35)}}

# define the data region: must match the named tables below exactly
ep_min, ep_max = 2.5, 350.0
eq_min, eq_max = 0.75, 200.0

likelihood = pdf.Ppq(ep_min, ep_max, eq_min, eq_max, ep_data, eq_data,
                      Ppq_table_NR="NR_ep2.5-350_eq0.75-200", Ppq_table_ER="ER_ep2.5-350_eq0.75-200",
                      prior_bounds=BOUNDS)                  # 1. fails now, not hours into the run

def log_prior(theta):                             # 2. -inf outside the hard bounds and the ordering
    p = unpack(theta)
    if not within(p, BOUNDS) or p["p10"] < p["p0"] or p["q10"] < p["q0"]:
        return -np.inf
    return soft_terms(p)                          # e.g. Gaussians on p0, q0 -- truncated by BOUNDS

def log_prob(theta):                              # 3. the prior first: the table never sees
    lp = log_prior(theta)                         #    a point outside it
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, likelihood)

likelihood.check_points(as_params(start))        # 4. starting walkers inside, before sampling
                                                  #    ({param: array}, laid out like BOUNDS)
```

A Gaussian prior (on `p0` or `q0`, say) must be truncated by the hard bounds:
`prior_bounds` rejects a parameter without finite bounds, since no table can cover it.
Walkers started in a small ball around a best fit can still land outside near an edge,
which is what step 4 catches.

### Fits (maximum likelihood / MAP)

- Give the optimizer bounds inside the tables (e.g. `bounds=` for `scipy.optimize.minimize`
  with L-BFGS-B, Powell or trust-constr), and check them once with `prior_bounds` or
  `table_bounds()`.
- Box bounds cannot express `p10 >= p0`. Either fit `dp = p10 - p0` and `dq = q10 - q0`
  directly, bounded by the tables' own `dp`/`dq` axes (`likelihood.Ppq_table_NR.box["dp"]`,
  0.18–0.59 by default; `dq` 0–0.353), and convert back when calling the PDF, or add the
  inequalities as constraints (trust-constr, SLSQP).
- A fit that ends on a bound is being limited by the box, not the data: widen it with
  `BOX=` and rebuild the tables.
