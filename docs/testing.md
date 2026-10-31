# Verification and validation

Fortran-level tests run via `fpm test` (see the compiler table under
[Building from source](building-from-source.md)). The sections below test the python
interface and the physics itself.

## Testing the python calls

This code builds a library that may be called within python (this is the original intent
of the code). Build and install the shared library first (see
[Building from source](building-from-source.md)). The python test scripts live in
`test/python/` and should be run from the repository root. To test the python calls, run

```bash
LD_LIBRARY_PATH=lib python test/python/test_PpqFort.py
```

or if you just want to test the vectorized functions `Ppq_vector_NR` and `Ppq_vector_ER`

```bash
LD_LIBRARY_PATH=lib python test/python/test_PpqFort_vectorFuncs.py
```

(`LD_LIBRARY_PATH=lib` is needed when running outside the docker containers, which set it
in their environment.)

`test/python/test_pip_packaging.py` is the one test here that's meant to run *without* the
compiled library: against a `pip install -e ".[test]"` of this repository and before the
library is built, it checks that `import phonon_charge_pdf` and `event_generator.py`'s
functions still work, and that `Ppq_compiled` fails clearly at construction instead of silently
succeeding.

See `test/python/README.md` for a map of the test scripts and their supporting modules,
including where the PDF-over-bin integration lives.

## PDF self-consistency tests

These verify that data sampled from the PDFs produces Pearson chi-square values that
follow the theoretical chi2(n_bins - 1) distribution. Reference plots are committed in
`test/figures/`.

```bash
python test/python/verify_sample_from_pdf.py                       # sampler moment checks (~2 s)
LD_LIBRARY_PATH=lib python test/python/test_chisquare_ppqn.py      # Fortran Ppq_NR PDF (~2 min first run)
```

`test_chisquare_ppqn.py` takes the same two optional positional arguments as the
simulator tests below: `n_throws` (default 100,000) and `n_bins` (default 64). For a quick
smoke test with fewer throws, run

```bash
LD_LIBRARY_PATH=lib python test/python/test_chisquare_ppqn.py 2000 64
```

It caches its PDF grid evaluation in `ppq_nr_vertex_grid.npz` (not committed) and reuses it
on later runs.

The `test_chisquare_ppqn.py` timing assumes an ifx or flang build; gfortran builds run the
band integrals serially (see the compiler notes under
[Building from source](building-from-source.md)) and take correspondingly longer.

## Physics-simulator validation tests

The tests above only check that the PDFs are *self-consistent* (samples drawn from a PDF
match that same PDF). The two tests below are the stronger check: they generate events
from an independent physics simulator ([`event_generator.py`](python-events.md), which
draws Erecoil from the recoil spectrum, N from a truncated normal, and applies detector
resolution) and compare the binned counts against the Fortran PDFs. A pass means the
Fortran `Ppq_NR`/`Ppq_ER` implementations correctly describe the physics.

```bash
LD_LIBRARY_PATH=lib python test/python/test_chisquare_simulator_nr.py [n_throws] [n_bins]   # NR band vs Ppq_NR
LD_LIBRARY_PATH=lib python test/python/test_chisquare_simulator_er.py [n_throws] [n_bins]   # ER band vs Ppq_ER
```

Both write their chi-square histograms to `test/figures/chisquare_simulator_nr.png` and
`test/figures/chisquare_simulator_er.png`. `n_throws` defaults to 1000 and `n_bins` to 400. The
one-time integration of the PDF over the bins dominates the wall time, so reducing
`n_bins` is the way to get a fast smoke test:

```bash
# smoke test: 100 throws, 64 bins
LD_LIBRARY_PATH=lib python test/python/test_chisquare_simulator_nr.py 100 64   # ~1 min
LD_LIBRARY_PATH=lib python test/python/test_chisquare_simulator_er.py 100 64   # ~1 min

# full validation: 10,000 throws, 400 bins
LD_LIBRARY_PATH=lib python test/python/test_chisquare_simulator_nr.py 10000    # ~4 min
LD_LIBRARY_PATH=lib python test/python/test_chisquare_simulator_er.py 10000    # ~5 min
```

(Timings measured with the ifx build, 18 workers, under x86 emulation on an Apple Silicon
Mac; native x86 hardware should be faster. gfortran builds run the band integrals
serially and will be dramatically slower here — use ifx or flang.)

To run inside the Intel docker container (built as `pdf` — see
[Docker/Apptainer images](building-from-source.md#dockerapptainer-images)), mount the
repository's `test/figures/` directory so the plot survives the container:

```bash
docker run --rm -v $(pwd)/test/figures:/app/test/figures pdf \
    bash --login -c "conda activate pdf && python /app/test/python/test_chisquare_simulator_nr.py 100 64"
```
