# Building from source

Most users don't need this — see [Two ways to use this](two-ways.md) for the published
containers and pip install. Build from source for local development, a compiler this
project doesn't publish, or to reproduce an exact local/uncommitted state.

## Fortran Package Manager (`fpm`)

This project uses the Fortran Package Manager (fpm). You'll need to install that to build
this project; please see <https://fpm.fortran-lang.org/install/index.html#install> for
instructions on installing fpm on your system. Currently (Oct 2026), both building from
source and installing via `conda` get the same version, 0.13.0.

**Shortcut:** `make test-gfortran`/`make test-ifx`/`make test-flang` and
`make install-gfortran`/`make install-ifx`/`make install-flang` (see the `Makefile`) wrap
the exact commands below, including generating `src/version.f90.inc` first. The commands
below remain the source of truth for the flags and are what you want for anything
non-default (a one-off flag override, a compiler the Makefile doesn't cover).

Before `fpm build`/`fpm test`, generate `src/version.f90.inc` (not committed —
`src/Ppq_fort_m.f90` `include`s it, so the build fails loudly if you skip this rather than
silently reporting a stale version):

```bash
python scripts/generate_version_include.py
```

Every Dockerfile in this repo runs this automatically; it's only a manual step for a
local, non-container build. `PpqFort_version()` then reports exactly `fpm.toml`'s
`version` field — there is no second copy to keep in sync by hand.

The code parallelizes its integration loops with `do concurrent` using locality
specifiers, including the Fortran 2023 `reduce` clause, so you need a recent compiler. The
versions below have been verified via the docker containers in this repository:

| Vendor | Compiler | Version(s) | Use for |
|---|---|---|---|
| Intel | `ifx` | 2025.2.1 | Production (MCMC, likelihood fits) |
| LLVM | `flang` | 22 | Production (MCMC, likelihood fits) |
| GNU | `gfortran` | 15.2.0 | Verification only (~25x slower, see below) |

Each compiler's commands are below, one block per command so they can be copied directly.
`fpm test` builds and runs the test suite; `fpm install --prefix=.` builds the shared
library the Python interface loads and places it in `./lib` (see
[Shared library for the python interface](#shared-library-for-the-python-interface)).
Run them from the repository root, after generating `src/version.f90.inc` as above.

### Intel (`ifx`)

Test (or `make test-ifx`):

```bash
fpm test --compiler ifx --profile release --flag "-fpp -O3 -qopenmp -DHAVE_MULTI_IMAGE_SUPPORT=0"
```

Install (or `make install-ifx`):

```bash
FPM_LDFLAGS="-liomp5" fpm install --prefix=. --compiler ifx --profile release --flag "-fpp -O3 -qopenmp -DHAVE_MULTI_IMAGE_SUPPORT=0"
```

Notes:

- `-fpp -DHAVE_MULTI_IMAGE_SUPPORT=0` are for the Julienne dependency. `-qopenmp` maps
  `do concurrent` onto the OpenMP thread pool.
- `FPM_LDFLAGS="-liomp5"` links the shared library against the Intel OpenMP runtime;
  without it the library builds but fails to load from python with undefined `__kmpc_*`
  symbols. The Intel container sets this for you.
- **ifx installed via conda** (`ifx_linux-64`, e.g. on a cluster): if every compile fails
  with

  ```
  ifx: error #10417: Problem setting up the Intel(R) Compiler compilation environment. Requires 'install path' setting gathered from 'gcc'
  ```

  ifx's internal call to `gcc` is failing. Run `gcc --version` in the same directory to
  see why. A common cause is a trailing colon in `LIBRARY_PATH` (`echo $LIBRARY_PATH`
  shows `.../compiler/lib:`), left by the conda activation script. gcc treats the empty
  entry as the current directory and looks there for a spec file named `specs`, so a
  `specs` *directory* in the build directory breaks it (`cannot read spec file
  './specs': Is a directory`). Remove the empty entry before building:

  ```bash
  export LIBRARY_PATH=$CONDA_PREFIX/compiler/lib
  ```

  If the build or tests then can't find Intel runtime libraries, also prepend
  `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH`:

  ```bash
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
  ```

### LLVM (`flang`)

Test (or `make test-flang`):

```bash
fpm test --compiler flang --profile release --flag "-O3 -fopenmp -fdo-concurrent-to-openmp=host"
```

Install (or `make install-flang`):

```bash
FPM_LDFLAGS="-fopenmp" fpm install --prefix=. --compiler flang --profile release --flag "-O3 -fopenmp -fdo-concurrent-to-openmp=host"
```

Notes:

- `-fdo-concurrent-to-openmp=host` is what parallelizes the `do concurrent` loops; without
  it they compile to serial loops.
- `FPM_LDFLAGS="-fopenmp"` links the shared library against `libomp`; without it the
  library fails to load from python with undefined `__kmpc_*` symbols. The LLVM container
  sets this for you.
- Intel and LLVM builds benchmark identically (~6 us per event in vector mode on 18
  emulated cores).

### GNU (`gfortran`) — verification only

Test (or `make test-gfortran`):

```bash
fpm test --compiler gfortran --profile release --flag "-march=native -fopenmp -ftree-parallelize-loops=4 -fcoarray=single -fPIC"
```

Install (or `make install-gfortran`):

```bash
fpm install --prefix=. --compiler gfortran --profile release --flag "-march=native -fopenmp -ftree-parallelize-loops=4 -fcoarray=single -fPIC"
```

Notes:

- gfortran is slow. It has no option for mapping `do concurrent` onto threads (the
  auto-parallelizer in `-ftree-parallelize-loops` cannot parallelize these loops because
  they contain function calls), so the band integrals run **serially** in gfortran
  builds — measured at 1.0 effective threads vs 17.6 for ifx on the same machine, making
  per-event likelihood evaluation ~25x slower there (roughly the core count times a
  ~1.5x per-core gap from the vector math library). The gfortran build is fine for
  verifying the physics and the Python interface, but use ifx or flang for production
  likelihood work (MCMC).

## Shared library for the python interface

The python wrappers load `lib/libphonon_charge_pdf.so`, which the `fpm install --prefix=.`
command for your compiler (above) builds and places under the repository root. For Intel
and LLVM, keep the `FPM_LDFLAGS` setting shown there: it makes the *shared library* link
its OpenMP runtime, and without it the library builds but fails to load from python with
undefined `__kmpc_*` symbols.

At run time — interactively or in a batch job script — the library and its compiler
runtime must be on the loader path. Outside the containers (which set this in their
environment), from the repository root:

```bash
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
```

For a conda-installed ifx, also add the conda environment's runtime libraries
(`libiomp5` etc.):

```bash
export LD_LIBRARY_PATH=$PWD/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

(`DYLD_LIBRARY_PATH` instead of `LD_LIBRARY_PATH` on macOS.)

## Docker/Apptainer images

There are multiple Dockerfiles, each building the code with a compiler from a different
vendor (GNU, Intel, and LLVM).

| Vendor | Dockerfile name | Notes |
|---|---|---|
| GNU | `Dockerfile_gfortran` | Slow: gfortran runs the band integrals serially, ~25x slower than ifx/flang (see the compiler notes above) |
| Intel | `Dockerfile_intel` | Recommended for work that needs speed |
| LLVM | `Dockerfile_llvm` | Same performance as the Intel build; preferred for any image meant to be published, since `environment.yaml` never installs Intel's ifx unless a Dockerfile explicitly adds it (`Dockerfile_intel`/`_tau_intel` do, via `conda install -n pdf ifx_linux-64` after the shared environment is created) |

Choose which compiler you want, determine the name of the dockerfile, and then issue the
following command:

```bash
docker build -f {dockerfile name} -t pdf .
```

By default this builds from your local checkout, including any uncommitted changes. To
instead build a specific released version — e.g. so someone else can reproduce your exact
results without needing their own clone lined up to the right commit — pass `GIT_REF` as
a build argument. It clones the repository fresh from GitHub and checks out that tag or
commit hash in place of the local files:

```bash
docker build -f {dockerfile name} -t pdf --build-arg GIT_REF=v0.9.0 .
```

`GIT_REF` accepts any tag or commit hash (run `git tag -l` to see available versions) and
works identically on all five Dockerfiles (`Dockerfile_gfortran`, `Dockerfile_intel`,
`Dockerfile_llvm`, `Dockerfile_jupyter`, `Dockerfile_tau_intel`). `v0.9.0` is the last
release before the Lindhard `(k, Z)` yield model's breaking interface change.

If you need to troubleshoot the docker build, you can shell into this container with the
command

```bash
docker run -it pdf
```

Now you have a docker container that contains the fortran binary, but this is not usable
on HPC systems. Run this command to create `pdf.sif`, an image file that can be used on
HPC systems. The command can be issued in any location (it is not directory dependent).

```bash
apptainer build pdf.sif docker-daemon://pdf:latest
```

## Jupyter image

```bash
docker build --rm -f Dockerfile_jupyter -t pdf_jupyter .
```

Run it the same way as the published image (see [Two ways to use this](two-ways.md)),
using `pdf_jupyter:latest` in place of the `ghcr.io` tag.

This image compiles with flang (LLVM), like `Dockerfile_llvm`, not ifx: it is meant to be
publishable, and `environment.yaml` never installs Intel's ifx unless a Dockerfile
explicitly adds it on top (see the compiler table above) — avoiding the Intel-redistribution
question entirely, the same reasoning that picked LLVM over Intel for the HPC container.

## Local development container

For local development, you most likely want the files available to you in a way that
persists once you close the container. In this case you need to supply arguments to
`docker run` that mount the top-level directory:

```bash
docker run -it --mount type=bind,src=.,dst=/app --entrypoint=/bin/bash pdf
```

## Profiling with TAU

Maintainer workflow for profiling the Fortran library itself, not needed to use the
library for a fit or MCMC — see [Profiling](PROFILING.md).
