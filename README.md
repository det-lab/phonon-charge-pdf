# phonon-charge-pdf

[![PyPI](https://img.shields.io/pypi/v/phonon-charge-pdf.svg)](https://pypi.org/project/phonon-charge-pdf/)
[![Tests](https://img.shields.io/github/actions/workflow/status/det-lab/phonon-charge-pdf/build.yml?branch=main&label=tests)](https://github.com/det-lab/phonon-charge-pdf/actions/workflows/build.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/det-lab/phonon-charge-pdf/docs.yml?branch=main&label=docs)](https://det-lab.github.io/phonon-charge-pdf/)
[![License](https://img.shields.io/github/license/det-lab/phonon-charge-pdf.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23049123.svg)](https://doi.org/10.5281/zenodo.23049123)

This code is useful for dark matter searches where detector output is Ep (total phonon energy) and Eq (total charge energy).  The point of the code is to provide the probability of an (Ep, Eq) pair given a set of detector parameters for both electron recoils (`Ppq_ER`) and nuclear recoils (`Ppq_NR`).

**Full documentation, including an API reference, is at <https://det-lab.github.io/phonon-charge-pdf/>.** This README covers the quickstart and a map of where everything lives.

# Quickstart

```bash
pip install phonon-charge-pdf
```

```python
import phonon_charge_pdf as pdf

events = pdf.generate_events(
    target_size=1000, fraction_ER=0.6, seed=1,
    ep_range=(2.5, 350.0), eq_range=(0.75, 200.0),
    k=0.18, Z=32.0, F0_NR=0.122, F0_ER=0.2, eps=3.0e-3, V=3.0,
    p0=0.06421907, p10=0.48998486, q0=0.23718488, q10=0.27093151,
    mix_frac_NR=0.53693208, scale_fast_NR=6.41515782, scale_slow_NR=23.71789286,
    mix_frac_ER=0.573211975, scale_fast_ER=0.169520023, scale_slow_ER=279.552394,
)
Ep, Eq = events["EP"].to_numpy(), events["EQ"].to_numpy()
```

This works with **no Fortran build** — pure Python. For a fit/MCMC you need the compiled
library and `Ppq_compiled` — see [Two ways to use this](#two-ways-to-use-this) below.

Published on PyPI as `phonon-charge-pdf`; `import phonon_charge_pdf` (hyphen becomes
underscore, the usual PyPI convention). For local development instead of a pinned release,
`pip install -e .` from a checkout does the same (add `[test]` for the test suite's
dependencies, or `[docs]` to build the documentation site locally with `mkdocs serve`).

# Citing

Citation metadata for the code is in `CITATION.cff` (GitHub's "Cite this repository" button); each release is archived on Zenodo.  The precomputed normalization tables are a separate dataset, [doi:10.5281/zenodo.23048215](https://doi.org/10.5281/zenodo.23048215) (all versions; one version per added fit window), which `python/cli/normtables.py` fetches by name.

# Two ways to use this

Everything in this library falls into one of two buckets, and they have different
dependencies.

**1. Pure Python — no Fortran, no container needed.** `pip install phonon-charge-pdf` is
enough. This covers generating simulated events ([`generate_events`](docs/python-events.md))
and the resolution/yield functions (`sigp`, `sigq`, `Y`, `Nbar`, ...), listed in the
[API reference](https://det-lab.github.io/phonon-charge-pdf/api/).

**2. The Fortran-backed PDF — needs the compiled library.** If you want normalized PDF
values or a likelihood — a fit, an MCMC, anything beyond generating or describing
simulated events — you need [`Ppq_compiled`](docs/fitting.md), and it always needs the
compiled library. Easiest first:

| Option | Command |
|---|---|
| Pip install (bundles the compiled library) | `pip install phonon-charge-pdf` |
| Published container -- HPC/library use | `docker pull ghcr.io/det-lab/phonon-charge-pdf-llvm:v2.1.1` |
| Published container -- Jupyter notebooks | `docker pull ghcr.io/det-lab/phonon-charge-pdf-jupyter:v2.1.1` |

Most users don't need to [build from source](docs/building-from-source.md) at all. For the
Jupyter container, run it with your own repository mounted so your notebooks and edits
persist (replace `/path/to/your/repo` with your own repository's path; leave
`/home/jovyan/work/nrFano` as-is), then select the **Python (pdf)** kernel:

```bash
docker run -it --rm -p 8888:8888 -v /path/to/your/repo:/home/jovyan/work/nrFano ghcr.io/det-lab/phonon-charge-pdf-jupyter:v2.1.1
```

For the HPC container, replace `v2.1.1` with any other release tag (see
[releases](https://github.com/det-lab/phonon-charge-pdf/releases) or `git tag -l`), or
`latest` for the newest; `apptainer build pdf.sif docker://ghcr.io/det-lab/phonon-charge-pdf-llvm:v2.1.1`
builds a `.sif` for HPC directly (what `slurm/pull_container.job` does).

# Map of the documentation

Everything below lives under [`docs/`](docs/) — read it there, on GitHub, or rendered at
<https://det-lab.github.io/phonon-charge-pdf/>.

| I want to... | See |
|---|---|
| Generate simulated (Ep, Eq) events, no Fortran build needed | [Generating simulated events](docs/python-events.md) |
| Use the resolution/yield functions directly (`sigp`, `sigq`, `Y`, `Nbar`, ...) | [Python API reference](https://det-lab.github.io/phonon-charge-pdf/api/) |
| Get normalized PDF values or a likelihood for a fit/MCMC | [Fitting](docs/fitting.md) |
| Run a fit/MCMC fast, without recomputing the normalization integral every step | [Precomputed normalization tables](docs/normalization-tables.md) |
| Build this project from source, or build/run its containers | [Building from source](docs/building-from-source.md) |
| Check that the physics is actually correct | [Verification and validation](docs/testing.md) |
| Profile the Fortran library itself (maintainer workflow) | [Profiling with TAU](docs/PROFILING.md) |
| Browse the auto-generated API reference, grouped by what you'd call, not by file | <https://det-lab.github.io/phonon-charge-pdf/api/> |
