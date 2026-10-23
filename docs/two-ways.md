# Two ways to use this

Everything in this library falls into one of two buckets, and they have different
dependencies:

## 1. Pure Python — no Fortran, no container needed

`pip install phonon-charge-pdf` is enough. This covers:

- **Generating simulated events** — [`generate_events`/`generate_NR_events`/`generate_ER_events`](python-events.md), built on `event_generator.py`.
- **The resolution/yield model** — [`sigp`, `sigq`, `Y`, `Nbar`, and friends](ppq-dist.md), built on `ppq_dist.py`, the pure-Python reference implementation.

Use this bucket to generate or describe simulated data, or to get resolution/yield model
values directly. It works in any environment the package installs into — no container,
no compiler.

## 2. The Fortran-backed PDF — needs the compiled library

**If you want normalized PDF values or a likelihood** — a fit, an MCMC, anything beyond
generating or describing simulated events — you need [`pqPDF`](fortran-pdf.md), and it
always needs the compiled Fortran library (`PpqN`/`PpqG`). Three ways to get it, easiest
first:

| Option | Command |
|---|---|
| Pip install (bundles the compiled library) | `pip install phonon-charge-pdf` |
| Published container — HPC/library use | `docker pull ghcr.io/det-lab/phonon-charge-pdf-llvm:v2.1.1` |
| Published container — Jupyter notebooks | `docker pull ghcr.io/det-lab/phonon-charge-pdf-jupyter:v2.1.1` |

Most users don't need to [build from source](building-from-source.md) at all — reach for
that only for local development, a compiler this project doesn't publish, or to reproduce
an exact uncommitted state.

## Which do I actually need?

```text
Do you need normalized PDF values or a likelihood (a fit, an MCMC)?
├── No  → pure Python is enough: event_generator.py / ppq_dist.py
└── Yes → you need pqPDF, so you need the compiled library:
          ├── pip install phonon-charge-pdf   (simplest)
          ├── a published container           (HPC / Jupyter)
          └── build from source                (only if you have a specific reason)
```
