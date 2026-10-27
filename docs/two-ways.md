# Two ways to use this

Everything in this library falls into one of two buckets, and they have different
dependencies — but the same single `import phonon_charge_pdf as pdf` gets you both; the
buckets differ in which pieces need the compiled Fortran library, not in how you import
them.

## 1. Pure Python — no Fortran, no container needed

`pip install phonon-charge-pdf` is enough. This covers:

- **Generating simulated events** — [`generate_events`/`generate_NR_events`/`generate_ER_events`](python-events.md), built on `event_generator.py`.
- **The resolution/yield model** — [`sigp`, `sigq`, `Y`, `Nbar`, and friends](ppq-py.md), built on `Ppq_py.py`, the pure-Python reference implementation.

Use the pure-Python library to generate or describe simulated data, or to get
resolution/yield model values directly. It works in any environment the package installs
into — no container, no compiler.

## 2. The Fortran-backed PDF — needs the compiled library

**If you want normalized PDF values or a likelihood** — a fit, an MCMC, anything beyond
generating or describing simulated events — you need [`Ppq_compiled`](ppq-fort.md), and it
always needs the compiled Fortran library (`Ppq_NR`/`Ppq_ER`). Three ways to get it, easiest
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
├── No  → pure Python is enough: generate_events / sigp, Y, Nbar, ...
└── Yes → you need Ppq_compiled, so you need the compiled library:
          ├── pip install phonon-charge-pdf   (simplest)
          ├── a published container           (HPC / Jupyter)
          └── build from source                (only if you have a specific reason)
```
