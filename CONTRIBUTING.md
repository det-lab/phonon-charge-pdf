# Contributing

Thanks for looking at contributing to `phonon-charge-pdf`. This covers the mechanics of
making a change; for what the library does and how to use it, see the
[README](README.md) and the [full documentation](https://det-lab.github.io/phonon-charge-pdf/).

## Branches and PRs

- Feature work, fixes, chores, docs -- anything that isn't a release -- branches off
  `develop` and opens a PR back into `develop`.
- Only `release/*` and `hotfix/*` branches may target `main` directly; this is enforced
  by CI (`.github/workflows/check-pr-base.yml`), not just a convention. If your PR is
  rejected for this, retarget it at `develop`.
- After anything lands on `main`, a PR back into `develop` opens automatically
  (`.github/workflows/sync-main-to-develop.yml`) so the two branches don't silently drift
  apart. You don't need to do anything for this yourself, but if you see one of these
  PRs open, it's real and needs a human to resolve any conflict, not just rubber-stamp.

## Building and testing

See [Building from source](https://det-lab.github.io/phonon-charge-pdf/building-from-source/)
for the full setup (compilers, `fpm`, Docker images). The short version for local
iteration:

```bash
python scripts/generate_version_include.py   # generates src/version.f90.inc, not committed
fpm test                                      # Fortran test suite
```

`gfortran` is the easiest compiler to get locally and is fine for correctness while
iterating, even though it builds the band integrals serially (~25x slower than `ifx`/
`flang` — see `src/Ppq_fort_m.f90`'s `Ppq_vector_NR`/`ER` doc comments, rendered in the
[Fortran API reference](https://det-lab.github.io/phonon-charge-pdf/fortran-api/)). CI
runs the full matrix (gfortran, ifx, flang) on every push and PR
(`.github/workflows/build.yml`) — that's what catches a compiler-specific issue, not
local iteration.

If your change touches the Python side, also run the Python validation against the
compiled library:

```bash
pip install -e ".[test]"
fpm install --prefix=. --profile release
LD_LIBRARY_PATH=lib python test/python/test_PpqFort.py
LD_LIBRARY_PATH=lib python test/python/test_PpqFort_vectorFuncs.py
```

(`DYLD_LIBRARY_PATH=lib` instead of `LD_LIBRARY_PATH` on macOS.) See `build.yml` for the
full list of Python validation scripts CI runs.

## Documentation

User-facing docs live under `docs/` and are built with mkdocs. Before opening a PR that
touches anything under `docs/` or `mkdocs.yml`, confirm it still builds clean:

```bash
mkdocs build --strict
```

`--strict` turns warnings (a broken internal link, a page not reachable from `nav:`) into
a build failure, so this is the actual gate, not just a sanity check.

## Recording a significant decision

If your change makes a real tradeoff a later maintainer could reasonably question --
not a routine implementation choice, and not something a docstring already explains --
see [Decision records (ADRs)](https://det-lab.github.io/phonon-charge-pdf/decision-records/)
for where that goes and the one rule that matters (records are immutable once accepted).

## Commit messages

Recent history mostly uses a short type prefix -- `feat:`, `fix:`, `docs:`, `refactor:`,
`chore:`, `ci:` -- followed by a concise summary. This isn't enforced by tooling, but
matching it keeps `git log` readable.

## License

By contributing, you agree your contribution is licensed under this project's
[BSD 3-Clause license](LICENSE).
