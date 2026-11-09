# 0007. The shared library finds its own dependencies (`RUNPATH $ORIGIN`)

Status: Proposed
Date: 2026-11-09

## Context

`fpm install --prefix=.` puts `libphonon_charge_pdf.so` in `lib/` next to its two
dependencies, `libassert.so` and `libjulienne.so`. But the library has no RUNPATH
and no SONAME (`readelf -d lib/libphonon_charge_pdf.so` shows only `NEEDED`
entries), so the dynamic loader can only find the dependencies through
`LD_LIBRARY_PATH`. `_load_library()`'s docstring says this, and every test
command, job script and doc example carries `LD_LIBRARY_PATH=lib`.

This has to be set *before* Python starts (the loader reads it once), so it can't
be fixed from inside Python. The failure (`libassert.so: cannot open shared
object file`) also points at the wrong library for most users.

## Decision

Link `libphonon_charge_pdf.so` with `-Wl,-rpath,'$ORIGIN'` (and
`-Wl,--enable-new-dtags`, so it's a RUNPATH, still overridable by
`LD_LIBRARY_PATH`). Add it to every documented `fpm install` line and the
Makefile/Dockerfiles: through `FPM_LDFLAGS` for ifx/flang, which already carry
`-liomp5`/`-fopenmp` there, and `--link-flag` for gfortran.

`$ORIGIN` is the directory containing the library itself, so `lib/` resolves
`lib/libassert.so` wherever the checkout or install lives. macOS's equivalent is
`-Wl,-rpath,@loader_path` plus `install_name` fixes, as a separate item if needed.

## Consequences

- `lib/` disappears from `LD_LIBRARY_PATH` everywhere. Only the *compiler's*
  runtime (`libiomp5`, `libimf`, ... for a conda ifx) may still need it,
  because that lives in the compiler's install, not in `lib/`. Docs should then
  say exactly that.
- The quoting of `$ORIGIN` through make/fpm/shell is the fiddly part: verify with
  `readelf -d` that the RUNPATH entry literally reads `$ORIGIN`, in CI.
- Verification: `env -u LD_LIBRARY_PATH python -c "import phonon_charge_pdf as p;
  p.Ppq_compiled"` and a version call succeed with the library in `lib/`.
