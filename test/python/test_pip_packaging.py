"""
Validates the pip-installable package's graceful-degradation promise:
pure-Python functionality (event_generator.py) works with no compiled
Fortran library anywhere on the system, while PpqPDF -- which always
needs it, even with a precomputed normalization table -- fails clearly
and immediately at construction rather than deep into a fit.

Must run against a `pip install -e .[test]` of this repository (imports
band_distribution as an installed package, not via sys.path hacks like
the rest of this test suite), and before the Fortran library is ever
built/installed to lib/ -- otherwise the "no Fortran available" half of
this test exercises nothing.

Run from the repository root:  python test/python/test_pip_packaging.py
"""

import sys

failures = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


import band_distribution as bd  # noqa: E402 -- the import itself is part of what's tested

check("import band_distribution with no compiled library present", True)
check("PpqPDF is exported", hasattr(bd, "PpqPDF"))
check("generate_events is exported", hasattr(bd, "generate_events"))
check("generate_NR_events is exported", hasattr(bd, "generate_NR_events"))
check("generate_ER_events is exported", hasattr(bd, "generate_ER_events"))

# ppq_dist.py's public API (pure Python, no Fortran -- see its module
# docstring): every name in its own __all__ must also be re-exported here.
from band_distribution.ppq_dist import __all__ as _ppq_dist_all  # noqa: E402

check("every ppq_dist.__all__ name is re-exported on band_distribution",
      all(hasattr(bd, name) for name in _ppq_dist_all),
      [name for name in _ppq_dist_all if not hasattr(bd, name)])

PARAMS = dict(
    k=0.18, Z=32.0, F0=0.122, eps=3.0e-3, V=3.0,
    p0=0.06421907, p10=0.48998486, q0=0.23718488, q10=0.27093151,
)
SPECTRUM = dict(PNa=0.53693208, PNb=6.41515782, PNd=23.71789286)

try:
    Ep, Eq, N0, seeds = bd.generate_NR_events(50, seed=1, **PARAMS, **SPECTRUM)
    check("generate_NR_events runs with no Fortran library",
          Ep.shape == Eq.shape == N0.shape == seeds.shape == (50,))
except Exception as e:
    check("generate_NR_events runs with no Fortran library", False, repr(e))

try:
    (val, err), _, peaks = bd.PpqN_safe_inspect(10.0, 5.0, **PARAMS, res=0.1)
    check("PpqN_safe_inspect runs with no Fortran library",
          val > 0 and len(peaks) >= 1, (val, len(peaks)))
except Exception as e:
    check("PpqN_safe_inspect runs with no Fortran library", False, repr(e))

try:
    bd.PpqPDF(0, 1, 0, 1, [], [], norm_epsrel=0.1, norm_epsabs=0.1)
    check("PpqPDF() raises at construction without the Fortran library", False,
          "construction succeeded -- it should have failed fast instead")
except RuntimeError as e:
    check("PpqPDF() raises at construction without the Fortran library", True, str(e)[:80])
except Exception as e:
    check("PpqPDF() raises at construction without the Fortran library", False,
          f"wrong exception type: {e!r}")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
