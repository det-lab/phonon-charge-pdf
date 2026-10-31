"""
Private: low-level ctypes bindings to the compiled Fortran
phonon_charge_pdf library (lib/libphonon_charge_pdf.so).

Leading underscore means what it usually does for everything else in this
module -- not a public API, no stability promise.  make_Ppq_densities_NR/
make_Ppq_densities_ER are the one exception: they're re-exported as public
API (phonon_charge_pdf.goodness_of_fit.make_Ppq_densities_NR/ER) for that
module's own pdf_func and for anyone else who wants the raw band density at
arbitrary points.  If you're doing a fit or MCMC instead, use Ppq_compiled in
python/Ppq_fort.py; that is the one intended entrypoint there.  Ppq_compiled
combines the pieces here (a density over the full physical plane and a
separate region-normalization integral) correctly, raises instead of
extrapolating when a precomputed table is used, and is what the rest of
this file's docstrings assume you already have if you're reading them.
Everything else here (Ppq_region_NR/ER, the ctypes setup) exists for
Ppq_compiled, python/cli/normgrid.py, python/goodness_of_fit.py, and the
test suite to build on.

make_Ppq_densities_NR / make_Ppq_densities_ER build vectorized
pdf_func(Ep_flat, Eq_flat) -> values_flat callables.  These are genuine
probability densities -- just over the full physical (Ep, Eq) plane, not
yet renormalized to any one analysis region.  Dividing by
Ppq_region_NR/Ppq_region_ER's integral over your region renormalizes
(conditions) the density onto that region -- a much smaller box -- which
is what a likelihood over data already cropped to it needs.

All physics parameters are required keyword arguments — there are no
defaults, so the caller must specify the model fully.

The shared library evaluates points sequentially (~3-25 ms per point,
scaling with max(Ep, Eq) because the internal Erecoil integration range grows
with energy).  Since ctypes releases the GIL during foreign calls, this
wrapper parallelizes large requests by splitting them into chunks and
dispatching Ppq_vector_NR calls across a thread pool.  Points are
interleaved across chunks after sorting by max(Ep, Eq) so each chunk
carries a similar share of expensive (high-energy) points.
"""

import ctypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# python/internal/ is two levels below the repo root now (it used to be one).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO_LIB_DIR = os.path.join(_REPO_ROOT, "lib")
# fpm/the OS linker name this .dylib on macOS and .so everywhere else fpm
# supports (Linux); there's no third platform to worry about.
_LIB_NAME = "libphonon_charge_pdf.dylib" if sys.platform == "darwin" else "libphonon_charge_pdf.so"

_DOUBLE_ARR = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")

_api = None


def _candidate_libs():
    """Where to look for libphonon_charge_pdf.so, in order.

    1. _REPO_LIB_DIR -- today's convention: running out of a git checkout
       with `fpm build`'s output at the repo root's lib/. This still makes
       sense once this file is relocated (e.g. pip-installed into
       site-packages): the path just won't exist there, and the next
       candidate is tried.
    2. A bare library name, so the OS's own loader searches
       LD_LIBRARY_PATH (Linux) / DYLD_LIBRARY_PATH (macOS) / rpath -- what
       a container or a pip install outside any git checkout is expected
       to rely on, same as every other shared library on the system.
    """
    return [os.path.join(_REPO_LIB_DIR, _LIB_NAME), _LIB_NAME]


def _load_library():
    """Load libphonon_charge_pdf.so, trying each of _candidate_libs() in turn.

    The library has no SONAME and its RUNPATH points at nix store paths,
    so its dependencies (libassert.so, libjulienne.so) can only be found
    via LD_LIBRARY_PATH, which must be set before the Python process
    starts:  LD_LIBRARY_PATH=lib python your_script.py
    (LD_LIBRARY_PATH -- or DYLD_LIBRARY_PATH on macOS -- is also how the
    bare-name candidate below finds the library itself, not just its
    dependencies, when there's no repo checkout to find lib/ in.)
    """
    global _api
    if _api is not None:
        return _api

    errors = []
    for candidate in _candidate_libs():
        try:
            _api = ctypes.CDLL(candidate)
            break
        except OSError as e:
            errors.append(f"  {candidate}: {e}")
    else:
        tried = "\n".join(errors)
        raise RuntimeError(
            f"Could not load {_LIB_NAME}. Tried:\n{tried}\n"
            f"Set LD_LIBRARY_PATH (or DYLD_LIBRARY_PATH on macOS) to the directory "
            f"containing it before Python starts, "
            f"e.g.: LD_LIBRARY_PATH=lib python your_script.py"
        )

    _api.Ppq_vector_NR.argtypes = [
        _DOUBLE_ARR, _DOUBLE_ARR, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
        _DOUBLE_ARR,
    ]
    _api.Ppq_vector_NR.restype = None

    _api.Ppq_vector_ER.argtypes = [
        _DOUBLE_ARR, _DOUBLE_ARR, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        _DOUBLE_ARR,
    ]
    _api.Ppq_vector_ER.restype = None

    _api.PpqFort_version.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    _api.PpqFort_version.restype = None

    _api.Ppq_region_NR.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # ep_min, ep_max, eq_min, eq_max
        ctypes.c_double, ctypes.c_double,                                    # epsrel, epsabs
        ctypes.c_double, ctypes.c_double, ctypes.c_double,                   # k, Z, F0
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # eps, V, p0, p10
        ctypes.c_double, ctypes.c_double,                                    # q0, q10
    ]
    _api.Ppq_region_NR.restype = ctypes.c_double

    _api.Ppq_region_ER.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # ep_min, ep_max, eq_min, eq_max
        ctypes.c_double, ctypes.c_double,                                    # epsrel, epsabs
        ctypes.c_double,                                                     # F0
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # eps, V, p0, p10
        ctypes.c_double, ctypes.c_double,                                    # q0, q10
    ]
    _api.Ppq_region_ER.restype = ctypes.c_double

    return _api


def version():
    """Return (major, minor, patch) reported by the loaded shared library."""
    api = _load_library()
    major, minor, patch = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    api.PpqFort_version(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
    return (major.value, minor.value, patch.value)


def make_Ppq_densities_NR(*, k, Z, F0, eps, V, p0, p10, q0, q10, n_workers=None):
    """
    Build pdf_func(Ep_flat, Eq_flat) -> values_flat backed by the Fortran
    Ppq_NR (nuclear recoil band PDF).  A genuine probability density, just
    over the full physical (Ep, Eq) plane rather than your analysis region
    -- a much bigger box.  Divide by Ppq_region_NR's integral over your
    region to renormalize (condition) it onto that region, which is what a
    likelihood over cropped data needs.  For a fit or MCMC, use Ppq_compiled
    (python/Ppq_fort.py) instead, which does that for you.

    All physics parameters are required:
      k, Z           : Lindhard ionization yield calibration constant and
                        target atomic number
      F0             : Fano factor (constant)
      eps            : energy per e/h pair [keV]
      V              : bias voltage [V]
      p0, p10        : phonon resolution parameters
      q0, q10        : charge resolution parameters

    n_workers : int or None
        Threads used for large requests.  Defaults to os.cpu_count().
        Requests of <= 16 points bypass the pool (e.g. scalar calls from
        scipy quadrature).

    Invalid physics parameters (e.g. p10 < p0) surface as NaN in the
    returned array rather than being silently zeroed -- Ppq_NR's own
    validation is limited (see Ppq_region_NR for the fuller guard a
    fit/MCMC's Ppq_compiled always runs behind this), so a NaN here is a
    sign the parameters themselves are wrong, not that the density is
    genuinely tiny.
    """
    api = _load_library()
    if n_workers is None:
        n_workers = os.cpu_count()
    scalars = (k, Z, F0, eps, V, p0, p10, q0, q10)

    def _eval_chunk(ep, eq, out):
        api.Ppq_vector_NR(ep, eq, ep.size, *scalars, out)

    return _make_threaded_pdf(_eval_chunk, n_workers)


def make_Ppq_densities_ER(*, F0, eps, V, p0, p10, q0, q10, n_workers=None):
    """Same as make_Ppq_densities_NR but for the Fortran Ppq_ER (gamma /
    ER band).  The yield is fixed at Y = 1 internally, so k and Z are not
    taken.  Also a full-plane density, not yet renormalized to a region --
    see that function's docstring."""
    api = _load_library()
    if n_workers is None:
        n_workers = os.cpu_count()
    scalars = (F0, eps, V, p0, p10, q0, q10)

    def _eval_chunk(ep, eq, out):
        api.Ppq_vector_ER(ep, eq, ep.size, *scalars, out)

    return _make_threaded_pdf(_eval_chunk, n_workers)


def Ppq_region_NR(ep_min, ep_max, eq_min, eq_max, *,
                 epsrel, epsabs,
                 k, Z, F0, eps, V, p0, p10, q0, q10):
    """
    Integral of Ppq_NR over [ep_min,ep_max] x [eq_min,eq_max], computed
    entirely in Fortran (a single call, no per-point ctypes round trips)
    with a doubling-verified nested Gauss-Legendre quadrature (see
    Ppq_fort_s.f90's region_integral_gl) -- the ridge location/width is
    already known exactly (ridge_eq_and_width), so this needs far fewer
    points than a naive grid or generic adaptive cubature for the same
    accuracy, and is much faster than nested scipy.integrate.quad.

    epsrel/epsabs are required (no defaults): they set how tightly two
    successive doubled quadrature orders must agree before the result is
    trusted, and directly control the accuracy of a number that feeds a
    likelihood normalization. The Fortran side error-stops rather than
    returning an unverified number if that isn't reached by its highest
    order.
    """
    api = _load_library()
    return api.Ppq_region_NR(ep_min, ep_max, eq_min, eq_max,
                            epsrel, epsabs,
                            k, Z, F0, eps, V, p0, p10, q0, q10)


def Ppq_region_ER(ep_min, ep_max, eq_min, eq_max, *,
                 epsrel, epsabs,
                 F0, eps, V, p0, p10, q0, q10):
    """Same as Ppq_region_NR but for Ppq_ER (electron-recoil band, Y=1)."""
    api = _load_library()
    return api.Ppq_region_ER(ep_min, ep_max, eq_min, eq_max,
                            epsrel, epsabs,
                            F0, eps, V, p0, p10, q0, q10)


def _make_threaded_pdf(eval_chunk, n_workers):
    def pdf(ep_flat, eq_flat):
        ep = np.ascontiguousarray(np.asarray(ep_flat, dtype=np.float64).ravel())
        eq = np.ascontiguousarray(np.asarray(eq_flat, dtype=np.float64).ravel())
        n = ep.size
        out = np.empty(n, dtype=np.float64)

        if n <= 16 or n_workers <= 1:
            eval_chunk(ep, eq, out)
        else:
            # Cost per point scales with max(Ep, Eq): sort by cost and deal
            # points round-robin so every chunk gets a balanced mix.
            order = np.argsort(-np.maximum(ep, eq), kind="stable")
            n_chunks = min(n_workers * 4, n)
            chunks = [order[i::n_chunks] for i in range(n_chunks)]

            def run(idx):
                ep_c = np.ascontiguousarray(ep[idx])
                eq_c = np.ascontiguousarray(eq[idx])
                out_c = np.empty(idx.size, dtype=np.float64)
                eval_chunk(ep_c, eq_c, out_c)
                out[idx] = out_c

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                list(pool.map(run, chunks))

        return out

    return pdf
