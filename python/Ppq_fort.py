"""
Ppq_compiled is the one intended entrypoint for using this library in a fit or
MCMC -- if you want normalized PDF values or a likelihood, start here:

    from Ppq_fort import Ppq_compiled
    likelihood = Ppq_compiled(ep_range, eq_range, ep_data, eq_data,
                 norm_epsrel=1e-4, norm_epsabs=1e-10)
    normalized = likelihood.Ppq_normalized_densities_NR(k=k, Z=Z, F0=F0, eps=eps, V=V,
                                             p0=p0, p10=p10, q0=q0, q10=q10)

Ppq_compiled always needs the compiled Fortran library (a table only replaces the
normalization integral, never the per-event PDF evaluation), so
construction eagerly loads it and raises immediately if it can't be found,
rather than succeeding and failing on the first likelihood call instead.

See docs/fitting.md for the full walkthrough and docs/mcmc.md for precomputed
tables for MCMC.  If you instead
want to generate simulated (Ep, Eq) events -- e.g. to validate a fit
pipeline or feed a downstream analysis -- see event_generator.py
(python/event_generator.py), a second, independent entrypoint with no
Fortran dependency.  Ppq_py.py (python/Ppq_py.py) is a third, public
entrypoint: the pure-Python reference implementation event_generator.py
and this module are themselves checked against -- see its own module
docstring.  Everything else under python/ -- python/internal/
_Ppq_fort_bindings.py (raw PDF evaluation over the full physical (Ep, Eq)
plane, not yet renormalized to any one analysis region) and python/cli/
normgrid.py (table machinery) -- is either what Ppq_compiled/event_generator.py
are built from or infrastructure for maintaining this library, not a
fourth way to use it.

Ppq_compiled: the fixed context for one fit/MCMC run -- an (Ep, Eq) region, the
observed dataset, and the normalization-integral's convergence tolerance
-- exposing, per parameter point, the density at the bound dataset (a
genuine probability density, just over the full physical plane rather
than your region -- a much bigger box), the region's normalization
integral (the probability mass inside your region), and their ratio
(the density renormalized to -- conditioned on -- your region, which is
what a likelihood over cropped data needs), for both the NR (Ppq_NR)
and ER (Ppq_ER) bands.

All physics parameters (k, Z, F0, eps, V, p0, p10, q0, q10) are passed
fresh to every method call rather than bound at construction: those are
what an MCMC step actually varies, while the region, dataset, and
tolerance don't change across a run.

Built entirely on the already-validated primitives in _Ppq_fort_bindings.py --
Ppq_region_NR/Ppq_region_ER (-> the Fortran symbols of the same name, a
doubling-verified nested Gauss-Legendre quadrature) and
make_Ppq_densities_NR/make_Ppq_densities_ER (-> Fortran
Ppq_vector_NR/Ppq_vector_ER) -- no new Fortran code and no new numerics.

For MCMC, pass precomputed normalization tables (python/cli/normgrid.py,
built once on a batch system) as Ppq_table_NR/Ppq_table_ER: the integral then
costs tens of microseconds instead of ~2 s, and raises rather than ever
extrapolating outside the table's parameter box.  A table is a file path or
a registered name (python/cli/normtables.py fetches and caches it).

At MCMC setup, pass the prior's hard bounds as prior_bounds so a prior that
reaches outside a table fails at construction, with the box to rebuild the
table with, rather than hours into the run; check the starting walker
positions with check_points.  table_bounds() gives a prior box the tables
cover.  In log_prob, reject points outside the prior before calling the
likelihood:

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf          # never reaches the table
        return lp + log_likelihood(theta)
"""

import os
import sys

import numpy as np

# _Ppq_fort_bindings.py and normgrid.py/normtables.py are no longer siblings
# of this file -- they live in internal/ and cli/ respectively.
_PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
_INTERNAL_DIR = os.path.join(_PYTHON_DIR, "internal")
_CLI_DIR = os.path.join(_PYTHON_DIR, "cli")

sys.path.insert(0, _INTERNAL_DIR)
from _Ppq_fort_bindings import (make_Ppq_densities_ER, make_Ppq_densities_NR,
                                 Ppq_region_ER, Ppq_region_NR, version as _fort_version)


class Ppq_compiled:
    """
    Requires the compiled Fortran library, usually by running inside a
    published container -- see the "Two ways to use this" page.

    Parameters
    ----------
    ep_range, eq_range : (float, float)
        The (Ep, Eq) region the PDF is normalized over. Scalars at this
        boundary (not a tuple) only below the C ABI, in
        python/internal/_Ppq_fort_bindings.py and the Fortran bind(c)
        interface itself -- ctypes/bind(c) only marshal simple scalars,
        not tuples, across that boundary.
    ep_data, eq_data : array_like
        The fixed observed dataset this fit is evaluated against.
    norm_epsrel, norm_epsabs : float or None
        Convergence tolerance for the normalization integral's doubling-
        verified quadrature (see Ppq_region_NR's doc comment in
        src/Ppq_fort_m.f90) -- unrelated to the size of ep_data/eq_data,
        hence the norm_ prefix.  Required for any band without a table.
    Ppq_table_NR, Ppq_table_ER : path, normgrid.NormInterpolator, or None
        Precomputed normalization table for that band (built by
        python/cli/normgrid.py for exactly this region), or a name registered in
        python/cli/table_registry.json.  If given, Ppq_integral_NR/Ppq_integral_ER
        interpolate it instead of running the quadrature.
    prior_bounds : dict or None
        The MCMC prior's hard bounds, {param: (lo, hi)} for k (NR), V, p0,
        p10, q0, q10 (Z/eps may be given as their fixed values), flat --
        same convention as generate_events()'s F0_NR/F0_ER keyword
        arguments, since F0 is the one parameter that genuinely differs
        per band:
            {"V": (2.7, 3.3), ..., "k": (0.13, 0.22),
             "F0_NR": (1e-5, 1), "F0_ER": (0.1, 0.35)}
        Checked against every table given, assuming the prior also enforces
        p10 >= p0 and q10 >= q0; raises normgrid.PriorNotCoveredError if it
        reaches outside one.
    n_workers : int or None
        Threads used for large batched Ppq_vector_NR/Ppq_vector_ER calls;
        see goodness_of_fit.make_Ppq_densities_NR.
    """

    def __init__(self, ep_range, eq_range, ep_data, eq_data, *,
                 norm_epsrel=None, norm_epsabs=None,
                 Ppq_table_NR=None, Ppq_table_ER=None, prior_bounds=None,
                 n_workers=None):
        # A table only ever replaces the normalization integral, never the
        # per-event PDF evaluation (_Ppq_densities_NR/_Ppq_densities_ER always need
        # Ppq_vector_NR/Ppq_vector_ER) -- so every Ppq_compiled needs the compiled
        # library eventually. Load it now and raise here, with the clear
        # LD_LIBRARY_PATH message from _load_library(), rather than
        # succeeding silently and failing on the first likelihood call
        # deep into a fit or MCMC run.
        _fort_version()
        self.ep_min, self.ep_max = ep_range
        self.eq_min, self.eq_max = eq_range
        self.ep_data = np.ascontiguousarray(np.asarray(ep_data, dtype=np.float64))
        self.eq_data = np.ascontiguousarray(np.asarray(eq_data, dtype=np.float64))
        self.norm_epsrel = norm_epsrel
        self.norm_epsabs = norm_epsabs
        self.n_workers = n_workers
        self.Ppq_table_NR = self._load_table(Ppq_table_NR, "NR")
        self.Ppq_table_ER = self._load_table(Ppq_table_ER, "ER")
        if prior_bounds is not None:
            if not self._tables():
                raise ValueError("prior_bounds is checked against the tables; no table was given")
            for t in self._tables():
                t.check_prior(self._band_view(prior_bounds, t.band))

    def _tables(self):
        return [t for t in (self.Ppq_table_NR, self.Ppq_table_ER) if t is not None]

    @staticmethod
    def _band_view(d, band):
        """d with F0_NR/F0_ER resolved to this band's bare F0 (whichever is
        missing is simply dropped, same as the other band's F0); k dropped
        for ER."""
        v = {p: x for p, x in d.items() if p not in ("F0_NR", "F0_ER")}
        f0_key = f"F0_{band}"
        if f0_key in d:
            v["F0"] = d[f0_key]
        if band == "ER":
            v.pop("k", None)
        return v

    def check_points(self, points):
        """Raise normgrid.PriorNotCoveredError unless every point is inside
        every table: e.g. the starting walker positions, {param: array} laid
        out like prior_bounds (flat, F0_NR/F0_ER instead of a bare F0)."""
        for t in self._tables():
            t.check_points(self._band_view(points, t.band))

    def table_bounds(self):
        """{param: (lo, hi)}: a single flat prior box every given table
        covers, same shape as prior_bounds -- pass straight back in as
        prior_bounds=table_bounds(), no reshaping needed. F0_NR/F0_ER are
        each band's own box; a parameter the bands share (V, p0, p10, q0,
        q10) is intersected across every table given (raises ValueError if
        two tables' boxes for a shared parameter don't overlap at all)."""
        tables = self._tables()
        if not tables:
            raise ValueError("no table was given")
        out = {}
        for t in tables:
            tb = t.table_bounds()                  # bare {param: (lo, hi)} for this band
            out[f"F0_{t.band}"] = tb["F0"]
            if t.band == "NR":
                out["k"] = tb["k"]
            for p in ("V", "p0", "p10", "q0", "q10"):
                lo, hi = tb[p]
                if p in out:
                    prev_lo, prev_hi = out[p]
                    lo, hi = max(lo, prev_lo), min(hi, prev_hi)
                    if lo > hi:
                        raise ValueError(f"{p}: NR/ER table boxes don't overlap "
                                         f"({(prev_lo, prev_hi)} vs {tb[p]})")
                out[p] = (lo, hi)
        return out

    def _load_table(self, table, band):
        if table is None:
            return None
        sys.path.insert(0, _CLI_DIR)
        import normgrid
        if not isinstance(table, normgrid.NormInterpolator):
            if not os.path.exists(str(table)):
                import normtables
                table = normtables.fetch(table)
            table = normgrid.NormInterpolator.from_hdf5(table)
        if table.band != band:
            raise ValueError(f"table is for the {table.band} band, expected {band}")
        ep_range, eq_range = (self.ep_min, self.ep_max), (self.eq_min, self.eq_max)
        if (not np.allclose(table.ep_range, ep_range, rtol=0, atol=1e-12)
                or not np.allclose(table.eq_range, eq_range, rtol=0, atol=1e-12)):
            raise ValueError(f"table was built for ep_range={table.ep_range}, eq_range={table.eq_range}, "
                             f"not ep_range={ep_range}, eq_range={eq_range}")
        return table

    def _quadrature_tolerances(self):
        if self.norm_epsrel is None or self.norm_epsabs is None:
            raise ValueError("norm_epsrel and norm_epsabs are required to compute a normalization "
                             "integral without a precomputed table")
        return dict(epsrel=self.norm_epsrel, epsabs=self.norm_epsabs)

    # ---- NR (Ppq_NR) band ----

    def Ppq_integral_NR(self, *, k, Z, F0, eps, V, p0, p10, q0, q10):
        """Normalization: integral of Ppq_NR over this region."""
        if self.Ppq_table_NR is not None:
            return self.Ppq_table_NR(k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
        return Ppq_region_NR(self.ep_min, self.ep_max, self.eq_min, self.eq_max,
                            **self._quadrature_tolerances(),
                            k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)

    def _Ppq_densities_NR(self, *, k, Z, F0, eps, V, p0, p10, q0, q10):
        """The full-plane density (see goodness_of_fit.make_Ppq_densities_NR) at the bound
        dataset, not yet renormalized to this region. Private: nothing
        outside Ppq_normalized_densities_NR needs this on its own -- a
        fit/MCMC always divides by Ppq_integral_NR(...), never uses this
        alone."""
        pdf_func = make_Ppq_densities_NR(k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10,
                                               q0=q0, q10=q10, n_workers=self.n_workers)
        return pdf_func(self.ep_data, self.eq_data)

    def Ppq_normalized_densities_NR(self, *, k, Z, F0, eps, V, p0, p10, q0, q10):
        """_Ppq_densities_NR(...) / Ppq_integral_NR(...), elementwise."""
        return (self._Ppq_densities_NR(k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
                / self.Ppq_integral_NR(k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10))

    # ---- ER (Ppq_ER) band ----

    def Ppq_integral_ER(self, *, F0, eps, V, p0, p10, q0, q10):
        """Normalization: integral of Ppq_ER over this region."""
        if self.Ppq_table_ER is not None:
            return self.Ppq_table_ER(F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
        return Ppq_region_ER(self.ep_min, self.ep_max, self.eq_min, self.eq_max,
                            **self._quadrature_tolerances(),
                            F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)

    def _Ppq_densities_ER(self, *, F0, eps, V, p0, p10, q0, q10):
        """The full-plane density (see goodness_of_fit.make_Ppq_densities_ER) at the bound
        dataset, not yet renormalized to this region. Private: nothing
        outside Ppq_normalized_densities_ER needs this on its own -- a
        fit/MCMC always divides by Ppq_integral_ER(...), never uses this
        alone."""
        pdf_func = make_Ppq_densities_ER(F0=F0, eps=eps, V=V, p0=p0, p10=p10,
                                               q0=q0, q10=q10, n_workers=self.n_workers)
        return pdf_func(self.ep_data, self.eq_data)

    def Ppq_normalized_densities_ER(self, *, F0, eps, V, p0, p10, q0, q10):
        """_Ppq_densities_ER(...) / Ppq_integral_ER(...), elementwise."""
        return (self._Ppq_densities_ER(F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
                / self.Ppq_integral_ER(F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10))
