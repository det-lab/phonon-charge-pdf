"""
Ppq_compiled is the one intended entrypoint for using this library in a fit or
MCMC -- if you want normalized PDF values or a likelihood, start here:

    from Ppq_fort import Ppq_compiled
    likelihood = Ppq_compiled(ep_min, ep_max, eq_min, eq_max, ep_data, eq_data,
                 norm_epsrel=1e-4, norm_epsabs=1e-10)
    normalized = likelihood.Ppq_normalized_values_NR(k=k, Z=Z, F0=F0, eps=eps, V=V,
                                             p0=p0, p10=p10, q0=q0, q10=q10)

Ppq_compiled always needs the compiled Fortran library (a table only replaces the
normalization integral, never the per-event PDF evaluation), so
construction eagerly loads it and raises immediately if it can't be found,
rather than succeeding and failing on the first likelihood call instead.

See the README's "Normalizing a likelihood fit to a region" section for the
full walkthrough, including precomputed tables for MCMC.  If you instead
want to generate simulated (Ep, Eq) events -- e.g. to validate a fit
pipeline or feed a downstream analysis -- see event_generator.py
(python/event_generator.py), a second, independent entrypoint with no
Fortran dependency.  Ppq_py.py (python/Ppq_py.py) is a third, public
entrypoint: the pure-Python reference implementation event_generator.py
and this module are themselves checked against -- see its own module
docstring.  Everything else under python/ -- python/internal/
_Ppq_fort_bindings.py (raw, unnormalized PDF evaluation) and python/cli/
normgrid.py (table machinery) -- is either what Ppq_compiled/event_generator.py
are built from or infrastructure for maintaining this library, not a
fourth way to use it.

Ppq_compiled: the fixed context for one fit/MCMC run -- an (Ep, Eq) region, the
observed dataset, and the normalization-integral's convergence tolerance
-- exposing, per parameter point, the un-normalized PDF at the bound
dataset, the region's normalization integral, and their ratio (normalized
PDF values), for both the NR (Ppq_NR) and ER (Ppq_ER) bands.

All physics parameters (k, Z, F0, eps, V, p0, p10, q0, q10) are passed
fresh to every method call rather than bound at construction: those are
what an MCMC step actually varies, while the region, dataset, and
tolerance don't change across a run.

Built entirely on the already-validated primitives in _Ppq_fort_bindings.py --
Ppq_region_NR/Ppq_region_ER (-> the Fortran symbols of the same name, a
doubling-verified nested Gauss-Legendre quadrature) and
make_Ppq_unnormalized_NR/make_Ppq_unnormalized_ER (-> Fortran
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
from _Ppq_fort_bindings import (make_Ppq_unnormalized_ER, make_Ppq_unnormalized_NR,
                                 Ppq_region_ER, Ppq_region_NR, version as _fort_version)


class Ppq_compiled:
    """
    ep_min, ep_max, eq_min, eq_max : float
        The (Ep, Eq) region the PDF is normalized over.
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
        The MCMC prior's hard bounds, {param: (lo, hi)} for k (NR), F0, V,
        p0, p10, q0, q10 (Z/eps may be given as their fixed values).  Keys
        shared by both bands go at the top level; a band's own parameters
        (e.g. its Fano factor) go under "NR"/"ER" and override them:
            {"V": (2.7, 3.3), ..., "NR": {"k": (0.13, 0.22), "F0": (1e-5, 1)},
                                   "ER": {"F0": (0.1, 0.35)}}
        Checked against every table given, assuming the prior also enforces
        p10 >= p0 and q10 >= q0; raises normgrid.PriorNotCoveredError if it
        reaches outside one.
    n_workers : int or None
        Threads used for large batched Ppq_vector_NR/Ppq_vector_ER calls;
        see make_Ppq_unnormalized_NR.
    """

    def __init__(self, ep_min, ep_max, eq_min, eq_max, ep_data, eq_data, *,
                 norm_epsrel=None, norm_epsabs=None,
                 Ppq_table_NR=None, Ppq_table_ER=None, prior_bounds=None,
                 n_workers=None):
        # A table only ever replaces the normalization integral, never the
        # per-event PDF evaluation (Ppq_values_NR/Ppq_values_ER always need
        # Ppq_vector_NR/Ppq_vector_ER) -- so every Ppq_compiled needs the compiled
        # library eventually. Load it now and raise here, with the clear
        # LD_LIBRARY_PATH message from _load_library(), rather than
        # succeeding silently and failing on the first likelihood call
        # deep into a fit or MCMC run.
        _fort_version()
        self.ep_min, self.ep_max = ep_min, ep_max
        self.eq_min, self.eq_max = eq_min, eq_max
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
        """The shared keys of d, overridden by d[band]; k only for NR."""
        v = {p: x for p, x in d.items() if p not in ("NR", "ER")}
        v.update(d.get(band, {}))
        if band == "ER":
            v.pop("k", None)
        return v

    def check_points(self, points):
        """Raise normgrid.PriorNotCoveredError unless every point is inside
        every table: e.g. the starting walker positions, {param: array} laid
        out like prior_bounds (band-specific arrays under "NR"/"ER")."""
        for t in self._tables():
            t.check_points(self._band_view(points, t.band))

    def table_bounds(self):
        """{band: {param: (lo, hi)}}: a prior box each given table covers,
        for a prior that enforces p10 >= p0 and q10 >= q0.  A parameter the
        bands share (V, p0, p10, q0, q10) must lie in both."""
        if not self._tables():
            raise ValueError("no table was given")
        return {t.band: t.table_bounds() for t in self._tables()}

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
        region = (self.ep_min, self.ep_max, self.eq_min, self.eq_max)
        if not np.allclose(table.region, region, rtol=0, atol=1e-12):
            raise ValueError(f"table was built for region {table.region}, not {region}")
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

    def Ppq_values_NR(self, *, k, Z, F0, eps, V, p0, p10, q0, q10):
        """Un-normalized Ppq_NR at the bound dataset."""
        pdf_func = make_Ppq_unnormalized_NR(k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10,
                                               q0=q0, q10=q10, n_workers=self.n_workers)
        return pdf_func(self.ep_data, self.eq_data)

    def Ppq_normalized_values_NR(self, *, k, Z, F0, eps, V, p0, p10, q0, q10):
        """Ppq_values_NR(...) / Ppq_integral_NR(...), elementwise."""
        return (self.Ppq_values_NR(k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
                / self.Ppq_integral_NR(k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10))

    # ---- ER (Ppq_ER) band ----

    def Ppq_integral_ER(self, *, F0, eps, V, p0, p10, q0, q10):
        """Normalization: integral of Ppq_ER over this region."""
        if self.Ppq_table_ER is not None:
            return self.Ppq_table_ER(F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
        return Ppq_region_ER(self.ep_min, self.ep_max, self.eq_min, self.eq_max,
                            **self._quadrature_tolerances(),
                            F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)

    def Ppq_values_ER(self, *, F0, eps, V, p0, p10, q0, q10):
        """Un-normalized Ppq_ER at the bound dataset."""
        pdf_func = make_Ppq_unnormalized_ER(F0=F0, eps=eps, V=V, p0=p0, p10=p10,
                                               q0=q0, q10=q10, n_workers=self.n_workers)
        return pdf_func(self.ep_data, self.eq_data)

    def Ppq_normalized_values_ER(self, *, F0, eps, V, p0, p10, q0, q10):
        """Ppq_values_ER(...) / Ppq_integral_ER(...), elementwise."""
        return (self.Ppq_values_ER(F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
                / self.Ppq_integral_ER(F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10))
