"""phonon_charge_pdf.erecoil_fit: a smooth, globally-analytic true-recoil-
energy spectrum (PErecoil) -- a sum of exponentials plus a constant floor,
fit to real calibration data -- for the compiled Fortran quadrature path.

Why a fit, not a table (adr/0012-perecoil-handles-register-a-fit-not-a-
table.md): a PCHIP-interpolated or smoothed table (erecoil_table.py) can
describe real calibration data perfectly and *still* make the
Gauss-Legendre region integral untrustworthy in part of the parameter
space. The quadrature's precision floor turned out to depend on how much
local structure the integrand carries -- genuine or just counting noise --
not on how well the representation describes the data; it got
measurably worse specifically near the phonon resolution model's
degenerate edge (p10 -> p0), where the integration window narrows and the
spectrum's own local shape starts to dominate. A function with no local
structure at any scale is immune to this, the same way the existing
closed-form biexponential always has been -- which is what every
supported family in this module is, by construction.

erecoil_table.PErecoilTable stays the right tool for simulation
(event_generator.py's sampling) and as the thing to check a fit against
(check_fit_against_events below) -- just not for the integration path,
which is the only thing this module's PErecoilFit feeds.

A small, curated set of families (FAMILY_FORMULAS), deliberately not an
arbitrary user-specified function (also adr/0012 -- an open-ended
expression language has the same problem a table does: nothing about it
guarantees the smoothness the quadrature actually needs, and it would
undo ADR 0001's point in evaluating it from Fortran besides):

    sum_of_exponentials_plus_const:
        density(Erecoil) = sum(a[i] * exp(-Erecoil / b[i]) for i in range(len(a))) + C

    turnon_times_sum_of_exponentials_plus_const (adr/0016):
        density(Erecoil) = (0.5*(1 + erf((Erecoil-x_on)/w_on))) *
                           (sum(a[i] * exp(-Erecoil / b[i]) for i in range(len(a))) + C)

Every term is globally smooth by construction (erf included -- smooth and
bounded everywhere). Add a new curated family to FAMILY_FORMULAS (and a
matching Fortran registration path) if a detector's real spectrum
genuinely needs one -- e.g. a term with a real interior mode -- never by
letting a caller supply an arbitrary expression.

Each PErecoilFit also carries its own valid_min/valid_max (adr/0016): the
range it's actually trustworthy over, separate from erecoil_max (the
normalization bound) and separate from where the underlying family stays
merely smooth (everywhere, for every family here). Evaluating outside
that range raises normgrid.OutOfBoxError rather than silently returning
a number -- see PErecoilFit.__call__ and adr/0016 for why. The same two
numbers also protect a direct Fortran caller once registered
(PErecoil_fit_register/PErecoil_turnon_fit_register): Ppq_NR/ER return a
quiet NaN instead, the same convention this library already uses for a
failed Gauss-Legendre doubling.
"""

import hashlib
import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.stats import kstest

__all__ = ["PErecoilFit", "fit_sum_of_exponentials", "fit_turnon_times_sum_of_exponentials",
          "check_fit_against_events", "FAMILY_FORMULAS",
          "CF252_BIEXPONENTIAL_NR", "CF252_BIEXPONENTIAL_ER"]

_PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
_CLI_DIR = os.path.join(_PYTHON_DIR, "cli")

# One entry per supported family, written once and reused verbatim by
# every fit of that family's own perecoil_identity() -- documentation
# embedded in data, never evaluated back (see this module's own
# docstring and adr/0012 for why not an eval-able expression).
FAMILY_FORMULAS = {
    "sum_of_exponentials_plus_const":
        "density(Erecoil) = sum(a[i] * exp(-Erecoil / b[i]) for i in range(len(a))) + C",
    "turnon_times_sum_of_exponentials_plus_const":
        "density(Erecoil) = (0.5*(1 + erf((Erecoil - x_on) / w_on))) * "
        "(sum(a[i] * exp(-Erecoil / b[i]) for i in range(len(a))) + C)",
}


def _density(x, a, b, C):
    x = np.asarray(x, dtype=float)
    return sum(ai * np.exp(-x / bi) for ai, bi in zip(a, b)) + C


def _antiderivative(x, a, b, C):
    """Exact antiderivative of _density -- used both to normalize a fit
    (integrate [0, erecoil_max]) and to build a conditional CDF for
    check_fit_against_events below. Closed-form, so only valid for the
    sum_of_exponentials_plus_const family -- the turnon family below has
    no closed-form antiderivative (erf times a sum of exponentials) and
    is normalized/CDF'd numerically instead (see _raw_integral)."""
    x = np.asarray(x, dtype=float)
    return sum(-ai * bi * np.exp(-x / bi) for ai, bi in zip(a, b)) + C * x


def _density_turnon(x, a, b, C, x_on, w_on):
    """density(Erecoil) for the turnon_times_sum_of_exponentials_plus_const
    family: a smooth rising step (the same erf-based device used for the
    falling step in the Cf ER step-floor model this mirrors) times the
    plain family's own sum-of-exponentials-plus-floor shape. Models a
    detector's own near-threshold trigger/ionization efficiency ramping
    up, rather than discarding that data and replacing it with a
    backward extrapolation of the above-threshold decline."""
    x = np.asarray(x, dtype=float)
    turnon = 0.5 * (1.0 + erf((x - x_on) / w_on))
    return turnon * _density(x, a, b, C)


def _raw_integral(family, lo, hi, a, b, C, x_on=None, w_on=None):
    """Integral of the (unnormalized) raw density over [lo, hi], dispatched
    by family: the closed-form antiderivative for the plain family, numeric
    quadrature for turnon (no closed form exists for erf times a sum of
    exponentials). Shared by PErecoilFit's own normalization (__init__,
    against the raw a/b/C before dividing by it) and check_fit_against_events
    (against a fit's already-normalized a/b/C, to build its CDF)."""
    if family == "turnon_times_sum_of_exponentials_plus_const":
        val, _ = quad(lambda t: _density_turnon(t, a, b, C, x_on, w_on), lo, hi, limit=200)
        return val
    return float(_antiderivative(hi, a, b, C) - _antiderivative(lo, a, b, C))


def _raw_density(family, x, a, b, C, x_on=None, w_on=None):
    """density(x) dispatched by family -- the plain closed form, or the
    turnon family's erf-windowed version. See _raw_integral for the
    matching integral dispatch."""
    if family == "turnon_times_sum_of_exponentials_plus_const":
        return _density_turnon(x, a, b, C, x_on, w_on)
    return _density(x, a, b, C)


class PErecoilFit:
    """A true-recoil-energy PDF: one of this module's small, curated set of
    families (see FAMILY_FORMULAS and this module's own docstring for why
    that's deliberate, not a limitation, and why only a curated set).
    Call with a single Erecoil (keV):

        fit = fit_sum_of_exponentials(erecoil, counts, raw_n_per_bin)
        fit(12.3)

    Unlike erecoil_table.PErecoilTable, this stays defined -- and smooth --
    for every Erecoil >= 0: no erecoil_range, nothing to clip a Fortran
    search window to (adr/0012). That smoothness is unconditional and has
    nothing to do with where this is *trustworthy* to evaluate -- see
    valid_min/valid_max below (adr/0016). The Erecoil < 0 convention below
    matches PErecoilTable/PErecoil_NR/ER's own for consistency, not
    because anything would otherwise go wrong there.

    Normalized to integrate to 1 over [0, erecoil_max] -- required
    explicitly at construction, since a nonzero constant floor C makes
    the density's integral over the full [0, infinity) diverge; there is
    no domain-free normalization for this family the way there is for
    the closed form's own two bare (already-normalized) exponentials.
    erecoil_max does not bound where this may be *evaluated* in principle
    (the underlying family stays smooth well past it) -- only what range
    the stored a/C were scaled against.

    valid_min/valid_max (default 0.0/erecoil_max) mark the range this fit
    is actually trustworthy over -- not a statement about smoothness, but
    about whether it was ever checked against real data there. Calling
    this fit outside [valid_min, valid_max] raises normgrid.OutOfBoxError
    rather than silently returning a number (adr/0016): the same
    mechanism erecoil_table.PErecoilTable.__call__ already uses for a
    table's own range. The same two numbers also cross into Fortran once
    this fit is registered (fortran_params below, or directly via
    PErecoil_fit_register/PErecoil_turnon_fit_register) -- a direct
    Fortran caller gets the identical guard (a quiet NaN in place of a
    raised exception, Fortran having no exceptions, but the same "don't
    trust this" signal). The defaults preserve this family's historical
    behavior (smooth and trustworthy by construction from 0 to
    erecoil_max); a narrower valid region is for a caller who has
    separately established that only part of that range is actually good
    -- e.g. a fit trained on a truncated subset of its own data, where
    erecoil_max itself should typically be narrowed to match rather than
    left at a wider default.
    """

    def __init__(self, a, b, C, *, erecoil_max, valid_min=0.0, valid_max=None,
                 family="sum_of_exponentials_plus_const", x_on=None, w_on=None,
                 detector="", band="", calibration_source="", source="", chi2_dof=None,
                 is_closed_form=False):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.ndim != 1 or a.shape != b.shape:
            raise ValueError("a and b must be 1-D arrays of the same length")
        if np.any(b <= 0):
            raise ValueError("b (decay lengths, keV) must be positive")
        if family not in FAMILY_FORMULAS:
            raise ValueError(f"unknown family {family!r}; known families: {sorted(FAMILY_FORMULAS)}")
        if family == "turnon_times_sum_of_exponentials_plus_const":
            if x_on is None or w_on is None:
                raise ValueError("the turnon family requires x_on and w_on")
            if not w_on > 0:
                raise ValueError(f"w_on must be > 0, got {w_on!r}")
        if not erecoil_max > 0:
            raise ValueError(f"erecoil_max must be > 0, got {erecoil_max!r}")
        if valid_max is None:
            valid_max = erecoil_max
        if not valid_min >= 0:
            raise ValueError(f"valid_min must be >= 0, got {valid_min!r}")
        if not valid_max > valid_min:
            raise ValueError(f"valid_max ({valid_max!r}) must be > valid_min ({valid_min!r})")
        raw_norm = _raw_integral(family, 0.0, erecoil_max, a, b, C, x_on, w_on)
        if not raw_norm > 0:
            raise ValueError(f"fit integrates to {raw_norm!r} over [0, {erecoil_max}]; cannot normalize")
        self.family = family
        self.a = a / raw_norm
        self.b = b
        self.C = float(C) / raw_norm
        self.x_on, self.w_on = x_on, w_on
        self.erecoil_max = float(erecoil_max)
        self.valid_min, self.valid_max = float(valid_min), float(valid_max)
        self.detector, self.band = detector, band
        self.calibration_source, self.source = calibration_source, source
        self.chi2_dof = chi2_dof
        # True only for CF252_BIEXPONENTIAL_NR/ER below: registering either
        # with Fortran gives a real handle (> 0, not 0) and costs nothing
        # extra numerically (eval_fit on these two terms computes exactly
        # the hardcoded closed form), but for normgrid's `perecoil`
        # provenance check (adr/0002) this needs to read as the SAME thing
        # a table built against the literal closed form recorded
        # ({"kind": "closed_form"}, no handle at all) -- they are the same
        # spectrum, just reached two different ways, and the check should
        # not see a difference that isn't physically there.
        self.is_closed_form = is_closed_form

    def __call__(self, Erecoil):
        """Evaluate the fit -- but first, for any Erecoil >= 0 outside
        [valid_min, valid_max], raise normgrid.OutOfBoxError rather than
        silently returning a number this fit was never checked against
        real data for (adr/0016). Mirrors erecoil_table.PErecoilTable's
        own call-time check (same tiny relative tolerance for float
        noise, same exception type); Erecoil < 0 keeps returning 0.0,
        matching that same convention."""
        Erecoil = np.asarray(Erecoil, dtype=float)
        tol = 1e-12 * (self.valid_max - self.valid_min)
        bad = (Erecoil >= 0.0) & ((Erecoil < self.valid_min - tol) | (Erecoil > self.valid_max + tol))
        if np.any(bad):
            sys.path.insert(0, _CLI_DIR)
            from normgrid import OutOfBoxError
            bad_vals = np.atleast_1d(Erecoil)[np.atleast_1d(bad)]
            source = f"{self.detector}/{self.band}" if self.detector or self.band else "(in memory)"
            raise OutOfBoxError(
                f"Erecoil={bad_vals!r} outside this fit's valid region "
                f"[{self.valid_min:.6g}, {self.valid_max:.6g}] for {source}\n  This fit was only "
                "ever checked against real data in that range (see valid_min/valid_max); "
                "evaluating outside it would trust an unchecked extrapolation, not a validated "
                "result. Restrict your query to the valid region, or build a fit that covers "
                "the range you actually need.",
                axis="Erecoil", value=bad_vals, box=(self.valid_min, self.valid_max), source=source)
        raw = _raw_density(self.family, Erecoil, self.a, self.b, self.C, self.x_on, self.w_on)
        return np.where(Erecoil < 0.0, 0.0, raw)

    @property
    def fortran_params(self):
        """(a, b, C, valid_min, valid_max) ready for PErecoil_fit_register
        (the plain family -- the turnon family's own extra x_on/w_on
        aren't part of this tuple, see _load_erecoil_fit) -- a/b/C already
        normalized, so no extra normalization constant needs to cross into
        Fortran. valid_min/valid_max (adr/0016) let a direct Fortran
        caller get the same out-of-range-evaluation guard
        PErecoilFit.__call__ already gives a Python one."""
        return self.a, self.b, self.C, self.valid_min, self.valid_max

    @property
    def sha256(self):
        """sha256 of this fit's family and (already-normalized) a/b/C --
        used by normgrid's `perecoil` provenance check (adr/0002, extended
        for the "fit" kind by adr/0012) to tell whether a normalization
        table's assumed PErecoil fit matches a given PErecoilFit. Only
        kind/sha256 are compared for a match; two fits built from
        identical final coefficients hash identically regardless of
        detector/source, and two with genuinely different coefficients
        (including from different normalization ranges) don't."""
        h = hashlib.sha256()
        h.update(self.family.encode())
        h.update(self.a.tobytes())
        h.update(self.b.tobytes())
        h.update(np.asarray(self.C, dtype=np.float64).tobytes())
        if self.family == "turnon_times_sum_of_exponentials_plus_const":
            h.update(np.asarray([self.x_on, self.w_on], dtype=np.float64).tobytes())
        return h.hexdigest()

    def perecoil_identity(self):
        """{"kind": "closed_form"} for CF252_BIEXPONENTIAL_NR/ER (see
        is_closed_form above); otherwise {"kind": "fit", "family",
        "formula", "a", "b", "C", "sha256", "detector",
        "calibration_source", "source"} -- this fit's identity as
        normgrid's `perecoil` provenance tag records it. `formula` is
        FAMILY_FORMULAS[self.family] verbatim: documentation for a human
        reading the HDF5 file directly, never read back or evaluated."""
        if self.is_closed_form:
            return {"kind": "closed_form"}
        identity = {"kind": "fit", "family": self.family, "formula": FAMILY_FORMULAS[self.family],
                   "a": self.a.tolist(), "b": self.b.tolist(), "C": self.C, "sha256": self.sha256,
                   "detector": self.detector, "calibration_source": self.calibration_source,
                   "source": self.source, "valid_min": self.valid_min, "valid_max": self.valid_max}
        if self.family == "turnon_times_sum_of_exponentials_plus_const":
            identity["x_on"], identity["w_on"] = self.x_on, self.w_on
        return identity


# The library's own closed-form biexponentials (src/Ppq_fort_s.f90's
# PErecoil_NR_impl/ER_impl, Ppq_py.py's PErecoil_NR/ER), exported by name so
# a caller can pass them explicitly instead of a band's PErecoil argument
# silently defaulting to them (adr/0011-recoil-spectrum-always-explicit.md:
# "nobody chooses the closed form; it's what you get by saying nothing").
# Ordinary PErecoilFit instances -- registering one costs nothing extra
# (eval_fit on these two terms computes exactly the hardcoded closed form,
# to floating-point precision) and needs no sampling support. The one thing
# that does need to know about them specifically: is_closed_form=True, so
# perecoil_identity() reads as {"kind": "closed_form"} -- the same identity
# a table built against the literal closed form already records -- rather
# than {"kind": "fit", ...}, which would otherwise make normgrid's
# provenance check (adr/0002) see a mismatch between two things that are
# actually the same spectrum. erecoil_max=1e5 keV is many thousands of
# decay lengths past PGd (ER's longest, 279.6 keV) -- the normalization
# integral below it differs from the true [0, infinity) integral by a
# wildly negligible amount.
_PNa, _PNb, _PNd = 0.53693208, 6.41515782, 23.71789286
_PGa, _PGb, _PGd = 0.573211975, 0.169520023, 279.552394
CF252_BIEXPONENTIAL_NR = PErecoilFit([_PNa / _PNb, (1.0 - _PNa) / _PNd], [_PNb, _PNd], 0.0,
                                     erecoil_max=1e5, calibration_source="Cf-252", is_closed_form=True)
CF252_BIEXPONENTIAL_ER = PErecoilFit([_PGa / _PGb, (1.0 - _PGa) / _PGd], [_PGb, _PGd], 0.0,
                                     erecoil_max=1e5, calibration_source="Cf-252", is_closed_form=True)


def fit_sum_of_exponentials(erecoil, counts, raw_n_per_bin, *, n_terms=2, detector="", band="",
                            calibration_source="", source="", n_restarts=30, rng=None):
    """Weighted least-squares fit of
    density(Erecoil) = sum(a[i]*exp(-Erecoil/b[i])) + C to (erecoil, counts)
    -- counts a density (count/bin-width, e.g. from an equal-count/quantile
    histogram), weighted by the TRUE per-bin uncertainty,
    counts * (1/sqrt(raw_n_per_bin)), not 1/sqrt(counts): counts is a
    density, not itself a Poisson count, so 1/sqrt(counts) is the wrong
    error model (it varies the wrong way with bin width/position) --
    getting this wrong made every functional family tried look inadequate
    until it was fixed; see adr/0012's own investigation history.
    raw_n_per_bin: the actual raw event count behind each bin (not
    density) -- pass the real per-bin value, not an assumed constant,
    even though an equal-count binning convention makes it approximately
    so by construction.

    Random multi-start (n_restarts tries, default rng: a new
    numpy.random.default_rng() each call) rather than one fixed initial
    guess: a sum of exponentials is a classically ill-conditioned
    nonlinear fit (components trade off against each other, and a
    one-term-collapses-to-a-constant degeneracy is easy to land on from a
    bad start) -- see adr/0012 for a worked example of exactly that
    failure mode.

    Fits in a reparametrized space -- a[i] and C carried as squared values,
    b[i] as exp(log_b[i]) -- rather than the raw a/b/C curve_fit sees
    externally. Without this, plain unconstrained curve_fit can (and did,
    found while checking this function against Cf's own ER-band data) land
    on two failure modes that still pass the old "pred > 0 at every bin
    center" check: two large, nearly-canceling same-decay-length terms
    (e.g. a ~= [-2.16, +2.16], b ~= [112.8, 113.4]), and a decay length so
    long it's a disguised constant canceling against a negative C. Both are
    invisible at the sparse bin centers but can cross zero between them --
    checked here on a dense 2000-point grid across the full
    [erecoil.min(), erecoil.max()] range, not just the bins. Squaring
    (not logging) the amplitudes specifically -- rather than reusing the
    log trick for everything -- lets a[i] or C reach zero smoothly if the
    data wants a term to drop out, instead of needing an unbounded log
    excursion to get there.

    Returns a PErecoilFit (erecoil_max = erecoil.max(); chi2_dof recorded
    for inspection -- check it, or better, check_fit_against_events with
    the real unbinned events, before trusting the fit)."""
    erecoil = np.asarray(erecoil, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    raw_n_per_bin = np.asarray(raw_n_per_bin, dtype=np.float64)
    if not (erecoil.shape == counts.shape == raw_n_per_bin.shape):
        raise ValueError("erecoil, counts, raw_n_per_bin must be the same shape")
    if np.any(raw_n_per_bin <= 0):
        raise ValueError("raw_n_per_bin must be positive everywhere (a bin with no real events "
                         "behind it has no basis for a weight)")
    sigma_abs = counts / np.sqrt(raw_n_per_bin)
    rng = np.random.default_rng() if rng is None else rng
    scale, erange = float(counts.max()), float(erecoil.max() - erecoil.min())
    xs_dense = np.linspace(erecoil.min(), erecoil.max(), 2000)

    def model_hybrid(x, *params):
        p_a = np.asarray(params[:n_terms])
        log_b = np.asarray(params[n_terms:2 * n_terms])
        p_C = params[-1]
        return _density(x, p_a ** 2, np.exp(log_b), p_C ** 2)

    best = None
    for _ in range(n_restarts):
        a0 = rng.uniform(0.1, 1.0, n_terms) * scale
        b0 = rng.uniform(0.05, 1.0, n_terms) * erange
        C0 = rng.uniform(0.0, 0.1) * scale
        p0 = [*np.sqrt(a0), *np.log(b0), np.sqrt(C0)]
        try:
            popt, _ = curve_fit(model_hybrid, erecoil, counts, p0=p0,
                                sigma=sigma_abs, absolute_sigma=True, method="lm", maxfev=20000)
        except Exception:
            continue
        pred_dense = model_hybrid(xs_dense, *popt)
        if np.any(~np.isfinite(pred_dense)) or np.any(pred_dense <= 0):
            continue
        pred = model_hybrid(erecoil, *popt)
        chi2 = float(np.sum(((pred - counts) / sigma_abs) ** 2))
        if best is None or chi2 < best[0]:
            best = (chi2, popt)
    if best is None:
        raise RuntimeError(f"fit_sum_of_exponentials: no successful fit in {n_restarts} restarts "
                           f"(n_terms={n_terms}) -- try more restarts or fewer terms")
    chi2, popt = best
    p_a, log_b, p_C = popt[:n_terms], popt[n_terms:2 * n_terms], popt[-1]
    a, b, C = p_a ** 2, np.exp(log_b), p_C ** 2
    dof = len(erecoil) - len(popt)
    return PErecoilFit(a, b, C, erecoil_max=float(erecoil.max()), detector=detector, band=band,
                       calibration_source=calibration_source, source=source,
                       chi2_dof=chi2 / dof if dof > 0 else float("nan"))


def _popt_distance(p1, p2):
    p1, p2 = np.asarray(p1), np.asarray(p2)
    return np.linalg.norm((p1 - p2) / (np.abs(p1) + np.abs(p2) + 1e-9))


def fit_turnon_times_sum_of_exponentials(erecoil, counts, raw_n_per_bin, erecoil_events, *,
                                         n_terms=2, detector="", band="", calibration_source="",
                                         source="", n_restarts=150, n_seeds=10, n_candidates=5,
                                         rng=None):
    """Weighted least-squares fit of the turnon_times_sum_of_exponentials_plus_const
    family -- a smooth rising step (modeling a detector's own near-threshold
    trigger/ionization efficiency ramping up) times the plain family's own
    sum-of-exponentials-plus-floor shape:

        density(Erecoil) = (0.5*(1 + erf((Erecoil-x_on)/w_on))) *
                           (sum(a[i]*exp(-Erecoil/b[i])) + C)

    Unlike fit_sum_of_exponentials, erecoil/counts/raw_n_per_bin should
    include the near-threshold bins that family's own production pipeline
    discards -- this family is meant to describe that region directly,
    not extrapolate past it.

    Fitting procedure mirrors fit_sum_of_exponentials (reparametrized
    space: a[i]/C carried as squared values, b[i] as exp(log_b[i]); x_on
    free, w_on as exp(log_w_on) since it must be positive) plus two
    refinements this family specifically needed when first tried on real
    Cf-252 NR data (see the nrFanoII_paper2022 PErecoil_Fit_vs_Data.ipynb
    notebook's own investigation): many independent rng seeds, not one
    seed's many restarts (a single seed's restarts can miss the real
    optimum); and selecting among the best few *distinct* local optima
    (deduplicated by relative distance in parameter space) by a
    Kolmogorov-Smirnov check against the real unbinned events
    (erecoil_events), not by chi2 alone -- the same selection-blind-spot
    fix adr/0012's own fit_sum_of_exponentials investigation found
    necessary for a sum of exponentials, which applies identically here.

    Returns a PErecoilFit (family="turnon_times_sum_of_exponentials_plus_const",
    erecoil_max=erecoil.max(), chi2_dof recorded for inspection -- check
    it, or better, check_fit_against_events with the real unbinned
    events, before trusting the fit)."""
    erecoil = np.asarray(erecoil, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    raw_n_per_bin = np.asarray(raw_n_per_bin, dtype=np.float64)
    if not (erecoil.shape == counts.shape == raw_n_per_bin.shape):
        raise ValueError("erecoil, counts, raw_n_per_bin must be the same shape")
    if np.any(raw_n_per_bin <= 0):
        raise ValueError("raw_n_per_bin must be positive everywhere (a bin with no real events "
                         "behind it has no basis for a weight)")
    sigma_abs = counts / np.sqrt(raw_n_per_bin)
    rng = np.random.default_rng() if rng is None else rng
    scale, erange = float(counts.max()), float(erecoil.max() - erecoil.min())
    xs_dense = np.linspace(erecoil.min(), erecoil.max(), 3000)

    def model_hybrid(x, *params):
        p_a = np.asarray(params[:n_terms])
        log_b = np.asarray(params[n_terms:2 * n_terms])
        p_C, x_on, log_w_on = params[-3], params[-2], params[-1]
        return _density_turnon(x, p_a ** 2, np.exp(log_b), p_C ** 2, x_on, np.exp(log_w_on))

    candidates = []
    for seed_offset in range(n_seeds):
        seed_rng = np.random.default_rng(rng.integers(0, 2**32 - 1)) if n_seeds > 1 else rng
        for _ in range(n_restarts):
            a0 = seed_rng.uniform(0.1, 1.0, n_terms) * scale
            b0 = seed_rng.uniform(0.05, 1.0, n_terms) * erange
            C0 = seed_rng.uniform(0.0, 0.1) * scale
            x_on0 = seed_rng.uniform(erecoil.min(), erecoil.min() + 0.1 * erange + 1.0)
            w_on0 = seed_rng.uniform(0.01, 0.2) * erange + 0.1
            p0 = [*np.sqrt(a0), *np.log(b0), np.sqrt(C0), x_on0, np.log(w_on0)]
            try:
                popt, _ = curve_fit(model_hybrid, erecoil, counts, p0=p0, sigma=sigma_abs,
                                   absolute_sigma=True, method="lm", maxfev=30000)
            except Exception:
                continue
            pred_dense = model_hybrid(xs_dense, *popt)
            if np.any(~np.isfinite(pred_dense)) or np.any(pred_dense <= 0):
                continue
            pred = model_hybrid(erecoil, *popt)
            chi2 = float(np.sum(((pred - counts) / sigma_abs) ** 2))
            candidates.append((chi2, popt))
    if not candidates:
        raise RuntimeError(f"fit_turnon_times_sum_of_exponentials: no successful fit in "
                           f"{n_seeds}x{n_restarts} restarts (n_terms={n_terms})")
    candidates.sort(key=lambda c: c[0])

    def unpack(popt):
        p_a, log_b = popt[:n_terms], popt[n_terms:2 * n_terms]
        p_C, x_on, log_w_on = popt[-3], popt[-2], popt[-1]
        return p_a ** 2, np.exp(log_b), p_C ** 2, x_on, np.exp(log_w_on)

    events = np.asarray(erecoil_events, dtype=np.float64)
    e_lo, e_hi = float(events.min()), float(events.max())

    def ksp_of(popt):
        a, b, C, x_on, w_on = unpack(popt)
        total = _raw_integral("turnon_times_sum_of_exponentials_plus_const", e_lo, e_hi,
                              a, b, C, x_on, w_on)

        def cdf(x):
            x = np.clip(np.atleast_1d(x), e_lo, e_hi)
            return np.array([_raw_integral("turnon_times_sum_of_exponentials_plus_const", e_lo, xx,
                                           a, b, C, x_on, w_on) for xx in x]) / total

        return kstest(events, cdf).pvalue

    distinct = []
    for chi2, popt in candidates:
        if all(_popt_distance(popt, dp) > 0.3 for _, dp in distinct):
            distinct.append((chi2, popt))
        if len(distinct) >= n_candidates:
            break
    best_chi2, best_popt = max(distinct, key=lambda cp: ksp_of(cp[1]))
    a, b, C, x_on, w_on = unpack(best_popt)
    dof = len(erecoil) - len(best_popt)
    return PErecoilFit(a, b, C, erecoil_max=float(erecoil.max()), valid_min=0.0,
                       family="turnon_times_sum_of_exponentials_plus_const", x_on=float(x_on),
                       w_on=float(w_on), detector=detector, band=band,
                       calibration_source=calibration_source, source=source,
                       chi2_dof=best_chi2 / dof if dof > 0 else float("nan"))


def check_fit_against_events(fit, erecoil_events):
    """Kolmogorov-Smirnov test of fit's own shape against real, unbinned
    Erecoil events -- the real calibration events themselves, or a
    synthetic sample drawn from the fuller, noisier
    erecoil_table.PErecoilTable.sample() it was meant to summarize.
    Restricted to [min(erecoil_events), max(erecoil_events)] (the fit's
    conditional shape there; this is independent of fit's own
    normalization range). Returns (ks_statistic, p_value) -- a small
    p-value (e.g. < 0.05) means fit is NOT a statistically adequate
    description of those events; see adr/0012 for why this check exists
    and matters before trusting a fit for integration."""
    events = np.asarray(erecoil_events, dtype=np.float64)
    lo, hi = float(events.min()), float(events.max())
    total = _raw_integral(fit.family, lo, hi, fit.a, fit.b, fit.C, fit.x_on, fit.w_on)

    def cdf(x):
        x = np.clip(np.atleast_1d(x), lo, hi)
        return np.array([_raw_integral(fit.family, lo, xx, fit.a, fit.b, fit.C, fit.x_on, fit.w_on)
                         for xx in x]) / total

    return kstest(events, cdf)
