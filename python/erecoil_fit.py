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
closed-form biexponential always has been -- which is what this module's
one supported family is, by construction.

erecoil_table.PErecoilTable stays the right tool for simulation
(event_generator.py's sampling) and as the thing to check a fit against
(check_fit_against_events below) -- just not for the integration path,
which is the only thing this module's PErecoilFit feeds.

One family, deliberately curated rather than an arbitrary user-specified
function (also adr/0012 -- an open-ended expression language has the same
problem a table does: nothing about it guarantees the smoothness the
quadrature actually needs, and it would undo ADR 0001's point in evaluating
it from Fortran besides): a sum of exponentials plus a constant,

    density(Erecoil) = sum(a[i] * exp(-Erecoil / b[i]) for i in range(len(a))) + C

Every term is globally smooth by construction. Add a new curated family to
FAMILY_FORMULAS (and a matching registration path) if a detector's real
spectrum genuinely needs one -- e.g. a term with a real interior mode --
never by letting a caller supply an arbitrary expression.
"""

import hashlib

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kstest

__all__ = ["PErecoilFit", "fit_sum_of_exponentials", "check_fit_against_events", "FAMILY_FORMULAS",
          "CF252_BIEXPONENTIAL_NR", "CF252_BIEXPONENTIAL_ER"]

# One entry per supported family, written once and reused verbatim by
# every fit of that family's own perecoil_identity() -- documentation
# embedded in data, never evaluated back (see this module's own
# docstring and adr/0012 for why not an eval-able expression).
FAMILY_FORMULAS = {
    "sum_of_exponentials_plus_const":
        "density(Erecoil) = sum(a[i] * exp(-Erecoil / b[i]) for i in range(len(a))) + C",
}


def _density(x, a, b, C):
    x = np.asarray(x, dtype=float)
    return sum(ai * np.exp(-x / bi) for ai, bi in zip(a, b)) + C


def _antiderivative(x, a, b, C):
    """Exact antiderivative of _density -- used both to normalize a fit
    (integrate [0, erecoil_max]) and to build a conditional CDF for
    check_fit_against_events below."""
    x = np.asarray(x, dtype=float)
    return sum(-ai * bi * np.exp(-x / bi) for ai, bi in zip(a, b)) + C * x


class PErecoilFit:
    """A true-recoil-energy PDF: sum of exponentials plus a constant floor
    (the one family this module supports -- see FAMILY_FORMULAS and this
    module's own docstring for why that's deliberate, not a limitation).
    Call with a single Erecoil (keV):

        fit = fit_sum_of_exponentials(erecoil, counts, raw_n_per_bin)
        fit(12.3)

    Unlike erecoil_table.PErecoilTable, this is defined -- and smooth --
    for every Erecoil >= 0: no erecoil_range, no OutOfBoxError, nothing
    to clip a Fortran search window to. That absence is the point
    (adr/0012); the Erecoil < 0 convention below matches
    PErecoilTable/PErecoil_NR/ER's own for consistency, not because
    anything would otherwise go wrong there.

    Normalized to integrate to 1 over [0, erecoil_max] -- required
    explicitly at construction, since a nonzero constant floor C makes
    the density's integral over the full [0, infinity) diverge; there is
    no domain-free normalization for this family the way there is for
    the closed form's own two bare (already-normalized) exponentials.
    erecoil_max does not bound where this may be *evaluated* (it stays
    perfectly smooth, and evaluable, well past it) -- only what range the
    stored a/C were scaled against.
    """

    def __init__(self, a, b, C, *, erecoil_max, family="sum_of_exponentials_plus_const",
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
        if not erecoil_max > 0:
            raise ValueError(f"erecoil_max must be > 0, got {erecoil_max!r}")
        raw_norm = float(_antiderivative(erecoil_max, a, b, C) - _antiderivative(0.0, a, b, C))
        if not raw_norm > 0:
            raise ValueError(f"fit integrates to {raw_norm!r} over [0, {erecoil_max}]; cannot normalize")
        self.family = family
        self.a = a / raw_norm
        self.b = b
        self.C = float(C) / raw_norm
        self.erecoil_max = float(erecoil_max)
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
        Erecoil = np.asarray(Erecoil, dtype=float)
        return np.where(Erecoil < 0.0, 0.0, _density(Erecoil, self.a, self.b, self.C))

    @property
    def fortran_params(self):
        """(a, b, C) ready for PErecoil_fit_register -- already normalized,
        so no extra normalization constant needs to cross into Fortran."""
        return self.a, self.b, self.C

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
        return {"kind": "fit", "family": self.family, "formula": FAMILY_FORMULAS[self.family],
                "a": self.a.tolist(), "b": self.b.tolist(), "C": self.C, "sha256": self.sha256,
                "detector": self.detector, "calibration_source": self.calibration_source,
                "source": self.source}


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

    def model(x, *params):
        a, b, C = params[:n_terms], params[n_terms:2 * n_terms], params[-1]
        return _density(x, a, b, C)

    best = None
    for _ in range(n_restarts):
        a0 = rng.uniform(0.1, 1.0, n_terms) * scale
        b0 = rng.uniform(0.05, 1.0, n_terms) * erange
        C0 = rng.uniform(0.0, 0.1) * scale
        try:
            popt, _ = curve_fit(model, erecoil, counts, p0=[*a0, *b0, C0],
                                sigma=sigma_abs, absolute_sigma=True, method="lm", maxfev=20000)
        except Exception:
            continue
        pred = model(erecoil, *popt)
        if np.any(~np.isfinite(pred)) or np.any(pred <= 0) or np.any(popt[n_terms:2 * n_terms] <= 0):
            continue
        chi2 = float(np.sum(((pred - counts) / sigma_abs) ** 2))
        if best is None or chi2 < best[0]:
            best = (chi2, popt)
    if best is None:
        raise RuntimeError(f"fit_sum_of_exponentials: no successful fit in {n_restarts} restarts "
                           f"(n_terms={n_terms}) -- try more restarts or fewer terms")
    chi2, popt = best
    a, b, C = popt[:n_terms], popt[n_terms:2 * n_terms], popt[-1]
    dof = len(erecoil) - len(popt)
    return PErecoilFit(a, b, C, erecoil_max=float(erecoil.max()), detector=detector, band=band,
                       calibration_source=calibration_source, source=source,
                       chi2_dof=chi2 / dof if dof > 0 else float("nan"))


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

    def cdf(x):
        F = lambda t: _antiderivative(np.clip(t, lo, hi), fit.a, fit.b, fit.C)
        return (F(x) - F(lo)) / (F(hi) - F(lo))

    return kstest(events, cdf)
