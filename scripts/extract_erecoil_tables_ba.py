"""
Build erecoil_tables/{DETECTOR}_ER_Ba.h5 from NRFano's raw Ba-133
calibration data (nrFanoII_paper2022's analysis_notebooks/data/
Ba_data_germanium/Ba_ptqi_<detector>.csv).

Ba-133 is a pure gamma source (no real NR population), so unlike
extract_erecoil_tables.py's Cf extraction, there is no NR/ER band to split
-- every event here is ER. What needs removing instead is surface/zero-
charge contamination that doesn't belong in the clean ER band, which is
exactly what Zack (ZW)'s established pipeline already does in
analysis_notebooks/Ba_data_resolution_fitting_ZW.ipynb (load_ba()/
band_outliers()): an ROI cut, then a 5-sigma-from-the-band-line robust
outlier cut, computed in log-spaced Ep slices. Reused verbatim here (same
constants, same algorithm) rather than inventing a different cut, so this
table and Zack's own resolution fits are built from the same cleaned
sample.

Ba-133 has four well-documented gamma lines (81, 276, 303, 356 keV -- see
Zack's own notebook markdown, which flags using the Cf-gamma biexponential
for Ba fits as a known source of bias). None of these showed up as a
feature in the *Cf* extraction's ER tables (checked directly). They are
NOT treated specially here; this script only builds the tabulated
continuum -- representing the four lines explicitly, rather than letting
the PCHIP continuum alone absorb them, is the separate "peaks
infrastructure" work this extraction is meant to feed into.

Ba's OWN raw data has a large, broad feature around 100-190 keV that is
NOT one of those four lines and does not belong in this table at all: it's
leftover Ge-75m (139.7 keV) neutron-activation contamination from a prior
Cf-252 run on the same crystal, not genuine Ba-133 physics. Three lines of
evidence support this (none alone conclusive, together fairly convincing):
(1) each detector's own hump position tracks its own Cf-derived Ge-75m-
candidate line position to within a few keV, for 4 of 5 detectors; (2) a
from-scratch Klein-Nishina calculation of the Compton continuum expected
from Ba-133's real higher-energy lines (276/303/356/384 keV, weighted by
their emission intensities) comes out far too small and the wrong shape to
explain it; (3) no published Ba-133 reference spectrum shows any line in
100-200 keV.

fit_cf_activation_mixture below removes it with a two-population mixture
model instead of a subtracted curve: a power-law background (Er^k) plus a
single Gaussian "peak" population, fit jointly by unbinned extended
maximum likelihood to the real events in MIXTURE_FIT_WINDOW_KEV. Each
event's own posterior probability of being background (Bayes' rule on the
two fitted population densities at that event's own Er -- the same idea
sPlot/sWeights uses in particle physics to statistically separate species)
becomes its histogram weight, rather than subtracting one global curve
from the binned density.

Two earlier versions were tried and both left visible artifacts. A hard
density-threshold cut, letting PCHIP bridge the resulting ~60-130 keV gap
from the surviving data on either side, was visibly wrong: PCHIP has
nothing inside that gap to anchor its curvature to, and the bridge didn't
track the two-sided falloff the data actually shows. Fitting the same
power-law background + Gaussian jointly by weighted least-squares on the
*binned density* and subtracting only the Gaussian back out did better,
but the background curve -- forced to fit the full 20-260 keV dynamic
range with one 2-parameter shape -- systematically overshot the data at
low Er and undershot it just below the peak, leaving a visible "double
peak" after subtraction (confirmed by inspection, not assumed): one real
bump (the contamination) and one artifact bump from the background
mismatch. The per-event mixture-weighting approach fixes this because the
weight at each event depends on the *local* ratio of the two fitted
densities at that event's own energy, not a single global curve that has
to simultaneously fit both tails; checked directly against power-law,
log-log-quadratic, and 1+c/Er background shapes, which gave nearly
identical weighted results -- much less sensitive to the exact background
functional form than the subtraction approach was, so the simplest
(power-law) shape is kept.

The fitted Gaussian's width (sigma ~35-48 keV, i.e. FWHM ~85-115 keV) is
far broader than a real detector energy resolution at 140 keV; this is
expected, not a sign the fit is wrong -- Er here is the crude, uncorrected
estimator (erecoil_of, no yield/resolution correction), so "sigma" is just
whatever width lets a single Gaussian track this estimator's own smearing
of the hump, not a claim about the true line width. Likewise the "peak
fraction" f (0.45-0.80 across detectors) is large only because the
Gaussian is broad relative to the fit window, not because most of the
window's events are contamination -- the weighted spectra themselves
(checked directly) stay smooth, positive, and consistent with the
surrounding clean data.

Weights outside MIXTURE_FIT_WINDOW_KEV are fixed at 1 (no reweighting);
the fitted populations evaluated at the window edges give a posterior
background-probability of ~0.96, not exactly 1, so there is a small
(~4%) discontinuity in the weight right at each edge -- checked directly
and found negligible against the surrounding bin-to-bin noise, not
corrected for further.

Same Erecoil estimate and quantile/log-log-slope-style binning as the Cf
script (imported from it, not reimplemented): Er = pt - V*(qi/eps)/1000,
equal-count bins. Does need the same near-threshold correction Cf's NR
band does (checked directly: Ba's first real bin is suppressed below the
next several, the same class of artifact, just milder -- a single
suppressed bin stepping up to a plateau, not a whole displaced peak).

Usage:

    python scripts/extract_erecoil_tables_ba.py --nrfano-dir ~/Repositories/nrFanoII_paper2022

Writes erecoil_tables/<detector>_ER_Ba.h5 for each of DETECTORS below.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import erf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from erecoil_table import save_erecoil_table
from extract_erecoil_tables import erecoil_of, histogram_density, nrfano_commit, N_BINS

# Same 5 detectors as the Cf extraction, for a matched NR(Cf)/ER(Cf)/ER(Ba)
# set per detector. Ba_ptqi_<code>.csv exists for more detectors (413, 422,
# 424, 434, 435, 436, 444, 445), same as Cf -- not pulled in here, same
# reasoning as extract_erecoil_tables.py's DETECTORS list.
DETECTORS = ["402", "405", "415", "426", "432"]

SOURCE_TAG = "Ba"
CALIBRATION_SOURCE = "Ba-133"

# The band_outliers() ALGORITHM is ported verbatim from
# Ba_data_resolution_fitting_ZW.ipynb -- but NOT its ROI (ep<=350, eq<=200).
# That ROI is tuned for Zack's purpose (fitting the band's resolution,
# which only needs the band's low-energy shape/width, not its high-energy
# tail: "smearing along the line carries little information" against a
# recoil spectrum spanning hundreds of keV, per his own notebook). Reusing
# it here silently cut off 3 of the 4 known Ba-133 lines (276, 303,
# 356 keV): on the ER band Er = Ep/2 (band_slope = 1 + V/(1000*eps) = 2 for
# V=3, eps=3e-3), so ep_max=350 caps Er at ~175 keV, well below 356. Widened
# to comfortably clear 356 keV (confirmed against the raw data: pt reaches
# well past 700 keV for every detector, if sparsely for some).
ROI = dict(ep_min=2.5, ep_max=800.0, eq_min=0.75, eq_max=450.0)
CLEAN_NSIG = 5.0
V_FOR_BAND = 3.0  # Zack's band_outliers always uses V=3.0, independent of the Er estimate's own V


def band_slope(V):
    from extract_erecoil_tables import EPS
    return 1.0 + V / (1000.0 * EPS)


def band_outliers(Ep, Eq, nsig=CLEAN_NSIG, n_slices=30):
    """True for events far from the ER band line -- verbatim port of
    Ba_data_resolution_fitting_ZW.ipynb's band_outliers()."""
    r = Eq - Ep / band_slope(V_FOR_BAND)
    edges = np.geomspace(ROI["ep_min"], ROI["ep_max"], n_slices + 1)
    bad = np.zeros(len(Ep), dtype=bool)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (Ep >= lo) & (Ep < hi) if hi < edges[-1] else (Ep >= lo) & (Ep <= hi)
        if m.sum() < 30:
            continue
        med = np.median(r[m])
        q25, q75 = np.percentile(r[m], [25, 75])
        bad[m] = np.abs(r[m] - med) > nsig * (q75 - q25) / 1.349
    return bad


# Where the mixture fit looks and where it's allowed to put the Gaussian
# population's center and width -- see this module's docstring for the
# evidence behind the model itself. The fit window (20-260 keV) extends
# well past the hump on both sides so the background population is
# anchored by real clean data, not just the few bins immediately flanking
# the peak; the mu/sigma bounds keep the fit from wandering onto the real
# 81 keV line (excluded below 60 keV) or past the falloff into the
# post-line region (above 250 keV).
MIXTURE_FIT_WINDOW_KEV = (20.0, 260.0)
PEAK_MU_BOUNDS_KEV = (60.0, 250.0)
PEAK_SIGMA_BOUNDS_KEV = (5.0, 100.0)
PEAK_FRACTION_BOUNDS = (1e-4, 0.95)
BACKGROUND_K_BOUNDS = (-3.0, 1.0)


def _powerlaw_pdf(Er, k, lo, hi):
    """Power law Er^k, normalized to a unit-integral density over [lo, hi]."""
    if abs(k + 1.0) < 1e-8:
        norm = np.log(hi / lo)
    else:
        norm = (hi ** (k + 1.0) - lo ** (k + 1.0)) / (k + 1.0)
    return Er**k / norm


def _gaussian_pdf(Er, mu, sigma, lo, hi):
    """Gaussian(mu, sigma), normalized to a unit-integral density over
    [lo, hi] (truncated, not the usual unbounded normalization) -- both
    populations need to integrate to 1 over the same window for f to be a
    meaningful mixing fraction."""
    z_lo = (lo - mu) / (sigma * np.sqrt(2.0))
    z_hi = (hi - mu) / (sigma * np.sqrt(2.0))
    norm = max(0.5 * (erf(z_hi) - erf(z_lo)), 1e-300)
    return np.exp(-0.5 * ((Er - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi)) / norm


def fit_cf_activation_mixture(er, window=MIXTURE_FIT_WINDOW_KEV):
    """Fit a two-population mixture -- power-law background + Gaussian
    "peak" -- to the real events in window, by unbinned extended maximum
    likelihood, then return each event's own posterior probability of
    being background (Bayes' rule on the two fitted densities at that
    event's own Er), to use as a histogram weight. See this module's
    docstring for why this replaces subtracting one global curve from the
    binned density.

    er : every real (Er >= 0) event for this detector, ascending or not.
    Returns (weights, info) -- weights has the same length/order as er
    (1.0 outside window, the posterior background-probability inside it),
    info holds the fit parameters for logging/provenance."""
    lo, hi = window
    in_window = (er > lo) & (er < hi)
    er_w = er[in_window]

    coarse_counts, coarse_edges = np.histogram(er_w, bins=30)
    mu0 = float(0.5 * (coarse_edges[np.argmax(coarse_counts)] + coarse_edges[np.argmax(coarse_counts) + 1]))
    mu0 = min(max(mu0, PEAK_MU_BOUNDS_KEV[0] + 1.0), PEAK_MU_BOUNDS_KEV[1] - 1.0)

    def neg_log_l(theta):
        k, f, mu, sigma = theta
        if not (PEAK_FRACTION_BOUNDS[0] < f < PEAK_FRACTION_BOUNDS[1]) or sigma <= 0:
            return 1e12
        bg = _powerlaw_pdf(er_w, k, lo, hi)
        pk = _gaussian_pdf(er_w, mu, sigma, lo, hi)
        pdf = f * pk + (1.0 - f) * bg
        if np.any(pdf <= 0):
            return 1e12
        return -np.sum(np.log(pdf))

    p0 = [-0.3, 0.15, mu0, 25.0]
    bounds = [BACKGROUND_K_BOUNDS, PEAK_FRACTION_BOUNDS, PEAK_MU_BOUNDS_KEV, PEAK_SIGMA_BOUNDS_KEV]
    res = minimize(neg_log_l, p0, method="Nelder-Mead", bounds=bounds,
                   options=dict(maxiter=20000, xatol=1e-7, fatol=1e-7))
    k, f, mu, sigma = res.x

    bg_w = _powerlaw_pdf(er_w, k, lo, hi)
    pk_w = _gaussian_pdf(er_w, mu, sigma, lo, hi)
    weights = np.ones_like(er)
    weights[in_window] = ((1.0 - f) * bg_w) / (f * pk_w + (1.0 - f) * bg_w)

    info = dict(background_k=float(k), peak_fraction=float(f),
               mu_keV=float(mu), sigma_keV=float(sigma), converged=bool(res.success))
    return weights, info


def extract_one(nrfano_dir, detector, out_dir, commit):
    source_file = f"Ba_ptqi_{detector}.csv"
    csv_path = os.path.join(nrfano_dir, "analysis_notebooks", "data",
                            "Ba_data_germanium", source_file)
    data = pd.read_csv(csv_path, skiprows=1, names=["pt", "qi"])
    pt, qi = data["pt"].to_numpy(), data["qi"].to_numpy()

    roi_mask = ((pt >= ROI["ep_min"]) & (pt <= ROI["ep_max"]) &
               (qi >= ROI["eq_min"]) & (qi <= ROI["eq_max"]) &
               np.isfinite(pt) & np.isfinite(qi))
    pt_roi, qi_roi = pt[roi_mask], qi[roi_mask]

    bad = band_outliers(pt_roi, qi_roi)
    pt_clean, qi_clean = pt_roi[~bad], qi_roi[~bad]

    er = erecoil_of(pt_clean, qi_clean)
    er = er[er >= 0]

    # Down-weight the leftover Cf-activation contamination before
    # histogramming at all: fit the two-population mixture (see
    # fit_cf_activation_mixture and this module's docstring), then give
    # each event its own posterior probability of being background as its
    # histogram weight.
    weights, peak_info = fit_cf_activation_mixture(er)
    print(f"    Cf-activation mixture fit: background Er^{peak_info['background_k']:.3g}, "
          f"peak fraction={peak_info['peak_fraction']:.3f}, mu={peak_info['mu_keV']:.1f} keV, "
          f"sigma={peak_info['sigma_keV']:.1f} keV"
          f"{'' if peak_info['converged'] else ' (optimizer did not report convergence)'}")

    # Despite the name (shared with the Cf NR extraction -- the mechanism
    # itself isn't NR-specific, just where it was first needed), this
    # applies here too: Ba's ER data has the same class of problem, just a
    # milder shape (a single suppressed bin stepping up to a plateau,
    # rather than a whole displaced peak) -- checked directly against the
    # raw data before enabling this, not assumed.
    x, y = histogram_density(er, n_bins=N_BINS, weights=weights, correct_nr_threshold=True)

    contam_note = (
        f"; a leftover Cf-252/Ge-75m activation population (see this script's module docstring "
        f"for the evidence) was modeled jointly with a power-law background (Er^"
        f"{peak_info['background_k']:.2g}) as a Gaussian (mu={peak_info['mu_keV']:.1f} keV, "
        f"sigma={peak_info['sigma_keV']:.1f} keV, peak fraction={peak_info['peak_fraction']:.2f}) "
        f"by unbinned extended maximum likelihood over {MIXTURE_FIT_WINDOW_KEV} keV; each event's "
        f"own posterior probability of being background was used as its histogram weight "
        f"(see fit_cf_activation_mixture), not genuine Ba-133 structure"
    )

    out_path = os.path.join(out_dir, f"{detector}_ER_{SOURCE_TAG}.h5")
    provenance = (
        f"NRFano (nrFanoII_paper2022) @ {commit}, "
        f"Ba_data_resolution_fitting_ZW.ipynb's ROI (ep {ROI['ep_min']}-{ROI['ep_max']}, "
        f"eq {ROI['eq_min']}-{ROI['eq_max']}) and band_outliers() cut ({CLEAN_NSIG:g} sigma, "
        f"V={V_FOR_BAND} for the band line only) verbatim, same Er = pt - V*(qi/eps)/1000 as "
        f"the Cf extraction; near-threshold bins corrected (see correct_nr_threshold_turnon)"
        f"{contam_note} -- extracted via scripts/extract_erecoil_tables_ba.py"
    )
    save_erecoil_table(out_path, x, y, detector=detector, band="ER",
                       calibration_source=CALIBRATION_SOURCE, source_file=source_file,
                       provenance=provenance)
    print(f"{detector}: {len(pt)} events, {len(pt_roi)} in ROI, "
          f"{bad.sum()} ({100 * bad.mean():.2f}%) removed as {CLEAN_NSIG:g}-sigma outliers, "
          f"{len(er)} with Er>=0, range [0, {er.max():.6g}] keV -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nrfano-dir", required=True,
                        help="Path to a nrFanoII_paper2022 checkout (contains analysis_notebooks/)")
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "erecoil_tables"))
    parser.add_argument("--detectors", nargs="+", default=DETECTORS)
    args = parser.parse_args()

    nrfano_dir = os.path.abspath(os.path.expanduser(args.nrfano_dir))
    os.makedirs(args.out_dir, exist_ok=True)
    commit = nrfano_commit(nrfano_dir)
    for detector in args.detectors:
        extract_one(nrfano_dir, detector, args.out_dir, commit)


if __name__ == "__main__":
    main()
