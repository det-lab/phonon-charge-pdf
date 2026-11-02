"""
Build erecoil_tables/{DETECTOR}_{BAND}_{SOURCE}.h5 from NRFano's raw Cf-252
calibration data (nrFanoII_paper2022's analysis_notebooks/data/
Cf_data_germanium/Cf_ptqi_<detector>.csv), following the same NR/ER band
cut and Erecoil estimate as Anthony Villano's
analysis_notebooks/NR_Energy_Distributions.ipynb and
ER_Energy_Distributions.ipynb -- but one table per detector per band
(equal-count/quantile bins, data-driven) instead of those notebooks'
pooled-across-detectors parametric fit. See docs/... (or just those two
notebooks) for the full derivation; the short version:

    Er = pt - V*(qi/eps)/1000            # pt=phonon/Ep, qi=ionization/Eq
    NR:  lbound < qi < ubound  (inside a hand-drawn band)
    ER:  everything else (not a clean gamma selection, just "not NR")
    lbound = 0.20*pt - 3, ubound = 0.35*pt + 0  -- "by eye", same for every
    detector in the source notebooks.

Those notebooks sum every detector's histogram together before fitting a
single closed-form biexponential -- that pooled fit is exactly this
project's current hardcoded PErecoil_NR/PErecoil_ER constants (confirmed:
the NR fit's (a, b, d) and the ER low-energy fit's (a, b, d) match
Ppq_py.py's _PNa/_PNb/_PNd and _PGa/_PGb/_PGd to 6+ figures). This script
instead keeps each detector's own histogram separate (no pooling, no
parametric fit) and writes it straight to an erecoil_table.PErecoilTable-
compatible HDF5 file, so the real per-detector spectrum shape is used
as-is rather than one compromise fit shared across every detector -- and,
for ER, covering its actual observed range instead of a fit calibrated
only out to 40 keV.

Bins are equal-count/quantile, not the source notebooks' 100 linear bins
over a capped display range, and not fixed log-spaced edges either: a
first attempt at fixed log-spaced edges down to an arbitrary 1e-3 keV
packed most of its bins below these detectors' real energy threshold,
where there is essentially no data, so counts/width was almost pure shot
noise there (confirmed by plotting it -- a sawtooth, not a real feature).
Equal-count bins put edges wherever the data actually is: narrow (fine
resolution) where events are dense -- including right where a real sharp
low-Er feature would be -- and wide (coarse, low-noise) in the sparse
tail, with no arbitrary low cutoff to pick. Raw bin counts are converted
to a density (divided by bin width) before writing -- these bins are not
uniform width, so skipping this would make the table's shape an artifact
of the binning, not the physics.

NR near-threshold correction: the raw NR histogram rises from a low value
at its lowest bins up to a peak around 0.3-1 keV before falling -- not
monotonically falling from Er=0 the way the true physics should (CDMS II's
nuclear-recoil energy-scale paper, arXiv:1803.02903, describes the Cf-252-
induced recoil spectrum as "characterized by a single smooth exponential,"
maximal at low Er, and shows the hardware trigger/ionization-threshold cut
efficiencies explicitly rolling off below the detector's analysis
threshold, Fig. 4-5). That rise is almost certainly this same threshold
turn-on, uncorrected in NRFano's raw (pt, qi) data -- not a real physical
feature -- so correct_nr_threshold_turnon below treats everything at or
below the observed peak as unreliable and replaces it with a single-
exponential fit to the (trustworthy, above-threshold) bins just past the
peak, extrapolated down through Er=0. ER's raw data shows no equivalent
rise (it's already flat near Er=0, consistent with a smoothly-declining
spectrum evaluated near its own origin), so it keeps the simpler flat-hold.

Below EP_MIN (2 keV in pt/Ep), raw data is dropped entirely before any other
step: a compact, non-physical origin blob (most plausibly detector pedestal/
zero-energy trigger noise) sits at almost exactly the same (pt, qi) as
genuine near-threshold signal, since the real NR band's own Eq -> 0 as
Ep -> 0 too -- no statistical cut in qi can separate them there without also
cutting real events, so this is a flat, deliberately conservative floor
rather than a fitted boundary (see EP_MIN's own comment below). Above that
floor, band_mode_outliers (also below) removes a further, separable
near-threshold tail.

Calibration source: Cf-252 is a neutron source (populates both NR and ER,
since ER here just means "not NR"). A detector's ER table from a different
source (e.g. Ba-133, a pure gamma source used for high-energy ER
calibration -- see the CDMS II paper above) needs its own table, not a
silent overwrite of this one -- hence _{SOURCE} in both the filename and
each table's own calibration_source/source_file metadata (see
erecoil_table.save_erecoil_table). Only Cf-252 is extracted here; add a
Ba-133 path (new raw-data location, almost certainly a different band cut
since Ba-133 produces no real NR population) as a separate, deliberate
piece of work, not folded into this one silently.

Usage:

    python scripts/extract_erecoil_tables.py --nrfano-dir ~/Repositories/nrFanoII_paper2022

Writes erecoil_tables/<detector>_NR_Cf.h5 and _ER_Cf.h5 for each of
DETECTORS below. Needs pandas, numpy, h5py (all already core/table
dependencies of this project).
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
from erecoil_table import save_erecoil_table

# The 5 detectors covered by NR_Energy_Distributions.ipynb/
# ER_Energy_Distributions.ipynb. More Cf_ptqi_<code>.csv files exist in
# NRFano (e.g. 413, 422, 424, 434, 435, 436, 444, 445) but are not pulled
# in here -- add to this list deliberately, not by discovering files on
# disk, so a stray/unrelated CSV never silently becomes a table.
DETECTORS = ["402", "405", "415", "426", "432"]

# Calibration source for every table this script writes -- see the
# "Calibration source" docstring section above. SOURCE_TAG is the short,
# filename-safe form (erecoil_tables/<detector>_<band>_<SOURCE_TAG>.h5);
# CALIBRATION_SOURCE is the fuller form recorded in each table's metadata.
SOURCE_TAG = "Cf"
CALIBRATION_SOURCE = "Cf-252"

# Fixed detector/material constants the source notebooks use for every
# detector (see Ppq_py.py's Nbar_NR/Nbar_ER -- this is the same eps/V).
EPS = 3.0e-3
V = 3.0

# The NR/ER band cut in (pt, qi), "by eye", identical for every detector --
# see this module's docstring. NR keeps events inside; ER keeps everything
# outside (not a clean gamma selection).
LSLOPE, LINT = 0.20, -3.0
HSLOPE, HINT = 0.35, 0.0

# Hard floor on pt (Ep), applied to the raw data before anything else, for
# both bands. Below this, there's a compact, non-physical "blob" of events
# at (pt, qi) ~ (0.1-0.5, ~0) -- most plausibly detector pedestal/zero-energy
# trigger noise (checked directly: a dense, roughly Gaussian cluster at the
# origin, distinct from either band's own rising trend). band_mode_outliers
# can't separate it statistically: the real NR band's own Eq -> 0 as Ep -> 0
# too (that's the physics -- NR turns on at Erecoil=0), so down here the
# blob and genuine near-threshold signal sit in the same place in (pt, qi).
# No cut in qi can tell them apart without also cutting real low-energy
# band events. EP_MIN=2.0 is comfortably above where the blob's density
# actually falls off (~0.5-0.7, confirmed via 2D density plots for all 5
# detectors) -- a deliberately conservative choice, not a tight boundary
# fit. Below it, we don't claim to know the true Erecoil shape at all;
# histogram_density's near-threshold extrapolation (correct_nr_threshold_turnon,
# NR only) still anchors the table down to Er=0, now built from this cleaner
# input. Checked across all 5 detectors: the blob's extent is consistent
# enough that one shared value was chosen over per-detector tuning.
EP_MIN = 2.0

# A second, NR-only noise population, surviving EP_MIN: a diffuse tail below
# the real NR band that band_mode_outliers can't catch -- not because it
# isn't noise, but because at low pt it's too large a fraction of each
# slice (43-79% of events below qi=0.5 for pt<5 keV) for a robust
# (IQR-based) sigma to stay small enough to flag it; the "outlier" isn't a
# minority there. Known to be noise regardless, so cut directly rather than
# tuning a statistical test to do a job it structurally can't: reject NR
# events with qi < NR_EQ_MIN.
#
# The actual valley between the real NR band and this noise population's
# own qi distribution, not a visual guess: per-pt-slice KDE of qi (bw=0.2)
# is genuinely bimodal over pt~4.6-13 keV in every detector (below that
# range the two populations overlap too much to resolve; above it the noise
# peak has vanished). The valley location varies by detector -- median 0.831
# pooling all 5, but detector 415 runs consistently high (~1.13-1.27) while
# the others cluster lower (~0.37-0.94) -- so a single shared value either
# over-cuts the detectors that don't need it or under-cuts 415. We care
# about keeping the lowest-energy information we can, so NR_EQ_MIN is set
# per detector at that detector's own median valley (not inflated above
# it): 402: 0.80 (median of 4 detections), 405: 0.77 (2 detections -- the
# fewest, so the least certain of the five), 415: 1.16 (4 detections),
# 426: 0.78 (5 detections), 432: 0.89 (5 detections). No pt dependence
# needed in the cut itself despite the valley's own mild pt-trend within a
# detector: the fraction of events below each detector's threshold falls
# off smoothly from pt~5 keV on and reaches 0% by pt~20-40 keV, well past
# where the real band has already separated, so a flat per-detector cut
# stops mattering there without needing an upper bound.
NR_EQ_MIN = {
    "402": 0.80,
    "405": 0.77,
    "415": 1.16,
    "426": 0.78,
    "432": 0.89,
}

# Quantile (equal-count) bins, not fixed log-spaced edges: a blind
# geomspace(1e-3, max, 500) packed ~180 of its 500 bins into 1e-3-0.1 keV
# alone, where these detectors have essentially no events below their real
# energy threshold -- most of those bins held 0 or 1 raw count, so
# counts/width was pure shot noise (confirmed by plotting it: a sawtooth
# below ~0.1-2 keV, not a real feature). Equal-count bins put bin edges
# wherever the data actually is: narrow (fine resolution) where events are
# dense -- including right where a real sharp low-Er feature would be --
# and wide (coarse, low-noise) in the sparse tail, automatically, with no
# arbitrary low cutoff to pick. ~30-60k events per table / 150 bins is
# ~200-400 events/bin on average, ~5-7% Poisson noise per bin.
N_BINS = 150

# Number of real bins immediately above the NR peak used to fit the
# single-exponential threshold correction (see this module's docstring).
NR_FIT_BINS = 15

# Robust-outlier sigma threshold for band_mode_outliers below, shared with
# extract_erecoil_tables_ba.py (imports this directly, doesn't redefine it) so
# both scripts' cuts use the same aggressiveness unless deliberately revisited
# together. Started at 5.0 (Zack's original choice for Ba's median-based cut);
# raised to 6.0 after checking the above-the-band/below-the-band split of what
# gets rejected: NR was already ~100% below-band at 5.0 (matches the
# charge-leakage-only-drifts-down physics expectation), but ER was often
# majority above-band (e.g. detector 402: 300/319; 415: 181/230) -- 6.0 roughly
# halves-to-thirds those above-band counts while barely touching the
# below-band ones, which is the asymmetry 5.0 was getting wrong.
CLEAN_NSIG = 6.0


def erecoil_of(pt, qi):
    """The same zeroth-order Erecoil estimate the source notebooks use:
    no yield correction, no resolution deconvolution -- just the measured
    ionization energy's Luke-phonon contribution subtracted off the
    measured phonon energy."""
    return pt - V * (qi / EPS) / 1000.0


def band_mask(pt, qi):
    """True where (pt, qi) falls inside the hand-drawn NR band; ER is the
    complement."""
    ubound = HSLOPE * pt + HINT
    lbound = LSLOPE * pt + LINT
    return (qi < ubound) & (qi > lbound)


def band_slope(V):
    """qi/pt ratio of the ER band's line (Y=1, Er = pt - V*(qi/eps)/1000
    -> qi = pt/band_slope at Er=0... equivalently qi = pt/(1 + V/(1000*eps))
    for any Er, since Y=1 makes qi a fixed fraction of pt). Shared with
    extract_erecoil_tables_ba.py, which imports this rather than redefining
    it (same relation, same eps/V)."""
    return 1.0 + V / (1000.0 * EPS)


def kde_mode(x, grid_n=400):
    """The KDE-estimated mode of x (scipy.stats.gaussian_kde, Scott's rule
    bandwidth, peak over a grid spanning the 1st-99th percentiles). Used
    instead of the median for band_mode_outliers below: checked directly
    (see this project's design discussion) that the median is systematically
    biased away from the data's own densest point even in the low-energy
    region that matters most here, for both NR's curved yield-band and ER's
    nominally straight line -- confirmed against two KDE bandwidths and a
    bandwidth-free half-sample-mode estimator, so this isn't a bandwidth
    artifact. Falls back to the median for very small slices, where a KDE
    is unreliable anyway."""
    if len(x) < 10:
        return float(np.median(x))
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return float(np.median(x))
    grid = np.linspace(lo, hi, grid_n)
    density = gaussian_kde(x)(grid)
    return float(grid[np.argmax(density)])


def band_mode_outliers(x, y, nsig=CLEAN_NSIG, n_slices=30, min_per_slice=30, x_floor=1e-3):
    """True for points far from the data's own robust per-slice mode of y
    vs log-spaced x: in each slice, flags |y - kde_mode(y)| > nsig * robust
    sigma (IQR/1.349). Generalizes
    extract_erecoil_tables_ba.py's band_outliers() (median of a residual
    from one fixed straight line) to band_mode_outliers (per-slice mode of
    y directly) so the same mechanism works for NR's curved yield-band too,
    not just ER's line -- for ER, pass y as the residual (qi - pt/band_slope)
    to match Ba's convention; for NR, pass y = qi directly.

    x_floor excludes x <= x_floor from slicing entirely (always kept, never
    flagged): log-spaced edges need x > 0, and there's too little data down
    there to cut reliably anyway. A slice (or the whole sliceable range,
    if too small overall) with fewer than min_per_slice points is also left
    unflagged, same as Ba's algorithm."""
    bad = np.zeros(len(x), dtype=bool)
    sliceable = x > x_floor
    if sliceable.sum() < min_per_slice:
        return bad
    idx_sliceable = np.flatnonzero(sliceable)
    xs, ys = x[sliceable], y[sliceable]
    edges = np.geomspace(xs.min(), xs.max(), n_slices + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (xs >= lo) & (xs <= hi) if hi == edges[-1] else (xs >= lo) & (xs < hi)
        if m.sum() < min_per_slice:
            continue
        center = kde_mode(ys[m])
        q25, q75 = np.percentile(ys[m], [25, 75])
        sigma = (q75 - q25) / 1.349
        bad[idx_sliceable[m]] = np.abs(ys[m] - center) > nsig * sigma
    return bad


def plot_band_outliers(pt, qi, bad, band, detector, source_tag, out_path, nsig=CLEAN_NSIG):
    """Diagnostic scatter of (pt, qi): kept points muted, rejected
    (band_mode_outliers) points in red, so nsig can be sanity-checked by eye
    before trusting it. x (pt) is log scale (it spans several orders of
    magnitude); y (qi) is symlog, not log -- most rejected points turn out
    to have qi < 0 (noise-floor events near threshold, checked directly),
    which plain log would silently drop from the plot entirely, hiding
    exactly what this is meant to show. matplotlib is a test-only
    dependency (see pyproject.toml) -- imported here, not at module level,
    so extraction itself never needs it unless this is called."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pt[~bad], qi[~bad], s=2, alpha=0.3, color="C0",
              label=f"kept ({(~bad).sum()})")
    ax.scatter(pt[bad], qi[bad], s=8, alpha=0.8, color="red",
              label=f"rejected ({bad.sum()})")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("pt, i.e. Ep (keV)")
    ax.set_ylabel("qi, i.e. Eq (keV)")
    ax.set_title(f"detector {detector}, {band} ({source_tag}) -- band_mode_outliers, nsig={nsig:g}")
    ax.legend(fontsize=8, markerscale=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def correct_nr_threshold_turnon(x, y, n_fit_bins=NR_FIT_BINS, search_window_keV=30.0):
    """Replace the spectrum's compromised near-threshold bins (at or below
    its observed peak, searched for only below search_window_keV -- see
    below for why) with a single-exponential fit to the trustworthy bins
    just past it. See this module's docstring for why: short version, a
    near-Er=0 rise (or, for Ba's ER, a single suppressed first bin) is the
    detector's own trigger/ionization-threshold efficiency turning on, not
    the true recoil spectrum, which should be maximal and smoothly falling
    from Er=0.

    The peak search is restricted to x < search_window_keV rather than a
    blind argmax(y) over the whole table: deep in a flat plateau (Ba's ER,
    ~20-300 keV), every equal-count bin holds almost exactly the same raw
    count, so density = count/width is dominated by bin-width noise, not
    signal -- a global argmax can and does land tens of keV away from the
    real near-threshold feature, picking up pure binning noise instead
    (confirmed directly: one such "peak" sat on bins all holding within 1
    of each other's raw count). 30 keV comfortably covers every real
    threshold feature seen so far (NR peaks at 0.2-0.8 keV, Ba's single
    suppressed bin at a few keV) while safely excluding the plateau and
    the ER lines (lowest at 81 keV).

    x, y : the real (non-zero-anchor) bin centers/densities, ascending.
    Returns (y_corrected, anchor0, info) -- anchor0 is the fit evaluated
    at Er=0 (for the explicit x=0 node), y_corrected has every bin at or
    below the peak replaced by the fit evaluated at that bin's own x, and
    info is a small dict of fit diagnostics for logging."""
    search_mask = x < search_window_keV
    peak_idx = int(np.flatnonzero(search_mask)[np.argmax(y[search_mask])])
    fit_x = x[peak_idx:peak_idx + n_fit_bins]
    fit_y = y[peak_idx:peak_idx + n_fit_bins]
    slope, intercept = np.polyfit(fit_x, np.log(fit_y), 1)
    tau = -1.0 / slope
    amplitude = np.exp(intercept)

    # NR's region just past its peak is a real declining tail, so an
    # exponential is the right model there and this always comes out
    # tau > 0. Ba's ER is different: right past its (milder) suppressed
    # region sits a near-flat plateau for tens of keV, not a decline --
    # fitting an exponential to a flat/noisy window is the wrong model and
    # goes unstable (confirmed directly: one detector's fit came back
    # tau=-690 keV, i.e. *rising*, not falling). tau <= 0 means exactly
    # that -- fall back to a flat anchor (the fit window's own median,
    # robust to the same noise that broke the exponential fit) rather than
    # trust a nonsensical extrapolation.
    if tau > 0:
        fit = lambda xx: amplitude * np.exp(-xx / tau)
    else:
        flat_value = float(np.median(fit_y))
        fit = lambda xx: np.full_like(np.asarray(xx, dtype=float), flat_value) if np.ndim(xx) else flat_value
        amplitude, tau = flat_value, np.inf

    y_corrected = y.copy()
    y_corrected[:peak_idx] = fit(x[:peak_idx])
    anchor0 = float(fit(0.0))
    info = dict(peak_Er_keV=float(x[peak_idx]), fit_amplitude=float(amplitude), fit_tau_keV=float(tau))
    return y_corrected, anchor0, info


def histogram_density(erecoil, n_bins=N_BINS, correct_nr_threshold=False, weights=None):
    """Equal-count (quantile) histogram of erecoil (keV, already >= 0),
    converted from raw bin counts to a density (divided by bin width).
    weights, if given, is a per-event array (same length/order as erecoil)
    summed per bin instead of raw counts -- used by the Ba extraction to
    down-weight events the Cf-activation mixture fit assigns to the
    contamination population (see extract_erecoil_tables_ba.py's
    fit_cf_activation_mixture); bin edges are still the unweighted
    quantile edges below, so this only changes what's counted in each bin,
    not where the bins are. correct_nr_threshold applies
    correct_nr_threshold_turnon (NR only -- see this module's docstring)
    before adding the explicit Erecoil=0 node; otherwise that node is held
    flat at the lowest bin's density (ER's case -- the data is already
    flat there, so this is a reasonable match to the visible trend, not a
    bare assumption). See the N_BINS comment above for why equal-count,
    not fixed log-spaced, edges. Returns (x, density), x strictly
    increasing, ready for erecoil_table.save_erecoil_table."""
    edges = np.unique(np.quantile(erecoil, np.linspace(0.0, 1.0, n_bins + 1)))
    counts, edges = np.histogram(erecoil, bins=edges, weights=weights)
    widths = np.diff(edges)
    density = counts / widths
    centers = 0.5 * (edges[:-1] + edges[1:])

    if correct_nr_threshold:
        density, anchor0, info = correct_nr_threshold_turnon(centers, density)
        print(f"    NR threshold correction: peak at {info['peak_Er_keV']:.3g} keV, "
              f"fit tau={info['fit_tau_keV']:.3g} keV, Er=0 anchor={anchor0:.4g}")
    else:
        anchor0 = density[0]

    x = np.concatenate(([0.0], centers))
    y = np.concatenate(([anchor0], density))
    return x, y


def nrfano_commit(nrfano_dir):
    try:
        out = subprocess.run(["git", "-C", nrfano_dir, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def extract_one(nrfano_dir, detector, out_dir, commit, plot_dir=None):
    source_file = f"Cf_ptqi_{detector}.csv"
    csv_path = os.path.join(nrfano_dir, "analysis_notebooks", "data",
                             "Cf_data_germanium", source_file)
    data = pd.read_csv(csv_path, skiprows=1, names=["pt", "qi"])
    pt, qi = data["pt"].to_numpy(), data["qi"].to_numpy()
    n_raw = len(pt)
    ep_floor_mask = pt > EP_MIN
    pt, qi = pt[ep_floor_mask], qi[ep_floor_mask]
    print(f"{detector}: {n_raw} raw events, {len(pt)} ({100 * len(pt) / n_raw:.1f}%) "
          f"with pt > EP_MIN={EP_MIN:g} keV (see this script's EP_MIN comment)")
    is_nr = band_mask(pt, qi)

    for band, mask in (("NR", is_nr), ("ER", ~is_nr)):
        pt_b, qi_b = pt[mask], qi[mask]
        n_eq_cut = 0
        if band == "NR":
            # Known noise, not something band_mode_outliers can be tuned to
            # catch (see NR_EQ_MIN's comment) -- cut directly.
            nr_eq_min = NR_EQ_MIN[detector]
            eq_floor_mask = qi_b >= nr_eq_min
            n_eq_cut = int((~eq_floor_mask).sum())
            pt_b, qi_b = pt_b[eq_floor_mask], qi_b[eq_floor_mask]

        # NR's band curves with the Lindhard yield, so there's no single line
        # to take a residual from -- use qi directly. ER's band is a single
        # line (Y=1), so use the residual from it, matching Ba's own
        # band_outliers() convention (see extract_erecoil_tables_ba.py).
        y_b = qi_b if band == "NR" else qi_b - pt_b / band_slope(V)
        bad = band_mode_outliers(pt_b, y_b)
        if plot_dir:
            plot_path = os.path.join(plot_dir, f"{detector}_{band}_{SOURCE_TAG}_outliers.png")
            plot_band_outliers(pt_b, qi_b, bad, band, detector, SOURCE_TAG, plot_path)

        er = erecoil_of(pt_b[~bad], qi_b[~bad])
        er = er[er >= 0]  # unphysical for this table's convention; see erecoil_table.py
        x, y = histogram_density(er, correct_nr_threshold=(band == "NR"))
        out_path = os.path.join(out_dir, f"{detector}_{band}_{SOURCE_TAG}.h5")
        provenance = (
            f"NRFano (nrFanoII_paper2022) @ {commit}, "
            f"raw data restricted to pt (Ep) > {EP_MIN:g} keV first (below this, a non-physical "
            f"origin blob and genuine near-threshold signal aren't statistically separable -- "
            f"see this script's EP_MIN comment), then "
            f"{band}_Energy_Distributions.ipynb's band cut (qi vs {LSLOPE}*pt{LINT:+g} / "
            f"{HSLOPE}*pt{HINT:+g}) and Er = pt - V*(qi/eps)/1000 (V={V}, eps={EPS}), "
            + (f"a qi >= {NR_EQ_MIN[detector]:g} keV cut (a diffuse near-threshold noise tail "
               f"that's too large a fraction of each low-pt slice for band_mode_outliers' robust "
               f"sigma to flag -- see this script's NR_EQ_MIN comment), then " if band == "NR" else "")
            + f"a {CLEAN_NSIG:g}-sigma band_mode_outliers cut (per-pt-slice robust mode, "
            f"IQR-based sigma -- see this script's docstring), "
            f"extracted per-detector (not pooled) via scripts/extract_erecoil_tables.py"
            + (f"; NR near-threshold bins below the observed peak replaced by a single-"
               f"exponential fit (see this script's correct_nr_threshold_turnon)" if band == "NR" else "")
        )
        save_erecoil_table(out_path, x, y, detector=detector, band=band,
                           calibration_source=CALIBRATION_SOURCE, source_file=source_file,
                           provenance=provenance)
        eq_cut_msg = f"{n_eq_cut} removed as qi < NR_EQ_MIN={nr_eq_min:g}, " if band == "NR" else ""
        print(f"{detector} {band}: {mask.sum()} events, {eq_cut_msg}"
              f"{bad.sum()} ({100 * bad.mean():.2f}% of what's left) "
              f"removed as {CLEAN_NSIG:g}-sigma band_mode_outliers, "
              f"{(er >= 0).sum()} with Er>=0, range [0, {er.max():.6g}] keV -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nrfano-dir", required=True,
                        help="Path to a nrFanoII_paper2022 checkout (contains analysis_notebooks/)")
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "erecoil_tables"))
    parser.add_argument("--detectors", nargs="+", default=DETECTORS)
    parser.add_argument("--plot-dir", default=None,
                        help="If given, save a per-detector/band band_mode_outliers "
                             "diagnostic scatter (red = rejected) here (needs matplotlib)")
    args = parser.parse_args()

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)

    nrfano_dir = os.path.abspath(os.path.expanduser(args.nrfano_dir))
    os.makedirs(args.out_dir, exist_ok=True)
    commit = nrfano_commit(nrfano_dir)
    for detector in args.detectors:
        extract_one(nrfano_dir, detector, args.out_dir, commit, plot_dir=args.plot_dir)


if __name__ == "__main__":
    main()
