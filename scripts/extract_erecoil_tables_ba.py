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
100-200 keV. exclude_cf_activation_contamination below removes it (per
detector, relative to that detector's own clean-plateau baseline) rather
than modeling or guessing a replacement -- PCHIP bridges the resulting gap
smoothly from the clean data on either side. This necessarily risks some
of the real 81 keV line's upper tail where the two overlap (confirmed to
still sit close for some detectors); accepted as a real trade-off, not
unnoticed.

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


# Where the Ge-75m-contamination search looks, and what counts as
# "contaminated" -- see this module's docstring for the evidence. The
# baseline window (40-60 keV) sits comfortably below where the hump starts
# for every detector checked; the search window's upper bound (250 keV)
# comfortably clears it on the high side too.
BASELINE_WINDOW_KEV = (40.0, 60.0)
CONTAMINATION_SEARCH_WINDOW_KEV = (60.0, 250.0)
CONTAMINATION_THRESHOLD = 1.1  # x the baseline density -- see this module's
# docstring/commit history: 1.5 was tried first and only cut the sharp
# peak, leaving the broader shoulders of the same feature in place (still
# visibly part of the same hump on a log-x plot, just under that
# threshold). 1.1 was chosen by eye against that plot: it comfortably
# brackets the full visual extent of the hump for all 5 detectors without
# cutting into the flat plateau on either side.


def exclude_cf_activation_contamination(x, y, baseline_window=BASELINE_WINDOW_KEV,
                                        search_window=CONTAMINATION_SEARCH_WINDOW_KEV,
                                        threshold=CONTAMINATION_THRESHOLD):
    """Remove bins in search_window where density exceeds threshold times
    the baseline (median density in baseline_window) -- the leftover
    Cf-252 neutron-activation contamination identified in this module's
    docstring, not genuine Ba-133 structure. Unlike
    correct_nr_threshold_turnon, this doesn't model or guess a replacement
    value for the compromised region -- it just removes those nodes and
    lets PCHIP bridge smoothly across the gap from the clean data on
    either side, since there's nothing in the data itself to say what the
    true continuum looks like under the contamination.

    x, y : the real (non-zero-anchor) bin centers/densities, ascending.
    Returns (x_clean, y_clean, info)."""
    baseline = float(np.median(y[(x > baseline_window[0]) & (x < baseline_window[1])]))
    search_mask = (x > search_window[0]) & (x < search_window[1])
    contaminated = search_mask & (y > threshold * baseline)
    info = dict(baseline=baseline, n_removed=int(contaminated.sum()))
    if contaminated.any():
        info["removed_range_keV"] = (float(x[contaminated].min()), float(x[contaminated].max()))
    return x[~contaminated], y[~contaminated], info


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
    # Despite the name (shared with the Cf NR extraction -- the mechanism
    # itself isn't NR-specific, just where it was first needed), this
    # applies here too: Ba's ER data has the same class of problem, just a
    # milder shape (a single suppressed bin stepping up to a plateau,
    # rather than a whole displaced peak) -- checked directly against the
    # raw data before enabling this, not assumed.
    x, y = histogram_density(er, n_bins=N_BINS, correct_nr_threshold=True)

    # Exclude the leftover Cf-activation contamination (x[0]/y[0] is the
    # synthetic Erecoil=0 anchor from histogram_density -- never in the
    # search window, but kept out of this step regardless to be explicit).
    x_real, y_real, contam_info = exclude_cf_activation_contamination(x[1:], y[1:])
    x, y = np.concatenate(([x[0]], x_real)), np.concatenate(([y[0]], y_real))
    if contam_info["n_removed"]:
        lo, hi = contam_info["removed_range_keV"]
        print(f"    Cf-activation contamination removed: {contam_info['n_removed']} bins, "
              f"[{lo:.1f}, {hi:.1f}] keV (baseline={contam_info['baseline']:.1f})")
        contam_note = (f"; the {lo:.0f}-{hi:.0f} keV region excluded as leftover Cf-252/Ge-75m "
                       f"activation contamination (see this script's module docstring for the "
                       f"evidence), not genuine Ba-133 structure")
    else:
        print(f"    no Cf-activation contamination found above {CONTAMINATION_THRESHOLD}x baseline "
              f"({contam_info['baseline']:.1f})")
        contam_note = "; no Cf-activation contamination found"

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
