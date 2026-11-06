"""
Build erecoil_tables/{DETECTOR}_ER_Ba.h5 from NRFano's raw Ba-133
calibration data (nrFanoII_paper2022's analysis_notebooks/data/
Ba_data_germanium/Ba_ptqi_<detector>.csv).

Ba-133 is a pure gamma source (no real NR population), so unlike
extract_erecoil_tables.py's Cf extraction, there is no NR/ER band to split
-- every event here is ER. What needs removing instead is surface/zero-
charge contamination that doesn't belong in the clean ER band. The ROI
below is reused from Zack (ZW)'s established pipeline in
analysis_notebooks/Ba_data_resolution_fitting_ZW.ipynb (load_ba()/
band_outliers()), widened as noted below -- but the actual outlier-rejection
ALGORITHM is extract_erecoil_tables.py's own band_mode_outliers() (the same
per-pt-slice, robust-mode cut used for Cf's NR and ER bands), not Zack's
median-based band_outliers(), by deliberate choice: this project's own
tables should all use one consistent cut, even though it means this table's
cleaned sample is no longer byte-identical to the one Zack's own resolution
fits use (both still remove the same class of near-threshold/surface
contamination; see extract_erecoil_tables.py's band_mode_outliers/kde_mode
docstrings for why mode over median).

Ba-133 has four well-documented gamma lines (81, 276, 303, 356 keV -- see
Zack's own notebook markdown, which flags using the Cf-gamma biexponential
for Ba fits as a known source of bias). None of these showed up as a
feature in the *Cf* extraction's ER tables (checked directly). They are
NOT treated specially here; this script only builds the tabulated
continuum -- representing the four lines explicitly, rather than letting
the PCHIP continuum alone absorb them, is the separate "peaks
infrastructure" work this extraction is meant to feed into.

Ba's OWN raw data has a large, broad feature around 100-190 keV that is
NOT one of those four lines: it's leftover Ge-75m (139.7 keV)
neutron-activation contamination from a prior Cf-252 run on the same
crystal, not genuine Ba-133 physics. Three lines of evidence support this
(none alone conclusive, together fairly convincing): (1) each detector's
own hump position tracks its own Cf-derived Ge-75m-candidate line
position to within a few keV, for 4 of 5 detectors; (2) a from-scratch
Klein-Nishina calculation of the Compton continuum expected from Ba-133's
real higher-energy lines (276/303/356/384 keV, weighted by their emission
intensities) comes out far too small and the wrong shape to explain it;
(3) no published Ba-133 reference spectrum shows any line in 100-200 keV.

DELIBERATE DECISION: this contamination is NOT removed. It is left in the
table as-measured, documented here and in the table's own provenance
string instead. This follows several attempts at removing it, all tried
and all rejected as worse than the problem they were meant to fix:

  1. A hard density-threshold cut, letting PCHIP bridge the resulting
     ~60-130 keV gap from the surviving data on either side -- visibly
     wrong: PCHIP has nothing inside that gap to anchor its curvature to.
  2. Fitting a power-law background + Gaussian jointly by weighted least
     squares on the binned density and subtracting only the Gaussian back
     out -- the background, forced to fit the full 20-260 keV dynamic
     range with one 2-parameter shape, systematically overshot the data at
     low Er and undershot it just below the peak, leaving a visible
     "double peak" artifact after subtraction.
  3. A two-population mixture (same power-law background + Gaussian) fit
     by unbinned extended maximum likelihood, weighting each event by its
     posterior probability of being background (sPlot-style) rather than
     subtracting a curve -- better, but still left real (not noise-level,
     confirmed against Poisson statistics), unexplained non-monotonic
     residual structure for 3 of 5 detectors that no amount of window or
     background-shape tuning removed.
  4. Both a charge-trapping-motivated asymmetric peak shape (exponentially
     modified Gaussian, tried in both tail directions) and fitting the
     same mixture natively in (Ep, Eq) space instead of the derived Er
     estimator -- neither reduced the residual structure; the (Ep, Eq) fit
     in particular failed structurally, because the band-relative outlier
     cut already restricts every surviving event (contamination and
     genuine Ba-133 alike) to within 5 sigma of the same ER band, so
     (Ep, Eq) carries no more separating information than the 1D Er
     estimate already did.

The common thread: with only energy information (no timing data
available -- checked), there is a hard ceiling on how cleanly these two
populations can be statistically separated, and every attempt at pushing
past it introduced a new artifact whose size isn't well understood. A
known, well-characterized problem (this hump, whose origin is established
above) is preferable to an unknown one (whatever residual bias a
not-fully-trusted removal method leaves behind). This also matches the
rest of this project's existing approach: erecoil_of is already a crude,
uncorrected zeroth-order Er estimator (no yield or resolution
deconvolution) standing in for the true recoil-energy spectrum, so using
"what this calibration run's data actually shows" as the table, rather
than a statistically reconstructed version of it, is a difference of
degree, not of kind.

Consequence: this table's PErecoil_ER(Ba) will not respond correctly to
changes in resolution parameters during an MCMC fit in the ~70-200 keV
region (the contamination's shape is baked in as measured, not as a true
recoil-energy spectrum that gets reconvolved), and the table is specific
to this calibration run -- it should not be assumed to generalize to a
different run without revisiting this contamination. An analysis whose
fit region is sensitive to this should exclude ~70-200 keV from its own
fit window; this script does not make that choice on the caller's behalf.
Properly separating this (rather than working around it) is exactly the
"peaks infrastructure" idea from this project's design discussions: model
Ge-75m explicitly via the project's own resolution physics once real
per-detector resolution parameters (F0, V, p0, p10, q0, q10) are
available, instead of an empirically-fit curve on derived, energy-only
data that's already shown it can't reliably separate the two populations.

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
from extract_erecoil_tables import (erecoil_of, histogram_density, nrfano_commit, N_BINS,
                                    CLEAN_NSIG, band_slope, band_mode_outliers, plot_band_outliers,
                                    DETECTORS)

# DETECTORS imported from extract_erecoil_tables, not redefined here: Ba's
# own Ba_ptqi_<code>.csv files cover the same 14 detectors as Cf's, and a
# matched NR(Cf)/ER(Cf)/ER(Ba) set per detector only happens if both
# scripts agree on the list -- importing one shared list means they can't
# drift apart.

SOURCE_TAG = "Ba"
CALIBRATION_SOURCE = "Ba-133"

# The ROI below is adapted from Ba_data_resolution_fitting_ZW.ipynb's own
# (ep<=350, eq<=200) -- not its band_outliers() algorithm (see this module's
# docstring for why this now uses band_mode_outliers instead). That original
# ROI is tuned for Zack's purpose (fitting the band's resolution,
# which only needs the band's low-energy shape/width, not its high-energy
# tail: "smearing along the line carries little information" against a
# recoil spectrum spanning hundreds of keV, per his own notebook). Reusing
# it here silently cut off 3 of the 4 known Ba-133 lines (276, 303,
# 356 keV): on the ER band Er = Ep/2 (band_slope = 1 + V/(1000*eps) = 2 for
# V=3, eps=3e-3), so ep_max=350 caps Er at ~175 keV, well below 356. Widened
# to comfortably clear 356 keV (confirmed against the raw data: pt reaches
# well past 700 keV for every detector, if sparsely for some).
ROI = dict(ep_min=2.5, ep_max=800.0, eq_min=0.75, eq_max=450.0)
V_FOR_BAND = 3.0  # same convention as Zack's band_outliers: fixed, independent of the Er estimate's own V

# CLEAN_NSIG, band_slope, band_mode_outliers, plot_band_outliers are imported
# from extract_erecoil_tables -- the same cut (and diagnostic plot) as Cf's
# NR/ER bands, not a separate algorithm. See this module's docstring for why
# that's a deliberate choice, not an oversight.


def extract_one(nrfano_dir, detector, out_dir, commit, plot_dir=None):
    source_file = f"Ba_ptqi_{detector}.csv"
    csv_path = os.path.join(nrfano_dir, "analysis_notebooks", "data",
                            "Ba_data_germanium", source_file)
    data = pd.read_csv(csv_path, skiprows=1, names=["pt", "qi"])
    pt, qi = data["pt"].to_numpy(), data["qi"].to_numpy()

    roi_mask = ((pt >= ROI["ep_min"]) & (pt <= ROI["ep_max"]) &
               (qi >= ROI["eq_min"]) & (qi <= ROI["eq_max"]) &
               np.isfinite(pt) & np.isfinite(qi))
    pt_roi, qi_roi = pt[roi_mask], qi[roi_mask]

    # ER band, same convention as the Cf extraction's own ER residual.
    r_roi = qi_roi - pt_roi / band_slope(V_FOR_BAND)
    bad = band_mode_outliers(pt_roi, r_roi)
    if plot_dir:
        plot_path = os.path.join(plot_dir, f"{detector}_ER_{SOURCE_TAG}_outliers.png")
        plot_band_outliers(pt_roi, qi_roi, bad, "ER", detector, SOURCE_TAG, plot_path)
    pt_clean, qi_clean = pt_roi[~bad], qi_roi[~bad]

    er = erecoil_of(pt_clean, qi_clean)
    er = er[er >= 0]

    # Despite the name (shared with the Cf NR extraction -- the mechanism
    # itself isn't NR-specific, just where it was first needed), this
    # applies here too: Ba's ER data has the same class of problem, just a
    # milder shape (a single suppressed bin stepping up to a plateau,
    # rather than a whole displaced peak) -- checked directly against the
    # raw data before enabling this, not assumed. model="linear", not the
    # default "exp": Ba's near-threshold region is a Compton continuum,
    # not a declining recoil spectrum, and published Ba-133/EDELWEISS
    # calibration analyses describe this specifically as a flat plateau at
    # these energies (arXiv:1607.04560, arXiv:1707.04308) -- no physics
    # motivation for an exponential here (see correct_nr_threshold_turnon's
    # own docstring for the full reasoning).
    #
    # No contamination removal beyond this: the ~70-200 keV Cf-252/Ge-75m
    # activation feature (see this module's docstring) is left in the
    # table as-measured. Several removal attempts were tried and rejected
    # -- see the docstring's DELIBERATE DECISION section for why.
    #
    # search_window_keV=12.0, narrower than the 30 keV default: Ba's flat
    # ER plateau is noisy enough that a 30 keV search window can and did
    # land on a spurious local maximum deep in the plateau instead of the
    # real near-threshold feature (detectors 435/436, found when this
    # script was expanded to the full 14-detector set -- see
    # correct_nr_threshold_turnon's own docstring for detail). 12.0 keV
    # comfortably covers every detector's real feature (3.6-8.7 keV
    # observed) with margin, well short of where the plateau-noise risk
    # actually showed up.
    x, y = histogram_density(er, n_bins=N_BINS, correct_nr_threshold=True, model="linear",
                             search_window_keV=12.0)

    out_path = os.path.join(out_dir, f"{detector}_ER_{SOURCE_TAG}.h5")
    provenance = (
        f"NRFano (nrFanoII_paper2022) @ {commit}, "
        f"Ba_data_resolution_fitting_ZW.ipynb's ROI (ep {ROI['ep_min']}-{ROI['ep_max']}, "
        f"eq {ROI['eq_min']}-{ROI['eq_max']}), cleaned with the same band_mode_outliers() cut "
        f"({CLEAN_NSIG:g} sigma, V={V_FOR_BAND} for the band line only) as the Cf extraction's "
        f"NR/ER bands (not Zack's own median-based band_outliers() -- a deliberate choice, see "
        f"this script's module docstring), same Er = pt - V*(qi/eps)/1000 as the Cf extraction; "
        f"near-threshold bins corrected with a linear (not exponential) fit -- see "
        f"correct_nr_threshold_turnon's docstring for the physics reasoning; "
        f"the ~70-200 keV region contains leftover Cf-252/Ge-75m (139.7 keV) neutron-activation "
        f"contamination from a prior Cf-252 run on this crystal, NOT genuine Ba-133 structure "
        f"(see this script's module docstring for the evidence) -- left unremoved after several "
        f"removal methods were tried and rejected as introducing worse, less-understood artifacts "
        f"than the contamination itself; an analysis sensitive to this region should exclude it "
        f"from its own fit window -- extracted via scripts/extract_erecoil_tables_ba.py"
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
    parser.add_argument("--plot-dir", default=None,
                        help="If given, save a per-detector band_mode_outliers "
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
