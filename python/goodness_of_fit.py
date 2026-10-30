"""
Goodness-of-fit testing for the NR/ER band PDFs: chi-square distribution
testing against equibin-adaptive bins, plus the band-specific breakpoints
that per-bin quadrature needs and a single-bin diagnostic.

Call these as `phonon_charge_pdf.goodness_of_fit.<name>`, not flattened to
the top level -- see docs/goodness-of-fit.md.

Workflow
--------
1. Generate a reference dataset; use equibin to define adaptive bins.
   The outer envelope of the bins is snapped to the declared region so
   every thrown event lands in exactly one bin.
2. Compute expected counts per bin by integrating pdf_func over each bin
   with adaptive quadrature, which is accurate regardless of bin size.
3. In batches, generate n_throws independent datasets of n_events each,
   count events in each bin, and accumulate the Pearson chi-square:
       chi2 = sum( (observed - expected)^2 / expected )
4. Return the array of chi-square values for downstream analysis.

run_chisquare_test/expected_counts_from_pdf are PDF-agnostic: supply
any pdf_func(Ep_flat, Eq_flat) -> values_flat callable and a matching
sampler(n_events, seed=None) -> (Ep, Eq) callable that returns events
already cropped to the region -- see run_chisquare_test's own docstring
for why generate_events_NR/generate_events_ER can't be passed directly
(they have no ROI cropping of their own) and docs/goodness-of-fit.md
for a sampler that wraps one correctly.
make_Ppq_densities_NR/make_Ppq_densities_ER (re-exported here from
python/internal/_Ppq_fort_bindings.py -- see that module's docstring) build
the one pdf_func this package actually ships, so they live here rather
than at the top level: nothing else in this package needs arbitrary-point
density evaluation.

Step 2's per-bin quadrature also requires an inner_points_func giving
the band-ridge breakpoints (make_ridge_breakpoints below); the NR/ER
band PDFs are narrow ridges, for which plain adaptive quadrature over a
wide bin can silently return ~0, so there is no "skip the breakpoints"
option.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import equibin
from scipy.integrate import quad, simpson

import os
import sys

_PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PYTHON_DIR)
import Ppq_py as ppq

try:
    # Relative import (not the sys.path trick above) so mkdocstrings/griffe
    # can resolve these as real re-exports when this module is loaded as
    # part of the phonon_charge_pdf package -- see the module docstring.
    from .internal._Ppq_fort_bindings import make_Ppq_densities_ER, make_Ppq_densities_NR  # noqa: F401
except ImportError:
    # Imported standalone (e.g. test/python/*.py via sys.path, with no
    # parent package) -- those callers get make_Ppq_densities_NR/ER
    # directly from _Ppq_fort_bindings instead, so this re-export isn't
    # needed here.
    pass

__all__ = ["run_chisquare_test", "expected_counts_from_pdf", "debug_bin", "make_ridge_breakpoints",
           "make_Ppq_densities_NR", "make_Ppq_densities_ER"]


# ---------------------------------------------------------------------------
# Band-specific breakpoints
# ---------------------------------------------------------------------------

def make_ridge_breakpoints(is_ER, *, k, Z, eps, V,
                           p0, p10,
                           q0, q10,
                           erecoil_max, n_window_widths):
    """
    Build an inner_points_func for expected_counts_from_pdf.

    The problem
    -----------
    The expected count in a bin is the integral of the PDF over the bin
    rectangle, computed in expected_counts_from_pdf with nested adaptive
    quadrature (scipy quad over Ep, inner quad over Eq). Adaptive
    quadrature starts from a fixed set of sample points across the
    integration interval: if the PDF is a narrow ridge inside a wide
    bin, every one of those initial points can land in the ~zero region
    on either side of the ridge, and quad confidently returns ~0 without
    ever subdividing toward the peak.

    The fix is quad's `points=` argument: telling the inner (Eq)
    quadrature where the ridge is forces a subdivision there, so the
    peak cannot be missed. This function computes those breakpoints
    from the band physics.

    The physics: where the ridge is
    -------------------------------
    A recoil of true energy Erecoil produces (noiselessly)

        N  = Y(Erecoil) * Erecoil / eps          e/h pairs, so
        Eq = Y(Erecoil) * Erecoil                (ionization channel)
        Ep = Erecoil + (V/1000) * N
           = Erecoil * (1 + Y(Erecoil) * V / (1000 * eps))   (phonon + Neganov-Luke)

    with Y(Erecoil) the Lindhard yield (k, Z) for nuclear recoils and Y = 1
    for electron recoils. These are parametric equations for the band
    centroid: each Erecoil gives one point (Ep(Erecoil), Eq(Erecoil)) on the ridge. We
    need the centroid in the form "most probable Eq at a given Ep" —
    that direction because the harness's *inner* quadrature runs over
    Eq at fixed Ep, so the breakpoints must be Eq values. Ep(Erecoil) = ep
    cannot be solved for Erecoil algebraically (Y makes it transcendental),
    but both Ep(Erecoil) and Eq(Erecoil) are monotone in Erecoil, so the centroid is
    tabulated on an Erecoil grid once and np.interp(ep, ep_tab, eq_tab)
    performs the inversion: find where the table's Ep equals ep, read
    off the Eq at the same Erecoil.

    The geometry: how wide the ridge is
    ------------------------------------
    The breakpoints must bracket the peak, so we need the band's local
    width in the Eq direction at fixed measured Ep. Picture the band as
    a tilted ridge in the (Ep, Eq) plane and slice it vertically at ep.
    Two independent noise sources spread events along that vertical
    slice:

    * Charge noise smears Eq directly: a vertical spread of sigq.
    * Phonon noise smears the measured Ep, i.e. it displaces events
      *horizontally*. An event that landed in the slice because its Ep
      fluctuated by delta really belongs to a ridge point whose Eq
      differs by (dEq_ridge/dEp) * delta — so horizontal noise, viewed
      through the tilt of the band, appears as a vertical spread of
      slope * sigp. A flat band gets no vertical smearing from phonon
      noise; a steep band gets a lot.

    The two contributions are independent Gaussians, so they add in
    quadrature (np.hypot is sqrt(a^2 + b^2), written that way only for
    numerical safety):

        width(Ep) = sqrt( sigq(Eq_ridge)^2 + (dEq_ridge/dEp * sigp(Ep))^2 )

    with the slope taken by central finite difference of the
    interpolated ridge. This width ignores the Fano broadening (the
    spread in N), which is harmless here: the breakpoints sit at ridge
    +/- 10 widths, so even a substantially underestimated width still
    brackets the whole peak, and the adaptive quadrature does the
    accurate work inside the bracket.

    The breakpoints handed to the harness are the ridge center and the
    window of +/- 10 local widths around it; the harness clips them to
    each bin.

    Parameters
    ----------
    is_ER : bool
        False (nuclear recoils) uses the Lindhard yield model Y(Erecoil, k, Z);
        True (electron recoils) uses Y = 1 (the k and Z arguments are then
        ignored).
    k, Z, eps, V, p0, p10, q0, q10 : float
        Detector parameters, matching the PDF being integrated.
    erecoil_max : float
        Upper end of the tabulated Erecoil range (keV); must comfortably
        exceed the largest Ep of interest.
    n_window_widths : float
        Half-width of the breakpoint window in units of the local band
        width.

    Returns
    -------
    inner_points_func : callable
        inner_points_func(ep, ymin, ymax) -> [lo, ridge, hi], the Eq
        breakpoints for the inner quadrature at phonon energy ep. The
        harness drops any that fall outside the bin's (ymin, ymax).
    """
    if is_ER:
        y_tab_func = lambda erecoil_tab: np.ones_like(erecoil_tab)
    else:
        y_tab_func = lambda erecoil_tab: ppq.Y(erecoil_tab, k=k, Z=Z)

    # Tabulate the noiseless band centroid Erecoil -> (Ep, Eq) and invert by
    # interpolation (both coordinates are monotone in Erecoil).
    erecoil_tab = np.geomspace(1e-3, erecoil_max, 4000)
    y_tab = y_tab_func(erecoil_tab)
    ep_tab = erecoil_tab * (1.0 + y_tab * V / (1000.0 * eps))
    eq_tab = y_tab * erecoil_tab

    def eq_ridge(ep):
        return np.interp(ep, ep_tab, eq_tab)

    def inner_points_func(ep, ymin, ymax):
        eqr = float(eq_ridge(ep))
        # local band width in Eq: charge resolution plus the phonon
        # resolution mapped through the ridge slope
        dep = max(1e-3, 0.01 * ep)
        slope = (float(eq_ridge(ep + dep)) - float(eq_ridge(ep - dep))) / (2 * dep)
        sigp_val = float(ppq.sigp(ep, eps=eps, V=V, p0=p0, p10=p10))
        sigq_val = float(ppq.sigq(eqr, q0=q0, q10=q10))
        width = np.hypot(sigq_val, slope * sigp_val)
        return [eqr - n_window_widths * width, eqr,
                eqr + n_window_widths * width]

    return inner_points_func


# ---------------------------------------------------------------------------
# Expected counts from the PDF
# ---------------------------------------------------------------------------

def expected_counts_from_pdf(pdf_func, bins, n_events, inner_points_func, *,
                             n_workers=1):
    """
    Integrate pdf_func over each bin rectangle using adaptive quadrature
    (scipy quad over Ep, nested quad over Eq).

    Parameters
    ----------
    pdf_func : callable
        pdf_func(Ep_flat, Eq_flat) -> values_flat
    bins : list of (xmin, xmax, ymin, ymax)
        Bin definitions from equibin.BinningResult.bins.
    n_events : int
        Events per throw; scales the returned expected counts.
    inner_points_func : callable
        inner_points_func(ep, ymin, ymax) -> sequence of Eq breakpoints.
        The NR/ER band PDFs are narrow ridges: plain adaptive
        quadrature starts from a fixed set of sample points, so a bin
        wider than the ridge can have every sample point land in the
        ~zero region on both sides and quad will confidently report a
        ~zero integral without ever subdividing toward the peak. Passing
        the ridge location as a breakpoint (see make_ridge_breakpoints)
        forces a subdivision there, so this argument is required rather
        than optional. Breakpoints outside (ymin, ymax) are dropped; an
        empty result for a bin is fine (scipy's points-based quadrature
        handles zero real breakpoints exactly like plain adaptive
        quadrature).
    n_workers : int
        Threads used to integrate bins concurrently. Only helps when
        pdf_func releases the GIL (e.g. calls into a compiled library).

    Returns
    -------
    expected : ndarray, shape (n_bins,)
        Expected event count in each bin for a throw of n_events.
    """
    def integrate_bin(bin_rect):
        xmin, xmax, ymin, ymax = bin_rect

        def integrand(eq, ep):
            return float(pdf_func(np.array([ep]), np.array([eq]))[0])

        def inner(ep):
            pts = sorted(p for p in inner_points_func(ep, ymin, ymax)
                        if ymin < p < ymax)
            val, _ = quad(integrand, ymin, ymax, args=(ep,),
                          points=pts, limit=200)
            return val

        val, _ = quad(inner, xmin, xmax, limit=200)
        return max(val, 0.0)

    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            integrals = np.array(list(pool.map(integrate_bin, bins)))
    else:
        integrals = np.array([integrate_bin(b) for b in bins])

    total = integrals.sum()
    if total == 0.0:
        raise ValueError("PDF integrates to zero over all bins.")

    return n_events * integrals / total


# ---------------------------------------------------------------------------
# Single-bin diagnostics
# ---------------------------------------------------------------------------

def debug_bin(pdf_func, bin_rect, inner_points_func, *,
              n_ep_probe=5, n_grid=400):
    """
    Diagnose the nested-quadrature integral of pdf_func over one bin.

    Use this when expected_counts_from_pdf reports a ~0 integral for a
    bin and it's unclear whether that's *physically correct* (the PDF
    really is negligible there) or a quadrature failure (quad missed
    the ridge). It prints, and returns, three independent estimates:

    - naive: nested quad with no breakpoints at all (what you'd get
      without inner_points_func -- reproduces the failure mode).
    - fixed: nested quad with inner_points_func, exactly as
      expected_counts_from_pdf computes it.
    - grid_ref: a brute-force fixed grid (Simpson's rule in both
      directions). This can't miss a feature the grid resolves,
      independently of quad's adaptive logic entirely, so a disagreement
      between grid_ref and fixed is a red flag even when breakpoints are
      in play (e.g. the breakpoint window is too narrow, or the ridge
      formula doesn't match the PDF being tested).

    It also prints the raw breakpoints inner_points_func returns at a
    few Ep values spanning the bin, and which of those survive the
    (ymin, ymax) clip -- if every point is being filtered out, the
    ridge has drifted entirely outside the bin's Eq range at that Ep,
    which is the first thing to check for a bin sitting at the edge of
    the band.

    Parameters
    ----------
    pdf_func : callable
        pdf_func(Ep_flat, Eq_flat) -> values_flat
    bin_rect : (xmin, xmax, ymin, ymax)
        The single bin to integrate.
    inner_points_func : callable or None
        Same as expected_counts_from_pdf's inner_points_func. Pass
        None to see only the naive and grid_ref estimates.
    n_ep_probe : int
        Number of Ep values, evenly spaced across the bin, to print
        breakpoints for.
    n_grid : int
        Grid resolution (n_grid x n_grid) for the brute-force reference.
        Must be fine enough to resolve the narrowest feature of the PDF
        inside the bin -- if grid_ref itself looks suspicious (e.g. 0
        when fixed is not), raise this before trusting either estimate.

    Returns
    -------
    dict with keys: naive, naive_err, fixed, fixed_err, grid_ref,
    breakpoints (list of (ep, raw_points, kept_points)).
    """
    xmin, xmax, ymin, ymax = bin_rect
    print(f"bin: Ep in [{xmin}, {xmax}], Eq in [{ymin}, {ymax}]")

    def integrand(eq, ep):
        return float(pdf_func(np.array([ep]), np.array([eq]))[0])

    def inner(ep, points_func):
        pts = None
        if points_func is not None:
            pts = [p for p in points_func(ep, ymin, ymax) if ymin < p < ymax]
            pts = sorted(pts) or None
        return quad(integrand, ymin, ymax, args=(ep,), points=pts, limit=200)

    naive, naive_err = quad(lambda ep: inner(ep, None)[0], xmin, xmax, limit=200)
    print(f"naive nested quad (no breakpoints) = {naive:.6g}  "
          f"(outer err est {naive_err:.2g})")

    fixed = fixed_err = None
    breakpoints_probed = []
    if inner_points_func is not None:
        fixed, fixed_err = quad(lambda ep: inner(ep, inner_points_func)[0],
                                xmin, xmax, limit=200)
        print(f"nested quad with breakpoints       = {fixed:.6g}  "
              f"(outer err est {fixed_err:.2g})")

        print(f"\nbreakpoints probed at {n_ep_probe} Ep values across the bin:")
        for ep in np.linspace(xmin, xmax, n_ep_probe):
            raw = list(inner_points_func(ep, ymin, ymax))
            kept = [p for p in raw if ymin < p < ymax]
            flag = ("  <-- all points fall outside [ymin, ymax]; "
                     "ridge is off the bin's Eq range at this Ep"
                     if not kept else "")
            print(f"  Ep={ep:12.5f}: raw={[round(p, 4) for p in raw]}  "
                  f"kept={[round(p, 4) for p in kept]}{flag}")
            breakpoints_probed.append((ep, raw, kept))
    else:
        print("no inner_points_func given -- skipping breakpointed quad")

    # Brute-force reference: fixed grid, Simpson in both directions.
    # Independent of quad's adaptive sampling entirely, so it's the
    # sanity check of last resort -- but only trustworthy if n_grid
    # resolves the PDF's narrowest feature inside the bin.
    ep_grid = np.linspace(xmin, xmax, n_grid)
    eq_grid = np.linspace(ymin, ymax, n_grid)
    Ep_g, Eq_g = np.meshgrid(ep_grid, eq_grid, indexing="ij")
    vals = np.asarray(pdf_func(Ep_g.ravel(), Eq_g.ravel())).reshape(Ep_g.shape)
    grid_ref = simpson(simpson(vals, x=eq_grid, axis=1), x=ep_grid)
    print(f"\nbrute-force grid reference ({n_grid}x{n_grid} Simpson) = {grid_ref:.6g}")
    print("(raise n_grid if this looks suspicious too -- it must resolve")
    print(" the narrowest feature of the PDF to be trustworthy)")

    return dict(naive=naive, naive_err=naive_err, fixed=fixed, fixed_err=fixed_err,
                grid_ref=grid_ref, breakpoints=breakpoints_probed)


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_chisquare_test(pdf_func, sampler_func, ep_range, eq_range,
                       n_events, n_throws, n_bins, inner_points_func,
                       *, n_ref=None, n_workers=1,
                       batch_size=5_000, seed=0):
    """
    Run a chi-square distribution test for a 2D PDF.

    Parameters
    ----------
    pdf_func : callable
        pdf_func(Ep_flat, Eq_flat) -> values_flat
    sampler_func : callable
        sampler_func(n, seed=None) -> (Ep, Eq), returning n events that are
        ALL already inside ep_range x eq_range. generate_events_NR/
        generate_events_ER have no ROI cropping of their own (by design --
        see event_generator.py), so passing either directly as sampler_func
        is wrong: this function reshapes sampler_func's output assuming
        every requested event lands in a bin, so any that don't just
        silently vanish from the observed counts while the expected counts
        below still assume the full n_events -- chi2 comes out inflated no
        matter how good the fit actually is, with no error to flag it.
        generate_events(fraction_ER=0.0 or 1.0, ep_range=..., eq_range=...)
        already crops to the region and is deterministically pure-NR/ER
        at those boundary values -- see docs/goodness-of-fit.md's
        sampler_NR for a worked example (it just unpacks the returned
        DataFrame's EP/EQ columns to the (Ep, Eq) tuple this needs).
    ep_range, eq_range : (float, float)
        Bounds of the sampling region, passed to equibin as hard limits.
        sampler_func's events must already be cropped to this box (see
        sampler_func above) -- this function does not crop for you.
    n_events : int
        Events per chi-square throw. All bins should have
        expected count >> 1; aim for n_events / n_bins >= 10.
    n_throws : int
        Total number of chi-square values to compute (e.g. 100_000).
    n_bins : int
        Target number of equibin bins.
    inner_points_func : callable
        Passed to expected_counts_from_pdf; see its docstring (and
        make_ridge_breakpoints) for why this is required rather than
        optional.
    n_ref : int or None
        Reference dataset size for computing bin boundaries.
        Defaults to max(50 * n_bins, 5 * n_events).
    n_workers : int
        Threads for the expected-count integration.
    batch_size : int
        Throws per batch. Each batch generates batch_size * n_events
        events at once; tune to fit available memory.
    seed : int
        Base random seed.

    Returns
    -------
    chi2_values : ndarray, shape (n_throws,)
    dof : int
        Degrees of freedom (n_bins - 1).
    bins : list of (xmin, xmax, ymin, ymax)
    expected : ndarray, shape (n_bins,)
        Expected counts per throw in each bin.
    """
    rng = np.random.default_rng(seed)

    if n_ref is None:
        n_ref = max(50 * n_bins, 5 * n_events)

    # --- step 1: reference data → equibin bins ---
    print(f"Generating {n_ref:,} reference events for binning...")
    Ep_ref, Eq_ref = sampler_func(n_ref, seed=int(rng.integers(0, 2**31)))
    print(f"Computing {n_bins} equibin bins...")
    result = equibin.bin_2d(
        Ep_ref, Eq_ref, n_bins=n_bins,
        xmin=ep_range[0], xmax=ep_range[1],
        ymin=eq_range[0], ymax=eq_range[1],
    )

    # equibin only tiles the extent of the reference data; snap the outer
    # envelope of the bins to the declared region so the bins tile it
    # fully and every thrown event lands in a bin.
    bins_arr = np.asarray(result.bins, dtype=np.float64)
    bins_arr[bins_arr[:, 0] == bins_arr[:, 0].min(), 0] = ep_range[0]
    bins_arr[bins_arr[:, 1] == bins_arr[:, 1].max(), 1] = ep_range[1]
    bins_arr[bins_arr[:, 2] == bins_arr[:, 2].min(), 2] = eq_range[0]
    bins_arr[bins_arr[:, 3] == bins_arr[:, 3].max(), 3] = eq_range[1]
    bins = [tuple(row) for row in bins_arr]
    n_bins_actual = len(bins)

    # --- step 2: expected counts from PDF ---
    print(f"Integrating PDF over {n_bins_actual} bins "
          f"(adaptive quadrature, {n_workers} workers)...")
    expected = expected_counts_from_pdf(
        pdf_func, bins, n_events, inner_points_func, n_workers=n_workers,
    )

    # --- step 3: batched chi-square throws ---
    print(f"Running {n_throws:,} throws of {n_events:,} events "
          f"(batch_size={batch_size:,})...")
    chi2_values = np.empty(n_throws)

    for batch_start in range(0, n_throws, batch_size):
        batch_end = min(batch_start + batch_size, n_throws)
        b = batch_end - batch_start                  # actual throws this batch

        # generate all events for this batch in one call
        seed_b = int(rng.integers(0, 2**31))
        Ep_b, Eq_b = sampler_func(b * n_events, seed=seed_b)
        Ep_b = Ep_b.reshape(b, n_events)            # (b, n_events)
        Eq_b = Eq_b.reshape(b, n_events)

        # count events per bin per throw: (b, n_bins)
        obs = np.zeros((b, n_bins_actual), dtype=np.intp)
        for j in range(n_bins_actual):
            xmin, xmax, ymin, ymax = bins_arr[j]
            mask = (
                (Ep_b >= xmin) & (Ep_b < xmax) &
                (Eq_b >= ymin) & (Eq_b < ymax)
            )
            obs[:, j] = mask.sum(axis=1)

        # Pearson chi-square: (b,)
        chi2_values[batch_start:batch_end] = (
            ((obs - expected) ** 2 / expected).sum(axis=1)
        )

        pct = 100 * batch_end / n_throws
        print(f"  {batch_end:>8,} / {n_throws:,}  ({pct:.0f}%)", end="\r", flush=True)

    print()
    return chi2_values, n_bins_actual - 1, bins, expected
