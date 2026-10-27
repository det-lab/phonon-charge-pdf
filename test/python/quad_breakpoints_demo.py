"""
Why scipy.integrate.quad can silently return ~0 for a narrow peak,
and how to fix it with the `points=` argument.

Background
----------
scipy.integrate.quad is *adaptive*: it starts from a fixed Gauss-Kronrod
sampling rule over the whole integration interval, compares two
polynomial-degree estimates, and only subdivides further where those
estimates disagree. That works great when the integrand is smooth on
the scale of the interval. But if your PDF is a narrow ridge/peak that
is much narrower than the interval, it's entirely possible for every
point in that initial sampling rule to land in the flat, ~0 region on
either side of the peak. In that case BOTH polynomial estimates agree
that the integral is ~0, quad reports a *tiny* error estimate (it has
no idea it missed anything), and it returns 0 instead of subdividing
toward the peak. This is the "quad returns something tiny/zero" failure
you're probably seeing on bins where the PDF ridge is narrow compared
to the bin.

The fix: `quad(f, a, b, points=[...])` tells quad to break [a, b] into
subintervals at the given points *before* starting the adaptive
search, and to refine each subinterval independently. If you tell it
roughly where the peak is (even just "peak center +/- N widths"), it's
guaranteed to place a subinterval around the peak and adaptively
refine inside it, rather than gambling on stumbling into it by luck.

This script demonstrates the failure and the fix in two steps:
  1. A single narrow 1D Gaussian in a wide interval (extreme, dramatic).
  2. A 2D case shaped like a real "integrate a PDF over a histogram
     bin" problem: a narrow ridge in y whose center y0(x) drifts with
     x, integrated with a nested quad(quad(...)) over an (x, y) box.
     This is the same technique used in this package's
     goodness_of_fit.py (see `inner_points_func`), with the
     band-specific breakpoints built in `make_ridge_breakpoints`.

No threads, no external dependencies beyond numpy/scipy.
"""

import numpy as np
from scipy.integrate import quad

# ---------------------------------------------------------------------------
# 1. Minimal 1D example
# ---------------------------------------------------------------------------

def demo_1d():
    print("=" * 70)
    print("1D demo: single narrow Gaussian in a wide interval")
    print("=" * 70)

    mu, sigma = 5.0, 0.01          # peak sits at y=5, width 0.01
    a, b = 0.0, 20.0                # interval is 2000 sigma wide

    def pdf(y):
        return np.exp(-0.5 * ((y - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    # True integral of a full Gaussian is 1.0 (it all fits inside [a, b]).

    naive, naive_err = quad(pdf, a, b)
    print(f"naive quad(pdf, {a}, {b})        = {naive:.6g}  "
          f"(reported error: {naive_err:.2g})")

    # Give quad the peak location as a breakpoint. A window of a few
    # sigma on either side is plenty; quad still adapts freely inside
    # each subinterval, this just guarantees one subinterval brackets
    # the peak.
    points = [mu - 5 * sigma, mu, mu + 5 * sigma]
    fixed, fixed_err = quad(pdf, a, b, points=points)
    print(f"quad(pdf, {a}, {b}, points={points})")
    print(f"  = {fixed:.6g}  (reported error: {fixed_err:.2g})")
    print(f"true value = 1.0")
    print()
    print("Notice the naive call's reported error is also tiny — quad isn't")
    print("just imprecise, it's *confidently wrong*. There's no warning sign")
    print("in the return value alone; you have to know to check.")
    print()


# ---------------------------------------------------------------------------
# 2. 2D case: PDF over a histogram bin, ridge center drifts with x
# ---------------------------------------------------------------------------

def demo_2d():
    print("=" * 70)
    print("2D demo: integrate a drifting ridge over an (x, y) bin")
    print("=" * 70)

    sigma = 0.01  # ridge width in y, held fixed for simplicity

    def y0(x):
        """Ridge center as a function of x — stand-in for a physical band."""
        return 5.0 + 0.5 * x

    def pdf(y, x):
        # NOTE: quad integrates over the *first* positional argument, so
        # the integration variable (y) must come first here, with x
        # passed through via args=(x,).
        return np.exp(-0.5 * ((y - y0(x)) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    xmin, xmax = 0.0, 10.0
    ymin, ymax = 0.0, 20.0
    # True integral: the ridge in y integrates to 1 for every x (it's a
    # normalized Gaussian, always inside [ymin, ymax]), so integrating
    # over x in [xmin, xmax] gives exactly (xmax - xmin) = 10.

    def inner_naive(x):
        val, _ = quad(pdf, ymin, ymax, args=(x,))
        return val

    naive, naive_err = quad(inner_naive, xmin, xmax, limit=200)
    print(f"naive nested quad  = {naive:.6g}  (reported error: {naive_err:.2g})")

    def inner_fixed(x):
        center = y0(x)
        pts = [center - 5 * sigma, center, center + 5 * sigma]
        pts = [p for p in pts if ymin < p < ymax]  # keep points inside the bin
        val, _ = quad(pdf, ymin, ymax, args=(x,), points=pts or None)
        return val

    fixed, fixed_err = quad(inner_fixed, xmin, xmax, limit=200)
    print(f"fixed nested quad  = {fixed:.6g}  (reported error: {fixed_err:.2g})")
    print(f"true value = {xmax - xmin:.6g}")
    print()
    print("The inner (y) integral gets breakpoints computed from the known")
    print("ridge location y0(x) at whatever x the outer quad happens to be")
    print("sampling. That's the general recipe: whenever you know roughly")
    print("where a narrow feature sits, hand its location to quad's")
    print("`points=` argument instead of hoping adaptive refinement finds")
    print("it on its own.")
    print()


if __name__ == "__main__":
    demo_1d()
    demo_2d()
