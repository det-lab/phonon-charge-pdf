# Testing goodness of fit

Once you have a fit (see [Fitting](fitting.md)), the next question is whether it's actually
a good fit. `pdf.goodness_of_fit` bins your data with [`equibin`](https://github.com/det-lab/equibin)
(equal-count 2D binning), computes the expected count in each bin by integrating the PDF
over it, and gives you a Pearson chi-square statistic to compare against. Its functions are
namespaced under `goodness_of_fit` rather than flattened onto `phonon_charge_pdf` directly
(`pdf.goodness_of_fit.run_chisquare_test`, not `pdf.run_chisquare_test`).

## Checking your fit

```python
import numpy as np
from scipy.stats import chi2 as chi2_dist
import equibin
import phonon_charge_pdf as pdf

# your analysis ROI, keV -- same as "Fitting"
ep_min, ep_max = 2.5, 350.0
eq_min, eq_max = 0.75, 200.0

# your best-fit parameters (see "Fitting")
FIT = dict(k=0.18, Z=32.0, F0=0.122, eps=3.0e-3, V=3.0,
           p0=0.06421907, p10=0.48998486, q0=0.23718488, q10=0.27093151)

# Ep_obs/Eq_obs below are generated, for this walkthrough to run end to end --
# use YOUR real measured events instead. generate_events_NR needs the NR
# band's true-recoil-energy spectrum too (see "Generating simulated events");
# this example reuses this project's own mix_frac/scale_fast/scale_slow.
Ep_obs, Eq_obs, _, _ = pdf.generate_events_NR(
    20_000, **FIT, mix_frac=0.53693208, scale_fast=6.41515782,
    scale_slow=23.71789286, seed=0)

# bin your own data -- counts[i] is the observed count in bins[i]
n_bins = 64
result = equibin.bin_2d(Ep_obs, Eq_obs, n_bins=n_bins,
                         xmin=ep_min, xmax=ep_max, ymin=eq_min, ymax=eq_max)

# the density at your fit, and where its ridge is (see "Why inner_points_func
# is required" below) -- both at FIT, not re-derived per bin
pdf_func = pdf.goodness_of_fit.make_Ppq_densities_NR(**FIT, n_workers=8)
ridge_breakpoints = pdf.goodness_of_fit.make_ridge_breakpoints(
    "NR", k=FIT["k"], Z=FIT["Z"], eps=FIT["eps"], V=FIT["V"],
    p0=FIT["p0"], p10=FIT["p10"], q0=FIT["q0"], q10=FIT["q10"],
    er_max=700.0, n_window_widths=10.0,
)

expected = pdf.goodness_of_fit.expected_counts_from_pdf(
    pdf_func, result.bins, n_events=len(Ep_obs),
    inner_points_func=ridge_breakpoints, n_workers=8,
)

chi2_stat = float(((result.counts - expected) ** 2 / expected).sum())
dof = n_bins - 1 - len(FIT)          # subtract fitted parameters, not just bins - 1
p_value = chi2_dist.sf(chi2_stat, dof)
print(f"chi2 = {chi2_stat:.1f}, dof = {dof}, p = {p_value:.3f}")
```

`make_Ppq_densities_NR`/`make_Ppq_densities_ER` give you the density at arbitrary (Ep, Eq)
points for fixed physics parameters — a genuine probability density, just over the full
physical plane rather than your analysis region. That's fine here: `expected_counts_from_pdf`
only cares about the *relative* shape across bins (it renormalizes the integrals to sum to
`n_events` itself), the same way it's fine for `Ppq_compiled`'s own normalization integral to
be computed over your region separately — see [Fitting](fitting.md) if that distinction
isn't familiar yet. Same shape for the ER band (`make_Ppq_densities_ER`, no `k`/`Z`).

`dof` subtracts the number of fitted parameters, not just `n_bins - 1`: parameters fit from
the same data you're testing already "used up" some of the bins' freedom to match.

## Why `inner_points_func` is required

The NR/ER band PDFs are narrow ridges in the (Ep, Eq) plane. Plain adaptive quadrature over
a bin starts from a fixed set of sample points; if a bin is wider than the ridge, every one
of those points can land in the ~zero region on either side and quadrature will confidently
report a ~zero integral without ever subdividing toward the peak — a silent wrong answer,
not an error. `make_ridge_breakpoints` locates the ridge from the band physics (the yield
model and Neganov-Luke amplification) and its local width (phonon and charge resolution
added in quadrature), so `expected_counts_from_pdf` can force quadrature to subdivide there.
There's no "skip this" option for these PDFs — passing `inner_points_func=None` is not
supported by `expected_counts_from_pdf`, only by `debug_bin` below, specifically so you can
see the failure mode it exists to prevent.

## Diagnosing a specific bin

If a bin's expected count comes out at 0 or suspiciously small, that isn't automatically a
quadrature bug — a wide tail bin can genuinely sit mostly or entirely off the band, in which
case a tiny expected count is correct. `debug_bin` tells the two cases apart by computing
the same bin three independent ways: `naive` (no breakpoints — reproduces the missed-peak
failure mode if there is one), `fixed` (with breakpoints, exactly what `expected_counts_from_pdf`
computes), and `grid_ref` (a brute-force fixed-grid Simpson integral, independent of
quadrature's adaptive logic entirely). If all three agree, the small value is real. If `naive`
disagrees with `fixed`/`grid_ref`, that bin hit the failure mode above.

```python
bin_rect = result.bins[np.argmin(expected)]     # e.g. the bin with the smallest expected count
diagnosis = pdf.goodness_of_fit.debug_bin(pdf_func, bin_rect, ridge_breakpoints)
```

It also prints where `inner_points_func`'s breakpoints land relative to the bin at a few Ep
values, flagging when all of them fall outside the bin's Eq range — a sign the ridge has
drifted off the bin there rather than quadrature having missed anything.

## Validating the procedure itself

The check above answers "is this one fit good." A different, useful question is "does this
binning-and-chi-square *procedure* behave the way it should" — if you simulate many
datasets under the model and compute a chi-square for each, those chi-square values should
themselves follow the chi2(dof) distribution. `run_chisquare_test` does exactly that: it
takes a `sampler_func(n, seed=None) -> (Ep, Eq)` instead of your own fixed dataset, and
returns the array of chi-square values across many simulated throws for you to check
against `scipy.stats.chi2`. This is how this library's own test suite validates that its
PDFs and this binning procedure agree with each other (`test/python/test_chisquare_*.py`)
— reach for it if you want that same confidence in your own binning/breakpoint choices
before trusting the single-fit check above.

```python
from scipy.stats import kstest

def sampler_NR(n, seed=None):
    Ep, Eq, _, _ = pdf.generate_events_NR(                 # drop the latent Er/N outputs --
        n, **FIT, mix_frac=0.53693208, scale_fast=6.41515782,  # sampler_func only returns (Ep, Eq)
        scale_slow=23.71789286, seed=seed)
    return Ep, Eq

chi2_values, dof, bins, expected = pdf.goodness_of_fit.run_chisquare_test(
    pdf_func, sampler_NR, (ep_min, ep_max), (eq_min, eq_max),
    n_events=1_000, n_throws=10_000, n_bins=64,
    inner_points_func=ridge_breakpoints, n_workers=8,
)

ks_stat, ks_p = kstest(chi2_values, chi2_dist(dof).cdf)
print(f"mean chi2 = {chi2_values.mean():.1f} (expect {dof}), KS p = {ks_p:.3f} (expect >> 0.05)")
```

Here `dof = n_bins - 1` (not minus the number of fitted parameters, unlike the single-fit
check above): every throw samples from `FIT` exactly, so there's nothing to fit and nothing
for those bins' freedom to have been "used up" by.
