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

# sampler_NR stands in for YOUR real measured events below, and is also
# what "Validating the procedure itself" passes as run_chisquare_test's
# sampler_func, further down -- it has to return events already inside
# (ep_min, ep_max) x (eq_min, eq_max). generate_events_NR has no ROI
# cropping of its own (see "Generating simulated events"), so this uses
# the top-level generate_events(fraction_ER=0.0) instead, which already
# crops to ep_range/eq_range and is deterministically pure-NR
# (Var(Binomial(n, p)) is exactly zero at p=0 -- see that function's
# docstring). F0_ER/mix_frac_ER/scale_fast_ER/scale_slow_ER are still
# required (generate_events() always needs both bands' parameters) but
# never actually used here, since zero ER events are ever generated.
def sampler_NR(n, seed=None):
    events = pdf.generate_events(
        n, fraction_ER=0.0, seed=seed, ep_range=(ep_min, ep_max), eq_range=(eq_min, eq_max),
        k=FIT["k"], Z=FIT["Z"], F0_NR=FIT["F0"], F0_ER=0.2,
        eps=FIT["eps"], V=FIT["V"], p0=FIT["p0"], p10=FIT["p10"], q0=FIT["q0"], q10=FIT["q10"],
        mix_frac_NR=0.53693208, scale_fast_NR=6.41515782, scale_slow_NR=23.71789286,
        mix_frac_ER=0.573211975, scale_fast_ER=0.169520023, scale_slow_ER=279.552394,
    )
    return events["EP"].to_numpy(), events["EQ"].to_numpy()

Ep_obs, Eq_obs = sampler_NR(20_000, seed=0)

# bin your own data -- counts[i] is the observed count in bins[i]
n_bins = 64
result = equibin.bin_2d(Ep_obs, Eq_obs, n_bins=n_bins,
                         xmin=ep_min, xmax=ep_max, ymin=eq_min, ymax=eq_max)

# the density at your fit -- make_Ppq_densities_NR takes exactly FIT's keys,
# so **FIT spreads cleanly here
pdf_func = pdf.goodness_of_fit.make_Ppq_densities_NR(**FIT, n_workers=8)

# ...and where its ridge is (see "Why inner_points_func is required" below),
# both at FIT, not re-derived per bin. Listed individually, not **FIT: this
# one doesn't take F0 at all (its width model ignores Fano broadening -- see
# its own docstring). Use make_ridge_breakpoints_ER (no k/Z) for the ER band.
ridge_breakpoints = pdf.goodness_of_fit.make_ridge_breakpoints_NR(
    k=FIT["k"], Z=FIT["Z"], eps=FIT["eps"], V=FIT["V"],
    p0=FIT["p0"], p10=FIT["p10"], q0=FIT["q0"], q10=FIT["q10"],
    erecoil_max=700.0, n_window_widths=10.0,
)

expected = pdf.goodness_of_fit.expected_counts_from_pdf(
    pdf_func, result.bins, n_events=len(Ep_obs),
    inner_points_func=ridge_breakpoints, n_workers=8,
)

chi2_stat = float(((result.counts - expected) ** 2 / expected).sum())
# len(result.bins), not n_bins: equibin.bin_2d can't always split the data into
# exactly n_bins bins (e.g. too few events, or duplicate/degenerate values on
# one axis) -- result.bins/result.counts/expected above are already sized to
# whatever it actually produced, so dof has to match that, not the request.
dof = len(result.bins) - 1 - len(FIT)          # subtract fitted parameters, not just bins - 1
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
not an error. `make_ridge_breakpoints_NR`/`ER` locates the ridge from the band physics (the yield
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

# sampler_NR is the same in-ROI sampler defined above, under "Checking your
# fit" -- reused as-is here, since run_chisquare_test needs exactly the
# sampler_func(n, seed=None) -> (Ep, Eq) shape it already has.
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
