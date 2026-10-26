"""phonon_charge_pdf.Ppq_py: pure-Python reference implementation of the
NR/ER (Ep, Eq) probability model.

Promoted from internal/ to a public, stable module alongside Ppq
(Ppq_fort.py) and event_generator.py -- it has no compiled-library
dependency at all (pure numpy/scipy), so it works anywhere this package
is installed, compiler or not.

This is deliberately the *slow*, brute-force path: Ppq_inspect_NR and
Ppq_inspect_ER scan a uniform Er grid (step set by `er_step`) rather than
using the Fortran side's analytic peak location and windowed quadrature,
specifically so they can serve as an independent cross-check of
Ppq/_Ppq_fort_bindings (see test/python/test_chisquare_ppqn.py and
test_PpqFort.py) -- never use these in a fit/MCMC inner loop; use Ppq
for that.

Public functions:
* Y, Nbar, sigp, sigq -- the ionization-yield, mean pair-count, and
  phonon/ionization resolution models.
* PEr_NR, PEr_ER -- the true-recoil-energy spectra (NR, ER).
* aN, bN, cN, N_exponent -- the Gaussian-in-N exponent's quadratic
  coefficients, and the exponent itself.
* pq_given_Er_NR, pq_given_Er_ER -- the unnormalized (Ep, Eq) density at one Er,
  integrated over N by a fixed 21-point Simpson rule.
* Ppq_inspect_NR(_vec), Ppq_inspect_ER(_vec) -- the full Ppq_NR/Ppq_ER value at
  (Ep, Eq), found by scanning for peak(s) in Er and integrating each
  with scipy.integrate.quad. The non-"_vec" form additionally returns
  the scan grid and per-peak diagnostics (width, location) for
  inspection -- the point of reaching for this over the Fortran side;
  "_vec" loops over paired (Ep, Eq) arrays and returns just the values.
* integrate_inspect -- the peak-finding/integration engine Ppq_inspect_NR/
  Ppq_inspect_ER are built on; also useful directly against a custom
  integrand (see test_integrate_inspect_multimodal.py). Its "g" argument
  is a generic placeholder for "the function being integrated" (called
  with pq_given_Er_NR for NR, pq_given_Er_ER for ER) -- not a reference to gamma/ER.
* consecutive_groups, gaussian_profile -- small helpers (grouping
  consecutive indices; evaluating a Gaussian) used by integrate_inspect's
  peak-width fit. Not part of the public API (not in __all__) -- purely
  implementation details of integrate_inspect.
"""

import warnings

import numpy as np
import scipy.integrate as integrate
from scipy.optimize import curve_fit

__all__ = [
    "Y", "Nbar", "sigp", "sigq", "PEr_NR", "PEr_ER",
    "aN", "bN", "cN", "N_exponent", "pq_given_Er_NR", "pq_given_Er_ER",
    "Ppq_inspect_NR", "Ppq_inspect_vec_NR",
    "Ppq_inspect_ER", "Ppq_inspect_vec_ER",
    "integrate_inspect",
]

# No module-level __version__ here anymore: the package version lives in
# fpm.toml (git tags, CITATION.cff) -- see the versioning bootstrap in the
# repo's git-flow setup. A second, hand-maintained copy here is exactly the
# per-file version drift this rename is meant to retire.


def Y(Er,
      *,
      k, Z):
    """The ionization yield, Lindhard model: Y(Er) = k*g(eps_L) / (1 + k*g(eps_L)),
    with eps_L = 11.5*Er*Z**(-7/3) and g(eps_L) = 3*eps_L**0.15 + 0.7*eps_L**0.29 + eps_L.
    Er in keV; Z is the target's atomic number."""
    eps_L = 11.5 * np.absolute(Er) * Z**(-7.0/3.0)
    g = 3.0*eps_L**0.15 + 0.7*eps_L**0.29 + eps_L
    return k*g / (1.0 + k*g)


def Nbar(Er,
         *,
         k, Z,
         eps,
         is_gamma):
    """Mean number of e/h pairs produced by a recoil of energy Er (keV).
    is_gamma=True selects the electron-recoil yield (Y=1) instead of the
    Lindhard NR yield Y(Er, k=k, Z=Z); k, Z are unused (pass any value)
    when is_gamma=True."""
    yield_val = 1.0 if is_gamma else Y(Er, k=k, Z=Z)
    return yield_val*Er/eps


def sigp(Ep,
         *,
         eps,
         V,
         p0, p10):
    """Phonon-energy resolution at measured phonon energy Ep (keV)."""
    e2pre = (p10**2 - p0**2)
    e2e = (Ep/(10*(1+(V/eps/1000))))**2
    #e2e = (Ep/(10*((V/eps/1000))))**2 #THIS ERROR WAS IN Ppq v1!! it makes a difference!
    return np.sqrt(p0**2+e2pre*e2e)


def sigq(Eq,
         *,
         q0, q10):
    """Ionization-energy resolution at measured charge energy Eq (keV)."""
    e2pre = (q10**2 - q0**2)
    e2e = (Eq/10)**2
    return np.sqrt(q0**2+e2pre*e2e)


# Fit parameters -- matches the Fortran PErN_impl constants exactly (not
# exposed as arguments there either: PEr_NR(Er) is the public signature)
_PNa, _PNb, _PNd = 0.53693208, 6.41515782, 23.71789286


def PEr_NR(Er):
    """Probability density of the *true* recoil energy Er (keV) for
    neutron recoils: a two-component exponential mixture, zero for Er < 0."""
    Er = np.asarray(Er)  # Convert input to a NumPy array
    result = np.where(Er < 0, 0, _PNa * (1 / _PNb) * np.exp(-Er / _PNb) + (1 - _PNa) * (1 / _PNd) * np.exp(-Er / _PNd))
    return result
#...TBD gammas


# Cfgammas -- matches the Fortran PErG_impl constants exactly
_PGa, _PGb, _PGd = 5.73211975e-01, 1.69520023e-01, 2.79552394e+02


def PEr_ER(Er):
    """Probability density of the *true* recoil energy Er (keV) for
    electron recoils (Cf gammas): a two-component exponential mixture,
    zero for Er < 0."""
    Er = np.asarray(Er)  # Convert Er to a NumPy array
    result = np.where(Er < 0, 0, _PGa * (1 / _PGb) * np.exp(-Er / _PGb) + (1 - _PGa) * (1 / _PGd) * np.exp(-Er / _PGd))
    return result


def aN(Er, Ep, Eq,
       *,
       F0,
       eps,
       V,
       p0, p10,
       q0, q10):
    """Linear coefficient of N_exponent's Gaussian-in-N exponent."""
    t1 = (V/1e3)*(Ep-Er)/sigp(Ep,eps=eps,V=V,p0=p0,p10=p10)**2
    t2 = eps*Eq/sigq(Eq,q0=q0,q10=q10)**2
    t3 = 1/F0
    return t1+t2+t3


def bN(Er, Ep, Eq,
       *,
       k, Z,
       F0,
       eps,
       V,
       p0, p10,
       q0, q10,
       is_gamma):
    """Quadratic coefficient of N_exponent's Gaussian-in-N exponent. is_gamma
    selects the electron-recoil yield (Y=1) instead of the Lindhard NR
    yield Y(Er, k=k, Z=Z); k, Z are unused (pass any value) when
    is_gamma=True."""
    Nbar_val = Nbar(Er,k=k,Z=Z,eps=eps,is_gamma=is_gamma)
    t1 = 1/(2*Nbar_val*F0)
    t2 = eps**2/(2*sigq(Eq,q0=q0,q10=q10)**2)
    t3 = V**2/(2*(sigp(Ep,eps=eps,V=V,p0=p0,p10=p10)*1e3)**2)
    return t1+t2+t3


def cN(Er, Ep, Eq,
       *,
       k, Z,
       F0,
       eps,
       V,
       p0, p10,
       q0, q10,
       is_gamma):
    """Constant term of N_exponent's Gaussian-in-N exponent. is_gamma selects
    the electron-recoil yield (Y=1) instead of the Lindhard NR yield
    Y(Er, k=k, Z=Z); k, Z are unused (pass any value) when is_gamma=True."""
    Nbar_val = Nbar(Er,k=k,Z=Z,eps=eps,is_gamma=is_gamma)
    t1 = -(Ep-Er)**2/(2*sigp(Ep,eps=eps,V=V,p0=p0,p10=p10)**2)
    t2 = -Eq**2/(2*sigq(Eq,q0=q0,q10=q10)**2)
    t3 = -Nbar_val/(2*F0)
    return t1+t2+t3


def N_exponent(Er, Ep, Eq,
           *,
           k, Z,
           F0,
           eps,
           V,
           p0, p10,
           q0, q10,
           is_gamma):
    """The exponent of the Gaussian-in-N integral that pq_given_Er_NR/pq_given_Er_ER
    evaluate -- completing the square of cN + aN*N + bN*N**2 in N gives
    cN + aN**2/(4*bN), which is what this returns. is_gamma selects the
    electron-recoil yield (Y=1) instead of the Lindhard NR yield
    Y(Er, k=k, Z=Z); k, Z are unused (pass any value) when is_gamma=True."""
    cN_val = cN(Er,Ep,Eq,k=k,Z=Z,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10,is_gamma=is_gamma)
    aN_val = aN(Er,Ep,Eq,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10)
    bN_val = bN(Er,Ep,Eq,k=k,Z=Z,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10,is_gamma=is_gamma)
    exponent = cN_val + (aN_val**2 / (4*bN_val))
    return exponent


def pq_given_Er_NR(Er, Ep, Eq,
             *,
             k, Z,
             F0,
             eps,
             V,
             p0, p10,
             q0, q10):
    """Unnormalized NR (Ep, Eq) density at one true recoil energy Er (keV):
    PEr_NR(Er) times the N-integral (a fixed 21-point Simpson rule over a
    +/-5-sigma window around the Gaussian-in-N peak)."""
    F_val    = F0
    Nbar_val = Nbar(Er, k=k, Z=Z, eps=eps, is_gamma=False)
    if Nbar_val <= 0.0 or F_val <= 0.0:
        return 0.0
    sigma_N    = np.sqrt(Nbar_val * F_val)
    norm_const = 1.0 / (2.0 * np.pi) ** 1.5
    sp0 = p0 ** 2
    sp1 = (p10 ** 2 - p0 ** 2) / (10.0 * (1.0 + V / (eps * 1.0e3))) ** 2
    sq0 = q0 ** 2
    sq1 = (q10 ** 2 - q0 ** 2) / 100.0
    EP_mean      = Er + (V * 1.0e-3) * Nbar_val
    EQ_mean      = eps * Nbar_val
    sigp_mean_sq = sp0 + sp1 * EP_mean ** 2
    sigq_mean_sq = sq0 + sq1 * EQ_mean ** 2
    a_coeff = ((V * 1.0e-3) * (Ep - Er) / sigp_mean_sq
               + eps * Eq / sigq_mean_sq
               + 1.0 / F_val)
    b_coeff = (1.0 / (2.0 * Nbar_val * F_val)
               + eps ** 2 / (2.0 * sigq_mean_sq)
               + V ** 2 / (2.0 * 1.0e6 * sigp_mean_sq))
    N_star     = a_coeff / (2.0 * b_coeff)
    sigma_Neff = 1.0 / np.sqrt(2.0 * b_coeff)
    if N_star - 5.0 * sigma_Neff < 0.0:
        N_lo, N_hi = 0.0, 10.0 * sigma_Neff
    else:
        N_lo, N_hi = N_star - 5.0 * sigma_Neff, N_star + 5.0 * sigma_Neff
    N_pts = np.linspace(N_lo, N_hi, 21)
    h_N   = (N_hi - N_lo) / 20.0
    EP_nl = Er + (V * 1.0e-3) * N_pts
    EQ_nl = eps * N_pts
    sig_p = np.sqrt(sp0 + sp1 * EP_nl ** 2)
    sig_q = np.sqrt(sq0 + sq1 * EQ_nl ** 2)
    exp_arg = (-0.5 * ((Ep - EP_nl) / sig_p) ** 2
               - 0.5 * ((Eq - EQ_nl) / sig_q) ** 2
               - 0.5 * ((N_pts - Nbar_val) / sigma_N) ** 2)
    integrand = np.zeros(21)
    mask = exp_arg >= -700.0
    integrand[mask] = (norm_const / (sigma_N * sig_p[mask] * sig_q[mask])
                       * np.exp(exp_arg[mask]))
    simps_w = np.array([1,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,1], dtype=float)
    return float((h_N / 3.0) * np.dot(simps_w, integrand) * PEr_NR(Er))


def Ppq_inspect_vec_NR(Ep, Eq,
              *,
              k, Z,
              F0,
              eps,
              V,
              p0, p10,
              q0, q10,
              er_step):
    """Vectorized Ppq_inspect_NR: Ppq_NR at each paired (Ep[i], Eq[i]),
    returned as a plain list of values (no scan grid/diagnostics -- use
    Ppq_inspect_NR directly if you need those)."""
    valuesArr = []
    for this_Ep, this_Eq in zip(Ep, Eq):
        thisVal, _, _ = Ppq_inspect_NR(this_Ep, this_Eq, k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10, er_step=er_step)
        valuesArr.append(thisVal[0])
    return valuesArr


def Ppq_inspect_NR(Ep, Eq,
              *,
              k, Z,
              F0,
              eps,
              V,
              p0, p10,
              q0, q10,
              er_step):
    """The full NR PDF value Ppq_NR(Ep, Eq), found by scanning a uniform Er
    grid (step `er_step`, in keV) for the Gaussian-in-N exponent's peak
    and integrating pq_given_Er_NR around it with scipy.integrate.quad -- the
    slow, brute-force reference this package's event_generator.py and
    the compiled Fortran Ppq_NR are cross-checked against (see
    test/python/test_chisquare_ppqn.py, test_PpqFort.py). Never use this
    in a fit/MCMC inner loop; use Ppq for that.

    Returns ((value, quad_error), (er_scan_grid, er_scan_integrand),
    peak_info_list): peak_info_list is one dict per located peak, with
    "minx"/"maxx" (the integration window), "integral"/"error" (that
    peak's own quad result), and "width"/"Ermx"/"integrand_max" from a
    Gaussian fit (nan if the fit didn't converge). Usage to just get the
    value: (value, _), _, _ = Ppq_inspect_NR(...).

    WARNING: the peak search only works if your peaks have a width
    greater than ~0.1 keV, which holds for all reasonable CDMS
    parameters; its advantage over an exact minimizer is that it never
    fails to find *a* peak (an exact minimizer sometimes does).
    """
    if F0 == 0:
        raise ValueError("Fano factor F(Er) = F0 is zero for all Er")

    # find Er_max
    f = lambda er,ep,eq: -1*N_exponent(er,ep,eq,k=k,Z=Z,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10,is_gamma=False)

    maxx = max(Ep, Eq) + 5
    er_arr = np.arange(5e-21, maxx, er_step)
    f_arr = np.zeros(len(er_arr), dtype=np.float64)
    for idx, er in enumerate(er_arr):
        try:
            f_arr[idx] = -1*N_exponent(er,Ep,Eq,k=k,Z=Z,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10,is_gamma=False)
        except Exception:
            # N_exponent can over/underflow at extreme Er; leaving this
            # sample at its default 0 and continuing the scan is correct
            # here, since a real peak is never isolated to a single
            # sample on a grid this fine.
            pass
    real_val_idx = np.where(f_arr > 0)[0]
    if len(real_val_idx) == 0:
        if np.all(f_arr == 0):
            return (0, 0), (er_arr, f_arr), []
        elif np.all(np.isnan(f_arr)):
            return (np.nan, np.nan), (er_arr, f_arr), []
    else:
        min_idx = np.argmin(f_arr[~np.isnan(f_arr)])
        Ermx = er_arr[min_idx]

    g = lambda er, ep, eq: pq_given_Er_NR(er, ep, eq, k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)

    return integrate_inspect(g, Ep, Eq, Ermx, er_step)


def pq_given_Er_ER(Er, Ep, Eq,
             *,
             F0,
             eps,
             V,
             p0, p10,
             q0, q10):
    """Unnormalized ER (Ep, Eq) density at one true recoil energy Er (keV):
    PEr_ER(Er) times the N-integral (a fixed 21-point Simpson rule over a
    +/-5-sigma window around the Gaussian-in-N peak), with the electron-
    recoil yield Y=1 (see pq_given_Er_NR for the analogous NR calculation)."""
    F_val    = F0
    Nbar_val = Nbar(Er, k=0.0, Z=1.0, eps=eps, is_gamma=True)   # k, Z unused: Y=1 for electron recoils
    if Nbar_val <= 0.0 or F_val <= 0.0:
        return 0.0
    sigma_N    = np.sqrt(Nbar_val * F_val)
    norm_const = 1.0 / (2.0 * np.pi) ** 1.5
    sp0 = p0 ** 2
    sp1 = (p10 ** 2 - p0 ** 2) / (10.0 * (1.0 + V / (eps * 1.0e3))) ** 2
    sq0 = q0 ** 2
    sq1 = (q10 ** 2 - q0 ** 2) / 100.0
    EP_mean      = Er + (V * 1.0e-3) * Nbar_val
    EQ_mean      = eps * Nbar_val
    sigp_mean_sq = sp0 + sp1 * EP_mean ** 2
    sigq_mean_sq = sq0 + sq1 * EQ_mean ** 2
    a_coeff = ((V * 1.0e-3) * (Ep - Er) / sigp_mean_sq
               + eps * Eq / sigq_mean_sq
               + 1.0 / F_val)
    b_coeff = (1.0 / (2.0 * Nbar_val * F_val)
               + eps ** 2 / (2.0 * sigq_mean_sq)
               + V ** 2 / (2.0 * 1.0e6 * sigp_mean_sq))
    N_star     = a_coeff / (2.0 * b_coeff)
    sigma_Neff = 1.0 / np.sqrt(2.0 * b_coeff)
    if N_star - 5.0 * sigma_Neff < 0.0:
        N_lo, N_hi = 0.0, 10.0 * sigma_Neff
    else:
        N_lo, N_hi = N_star - 5.0 * sigma_Neff, N_star + 5.0 * sigma_Neff
    N_pts = np.linspace(N_lo, N_hi, 21)
    h_N   = (N_hi - N_lo) / 20.0
    EP_nl = Er + (V * 1.0e-3) * N_pts
    EQ_nl = eps * N_pts
    sig_p = np.sqrt(sp0 + sp1 * EP_nl ** 2)
    sig_q = np.sqrt(sq0 + sq1 * EQ_nl ** 2)
    exp_arg = (-0.5 * ((Ep - EP_nl) / sig_p) ** 2
               - 0.5 * ((Eq - EQ_nl) / sig_q) ** 2
               - 0.5 * ((N_pts - Nbar_val) / sigma_N) ** 2)
    integrand = np.zeros(21)
    mask = exp_arg >= -700.0
    integrand[mask] = (norm_const / (sigma_N * sig_p[mask] * sig_q[mask])
                       * np.exp(exp_arg[mask]))
    simps_w = np.array([1,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,1], dtype=float)
    return float((h_N / 3.0) * np.dot(simps_w, integrand) * PEr_ER(Er))


def consecutive_groups(data, stepsize=1):
    """Split a sorted integer array into groups of consecutive values
    (each group's elements differ from the next by exactly `stepsize`).
    Used by integrate_inspect to group a peak-scan's nonzero sample
    indices into one or more peaks. Not part of the public API (not in
    __all__) -- an implementation detail of integrate_inspect."""
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def gaussian_profile(x, A, mu, sigma):
    """A*exp(-(x-mu)**2 / (2*sigma**2)) -- the model integrate_inspect fits
    to a located peak to estimate its width. Not part of the public API
    (not in __all__) -- an implementation detail of integrate_inspect."""
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def integrate_inspect(g, Ep, Eq, Ermx, er_step):
    """Integrate g(Er, Ep, Eq) over Er by locating its peak(s) on a
    uniform scan grid (step `er_step/2`, in keV, starting from an
    approximate peak location Ermx) and integrating each located peak
    with scipy.integrate.quad. This is the engine Ppq_inspect_NR and
    Ppq_inspect_ER are built on; it also works directly against any custom
    integrand g (see test/python/test_integrate_inspect_multimodal.py for
    a multi-peak example). The "g" here is a generic placeholder for
    "the function being integrated" (it's called with pq_given_Er_NR for NR,
    pq_given_Er_ER for ER) -- not a reference to gamma/ER, which is why this
    function's own name doesn't carry an N/G suffix the way Ppq_inspect_NR/
    Ppq_inspect_ER do.

    Returns ((integral, quad_error), (er_scan_grid, er_scan_integrand),
    peak_info_list) -- see Ppq_inspect_NR's docstring for the shape of each
    peak_info_list entry.
    """
    minx = 5e-21
    maxx = max(Ep, Eq) + 5

    # This early-out (Ermx's own value is exactly zero or NaN) has not
    # been confirmed to be reachable given how Ermx is located by the
    # callers above -- kept rather than removed without first proving it
    # dead, but flagged here since it predates this module's public-API
    # promotion.
    if g(Ermx, Ep, Eq) == 0 or np.isnan(g(Ermx, Ep, Eq)):
        return (0, 0), (None, None), []

    # Scan the integrand on a uniform grid.  If it is still non-zero
    # near the end of the scan range, extend the range and re-scan the
    # whole grid, so that all group indices below refer to one
    # consistent array.
    scan_min = minx
    while True:
        er_arr = np.arange(scan_min, maxx, er_step / 2)
        er_integrand_arr = np.zeros(np.size(er_arr))
        for idx, er in enumerate(er_arr):
            er_integrand_arr[idx] = g(er, Ep, Eq)
        nonZero_idx = np.where(er_integrand_arr > 0)[0]
        if len(nonZero_idx) == 0 or nonZero_idx[-1] + 5 < len(er_arr) - 1:
            break
        maxx = 1.5 * maxx

    if len(nonZero_idx) == 0:
        # the peak is narrower than the scan step and fell between grid
        # points; integrate around the known maximum instead
        val, err = integrate.quad(g, er_arr[0], er_arr[-1], args=(Ep, Eq),
                                  points=[Ermx], epsabs=0, epsrel=1e-12,
                                  limit=200)
        peak_info = {"minx": er_arr[0], "maxx": er_arr[-1],
                     "integral": val, "error": err,
                     "width": np.nan, "Ermx": Ermx, "integrand_max": np.nan}
        return (val, err), (er_arr, er_integrand_arr), [peak_info]

    # We expect a structure of peaks surrounded by zeros: group the
    # consecutive non-zero samples and pad each group's range by 5 grid
    # points on both sides.
    nonZero_idx_groups = consecutive_groups(nonZero_idx)

    peak_info_array = []
    for peak_idx in nonZero_idx_groups:
        min_idx = max(0, min(peak_idx) - 5)
        max_idx = min(np.size(er_integrand_arr) - 1, max(peak_idx) + 5)
        peak_info_array.append({"minx": er_arr[min_idx],
                                "maxx": er_arr[max_idx]})

    # Merge overlapping ranges (union), so that quad below neither
    # double-counts the overlap nor drops a peak.  The groups are in
    # ascending Er order by construction.
    merged = [peak_info_array[0]]
    for peak_info in peak_info_array[1:]:
        if peak_info["minx"] <= merged[-1]["maxx"]:
            merged[-1]["maxx"] = max(merged[-1]["maxx"], peak_info["maxx"])
        else:
            merged.append(peak_info)
    peak_info_array = merged

    # Integrate using quad now that you have accurate limits
    for jdx, peak_info in enumerate(peak_info_array):
        minx = peak_info["minx"]
        maxx = peak_info["maxx"]
        # integrate and record the value
        # epsabs=0 forces quad to meet the *relative* tolerance: these
        # integrands can be ~1e-20, far below the default epsabs=1.49e-8,
        # so with the default quad accepts its first coarse estimate over
        # the wide window without ever subdividing around the narrow peak.
        # epsrel=1e-12 (vs the 1.49e-8 default) so this reference is
        # guaranteed tighter than the ~1e-12 accuracy of the windowed
        # Fortran integration it validates, rather than only typically so
        ans = integrate.quad(g, minx, maxx, args = (Ep,Eq,), epsabs=0, epsrel=1e-12, limit=200)
        peak_info_array[jdx]["integral"] = ans[0]
        peak_info_array[jdx]["error"] = ans[1]

        # this array is for the fitting
        er_arr = np.linspace(minx, maxx, 2000)
        er_integrand_arr = np.zeros(np.size(er_arr))
        for idx, er in enumerate(er_arr):
            er_integrand_arr[idx] = g(er, Ep, Eq)

        # find the width
        # Find maximum function value and corresponding energy
        er_arr = er_arr.flatten()

        # Filter data to only include points where the value is not NaN
        mask_nan = np.isnan(er_integrand_arr)
        mask_zeros = er_integrand_arr == 0
        filtered_energies = er_arr[~mask_nan & ~mask_zeros]
        filtered_values = er_integrand_arr[~mask_nan & ~mask_zeros]

        max_index = np.argmax(filtered_values)
        max_value = filtered_values[max_index]
        max_energy = filtered_energies[max_index]
        width_estimate = np.nan

        try:
            # estimate the width of the function
            curvature = np.diff(np.diff(filtered_values))

            if np.all(curvature == 0):
                warnings.warn(
                    f"integrate_inspect: peak {jdx} at Ep={Ep}, Eq={Eq} "
                    "is a flat line over its scan window -- width/Ermx left as "
                    "the plain max, no Gaussian fit attempted.",
                    stacklevel=2,
                )
                peak_info_array[jdx]["width"] = width_estimate
                peak_info_array[jdx]["Ermx"] = max_energy
                peak_info_array[jdx]['integrand_max'] = max_value

            else:
                # Walk to the right of max_index
                for i in range(max_index + 1, len(curvature) - 1):
                    if curvature[i] > 0 and curvature[i-1] <= 0:
                        inflection_index = i + 2 # Return the index where it switches from negative to positive
                        break

                width_estimate = filtered_energies[inflection_index] - filtered_energies[max_index]

                # Fit Gaussian to filtered data
                # note that we fit the values filtered_values * 1/max_value
                # this is so that when the values are small the fit still works!
                popt, _ = curve_fit(gaussian_profile, filtered_energies, filtered_values * 1/max_value, p0=[1, max_energy, width_estimate])

                # Er_max is the value of Er for which g is maximum
                peak_info_array[jdx]['integrand_max'] = popt[0] * max_value
                peak_info_array[jdx]["width"] = popt[2]
                peak_info_array[jdx]["Ermx"] = popt[1]
        except Exception as e:
            # Not setting an Ermx here means a caller relying on it will
            # fail noisily rather than silently continuing with a wrong
            # peak location.
            warnings.warn(
                f"integrate_inspect: could not fit peak {jdx}'s width "
                f"at Ep={Ep}, Eq={Eq} (max_value={max_value}, "
                f"max_energy={max_energy}, width_estimate={width_estimate}): {e}",
                stacklevel=2,
            )
            peak_info_array[jdx]["width"] = width_estimate

    # sum the integrals from all the peaks to get the final total integral and error
    integral = 0
    error = 0
    for peak_info in peak_info_array:
        integral += peak_info["integral"]
        error += peak_info["error"]

    # find the overall minimum and maximum and evaluate er_integrand_arr over that range
    min_minx = min(d["minx"] for d in peak_info_array)
    max_maxx = max(d["maxx"] for d in peak_info_array)
    er_arr = np.arange(min_minx, max_maxx, er_step / 2)
    er_integrand_arr = np.zeros(np.size(er_arr))
    for idx, er in enumerate(er_arr):
        val = g(er, Ep, Eq)
        er_integrand_arr[idx] = val

    # now we're ready to return everything to the user
    return (integral, error), (er_arr, er_integrand_arr), peak_info_array


def Ppq_inspect_vec_ER(Ep, Eq,
              *,
              F0,
              eps,
              V,
              p0, p10,
              q0, q10,
              er_step):
    """Vectorized Ppq_inspect_ER: Ppq_ER at each paired (Ep[i], Eq[i]),
    returned as a plain list of values (no scan grid/diagnostics -- use
    Ppq_inspect_ER directly if you need those)."""
    valuesArr = []
    for this_Ep, this_Eq in zip(Ep, Eq):
        thisVal, _, _ = Ppq_inspect_ER(this_Ep, this_Eq, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10, er_step=er_step)
        valuesArr.append(thisVal[0])
    return valuesArr


def Ppq_inspect_ER(Ep, Eq,
              *,
              F0,
              eps,
              V,
              p0, p10,
              q0, q10,
              er_step):
    """The full ER PDF value Ppq_ER(Ep, Eq) -- the ER counterpart of
    Ppq_inspect_NR (see its docstring for the method, return shape, and
    the "never use this in a fit/MCMC loop" caveat). `er_step` (keV)
    should be scanned finer than the narrowest expected peak: ER peaks
    can be as narrow as the zero-energy ionization resolution q0 (~0.06
    keV for typical CDMS parameters), so e.g. er_step=0.01 (a 0.005 keV
    scan step) is finer than the 0.01 keV grid the Fortran Ppq_ER
    integrates on. Usage to just get the value:
    (value, _), _, _ = Ppq_inspect_ER(...).
    """
    if F0 == 0:
        raise ValueError("Fano factor F(Er) = F0 is zero for all Er")

    g = lambda er, ep, eq: pq_given_Er_ER(er, ep, eq, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)

    # k, Z are unused placeholders here: is_gamma=True forces Y=1 in N_exponent
    f = lambda er,ep,eq: -1*N_exponent(er,ep,eq,k=0.0,Z=1.0,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10,is_gamma=True)

    # WARNING: this will only work if your peaks have a width greater
    # than 0.1 keV, which seems to be true for all reasonable CDMS
    # parameters (see Ppq_inspect_NR's docstring).
    maxx = max(Ep, Eq) + 5
    er_arr = np.arange(5e-21, maxx, er_step)
    f_arr = np.zeros(len(er_arr))
    for idx, er in enumerate(er_arr):
        f_arr[idx] = f(er, Ep, Eq)

    if not np.any(f_arr > 0):
        if np.all(f_arr == 0):
            return (0, 0), (er_arr, f_arr), []
        elif np.all(np.isnan(f_arr)):
            return (np.nan, np.nan), (er_arr, f_arr), []
    else:
        min_idx = np.argmin(f_arr[~np.isnan(f_arr)])
        Ermx = er_arr[min_idx]

    return integrate_inspect(g, Ep, Eq, Ermx, er_step)
