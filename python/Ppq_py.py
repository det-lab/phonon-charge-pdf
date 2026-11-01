"""phonon_charge_pdf.Ppq_py: pure-Python reference implementation of the
NR/ER (Ep, Eq) probability model.

Promoted from internal/ to a public, stable module alongside Ppq_compiled
(Ppq_fort.py) and event_generator.py -- it has no compiled-library
dependency at all (pure numpy/scipy), so it works anywhere this package
is installed, compiler or not.

This is deliberately the *slow*, brute-force path: Ppq_inspect_NR and
Ppq_inspect_ER scan a uniform Erecoil grid (step set by `erecoil_step`) rather than
using the Fortran side's analytic peak location and windowed quadrature,
specifically so they can serve as an independent cross-check of
Ppq_compiled/_Ppq_fort_bindings (see test/python/test_chisquare_ppqn.py and
test_PpqFort.py) -- never use these in a fit/MCMC inner loop; use Ppq_compiled
for that.

Public functions:
* Y, Nbar_NR, Nbar_ER, sigp, sigq -- the ionization-yield, mean pair-count, and
  phonon/ionization resolution models.
* PErecoil_NR, PErecoil_ER -- the true-recoil-energy spectra (NR, ER).
* Ppq_inspect_NR, Ppq_inspect_ER -- the full Ppq_NR/Ppq_ER value at (Ep, Eq),
  found by scanning for peak(s) in Erecoil and integrating each with
  scipy.integrate.quad, and returning the scan grid and per-peak
  diagnostics (width, location) alongside the value -- the point of
  reaching for this over the Fortran side.

Not part of the public API (not in __all__) -- no end user needs these directly:
* aN, bN, cN -- the Gaussian-in-N exponent's quadratic coefficients, used only
  to build N_exponent below.
* N_exponent, pq_given_Erecoil_NR, pq_given_Erecoil_ER -- internal pieces Ppq_inspect_NR/ER
  are built from; still importable from this module by name (and from
  ctypes-level tests cross-checking the Fortran side) if you need them.
  pq_given_Erecoil_NR/ER are themselves thin wrappers (an Nbar_val, then
  a PErecoil_NR/ER multiply) around pq_N_integral, their one shared piece
  of math.
* integrate_inspect -- the peak-finding/integration engine Ppq_inspect_NR/
  Ppq_inspect_ER are built on. Its "g" argument is a generic placeholder for
  "the function being integrated" (called with pq_given_Erecoil_NR for NR,
  pq_given_Erecoil_ER for ER) -- not a reference to gamma/ER. Exercised directly
  against a custom multimodal integrand only by
  test_integrate_inspect_multimodal.py, to prove it finds every peak rather
  than just the one Ppq_inspect_NR/ER happen to feed it -- not something an
  end user needs to call themselves.
* consecutive_groups, gaussian_profile -- small helpers (grouping
  consecutive indices; evaluating a Gaussian) used by integrate_inspect's
  peak-width fit.
"""

import os
import sys
import warnings

import numpy as np
import scipy.integrate as integrate
from scipy.optimize import curve_fit

# cli/'s normgrid.py (table machinery) is imported lazily, only from inside
# integrate_inspect's erecoil_bounds error path below -- same reasoning and
# the same sys.path setup Ppq_fort.py uses for its own lazy `import normgrid`:
# this keeps this module's "no compiled-library dependency at all" promise,
# since that import only has to happen for a tabulated PErecoil's error case.
_CLI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cli")

__all__ = [
    "Y", "Nbar_NR", "Nbar_ER", "sigp", "sigq", "PErecoil_NR", "PErecoil_ER",
    "Ppq_inspect_NR", "Ppq_inspect_ER",
]

# No module-level __version__ here anymore: the package version lives in
# fpm.toml (git tags, CITATION.cff) -- see the versioning bootstrap in the
# repo's git-flow setup. A second, hand-maintained copy here is exactly the
# per-file version drift this rename is meant to retire.


def Y(Erecoil,
      *,
      k, Z):
    """The ionization yield, Lindhard model: Y(Erecoil) = k*g(eps_L) / (1 + k*g(eps_L)),
    with eps_L = 11.5*Erecoil*Z**(-7/3) and g(eps_L) = 3*eps_L**0.15 + 0.7*eps_L**0.29 + eps_L.
    Erecoil in keV; Z is the target's atomic number."""
    eps_L = 11.5 * np.absolute(Erecoil) * Z**(-7.0/3.0)
    g = 3.0*eps_L**0.15 + 0.7*eps_L**0.29 + eps_L
    return k*g / (1.0 + k*g)


def Nbar_NR(Erecoil,
            *,
            k, Z,
            eps):
    """Mean number of e/h pairs produced by a nuclear recoil of energy
    Erecoil (keV): the Lindhard yield Y(Erecoil, k=k, Z=Z) times Erecoil/eps."""
    return Y(Erecoil, k=k, Z=Z)*Erecoil/eps


def Nbar_ER(Erecoil,
            *,
            eps):
    """Same as Nbar_NR but for the electron-recoil band (Y=1 everywhere, so
    there's no k/Z to pass): Erecoil/eps."""
    return Erecoil/eps


def sigp(Ep,
         *,
         eps,
         V,
         p0, p10):
    """Phonon-energy resolution at measured phonon energy Ep (keV)."""
    e2pre = (p10**2 - p0**2)
    # The "1+" here matters: Ppq v1 omitted it (e2e = (Ep/(10*(V/eps/1000)))**2),
    # which changes the result -- don't drop it when touching this line.
    e2e = (Ep/(10*(1+(V/eps/1000))))**2
    return np.sqrt(p0**2+e2pre*e2e)


def sigq(Eq,
         *,
         q0, q10):
    """Ionization-energy resolution at measured charge energy Eq (keV)."""
    e2pre = (q10**2 - q0**2)
    e2e = (Eq/10)**2
    return np.sqrt(q0**2+e2pre*e2e)


# Fit parameters -- matches the Fortran PErecoil_NR_impl constants exactly
# (not exposed as arguments there either: PErecoil_NR(Erecoil) is the public signature)
_PNa, _PNb, _PNd = 0.53693208, 6.41515782, 23.71789286


def PErecoil_NR(Erecoil):
    """Probability density of the *true* recoil energy Erecoil (keV) for
    neutron recoils: a two-component exponential mixture, zero for Erecoil < 0."""
    Erecoil = np.asarray(Erecoil)  # Convert input to a NumPy array
    result = np.where(Erecoil < 0, 0, _PNa * (1 / _PNb) * np.exp(-Erecoil / _PNb) + (1 - _PNa) * (1 / _PNd) * np.exp(-Erecoil / _PNd))
    return result


# Cfgammas -- matches the Fortran PErecoil_ER_impl constants exactly
_PGa, _PGb, _PGd = 5.73211975e-01, 1.69520023e-01, 2.79552394e+02


def PErecoil_ER(Erecoil):
    """Probability density of the *true* recoil energy Erecoil (keV) for
    electron recoils (Cf gammas): a two-component exponential mixture,
    zero for Erecoil < 0."""
    Erecoil = np.asarray(Erecoil)  # Convert Erecoil to a NumPy array
    result = np.where(Erecoil < 0, 0, _PGa * (1 / _PGb) * np.exp(-Erecoil / _PGb) + (1 - _PGa) * (1 / _PGd) * np.exp(-Erecoil / _PGd))
    return result


def aN(Erecoil, Ep, Eq,
       *,
       F0,
       eps,
       V,
       p0, p10,
       q0, q10):
    """Linear coefficient of N_exponent's Gaussian-in-N exponent."""
    t1 = (V/1e3)*(Ep-Erecoil)/sigp(Ep,eps=eps,V=V,p0=p0,p10=p10)**2
    t2 = eps*Eq/sigq(Eq,q0=q0,q10=q10)**2
    t3 = 1/F0
    return t1+t2+t3


def bN(Erecoil, Ep, Eq,
       *,
       k, Z,
       F0,
       eps,
       V,
       p0, p10,
       q0, q10,
       is_ER):
    """Quadratic coefficient of N_exponent's Gaussian-in-N exponent. is_ER
    selects the electron-recoil yield (Y=1) instead of the Lindhard NR
    yield Y(Erecoil, k=k, Z=Z); k, Z are unused (pass any value) when
    is_ER=True."""
    Nbar_val = Nbar_ER(Erecoil,eps=eps) if is_ER else Nbar_NR(Erecoil,k=k,Z=Z,eps=eps)
    t1 = 1/(2*Nbar_val*F0)
    t2 = eps**2/(2*sigq(Eq,q0=q0,q10=q10)**2)
    t3 = V**2/(2*(sigp(Ep,eps=eps,V=V,p0=p0,p10=p10)*1e3)**2)
    return t1+t2+t3


def cN(Erecoil, Ep, Eq,
       *,
       k, Z,
       F0,
       eps,
       V,
       p0, p10,
       q0, q10,
       is_ER):
    """Constant term of N_exponent's Gaussian-in-N exponent. is_ER selects
    the electron-recoil yield (Y=1) instead of the Lindhard NR yield
    Y(Erecoil, k=k, Z=Z); k, Z are unused (pass any value) when is_ER=True."""
    Nbar_val = Nbar_ER(Erecoil,eps=eps) if is_ER else Nbar_NR(Erecoil,k=k,Z=Z,eps=eps)
    t1 = -(Ep-Erecoil)**2/(2*sigp(Ep,eps=eps,V=V,p0=p0,p10=p10)**2)
    t2 = -Eq**2/(2*sigq(Eq,q0=q0,q10=q10)**2)
    t3 = -Nbar_val/(2*F0)
    return t1+t2+t3


def N_exponent(Erecoil, Ep, Eq,
           *,
           k, Z,
           F0,
           eps,
           V,
           p0, p10,
           q0, q10,
           is_ER):
    """The exponent of the Gaussian-in-N integral that pq_given_Erecoil_NR/pq_given_Erecoil_ER
    evaluate -- completing the square of cN + aN*N + bN*N**2 in N gives
    cN + aN**2/(4*bN), which is what this returns. is_ER selects the
    electron-recoil yield (Y=1) instead of the Lindhard NR yield
    Y(Erecoil, k=k, Z=Z); k, Z are unused (pass any value) when is_ER=True."""
    cN_val = cN(Erecoil,Ep,Eq,k=k,Z=Z,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10,is_ER=is_ER)
    aN_val = aN(Erecoil,Ep,Eq,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10)
    bN_val = bN(Erecoil,Ep,Eq,k=k,Z=Z,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10,is_ER=is_ER)
    exponent = cN_val + (aN_val**2 / (4*bN_val))
    return exponent


def pq_N_integral(Erecoil, Ep, Eq, Nbar_val, F_val, *, eps, V, p0, p10, q0, q10):
    """The N-integral shared by pq_given_Erecoil_NR/pq_given_Erecoil_ER: a
    fixed 21-point Simpson rule over a +/-5-sigma window around the
    Gaussian-in-N peak, given that peak's own Nbar_val (NR vs ER differ
    only in how Nbar_val is computed -- see Nbar_NR/Nbar_ER) and the Fano
    factor F_val. Not part of the public API (not in __all__); an
    implementation detail both pq_given_Erecoil_NR and pq_given_Erecoil_ER
    are built on."""
    if Nbar_val <= 0.0 or F_val <= 0.0:
        return 0.0
    sigma_N    = np.sqrt(Nbar_val * F_val)
    norm_const = 1.0 / (2.0 * np.pi) ** 1.5
    sp0 = p0 ** 2
    sp1 = (p10 ** 2 - p0 ** 2) / (10.0 * (1.0 + V / (eps * 1.0e3))) ** 2
    sq0 = q0 ** 2
    sq1 = (q10 ** 2 - q0 ** 2) / 100.0
    EP_mean      = Erecoil + (V * 1.0e-3) * Nbar_val
    EQ_mean      = eps * Nbar_val
    sigp_mean_sq = sp0 + sp1 * EP_mean ** 2
    sigq_mean_sq = sq0 + sq1 * EQ_mean ** 2
    a_coeff = ((V * 1.0e-3) * (Ep - Erecoil) / sigp_mean_sq
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
    EP_nl = Erecoil + (V * 1.0e-3) * N_pts
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
    return (h_N / 3.0) * np.dot(simps_w, integrand)


def pq_given_Erecoil_NR(Erecoil, Ep, Eq,
             *,
             k, Z,
             F0,
             eps,
             V,
             p0, p10,
             q0, q10,
             p_erecoil=None):
    """Unnormalized NR (Ep, Eq) density at one true recoil energy Erecoil (keV):
    p_erecoil(Erecoil) (PErecoil_NR by default) times the N-integral (a fixed
    21-point Simpson rule over a +/-5-sigma window around the Gaussian-in-N
    peak).

    p_erecoil overrides the closed-form PErecoil_NR -- pass an
    erecoil_table.PErecoilTable (or anything else matching its Erecoil ->
    density signature) to use a tabulated true-recoil-energy spectrum
    instead. This function does not itself keep Erecoil inside the table's
    range: call it through Ppq_inspect_NR (its own p_erecoil argument), which
    clips the scan/integration windows to erecoil_range first."""
    p_erecoil = PErecoil_NR if p_erecoil is None else p_erecoil
    Nbar_val = Nbar_NR(Erecoil, k=k, Z=Z, eps=eps)
    return float(pq_N_integral(Erecoil, Ep, Eq, Nbar_val, F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
                 * p_erecoil(Erecoil))


def Ppq_inspect_NR(Ep, Eq,
              *,
              k, Z,
              F0,
              eps,
              V,
              p0, p10,
              q0, q10,
              erecoil_step,
              p_erecoil=None):
    """The full NR PDF value Ppq_NR(Ep, Eq), found by scanning a uniform Erecoil
    grid (step `erecoil_step`, in keV) for the Gaussian-in-N exponent's peak
    and integrating pq_given_Erecoil_NR around it with scipy.integrate.quad -- the
    slow, brute-force reference this package's event_generator.py and
    the compiled Fortran Ppq_NR are cross-checked against (see
    test/python/test_chisquare_ppqn.py, test_PpqFort.py). Never use this
    in a fit/MCMC inner loop; use Ppq_compiled for that.

    `erecoil_step` sets both this function's own scan (above, a rough
    starting guess for where the peak is) and the finer scan integrate_inspect
    runs next to locate the integration windows handed to quad (see its
    docstring) -- pick it well under your narrowest expected peak width, the
    same reasoning behind quad's own `points=` argument (see
    test/python/quad_breakpoints_demo.py): a peak narrower than erecoil_step
    can fall entirely between grid points and be missed.

    p_erecoil overrides the closed-form PErecoil_NR (see pq_given_Erecoil_NR).
    When it has an erecoil_range (an erecoil_table.PErecoilTable does), this
    function's own scan and integrate_inspect's window are both clipped to
    it, and integrate_inspect raises normgrid.OutOfBoxError rather than
    silently truncating if the integrand still needs more room than the
    table's edge allows.

    Returns a dict:
      value, value_err : the PDF value and its quadrature error.
      erecoil_scan_grid, erecoil_scan_integrand : the Erecoil scan arrays.
      peaks : one dict per located peak, with "minx"/"maxx" (the
        integration window), "integral"/"integral_err" (that peak's own
        quad result), and "width"/"Erecoil_peak"/"integrand_max" from a Gaussian
        fit (nan if the fit didn't converge).
    Usage to just get the value: Ppq_inspect_NR(...)["value"].

    WARNING: the peak search only works if your peaks have a width
    greater than ~0.1 keV, which holds for all reasonable CDMS
    parameters; its advantage over an exact minimizer is that it never
    fails to find *a* peak (an exact minimizer sometimes does).
    """
    if F0 == 0:
        raise ValueError("Fano factor F(Erecoil) = F0 is zero for all Erecoil")

    p_erecoil = PErecoil_NR if p_erecoil is None else p_erecoil
    erecoil_bounds = getattr(p_erecoil, "erecoil_range", None)

    # find Erecoil_max
    f = lambda erecoil,ep,eq: -1*N_exponent(erecoil,ep,eq,k=k,Z=Z,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10,is_ER=False)

    # Deliberately NOT clipped to erecoil_bounds: this scan never calls
    # p_erecoil (N_exponent alone), and the bounds check right after locating
    # Erecoil_peak below needs that peak's true, unclamped location to tell
    # whether the table actually reaches it.
    maxx = max(Ep, Eq) + 5
    erecoil_arr = np.arange(5e-21, maxx, erecoil_step)
    f_arr = np.zeros(len(erecoil_arr), dtype=np.float64)
    for idx, erecoil in enumerate(erecoil_arr):
        try:
            f_arr[idx] = -1*N_exponent(erecoil,Ep,Eq,k=k,Z=Z,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10,is_ER=False)
        except Exception:
            # N_exponent can over/underflow at extreme Erecoil; leaving this
            # sample at its default 0 and continuing the scan is correct
            # here, since a real peak is never isolated to a single
            # sample on a grid this fine.
            pass
    # valid: scan points that didn't over/underflow above. If none did,
    # there's nothing to locate a peak from. If every valid point is
    # exactly 0 (the untouched np.zeros default), every one of them hit
    # the except branch above, so there's no real peak in this window
    # either -- a value of 0, not an attempt at a Gaussian fit.
    valid = ~np.isnan(f_arr)
    if not valid.any():
        return {"value": np.nan, "value_err": np.nan, "erecoil_scan_grid": erecoil_arr,
                "erecoil_scan_integrand": f_arr, "peaks": []}
    if np.all(f_arr[valid] == 0):
        return {"value": 0, "value_err": 0, "erecoil_scan_grid": erecoil_arr,
                "erecoil_scan_integrand": f_arr, "peaks": []}
    # argmin over the valid subset, then map back through its own indices
    # to erecoil_arr's -- argmin's return index is only valid within the
    # filtered subset, not the original array, whenever any point above
    # overflowed to NaN.
    valid_idx = np.flatnonzero(valid)
    Erecoil_peak = erecoil_arr[valid_idx[np.argmin(f_arr[valid_idx])]]
    if erecoil_bounds is not None:
        # This scan never calls p_erecoil (it locates the resolution-driven
        # peak of N_exponent alone, independent of the recoil spectrum), so
        # it cannot raise on its own -- but if *that* peak sits outside the
        # table's range, this (Ep, Eq) point's dominant contribution is
        # somewhere the table has no data for. Clamping it to the edge and
        # carrying on would hand integrate_inspect a seed whose own g() value
        # there can underflow to exactly 0, which looks exactly like "no
        # density here" instead of "the table doesn't reach where it is" --
        # so raise here, before that happens, rather than downstream.
        lo, hi = erecoil_bounds
        tol = 1e-9 * (hi - lo)
        if not (lo - tol <= Erecoil_peak <= hi + tol):
            sys.path.insert(0, _CLI_DIR)
            from normgrid import OutOfBoxError
            raise OutOfBoxError(
                f"Ppq_inspect_NR(Ep={Ep!r}, Eq={Eq!r}): the dominant Erecoil={Erecoil_peak!r} "
                f"peak sits outside the PErecoil table's range [{lo:.6g}, {hi:.6g}] -- this point's "
                "density depends on a recoil energy the table has no data for.",
                axis="Erecoil", value=Erecoil_peak, box=(lo, hi), query={"Ep": Ep, "Eq": Eq})
        Erecoil_peak = min(max(Erecoil_peak, lo), hi)

    g = lambda erecoil, ep, eq: pq_given_Erecoil_NR(erecoil, ep, eq, k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10, p_erecoil=p_erecoil)

    return integrate_inspect(g, Ep, Eq, Erecoil_peak, erecoil_step, erecoil_bounds=erecoil_bounds)


def pq_given_Erecoil_ER(Erecoil, Ep, Eq,
             *,
             F0,
             eps,
             V,
             p0, p10,
             q0, q10,
             p_erecoil=None):
    """Unnormalized ER (Ep, Eq) density at one true recoil energy Erecoil (keV):
    p_erecoil(Erecoil) (PErecoil_ER by default) times the N-integral (a fixed
    21-point Simpson rule over a +/-5-sigma window around the Gaussian-in-N
    peak), with the electron-recoil yield Y=1 (see pq_given_Erecoil_NR for
    the analogous NR calculation, including what p_erecoil overrides)."""
    p_erecoil = PErecoil_ER if p_erecoil is None else p_erecoil
    Nbar_val = Nbar_ER(Erecoil, eps=eps)
    return float(pq_N_integral(Erecoil, Ep, Eq, Nbar_val, F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
                 * p_erecoil(Erecoil))


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


def integrate_inspect(g, Ep, Eq, Erecoil_peak, erecoil_step, erecoil_bounds=None):
    """Integrate g(Erecoil, Ep, Eq) over Erecoil by locating its peak(s) on a
    uniform scan grid (step `erecoil_step`, in keV, starting from an
    approximate peak location Erecoil_peak) and integrating each located peak
    with scipy.integrate.quad. This is the engine Ppq_inspect_NR and
    Ppq_inspect_ER are built on. Not part of the public API (not in
    __all__); test/python/test_integrate_inspect_multimodal.py exercises it
    directly against a synthetic multi-peak integrand to prove the
    peak-finding itself is correct, independent of either band's physics.
    The "g" here is a generic placeholder for
    "the function being integrated" (it's called with pq_given_Erecoil_NR for NR,
    pq_given_Erecoil_ER for ER) -- not a reference to gamma/ER, which is why this
    function's own name doesn't carry an N/G suffix the way Ppq_inspect_NR/
    Ppq_inspect_ER do.

    erecoil_bounds, if given, is g's PErecoil table's own erecoil_range: the
    scan/integration window is clipped to it and never grows past its upper
    edge, since a tabulated PErecoil raises OutOfBoxError rather than
    extrapolate -- g must never be called outside this range. None (the
    default) keeps the unrestricted closed-form-PErecoil behavior.

    Returns a dict: value, value_err (the integral and its quadrature
    error), erecoil_scan_grid, erecoil_scan_integrand, and peaks -- see
    Ppq_inspect_NR's docstring for the shape of each peaks entry.
    """
    erecoil_hi = None if erecoil_bounds is None else erecoil_bounds[1]
    minx = 5e-21 if erecoil_bounds is None else max(5e-21, erecoil_bounds[0])
    maxx = max(Ep, Eq) + 5
    if erecoil_hi is not None:
        maxx = min(maxx, erecoil_hi)

    # This early-out (Erecoil_peak's own value is exactly zero or NaN) has not
    # been confirmed to be reachable given how Erecoil_peak is located by the
    # callers above -- kept rather than removed without first proving it
    # dead, but flagged here since it predates this module's public-API
    # promotion.
    if g(Erecoil_peak, Ep, Eq) == 0 or np.isnan(g(Erecoil_peak, Ep, Eq)):
        return {"value": 0, "value_err": 0, "erecoil_scan_grid": None,
                "erecoil_scan_integrand": None, "peaks": []}

    # Scan the integrand on a uniform grid.  If it is still non-zero
    # near the end of the scan range, extend the range and re-scan the
    # whole grid, so that all group indices below refer to one
    # consistent array.
    scan_min = minx
    while True:
        erecoil_arr = np.arange(scan_min, maxx, erecoil_step)
        erecoil_integrand_arr = np.zeros(np.size(erecoil_arr))
        for idx, erecoil in enumerate(erecoil_arr):
            erecoil_integrand_arr[idx] = g(erecoil, Ep, Eq)
        nonZero_idx = np.where(erecoil_integrand_arr > 0)[0]
        if len(nonZero_idx) == 0 or nonZero_idx[-1] + 5 < len(erecoil_arr) - 1:
            break
        if erecoil_hi is not None and maxx >= erecoil_hi:
            # The integrand is still significantly nonzero right at the
            # table's own edge: real probability mass sits past erecoil_hi,
            # which a tabulated PErecoil cannot supply (it raises rather than
            # extrapolate). Silently accepting this window would silently
            # under-integrate this (Ep, Eq) point -- raise instead, the same
            # "table doesn't cover what's needed" failure as a direct
            # out-of-range PErecoil query, just discovered here instead of
            # inside a single call to g.
            sys.path.insert(0, _CLI_DIR)
            from normgrid import OutOfBoxError
            raise OutOfBoxError(
                f"Erecoil integration window for Ep={Ep!r}, Eq={Eq!r} still needs to extend past "
                f"{erecoil_hi:.6g} keV (the PErecoil table's own upper edge) to capture the full "
                "integrand -- refusing to silently truncate it there. Either this (Ep, Eq) point "
                "is outside what the table's recoil-energy range can support, or the table needs "
                "to extend further.",
                axis="Erecoil", value=maxx, box=(minx, erecoil_hi), query={"Ep": Ep, "Eq": Eq})
        maxx = 1.5 * maxx if erecoil_hi is None else min(1.5 * maxx, erecoil_hi)

    if len(nonZero_idx) == 0:
        # the peak is narrower than the scan step and fell between grid
        # points; integrate around the known maximum instead
        val, err = integrate.quad(g, erecoil_arr[0], erecoil_arr[-1], args=(Ep, Eq),
                                  points=[Erecoil_peak], epsabs=0, epsrel=1e-12,
                                  limit=200)
        peak_info = {"minx": erecoil_arr[0], "maxx": erecoil_arr[-1],
                     "integral": val, "integral_err": err,
                     "width": np.nan, "Erecoil_peak": Erecoil_peak, "integrand_max": np.nan}
        return {"value": val, "value_err": err, "erecoil_scan_grid": erecoil_arr,
                "erecoil_scan_integrand": erecoil_integrand_arr, "peaks": [peak_info]}

    # We expect a structure of peaks surrounded by zeros: group the
    # consecutive non-zero samples and pad each group's range by 5 grid
    # points on both sides.
    nonZero_idx_groups = consecutive_groups(nonZero_idx)

    peak_info_array = []
    for peak_idx in nonZero_idx_groups:
        min_idx = max(0, min(peak_idx) - 5)
        max_idx = min(np.size(erecoil_integrand_arr) - 1, max(peak_idx) + 5)
        peak_info_array.append({"minx": erecoil_arr[min_idx],
                                "maxx": erecoil_arr[max_idx]})

    # Merge overlapping ranges (union), so that quad below neither
    # double-counts the overlap nor drops a peak.  The groups are in
    # ascending Erecoil order by construction.
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
        peak_info_array[jdx]["integral_err"] = ans[1]

        # this array is for the fitting
        erecoil_arr = np.linspace(minx, maxx, 2000)
        erecoil_integrand_arr = np.zeros(np.size(erecoil_arr))
        for idx, erecoil in enumerate(erecoil_arr):
            erecoil_integrand_arr[idx] = g(erecoil, Ep, Eq)

        # find the width
        # Find maximum function value and corresponding energy
        erecoil_arr = erecoil_arr.flatten()

        # Filter data to only include points where the value is not NaN
        mask_nan = np.isnan(erecoil_integrand_arr)
        mask_zeros = erecoil_integrand_arr == 0
        filtered_energies = erecoil_arr[~mask_nan & ~mask_zeros]
        filtered_values = erecoil_integrand_arr[~mask_nan & ~mask_zeros]

        # Prefixed grid_* since these are the raw scan-grid estimate of the
        # peak's location/height/index -- Erecoil_peak/integrand_max below are
        # the (usually Gaussian-fit-refined) values actually returned.
        grid_max_idx = np.argmax(filtered_values)
        grid_max_value = filtered_values[grid_max_idx]
        grid_max_energy = filtered_energies[grid_max_idx]
        width_estimate = np.nan

        try:
            # estimate the width of the function
            curvature = np.diff(np.diff(filtered_values))

            if np.all(curvature == 0):
                warnings.warn(
                    f"integrate_inspect: peak {jdx} at Ep={Ep}, Eq={Eq} "
                    "is a flat line over its scan window -- width/Erecoil_peak left as "
                    "the plain max, no Gaussian fit attempted.",
                    stacklevel=2,
                )
                peak_info_array[jdx]["width"] = width_estimate
                peak_info_array[jdx]["Erecoil_peak"] = grid_max_energy
                peak_info_array[jdx]['integrand_max'] = grid_max_value

            else:
                # Walk to the right of grid_max_idx
                for i in range(grid_max_idx + 1, len(curvature) - 1):
                    if curvature[i] > 0 and curvature[i-1] <= 0:
                        inflection_index = i + 2 # Return the index where it switches from negative to positive
                        break

                width_estimate = filtered_energies[inflection_index] - filtered_energies[grid_max_idx]

                # Fit Gaussian to filtered data
                # note that we fit the values filtered_values * 1/grid_max_value
                # this is so that when the values are small the fit still works!
                popt, _ = curve_fit(gaussian_profile, filtered_energies, filtered_values * 1/grid_max_value, p0=[1, grid_max_energy, width_estimate])

                # Erecoil_max is the value of Erecoil for which g is maximum
                peak_info_array[jdx]['integrand_max'] = popt[0] * grid_max_value
                peak_info_array[jdx]["width"] = popt[2]
                peak_info_array[jdx]["Erecoil_peak"] = popt[1]
        except Exception as e:
            # Not setting an Erecoil_peak here means a caller relying on it will
            # fail noisily rather than silently continuing with a wrong
            # peak location.
            warnings.warn(
                f"integrate_inspect: could not fit peak {jdx}'s width "
                f"at Ep={Ep}, Eq={Eq} (grid_max_value={grid_max_value}, "
                f"grid_max_energy={grid_max_energy}, width_estimate={width_estimate}): {e}",
                stacklevel=2,
            )
            peak_info_array[jdx]["width"] = width_estimate

    # sum the integrals from all the peaks to get the final total integral and error
    integral = 0
    error = 0
    for peak_info in peak_info_array:
        integral += peak_info["integral"]
        error += peak_info["integral_err"]

    # find the overall minimum and maximum and evaluate erecoil_integrand_arr over that range
    min_minx = min(d["minx"] for d in peak_info_array)
    max_maxx = max(d["maxx"] for d in peak_info_array)
    erecoil_arr = np.arange(min_minx, max_maxx, erecoil_step)
    erecoil_integrand_arr = np.zeros(np.size(erecoil_arr))
    for idx, erecoil in enumerate(erecoil_arr):
        val = g(erecoil, Ep, Eq)
        erecoil_integrand_arr[idx] = val

    # now we're ready to return everything to the user
    return {"value": integral, "value_err": error, "erecoil_scan_grid": erecoil_arr,
            "erecoil_scan_integrand": erecoil_integrand_arr, "peaks": peak_info_array}


def Ppq_inspect_ER(Ep, Eq,
              *,
              F0,
              eps,
              V,
              p0, p10,
              q0, q10,
              erecoil_step,
              p_erecoil=None):
    """The full ER PDF value Ppq_ER(Ep, Eq) -- the ER counterpart of
    Ppq_inspect_NR (see its docstring for the method, return shape, the
    "never use this in a fit/MCMC loop" caveat, and what p_erecoil
    overrides). `erecoil_step` (keV) should be scanned finer than the
    narrowest expected peak: ER peaks can be as narrow as the zero-energy
    ionization resolution q0 (~0.06 keV for typical CDMS parameters), so
    e.g. erecoil_step=0.005 is finer than the 0.01 keV grid the Fortran
    Ppq_ER integrates on. Usage to just get the value:
    Ppq_inspect_ER(...)["value"].
    """
    if F0 == 0:
        raise ValueError("Fano factor F(Erecoil) = F0 is zero for all Erecoil")

    p_erecoil = PErecoil_ER if p_erecoil is None else p_erecoil
    erecoil_bounds = getattr(p_erecoil, "erecoil_range", None)

    g = lambda erecoil, ep, eq: pq_given_Erecoil_ER(erecoil, ep, eq, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10, p_erecoil=p_erecoil)

    # k, Z are unused placeholders here: is_ER=True forces Y=1 in N_exponent
    f = lambda erecoil,ep,eq: -1*N_exponent(erecoil,ep,eq,k=0.0,Z=1.0,F0=F0,eps=eps,V=V,p0=p0,p10=p10,q0=q0,q10=q10,is_ER=True)

    # WARNING: this will only work if your peaks have a width greater
    # than 0.1 keV, which seems to be true for all reasonable CDMS
    # parameters (see Ppq_inspect_NR's docstring).
    #
    # Deliberately NOT clipped to erecoil_bounds -- see the matching comment
    # in Ppq_inspect_NR.
    maxx = max(Ep, Eq) + 5
    erecoil_arr = np.arange(5e-21, maxx, erecoil_step)
    f_arr = np.zeros(len(erecoil_arr))
    for idx, erecoil in enumerate(erecoil_arr):
        try:
            f_arr[idx] = f(erecoil, Ep, Eq)
        except Exception:
            # N_exponent can over/underflow at extreme Erecoil; leaving this
            # sample at its default 0 and continuing the scan is correct
            # here, since a real peak is never isolated to a single
            # sample on a grid this fine. (See the matching guard in
            # Ppq_inspect_NR.)
            pass

    # valid: scan points that didn't over/underflow above. If none did,
    # there's nothing to locate a peak from. If every valid point is
    # exactly 0 (the untouched np.zeros default), every one of them hit
    # the except branch above, so there's no real peak in this window
    # either -- a value of 0, not an attempt at a Gaussian fit. (See the
    # matching guard in Ppq_inspect_NR.)
    valid = ~np.isnan(f_arr)
    if not valid.any():
        return {"value": np.nan, "value_err": np.nan, "erecoil_scan_grid": erecoil_arr,
                "erecoil_scan_integrand": f_arr, "peaks": []}
    if np.all(f_arr[valid] == 0):
        return {"value": 0, "value_err": 0, "erecoil_scan_grid": erecoil_arr,
                "erecoil_scan_integrand": f_arr, "peaks": []}
    # argmin over the valid subset, then map back through its own indices
    # to erecoil_arr's -- argmin's return index is only valid within the
    # filtered subset, not the original array, whenever any point above
    # overflowed to NaN.
    valid_idx = np.flatnonzero(valid)
    Erecoil_peak = erecoil_arr[valid_idx[np.argmin(f_arr[valid_idx])]]
    if erecoil_bounds is not None:
        # See the matching guard (and its full rationale) in Ppq_inspect_NR.
        lo, hi = erecoil_bounds
        tol = 1e-9 * (hi - lo)
        if not (lo - tol <= Erecoil_peak <= hi + tol):
            sys.path.insert(0, _CLI_DIR)
            from normgrid import OutOfBoxError
            raise OutOfBoxError(
                f"Ppq_inspect_ER(Ep={Ep!r}, Eq={Eq!r}): the dominant Erecoil={Erecoil_peak!r} "
                f"peak sits outside the PErecoil table's range [{lo:.6g}, {hi:.6g}] -- this point's "
                "density depends on a recoil energy the table has no data for.",
                axis="Erecoil", value=Erecoil_peak, box=(lo, hi), query={"Ep": Ep, "Eq": Eq})
        Erecoil_peak = min(max(Erecoil_peak, lo), hi)

    return integrate_inspect(g, Ep, Eq, Erecoil_peak, erecoil_step, erecoil_bounds=erecoil_bounds)
