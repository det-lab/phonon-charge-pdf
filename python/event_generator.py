"""
event_generator is the public entrypoint for generating simulated (Ep, Eq)
detector events -- an independent physics simulator, not a second way to
evaluate the PDF (see "Physical picture" below for why that independence
matters). If you want normalized PDF values or a likelihood instead, use
Ppq_compiled (python/Ppq_fort.py).

Unlike Ppq_compiled, this module has no Fortran dependency at all -- it is pure
Python/numpy built on python/Ppq_py.py's Y/Nbar_NR/Nbar_ER/sigp/sigq (itself a
public entrypoint; see its module docstring), so it works in any
environment that can `pip install` this package, compiled library or not.

Quickstart
----------
    from event_generator import generate_events
    events = generate_events(
        target_size=20_000, fraction_ER=0.6, seed=8,
        ep_range=(2.5, 350.0), eq_range=(0.75, 200.0),
        k=0.18, Z=32.0, F0_NR=0.122, F0_ER=0.2, eps=3.0e-3, V=3.0,
        p0=0.06421907, p10=0.48998486, q0=0.23718488, q10=0.27093151,
        mix_frac_NR=0.53693208, scale_fast_NR=6.41515782, scale_slow_NR=23.71789286,
        mix_frac_ER=0.573211975, scale_fast_ER=0.169520023, scale_slow_ER=279.552394,
    )
    Ep, Eq = events["EP"].to_numpy(), events["EQ"].to_numpy()

All physics parameters are required keyword arguments -- there are no
defaults, matching the rest of this library (Ppq_compiled, _Ppq_fort_bindings.py,
Ppq_py.py). fraction_ER=1.0 generates ER-only, fraction_ER=0.0 generates NR-only,
and anything in between generates a mix.

Physical picture
-----------------
Each event starts from a true recoil energy Erecoil, drawn from the species'
biexponential spectrum (PErecoil_ER for electron recoils, PErecoil_NR for nuclear
recoils). Erecoil then produces a Fano-broadened number of e/h pairs N, which
in turn sets the phonon energy Ep (recoil + Neganov-Luke) and charge
energy Eq (N * eps), each further smeared by its own detector resolution:

    Erecoil ~ PErecoil_NR or PErecoil_ER        (biexponential true-recoil-energy spectrum)
    N  | Erecoil ~ TruncNormal     (Fano-factor fluctuation, truncated at N >= 0)
    Ep | Erecoil, N ~ Normal       (phonon detector: Neganov-Luke + thermal)
    Eq | N ~ Normal           (charge detector)

The analytic PDFs Ppq_NR/Ppq_ER are obtained by marginalizing this same chain
analytically over N and numerically over Erecoil -- so agreement between events
generated here and Ppq_NR/Ppq_ER (see test/python/test_chisquare_{nr,er}_simulator.py)
is a real physics check, not a tautology: this module never touches Ppq_NR/Ppq_ER.

Returned DataFrame
-------------------
generate_events() returns columns EP, EQ, ERECOIL, N, is_ER, seed, is_bkg
(is_bkg is always False here -- it exists only so these rows line up with
a separate zero-charge/surface-background generator, e.g. the KDE-based
one in nrFanoII_paper2022's ba_bknd_lines.py, which can be concatenated
directly as long as it produces the same columns -- note that N was
previously named N0, misleadingly, since it's the same Fano-fluctuated N
generate_events_NR/generate_events_ER return, not a distinct quantity;
anything consuming the old N0 name needs updating). ERECOIL is the
latent true recoil energy Erecoil -- named to avoid reading as is_ER (is this
event an electron recoil?), a completely different thing.

Reproducibility
----------------
Every retry batch (see _species_in_roi below) needs its own independent
random stream, but "independent" from a single seed has to be done
carefully. np.random.SeedSequence.spawn() deterministically derives any
number of child streams from one seed via a proper hash-based mix,
guaranteeing (for practical purposes) no correlation between them or with
streams spawned from a different seed.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

import os
import sys

_PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PYTHON_DIR)
import Ppq_py as ppq

# generate_events_NR/generate_events_ER are deliberately not here (see the
# module docstring's "Returned DataFrame" section and generate_events()'s
# own docstring for why calling them directly is a footgun) -- still
# importable by name if you genuinely want them.
__all__ = ["generate_events"]


def _sample_biexponential(n, rng, *, mix_frac, scale_fast, scale_slow):
    """n draws from a two-component exponential mixture -- the shared shape
    of both PErecoil_NR (nuclear recoils) and PErecoil_ER (electron recoils)."""
    is_fast = rng.random(n) < mix_frac
    return np.where(is_fast, rng.exponential(scale_fast, n), rng.exponential(scale_slow, n))


def _sample_N(Nbar_arr, F_arr, rng):
    """
    Sample N (number of e/h pairs) for each event from a half-normal
    truncated at 0. Mean = Nbar, Var = Nbar * F.

    Uses scipy.stats.truncnorm vectorised over events.
    """
    sigma_N = np.sqrt(np.maximum(Nbar_arr * F_arr, 0.0))

    # scalar path: avoid divide-by-zero when sigma is zero
    N = np.empty_like(Nbar_arr)
    zero_mask = sigma_N == 0.0
    N[zero_mask] = Nbar_arr[zero_mask]

    nonzero = ~zero_mask
    if nonzero.any():
        a_trunc = -Nbar_arr[nonzero] / sigma_N[nonzero]   # lower clip in std units
        b_trunc = np.inf
        N[nonzero] = truncnorm.rvs(a_trunc, b_trunc,
                                   loc=Nbar_arr[nonzero],
                                   scale=sigma_N[nonzero],
                                   random_state=rng)
    return N


def _generate_species_events(n_events, *, Nbar_func,
                              F0, eps, V, p0, p10, q0, q10,
                              mix_frac, scale_fast, scale_slow,
                              seed):
    """
    Shared generator: draws Erecoil from the biexponential mix_frac/scale_fast/
    scale_slow spectrum and mean e/h pair count Nbar_func(Erecoil), then
    simulates the rest of the detector chain. NR/ER differ only in
    Nbar_func and which spectrum parameters they pass (see
    generate_events_NR/generate_events_ER).
    """
    rng = np.random.default_rng(seed)

    Erecoil = _sample_biexponential(n_events, rng, mix_frac=mix_frac,
                                scale_fast=scale_fast, scale_slow=scale_slow)

    Nbar = Nbar_func(Erecoil)                              # mean e/h pairs
    F_val = np.full_like(Erecoil, F0)                      # Fano factor (constant)

    N = _sample_N(Nbar, F_val, rng)

    Ep_true = Erecoil + N * (V / 1e3)                      # Neganov-Luke + recoil
    Eq_true = N * eps                                 # ionization

    # Detector smearing: evaluate resolution at the true (noise-free) signal.
    # sigp and sigq vary slowly, so this closely approximates the heteroscedastic
    # PDF where resolutions are evaluated at the measured values.
    sigp_val = ppq.sigp(Ep_true, eps=eps, V=V, p0=p0, p10=p10)
    sigq_val = ppq.sigq(Eq_true, q0=q0, q10=q10)

    Ep = rng.normal(Ep_true, sigp_val)
    Eq = rng.normal(Eq_true, sigq_val)

    return Ep, Eq, Erecoil, N


def generate_events_NR(n_events, *,
                        k, Z,
                        F0,
                        eps, V,
                        p0, p10,
                        q0, q10,
                        mix_frac, scale_fast, scale_slow,
                        seed):
    """
    Generate (Ep, Eq) pairs distributed according to Ppq_NR -- no ROI
    cropping, no species mixing. Not exported: generate_events(fraction_ER=0.0)
    gives you a pure-NR sample with the same columns, already cropped to
    your ROI, for every normal use (including as a chi-square sampler_func
    -- calling this directly there is a real footgun, see
    run_chisquare_test's docstring). Reach for this directly only if you
    genuinely want the uncropped full physical model.

    mix_frac, scale_fast, scale_slow are PErecoil_NR's biexponential
    true-recoil-energy spectrum parameters (see Ppq_py.py's PErecoil_NR).

    Returns
    -------
    Ep, Eq : ndarray, shape (n_events,)
    Erecoil : ndarray, shape (n_events,)   -- latent true recoil energies
    N  : ndarray, shape (n_events,)   -- latent e/h pair counts
    """
    return _generate_species_events(
        n_events, Nbar_func=lambda Erecoil: ppq.Nbar_NR(Erecoil, k=k, Z=Z, eps=eps),
        F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10,
        mix_frac=mix_frac, scale_fast=scale_fast, scale_slow=scale_slow, seed=seed)


def generate_events_ER(n_events, *,
                        F0,
                        eps, V,
                        p0, p10,
                        q0, q10,
                        mix_frac, scale_fast, scale_slow,
                        seed):
    """
    Generate (Ep, Eq) pairs distributed according to Ppq_ER -- no ROI
    cropping, no species mixing. Not exported: generate_events(fraction_ER=1.0)
    gives you a pure-ER sample with the same columns, already cropped to
    your ROI, for every normal use (see generate_events_NR's docstring for
    why calling this directly, e.g. as a chi-square sampler_func, is a
    footgun).

    Gamma / electron-recoil events have ionization yield Y = 1. mix_frac,
    scale_fast, scale_slow are PErecoil_ER's biexponential true-recoil-energy
    spectrum parameters (see Ppq_py.py's PErecoil_ER).

    Returns
    -------
    Ep, Eq : ndarray, shape (n_events,)
    Erecoil : ndarray, shape (n_events,)
    N  : ndarray, shape (n_events,)
    """
    return _generate_species_events(
        n_events, Nbar_func=lambda Erecoil: ppq.Nbar_ER(Erecoil, eps=eps),
        F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10,
        mix_frac=mix_frac, scale_fast=scale_fast, scale_slow=scale_slow, seed=seed)


# Initial guess for each species' ROI acceptance rate, refined from batch 1
# onward using the actually-observed rate. Only affects how generously the
# FIRST retry batch is sized -- a bad guess costs one extra retry, not
# correctness. ER's band is intrinsically narrower in a typical ROI than
# NR's, hence the lower starting guess.
_DEFAULT_PASS_RATE = {"ER": 0.08, "NR": 0.20}


def _species_in_roi(n_target, species, seed_sequence, *,
                     ep_range, eq_range, max_retries,
                     sampler, sampler_kwargs):
    """
    Generate exactly n_target ROI-passing (Ep, Eq, Erecoil, N) events of one
    species, by calling sampler(n, seed=..., **sampler_kwargs) in retry
    batches (generate_events_NR or generate_events_ER).

    Each retry batch draws a fresh set of events (sized from a running
    estimate of the ROI acceptance rate) using its own independent seed
    spawned from seed_sequence, so retries never reuse a random stream.
    """
    if n_target == 0:
        empty = np.array([])
        return empty, empty, empty, empty

    child_seeds = iter(seed_sequence.spawn(max_retries))
    ep_min, ep_max = ep_range
    eq_min, eq_max = eq_range

    ep_parts, eq_parts, erecoil_parts, n_parts = [], [], [], []
    n_have = 0
    pass_rate = _DEFAULT_PASS_RATE[species]

    for _ in range(max_retries):
        if n_have >= n_target:
            break
        seed = int(next(child_seeds).generate_state(1)[0])

        remaining = n_target - n_have
        want = int(remaining / pass_rate * 1.3) + 2000

        Ep, Eq, Erecoil, N = sampler(want, seed=seed, **sampler_kwargs)

        mask = (Ep >= ep_min) & (Ep <= ep_max) & (Eq >= eq_min) & (Eq <= eq_max)
        passed = int(mask.sum())
        pass_rate = max(passed / want, 0.01)

        keep = min(passed, remaining)
        idx = np.flatnonzero(mask)[:keep]
        ep_parts.append(Ep[idx])
        eq_parts.append(Eq[idx])
        erecoil_parts.append(Erecoil[idx])
        n_parts.append(N[idx])
        n_have += keep

    if n_have < n_target:
        warnings.warn(f"{species}: only generated {n_have}/{n_target} events "
                      f"within {max_retries} retry batches -- raise max_retries.")

    return (np.concatenate(ep_parts), np.concatenate(eq_parts),
            np.concatenate(erecoil_parts), np.concatenate(n_parts))


def generate_events(target_size, fraction_ER, seed, *,
                     ep_range, eq_range,
                     k, Z, F0_NR, F0_ER, eps, V, p0, p10, q0, q10,
                     mix_frac_NR, scale_fast_NR, scale_slow_NR,
                     mix_frac_ER, scale_fast_ER, scale_slow_ER,
                     max_retries=40):
    """
    Generate target_size (Ep, Eq) events, fraction_ER of them (in expectation)
    electron recoils and the rest nuclear recoils, cropped to ep_range x
    eq_range.

    fraction_ER=1.0 / 0.0 generate pure ER / pure NR samples.

    Parameters
    ----------
    target_size : int
        Total number of ROI-passing events to generate.
    fraction_ER : float
        Probability that a given event is an electron recoil; the actual
        ER count is Binomial(target_size, fraction_ER), not a fixed split.
        Var(Binomial(n, p)) = n*p*(1-p) is exactly zero at p=0 or p=1, so
        fraction_ER=1.0 / 0.0 are still fully deterministic (ER count = exactly
        target_size / 0, every seed) -- no separate ER_only/NR_only flag
        is needed to get a pure sample with zero chance of a stray event
        from the other species.
    seed : int
        Seed for the whole call. Reproducible: the same seed always
        produces the same events, independent of what other seeds are
        used elsewhere in the same run.
    ep_range, eq_range : (float, float)
        ROI bounds, keV -- required, no default: this must be your real
        analysis region, not an arbitrary wide range.
    k, Z : float
        Lindhard ionization yield calibration constant and target atomic
        number (nuclear recoils only; unused but still required when
        fraction_ER=1.0, for the same reason Ppq_compiled always requires them).
    F0_NR, F0_ER : float
        Fano factor (constant), one per species -- ER and NR generally
        have different Fano factors even at the same operating point.
    eps, V, p0, p10, q0, q10 : float
        Shared detector parameters (energy per e/h pair, bias voltage,
        phonon/charge resolution parameters) -- see Ppq_py.py's
        sigp/sigq.
    mix_frac_NR, scale_fast_NR, scale_slow_NR : float
        NR true-recoil-energy (PErecoil_NR) biexponential spectrum parameters.
    mix_frac_ER, scale_fast_ER, scale_slow_ER : float
        ER true-recoil-energy (PErecoil_ER) biexponential spectrum parameters.
    max_retries : int
        Cap on retry batches per species before giving up with a warning.

    Returns
    -------
    pandas.DataFrame with columns EP, EQ, ERECOIL, N, is_ER, seed, is_bkg --
    shuffled, so downstream code doesn't need to worry about ER/NR
    ordering. is_bkg is always False; see the module docstring.
    """
    if not 0.0 <= fraction_ER <= 1.0:
        raise ValueError(f"fraction_ER must be in [0, 1], got {fraction_ER}")

    ss = np.random.SeedSequence(seed)
    seed_split, seed_ER, seed_NR, seed_shuffle = ss.spawn(4)

    # deterministic (no fluctuation) at fraction_ER=0.0/1.0 -- see docstring
    n_ER = int(np.random.default_rng(seed_split).binomial(target_size, fraction_ER))
    n_NR = target_size - n_ER

    roi = dict(ep_range=ep_range, eq_range=eq_range)

    Ep_er, Eq_er, Erecoil_er, N_er = _species_in_roi(
        n_ER, "ER", seed_ER, max_retries=max_retries, **roi,
        sampler=generate_events_ER,
        sampler_kwargs=dict(F0=F0_ER, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10,
                            mix_frac=mix_frac_ER, scale_fast=scale_fast_ER, scale_slow=scale_slow_ER))
    Ep_nr, Eq_nr, Erecoil_nr, N_nr = _species_in_roi(
        n_NR, "NR", seed_NR, max_retries=max_retries, **roi,
        sampler=generate_events_NR,
        sampler_kwargs=dict(k=k, Z=Z, F0=F0_NR, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10,
                            mix_frac=mix_frac_NR, scale_fast=scale_fast_NR, scale_slow=scale_slow_NR))

    Ep = np.concatenate([Ep_er, Ep_nr])
    Eq = np.concatenate([Eq_er, Eq_nr])
    Erecoil = np.concatenate([Erecoil_er, Erecoil_nr])
    N = np.concatenate([N_er, N_nr])
    is_ER = np.concatenate([np.ones(len(Ep_er), dtype=bool), np.zeros(len(Ep_nr), dtype=bool)])

    order = np.random.default_rng(seed_shuffle).permutation(len(Ep))
    df = pd.DataFrame({
        "EP": Ep[order], "EQ": Eq[order], "ERECOIL": Erecoil[order],
        "N": N[order], "is_ER": is_ER[order],
    })
    df["seed"] = seed
    df["is_bkg"] = False

    n_ok = len(df)
    print(f"Generated {n_ok}/{target_size} events "
          f"(ER {int(df['is_ER'].sum())}, NR {n_ok - int(df['is_ER'].sum())}; "
          f"target ER count was {n_ER} of {target_size} via Binomial(fraction_ER={fraction_ER})); "
          f"EP:[{ep_range[0]},{ep_range[1]}], EQ:[{eq_range[0]},{eq_range[1]}] keV")

    return df
