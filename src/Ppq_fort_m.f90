module Ppq_fort_m
  !! TODO: Add comment for FORD describing the module's purpose
  use iso_c_binding, only : c_double, c_int
  implicit none

  interface

  pure module subroutine Ppq_vector_NR(Ep_arr, Eq_arr, n, k, Z, F0, eps, V, p0, p10, q0, q10, &
      table_handle, res_arr) bind(c, name="Ppq_vector_NR")
    !! Ppq_NR evaluated at n (Ep, Eq) points via do concurrent, which compiles to
    !! genuine multi-threaded parallelism under ifx/flang but runs serially under
    !! gfortran. Measured at roughly 6 microseconds per point under ifx/flang (18
    !! threads, x86 emulation, over a representative grid); gfortran builds run this
    !! loop serially and are roughly 25x slower.
    !!
    !! table_handle: 0 for the closed-form PErecoil_NR spectrum, or a handle
    !! returned by PErecoil_fit_register for a fitted one -- an
    !! explicit argument of this call, not ambient state, so two callers
    !! with different fits (e.g. a joint Cf + Ba fit's two Ppq_compiled
    !! instances) can't interfere with each other, under any concurrency
    !! pattern. See adr/0003-per-instance-perecoil-tables.md and
    !! adr/0011-perecoil-handles-register-a-fit-not-a-table.md.
    integer(c_int), value :: n
    real(c_double), intent(in) :: Ep_arr(n), Eq_arr(n)
    real(c_double), value :: k, Z, F0, eps, V, p0, p10, q0, q10
    integer(c_int), value :: table_handle
    real(c_double), intent(out) :: res_arr(n)
  end subroutine Ppq_vector_NR

  pure module subroutine Ppq_vector_ER(Ep_arr, Eq_arr, n, F0, eps, V, p0, p10, q0, q10, &
      table_handle, res_arr) bind(c, name="Ppq_vector_ER")
    !! Same as Ppq_vector_NR but for Ppq_ER.
    integer(c_int), value :: n
    real(c_double), intent(in) :: Ep_arr(n), Eq_arr(n)
    real(c_double), value :: F0, eps, V, p0, p10, q0, q10
    integer(c_int), value :: table_handle
    real(c_double), intent(out) :: res_arr(n)
  end subroutine Ppq_vector_ER

  pure module function Ppq_NR(Ep, Eq, k, Z, F0, eps, V, p0, p10, q0, q10, table_handle) &
      result(res) bind(c, name="Ppq_NR")
    !! The nuclear-recoil band probability density at one (Ep, Eq) point. Integrates
    !! over a window placed around the located peak(s) of the Erecoil integrand
    !! rather than sampling the full physical range, since the ridge location/width
    !! is already known exactly (see Ppq_region_NR).
    !!
    !! table_handle: see Ppq_vector_NR's own comment -- this takes it only
    !! because Ppq_vector_NR's per-point loop calls this and must forward
    !! it, not because this is meant to be called directly with a table in
    !! mind.
    real(c_double), value :: Ep, Eq, k, Z, F0, eps, V, p0, p10, q0, q10
    integer(c_int), value :: table_handle
    real(c_double) :: res
  end function Ppq_NR

  pure module function Ppq_ER(Ep, Eq, F0, eps, V, p0, p10, q0, q10, table_handle) &
      result(res) bind(c, name="Ppq_ER")
    !! Same as Ppq_NR but for the electron-recoil band (Y=1).
    real(c_double), value :: Ep, Eq, F0, eps, V, p0, p10, q0, q10
    integer(c_int), value :: table_handle
    real(c_double) :: res
  end function Ppq_ER

  pure module function Y(Erecoil, k, Z) result(res) bind(c, name="Y")
    !! the ionization yield, using the Lindhard model
    !! Y(Erecoil) = k*g(eps_L) / (1 + k*g(eps_L)), with
    !! eps_L = 11.5*Erecoil*Z**(-7/3) and g(eps_L) = 3*eps_L**0.15 + 0.7*eps_L**0.29 + eps_L
    real(c_double), value :: Erecoil, k, Z
    real(c_double) :: res
  end function Y

  pure module function sigp(Ep, eps, V, p0, p10) result(res) bind(c, name="sigp")
    !! phonon sensor resolution
    real(c_double), value :: Ep, eps, V, p0, p10
    real(c_double) :: res
  end function sigp

  pure module function sigq(Eq, q0, q10) result(res) bind(c, name="sigq")
    !! charge sensor resolution
    real(c_double), value :: Eq, q0, q10
    real(c_double) :: res
  end function sigq

  pure module function PErecoil_NR(Erecoil) result(res) bind(c, name="PErecoil_NR")
    !! the Erecoil (energy) distribution of neutrons
    real(c_double), value :: Erecoil
    real(c_double) :: res
  end function PErecoil_NR

  pure module function PErecoil_ER(Erecoil) result(res) bind(c, name="PErecoil_ER")
    !! the Erecoil (energy) distribution of gamma events/electron recoils
    real(c_double), value :: Erecoil
    real(c_double) :: res
  end function PErecoil_ER

  module function PErecoil_fit_register(a, b, n, C) result(handle) &
      bind(c, name="PErecoil_fit_register")
    !! Register a fitted true-recoil-energy spectrum and get back a
    !! **handle** -- that's `handle`, this function's *return value*
    !! (`result(handle)` above), not any of the four listed arguments.
    !! Pass the returned handle, in place of `0` (which always means the
    !! closed-form spectrum instead), to Ppq_NR/ER, Ppq_vector_NR/ER, or
    !! Ppq_region_NR/ER(_with_error) to evaluate against this fit.
    !!
    !! The spectrum is a sum of `n` exponentials plus a constant floor:
    !!
    !! `density(Erecoil) = sum(a(i) * exp(-Erecoil / b(i)), i=1..n) + C`
    !!
    !! -- deliberately a *small, curated* family, not an arbitrary
    !! user-specified function: every term is globally smooth (no local
    !! structure at any scale), which is the actual property this
    !! library's quadrature needs (see
    !! adr/0011-perecoil-handles-register-a-fit-not-a-table.md). A
    !! PCHIP-interpolated or smoothed *table* built from real calibration
    !! data does not have that property, and was found to introduce a
    !! parameter-dependent Gauss-Legendre precision floor near the phonon
    !! resolution model's degenerate edge (p10 -> p0) -- this replaces
    !! that table-registration path entirely (adr/0003's handle pattern
    !! stays; what a handle now points at changes).
    !!
    !! Arguments, concretely:
    !!
    !! - `n`: number of exponential terms, i.e. `size(a) == size(b)`.
    !! - `a(n)`, `b(n)`: each term's amplitude and decay length (keV).
    !! - `C`: the constant floor.
    !!
    !! Nobody hand-types `a`/`b`/`C` in practice -- they come from a
    !! weighted least-squares fit to real calibration data, checked
    !! against the data it was fit to (a Kolmogorov-Smirnov test against
    !! the real, unbinned events is the recommended check -- see the
    !! Python-side fit module). Called directly from Fortran, though, it
    !! looks like:
    !!
    !! ```fortran
    !! use iso_c_binding, only: c_double, c_int
    !! use Ppq_fort_m, only: PErecoil_fit_register, Ppq_NR
    !! integer(c_int) :: handle
    !! real(c_double) :: a(n), b(n), C, density
    !! ! ... a, b, C come from a weighted fit to real calibration data ...
    !! handle = PErecoil_fit_register(a, b, n, C)   ! the RESULT
    !! density = Ppq_NR(Ep, Eq, k, Z, F0, eps, V, p0, p10, q0, q10, handle)
    !! ```
    !!
    !! The only non-pure procedure in this file: copies a/b/C into a new,
    !! append-only registry slot (never overwriting one an earlier caller
    !! already holds a handle to -- see adr/0003-per-instance-perecoil-
    !! tables.md), fixed capacity, not grown by reallocation. Call once
    !! per fit a Ppq_compiled/Ppq_densities registers at construction.
    integer(c_int), value :: n
    real(c_double), intent(in) :: a(n), b(n)
    real(c_double), value :: C
    integer(c_int) :: handle
  end function PErecoil_fit_register

  pure module function pq_given_Erecoil_NR(Erecoil, Ep, Eq, k, Z, F0, eps, V, p0, p10, q0, q10) result(res) bind(c, name="pq_given_Erecoil_NR")
    !! An internal bind(c) handle on the per-Erecoil N-integral piece of
    !! Ppq_NR, exposed only so test_PpqFort.py can cross-check it against
    !! Python at machine precision -- not a stable public API, no end
    !! user needs this.
    real(c_double), value :: Erecoil, Ep, Eq, k, Z, F0, eps, V, p0, p10, q0, q10
    real(c_double) :: res
  end function pq_given_Erecoil_NR

  pure module function pq_given_Erecoil_ER(Erecoil, Ep, Eq, F0, eps, V, p0, p10, q0, q10) result(res) bind(c, name="pq_given_Erecoil_ER")
    !! Same as pq_given_Erecoil_NR but for the ER band -- see its comment.
    real(c_double), value :: Erecoil, Ep, Eq, F0, eps, V, p0, p10, q0, q10
    real(c_double) :: res
  end function pq_given_Erecoil_ER

  pure module function Ppq_region_NR(ep_min, ep_max, eq_min, eq_max, epsrel, epsabs, &
      k, Z, F0, eps, V, p0, p10, q0, q10, table_handle) result(res) bind(c, name="Ppq_region_NR")
    !! Integral of Ppq_NR over the rectangle [ep_min,ep_max] x [eq_min,eq_max],
    !! e.g. for normalizing a likelihood to its fit region.  Computed with
    !! a doubling-verified nested Gauss-Legendre quadrature: the ridge
    !! location/width machinery (ridge_eq_and_width in Ppq_fort_s.f90)
    !! already tells this exactly where the band is, so it needs far
    !! fewer points than a naive fixed grid for the same accuracy.
    !! epsrel/epsabs set how tightly two successive doubled quadrature
    !! orders (32 vs 64, 64 vs 128, 128 vs 256) must agree before the
    !! result is trusted: this stops refining once
    !! |result_2N - result_N| <= max(epsabs, epsrel*|result_2N|).  Returns
    !! NaN (not a value you can mistake for a real one), rather than an
    !! unverified number, if order 256 still hasn't converged -- the
    !! Python wrapper (_Ppq_fort_bindings.Ppq_region_NR) turns that into a
    !! catchable GLConvergenceError.  Parameter sets the model cannot
    !! evaluate (p10 < p0 or q10 < q0, which make the resolution variance
    !! negative at high energy; F0, eps, p0, q0, k, Z <= 0; empty region;
    !! NaN inputs) error-stop immediately instead of grinding -- that guard
    !! is a different, genuinely fatal case (a logic error in the caller),
    !! unlike ordinary non-convergence above.
    !!
    !! table_handle: see Ppq_vector_NR's own comment.
    real(c_double), value :: ep_min, ep_max, eq_min, eq_max, epsrel, epsabs
    real(c_double), value :: k, Z, F0, eps, V, p0, p10, q0, q10
    integer(c_int), value :: table_handle
    real(c_double) :: res
  end function Ppq_region_NR

  pure module function Ppq_region_ER(ep_min, ep_max, eq_min, eq_max, epsrel, epsabs, &
      F0, eps, V, p0, p10, q0, q10, table_handle) result(res) bind(c, name="Ppq_region_ER")
    !! Same as Ppq_region_NR but for Ppq_ER (electron-recoil band, Y=1).
    real(c_double), value :: ep_min, ep_max, eq_min, eq_max, epsrel, epsabs
    real(c_double), value :: F0, eps, V, p0, p10, q0, q10
    integer(c_int), value :: table_handle
    real(c_double) :: res
  end function Ppq_region_ER

  pure module subroutine Ppq_region_NR_with_error(ep_min, ep_max, eq_min, eq_max, epsrel, epsabs, &
      k, Z, F0, eps, V, p0, p10, q0, q10, table_handle, res, error_estimate) &
      bind(c, name="Ppq_region_NR_with_error")
    !! Same as Ppq_region_NR, plus the achieved Gauss-Legendre doubling
    !! residual in error_estimate: |result_2N - result_N| for whichever
    !! pair of successive orders (32/64, 64/128, 128/256) satisfied
    !! epsrel/epsabs (or, if none did, the 128/256 pair just before
    !! returning NaN) -- a numerical convergence diagnostic, in the same
    !! units as res, not a statistical uncertainty on the PErecoil model
    !! itself (a tabulated PErecoil's own calibration-statistics
    !! uncertainty is a separate question this says nothing about -- see
    !! erecoil_table.py). By design this should be far smaller than
    !! epsrel/epsabs asked for; a caller propagating it into e.g. a
    !! likelihood's error budget is checking that it's negligible, not
    !! normally expecting it to matter. A second, separate entry point
    !! (not an extra argument on Ppq_region_NR itself) so that function's
    !! existing bind(c) signature never changes. A subroutine, not a
    !! function: a pure FUNCTION's dummy arguments must all be
    !! intent(in)/value, unlike a pure SUBROUTINE (same reason
    !! PpqFort_version is a subroutine) -- res is this entry point's
    !! equivalent of Ppq_region_NR's own function result, just passed out
    !! as an ordinary argument instead.
    real(c_double), value :: ep_min, ep_max, eq_min, eq_max, epsrel, epsabs
    real(c_double), value :: k, Z, F0, eps, V, p0, p10, q0, q10
    integer(c_int), value :: table_handle
    real(c_double), intent(out) :: res, error_estimate
  end subroutine Ppq_region_NR_with_error

  pure module subroutine Ppq_region_ER_with_error(ep_min, ep_max, eq_min, eq_max, epsrel, epsabs, &
      F0, eps, V, p0, p10, q0, q10, table_handle, res, error_estimate) &
      bind(c, name="Ppq_region_ER_with_error")
    !! Same as Ppq_region_NR_with_error but for Ppq_ER (electron-recoil
    !! band, Y=1) -- see that function's comment.
    real(c_double), value :: ep_min, ep_max, eq_min, eq_max, epsrel, epsabs
    real(c_double), value :: F0, eps, V, p0, p10, q0, q10
    integer(c_int), value :: table_handle
    real(c_double), intent(out) :: res, error_estimate
  end subroutine Ppq_region_ER_with_error

  pure module subroutine PpqFort_version(major, minor, patch) bind(c, name="PpqFort_version")
    !! Report the package version: fpm.toml's version field, verbatim --
    !! see version.f90.inc (generated by scripts/generate_version_include.py,
    !! not committed; run it before building, same as every Dockerfile does).
    integer(c_int), intent(out) :: major, minor, patch
  end subroutine PpqFort_version

  end interface

  ! Package version (semver): version_major/minor/patch, generated from
  ! fpm.toml by scripts/generate_version_include.py.  Not hand-typed here
  ! any more -- that was two copies of the same number that had to be kept
  ! in sync by hand, and they drifted (v1.1.0 and v1.1.1 both shipped still
  ! reporting 1.1.0).
  include "version.f90.inc"

end module Ppq_fort_m
