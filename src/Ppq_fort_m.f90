module Ppq_fort_m
  !! TODO: Add comment for FORD describing the module's purpose
  use iso_c_binding, only : c_double, c_int
  implicit none

  interface

  pure module subroutine Ppq_vector_NR(Ep_arr, Eq_arr, n, k, Z, F0, eps, V, p0, p10, q0, q10, res_arr) bind(c, name="Ppq_vector_NR")
    !! Ppq_NR evaluated at n (Ep, Eq) points via do concurrent, which compiles to
    !! genuine multi-threaded parallelism under ifx/flang but runs serially under
    !! gfortran. Measured at roughly 6 microseconds per point under ifx/flang (18
    !! threads, x86 emulation, over a representative grid); gfortran builds run this
    !! loop serially and are roughly 25x slower.
    integer(c_int), value :: n
    real(c_double), intent(in) :: Ep_arr(n), Eq_arr(n)
    real(c_double), value :: k, Z, F0, eps, V, p0, p10, q0, q10
    real(c_double), intent(out) :: res_arr(n)
  end subroutine Ppq_vector_NR

  pure module subroutine Ppq_vector_ER(Ep_arr, Eq_arr, n, F0, eps, V, p0, p10, q0, q10, res_arr) bind(c, name="Ppq_vector_ER")
    !! Same as Ppq_vector_NR but for Ppq_ER.
    integer(c_int), value :: n
    real(c_double), intent(in) :: Ep_arr(n), Eq_arr(n)
    real(c_double), value :: F0, eps, V, p0, p10, q0, q10
    real(c_double), intent(out) :: res_arr(n)
  end subroutine Ppq_vector_ER

  pure module function Ppq_NR(Ep, Eq, k, Z, F0, eps, V, p0, p10, q0, q10) result(res) bind(c, name="Ppq_NR")
    !! The nuclear-recoil band probability density at one (Ep, Eq) point. Integrates
    !! over a window placed around the located peak(s) of the Erecoil integrand
    !! rather than sampling the full physical range, since the ridge location/width
    !! is already known exactly (see Ppq_region_NR).
    real(c_double), value :: Ep, Eq, k, Z, F0, eps, V, p0, p10, q0, q10
    real(c_double) :: res
  end function Ppq_NR

  pure module function Ppq_ER(Ep, Eq, F0, eps, V, p0, p10, q0, q10) result(res) bind(c, name="Ppq_ER")
    !! Same as Ppq_NR but for the electron-recoil band (Y=1).
    real(c_double), value :: Ep, Eq, F0, eps, V, p0, p10, q0, q10
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

  module subroutine PErecoil_set_table_NR(x, c0, c1, c2, c3, n) bind(c, name="PErecoil_set_table_NR")
    !! Switch PErecoil_NR to a tabulated true-recoil-energy spectrum: a
    !! piecewise cubic in power-basis form (scipy PPoly convention; see
    !! erecoil_table.PErecoilTable, which builds x/c0..c3 with PCHIP and is
    !! already normalized to integrate to 1), replacing the closed-form
    !! biexponential until PErecoil_clear_table_NR is called. NOT pure: this
    !! is global, process-wide state (see the module-level comment on
    !! nr_table_x in Ppq_fort_s.f90) -- call it once, before any density
    !! evaluation, never from inside a parallel region. x has n points
    !! (strictly increasing); c0/c1/c2/c3 each have n-1, one per interval.
    integer(c_int), value :: n
    real(c_double), intent(in) :: x(n), c0(n - 1), c1(n - 1), c2(n - 1), c3(n - 1)
  end subroutine PErecoil_set_table_NR

  module subroutine PErecoil_clear_table_NR() bind(c, name="PErecoil_clear_table_NR")
    !! Undo PErecoil_set_table_NR: PErecoil_NR goes back to the closed-form
    !! biexponential. NOT pure; see PErecoil_set_table_NR.
  end subroutine PErecoil_clear_table_NR

  module subroutine PErecoil_set_table_ER(x, c0, c1, c2, c3, n) bind(c, name="PErecoil_set_table_ER")
    !! Same as PErecoil_set_table_NR but for PErecoil_ER.
    integer(c_int), value :: n
    real(c_double), intent(in) :: x(n), c0(n - 1), c1(n - 1), c2(n - 1), c3(n - 1)
  end subroutine PErecoil_set_table_ER

  module subroutine PErecoil_clear_table_ER() bind(c, name="PErecoil_clear_table_ER")
    !! Same as PErecoil_clear_table_NR but for PErecoil_ER.
  end subroutine PErecoil_clear_table_ER

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
      k, Z, F0, eps, V, p0, p10, q0, q10) result(res) bind(c, name="Ppq_region_NR")
    !! Integral of Ppq_NR over the rectangle [ep_min,ep_max] x [eq_min,eq_max],
    !! e.g. for normalizing a likelihood to its fit region.  Computed with
    !! a doubling-verified nested Gauss-Legendre quadrature: the ridge
    !! location/width machinery (ridge_eq_and_width in Ppq_fort_s.f90)
    !! already tells this exactly where the band is, so it needs far
    !! fewer points than a naive fixed grid for the same accuracy.
    !! epsrel/epsabs set how tightly two successive doubled quadrature
    !! orders (32 vs 64, 64 vs 128, 128 vs 256) must agree before the
    !! result is trusted: this stops refining once
    !! |result_2N - result_N| <= max(epsabs, epsrel*|result_2N|), and
    !! error-stops rather than returning an unverified number if order 256
    !! still hasn't converged.  Parameter sets the model cannot evaluate
    !! (p10 < p0 or q10 < q0, which make the resolution variance negative
    !! at high energy; F0, eps, p0, q0, k, Z <= 0; empty region; NaN
    !! inputs) error-stop immediately instead of grinding.
    real(c_double), value :: ep_min, ep_max, eq_min, eq_max, epsrel, epsabs
    real(c_double), value :: k, Z, F0, eps, V, p0, p10, q0, q10
    real(c_double) :: res
  end function Ppq_region_NR

  pure module function Ppq_region_ER(ep_min, ep_max, eq_min, eq_max, epsrel, epsabs, &
      F0, eps, V, p0, p10, q0, q10) result(res) bind(c, name="Ppq_region_ER")
    !! Same as Ppq_region_NR but for Ppq_ER (electron-recoil band, Y=1).
    real(c_double), value :: ep_min, ep_max, eq_min, eq_max, epsrel, epsabs
    real(c_double), value :: F0, eps, V, p0, p10, q0, q10
    real(c_double) :: res
  end function Ppq_region_ER

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
