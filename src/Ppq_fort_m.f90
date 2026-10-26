module Ppq_fort_m
  !! TODO: Add comment for FORD describing the module's purpose
  use iso_c_binding, only : c_double, c_int
  implicit none

  interface

    pure module subroutine Ppq_vector_NR(Ep_arr, Eq_arr, n, k, Z, F0, eps, V, p0, p10, q0, q10, res_arr) bind(c, name="Ppq_vector_NR")
      !! TODO: Add comment for FORD describing the function's purpose
      integer(c_int), value :: n
      real(c_double), intent(in) :: Ep_arr(n), Eq_arr(n)
      real(c_double), value :: k, Z, F0, eps, V, p0, p10, q0, q10
      real(c_double), intent(out) :: res_arr(n)
    end subroutine Ppq_vector_NR

    pure module subroutine Ppq_vector_ER(Ep_arr, Eq_arr, n, F0, eps, V, p0, p10, q0, q10, res_arr) bind(c, name="Ppq_vector_ER")
      !! TODO: Add comment for FORD describing the function's purpose
      integer(c_int), value :: n
      real(c_double), intent(in) :: Ep_arr(n), Eq_arr(n)
      real(c_double), value :: F0, eps, V, p0, p10, q0, q10
      real(c_double), intent(out) :: res_arr(n)
    end subroutine Ppq_vector_ER

    pure module function Ppq_NR(Ep, Eq, k, Z, F0, eps, V, p0, p10, q0, q10) result(res) bind(c, name="Ppq_NR")
      !! TODO: Add comment for FORD describing the function's purpose
      real(c_double), value :: Ep, Eq, k, Z, F0, eps, V, p0, p10, q0, q10
      real(c_double) :: res
    end function Ppq_NR

    pure module function Ppq_ER(Ep, Eq, F0, eps, V, p0, p10, q0, q10) result(res) bind(c, name="Ppq_ER")
      !! TODO: Add comment for FORD describing the function's purpose
      real(c_double), value :: Ep, Eq, F0, eps, V, p0, p10, q0, q10
      real(c_double) :: res
    end function Ppq_ER

   pure module function Y(Er, k, Z) result(res) bind(c, name="Y")
       !! the ionization yield, using the Lindhard model
       !! Y(Er) = k*g(eps_L) / (1 + k*g(eps_L)), with
       !! eps_L = 11.5*Er*Z**(-7/3) and g(eps_L) = 3*eps_L**0.15 + 0.7*eps_L**0.29 + eps_L
       real(c_double), value :: Er, k, Z
       real(c_double) :: res
    end function Y

   pure module function Nbar(Er, k, Z, eps) result(res) bind(c, name="Nbar")
     !! average number of electron-hole pairs for a given Er
     real(c_double), value :: Er, k, Z, eps
     real(c_double) :: res
   end function Nbar

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

  pure module function PEr_NR(Er) result(res) bind(c, name="PEr_NR")
    !! the Er (energy) distribution of neutrons
    real(c_double), value :: Er
    real(c_double) :: res
  end function PEr_NR

  pure module function PEr_ER(Er) result(res) bind(c, name="PEr_ER")
    !! the Er (energy) distribution of gamma events/electron recoils
    real(c_double), value :: Er
    real(c_double) :: res
  end function PEr_ER

  pure module function pq_given_Er_NR(Er, Ep, Eq, k, Z, F0, eps, V, p0, p10, q0, q10) result(res) bind(c, name="pq_given_Er_NR")
    ! Input arguments
    real(c_double), value :: Er, Ep, Eq, k, Z, F0, eps, V, p0, p10, q0, q10
    real(c_double) :: res
  end function pq_given_Er_NR

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

  ! Define pi
  real(c_double), parameter :: pi = 3.14159265358979323846_c_double
    ! TODO: move the pi definition to just above the submodule's "contains" statement

  ! Package version (semver): version_major/minor/patch, generated from
  ! fpm.toml by scripts/generate_version_include.py.  Not hand-typed here
  ! any more -- that was two copies of the same number that had to be kept
  ! in sync by hand, and they drifted (v1.1.0 and v1.1.1 both shipped still
  ! reporting 1.1.0).
  include "version.f90.inc"

end module Ppq_fort_m
