module Ppq_fort_test_m
  use iso_c_binding, only : c_double, c_int
  use julienne_m, only : &
     test_t, test_description_t, test_diagnosis_t, test_result_t &
    ,operator(.approximates.), operator(.within.), operator(.all.), operator(//)
  use Ppq_fort_m, only :  Ppq_vector_ER, Ppq_vector_NR
  implicit none

  type, extends(test_t) :: Ppq_fort_test_t
  contains
    procedure, nopass :: subject
    procedure, nopass :: results
  end type

  ! Parameters
  real(c_double), parameter :: k   = 0.18d0
  real(c_double), parameter :: Z   = 32.0d0
  real(c_double), parameter :: F0  = 0.122d0
  real(c_double), parameter :: eps = 3.0d-3
  real(c_double), parameter :: V   = 3.0d0
  real(c_double), parameter :: p0  = 0.06421907d0
  real(c_double), parameter :: p10 = 0.48998486d0
  real(c_double), parameter :: q0  = 0.06421907d0
  real(c_double), parameter :: q10 = 0.48998486d0

  ! Arrays
 real(c_double), parameter :: Eq_arr(*) = [100.0d0, 100.0d0, 100.0d0]
 real(c_double), parameter :: Ep_arr(*) = [347.0d0, 346.0d0, 348.0d0]
 integer(c_int), parameter :: ni = int(size(Eq_arr), c_int)

contains

  pure function subject() result(test_subject)
    character(len=:), allocatable :: test_subject
    test_subject = 'The Ppq_fort module'
  end function

  function results() result(test_results)
    type(test_result_t), allocatable :: test_results(:)
    type(Ppq_fort_test_t) :: Ppq_fort_test

    test_results = Ppq_fort_test%run([ &
       test_description_t('computing the resNR array', check_Ppq_vector_NR) &
      ,test_description_t('computing the resER array', check_Ppq_vector_ER) &
    ])
  end function

  function check_Ppq_vector_NR() result(test_diagnosis)
    type(test_diagnosis_t) test_diagnosis
    real(c_double), parameter :: expected_resNR(*) = [real(c_double) :: 2.0555519070930D-008, 2.2433719405230D-008, 1.8781056966950D-008]
    real(c_double), parameter :: tolerance(*) = expected_resNR/100
    real(c_double) resNR(size(Eq_arr))

    call Ppq_vector_NR(Ep_arr, Eq_arr, ni, k, Z, F0, eps, V, p0, p10, q0, q10, resNR)
    test_diagnosis =  .all. ( resNR .approximates. expected_resNR .within. tolerance)  // ' (Ppq_NR)'
  end function

  function check_Ppq_vector_ER() result(test_diagnosis)
    type(test_diagnosis_t) test_diagnosis
    real(c_double), parameter :: expected_resER(*) = [real(c_double) :: 1.0929369661820D-020, 1.4684710987090D-020, 8.1375591868460D-021]
    real(c_double), parameter :: tolerance(*) = expected_resER/100
    real(c_double) resER(size(Eq_arr))

    call Ppq_vector_ER(Ep_arr, Eq_arr, ni, F0, eps, V, p0, p10, q0, q10, resER)
    test_diagnosis =  .all. ( resER .approximates. expected_resER .within. tolerance)  // ' (Ppq_ER)'
  end function

end module Ppq_fort_test_m
