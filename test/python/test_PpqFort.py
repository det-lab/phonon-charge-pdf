import numpy as np
import ctypes
import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path to the directory containing Ppq.py
module_dir = os.path.join(repo_root, 'python')
sys.path.append(module_dir)

# Ppq contains the Python implementation with the numerical N integral
# (21-point Simpson's rule, sigp/sigq at noiseless energies), matching Fortran Ppq_NR.
import Ppq as ppq

folderpath = repo_root
if sys.platform == 'darwin':
    DLLname = 'lib/libphonon_charge_pdf.dylib'
elif os.name == 'posix':  # Linux
    DLLname = 'lib/libphonon_charge_pdf.so'
else:
    print('OS other than Linux/Mac are not supported')

api = np.ctypeslib.load_library(DLLname,folderpath)

failures = []


def check(name, fort_val, py_val, *, rtol=1e-9, atol=0.0):
    """Record and print a PASS/FAIL comparison of the Fortran and Python results."""
    ok = bool(np.isclose(fort_val, py_val, rtol=rtol, atol=atol))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: fortran={fort_val!r}  python={py_val!r}")
    if not ok:
        failures.append(name)


# Define all our parameters and variables
k = 0.18
Z = 32.0
F0 = 0.122
eps = 3E-3
V=3.0
p0=0.06421907
p10=0.48998486
q0=0.06421907
q10=0.48998486

Er = 234.0
Ep = 337.5
Eq = 100

# Define argument and return types for the function Y
api.Y.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
api.Y.restype = ctypes.c_double

result_fort = api.Y(ctypes.c_double(Er), ctypes.c_double(k), ctypes.c_double(Z))
result_py = ppq.Y(Er, k=k, Z=Z)
print('back in python after running function Y')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)
check("Y", result_fort, result_py)

# Define argument and return types for the function Nbar
api.Nbar.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
api.Nbar.restype = ctypes.c_double

result_fort = api.Nbar(ctypes.c_double(Er), ctypes.c_double(k), ctypes.c_double(Z), ctypes.c_double(eps))
result_py = ppq.Nbar(Er, k=k, Z=Z, eps=eps, is_gamma=False)
print('back in python after running function Nbar')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)
check("Nbar", result_fort, result_py)

# Define argument and return types for the function sigp
api.sigp.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
api.sigp.restype = ctypes.c_double

result_fort = api.sigp(ctypes.c_double(Ep), ctypes.c_double(eps), ctypes.c_double(V),
                       ctypes.c_double(p0), ctypes.c_double(p10))
result_py = ppq.sigp(Ep, eps=eps, V=V, p0=p0, p10=p10)
print('back in python after running function sigp')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)
check("sigp", result_fort, result_py)

# Define argument and return types for the function sigq
api.sigq.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
api.sigq.restype = ctypes.c_double

result_fort = api.sigq(ctypes.c_double(Eq), ctypes.c_double(q0), ctypes.c_double(q10))
result_py = ppq.sigq(Eq, q0=q0, q10=q10)
print('back in python after running function sigq')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)
check("sigq", result_fort, result_py)

# Define argument and return types for the function PEr_NR
api.PEr_NR.argtypes = [ctypes.c_double]
api.PEr_NR.restype = ctypes.c_double

result_fort = api.PEr_NR(ctypes.c_double(Er))
result_py = ppq.PEr_NR(Er)
print('back in python after running function PEr_NR')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)
check("PEr_NR", result_fort, result_py)

# Define argument and return types for the function PEr_ER
api.PEr_ER.argtypes = [ctypes.c_double]
api.PEr_ER.restype = ctypes.c_double

result_fort = api.PEr_ER(ctypes.c_double(Er))
result_py = ppq.PEr_ER(Er)
print('back in python after running function PEr_ER')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)
check("PEr_ER", result_fort, result_py)

# Define argument and return types for the function pq_given_Er_NR
api.pq_given_Er_NR.argtypes = [ctypes.c_double] * 12
api.pq_given_Er_NR.restype = ctypes.c_double

result_fort = api.pq_given_Er_NR(ctypes.c_double(Er), ctypes.c_double(Ep), ctypes.c_double(Eq),
                           ctypes.c_double(k), ctypes.c_double(Z),
                           ctypes.c_double(F0), ctypes.c_double(eps),
                           ctypes.c_double(V), ctypes.c_double(p0), ctypes.c_double(p10),
                           ctypes.c_double(q0), ctypes.c_double(q10))
result_py = ppq.pq_given_Er_NR(Er, Ep, Eq, k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
print('back in python after running function pq_given_Er_NR')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)
check("pq_given_Er_NR", result_fort, result_py, rtol=1e-6, atol=1e-12)

# Define argument and return types for the function Ppq_NR
api.Ppq_NR.argtypes = [ctypes.c_double] * 11
api.Ppq_NR.restype = ctypes.c_double

result_fort = api.Ppq_NR(ctypes.c_double(Ep), ctypes.c_double(Eq),
                       ctypes.c_double(k), ctypes.c_double(Z),
                       ctypes.c_double(F0), ctypes.c_double(eps),
                       ctypes.c_double(V), ctypes.c_double(p0), ctypes.c_double(p10),
                       ctypes.c_double(q0), ctypes.c_double(q10))
(result_py, _), _, _ = ppq.Ppq_inspect_NR(Ep, Eq, k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10, er_step=0.1)
print('back in python after running function Ppq_NR')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)
# Looser tolerance: the Fortran side and Ppq_inspect_NR's scipy.integrate.quad
# both integrate pq_given_Er_NR over Er, but with different quadrature schemes --
# "machine precision" only applies to pq_given_Er_NR itself (checked above), not
# to two independently-converged numerical integrals of it.
check("Ppq_NR", result_fort, result_py, rtol=1e-4, atol=1e-12)

# Define argument and return types for the function Ppq_ER
api.Ppq_ER.argtypes = [ctypes.c_double] * 9
api.Ppq_ER.restype = ctypes.c_double

result_fort = api.Ppq_ER(ctypes.c_double(Ep), ctypes.c_double(Eq),
                       ctypes.c_double(F0), ctypes.c_double(eps),
                       ctypes.c_double(V), ctypes.c_double(p0), ctypes.c_double(p10),
                       ctypes.c_double(q0), ctypes.c_double(q10))
(result_py, _), _, _ = ppq.Ppq_inspect_ER(Ep, Eq, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10, er_step=0.01)
print('back in python after running function Ppq_ER')
print('The fortran result is ', result_fort)
print('The python result is ', result_py)
check("Ppq_ER", result_fort, result_py, rtol=1e-4, atol=1e-12)

# Set argtypes and restype for the function Ppq_vector_NR
api.Ppq_vector_NR.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # Ep_arr
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),  # Eq_arr
    ctypes.c_int,              # n
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double,  # scalar params
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")   # res_arr (output)
]
api.Ppq_vector_NR.restype = None

# Inputs
Eq_arr = np.array([100.0, 100.0, 100.0], dtype=np.float64)
Ep_arr = np.array([347.0, 346.0, 348.0], dtype=np.float64)
n = Ep_arr.size

# Output array
res_arr = np.empty(n, dtype=np.float64)

# Call the Fortran vectorized function
api.Ppq_vector_NR(Ep_arr, Eq_arr, n, k, Z, F0, eps, V, p0, p10, q0, q10, res_arr)

# Set argtypes and restype for the function Ppq_vector_ER
api.Ppq_vector_ER.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),      # Ep_arr
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),      # Eq_arr
    ctypes.c_int,                                                        # n
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # scalar params
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")       # res_arr (output)
]
api.Ppq_vector_ER.restype = None

# Inputs
Eq_arr = np.array([100.0, 100.0, 100.0], dtype=np.float64)
Ep_arr = np.array([347.0, 346.0, 348.0], dtype=np.float64)
n = Ep_arr.size

# Output array
res_arr = np.empty(n, dtype=np.float64)

# Call the Fortran vectorized function
api.Ppq_vector_ER(Ep_arr, Eq_arr, n, F0, eps, V, p0, p10, q0, q10, res_arr)

print("Vectorized result:", res_arr)

# Package version
api.PpqFort_version.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3
api.PpqFort_version.restype = None
vmaj, vmin, vpatch = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
api.PpqFort_version(ctypes.byref(vmaj), ctypes.byref(vmin), ctypes.byref(vpatch))
print("Fortran version: ", (vmaj.value, vmin.value, vpatch.value))

print("Parameters Are ####################")
print("Er: ", Er)
print("Ep: ", Ep)
print("Eq: ", Eq)
print("k: ", k)
print("Z: ", Z)
print("F0: ", F0)
print("eps: ", eps)
print("V: ", V)
print("p0: ", p0)
print("p10: ", p10)
print("q0: ", q0)
print("q10: ", q10)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
