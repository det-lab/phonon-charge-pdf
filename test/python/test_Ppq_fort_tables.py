"""
Tests for Ppq_compiled's MCMC setup checks with normalization tables: prior_bounds
(flat, F0_NR/F0_ER for the one band-specific parameter), check_points,
table_bounds, and table names that are neither files nor registered.  Small
in-memory tables stand in for
real ones, so no data files are needed -- but importing Ppq_fort loads the
Fortran library, so run it where that is built (e.g. in the container):

    LD_LIBRARY_PATH=lib python test/python/test_Ppq_fort_tables.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(REPO_ROOT, "python", "cli"))
import normgrid as ng
from Ppq_fort import Ppq_compiled

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail and not cond else ''}")
    if not cond:
        failures.append(name)


def error_of(exc, fn):
    try:
        fn()
    except exc as e:
        return str(e)
    return None


REGION = (2.5, 350.0, 0.75, 200.0)


def fake_table(band):
    """A 2-node-per-axis table over the band's default box (values are
    irrelevant here; only the box and metadata are checked)."""
    axes = ng.BAND_AXES[band]
    box = ng.default_box(band)
    nodes = {a: ng.chebyshev_lobatto(*box[a], 2) for a in axes}
    return ng.NormInterpolator(axes, nodes, np.ones([2] * len(axes)), band=band, region=REGION,
                               fixed=ng._base_spec("grid", band, REGION, ng.DEFAULT_FIXED, box, 1e-7, 1e-13)["fixed"])


nr, er = fake_table("NR"), fake_table("ER")
make = lambda **kw: Ppq_compiled(*REGION, [10.0], [5.0], Ppq_table_NR=nr, Ppq_table_ER=er, **kw)

shared = dict(V=(2.7, 3.3), p0=(0.2 * ng.P0_MEAN, 1.8 * ng.P0_MEAN), p10=(0.3, 0.6),
              q0=(0.2 * ng.Q0_MEAN, 1.8 * ng.Q0_MEAN), q10=(0.2, 0.4))
prior = dict(shared, k=(0.13, 0.22), F0_NR=(1e-5, 1.0), F0_ER=(0.15, 0.3))

make(prior_bounds=prior)
check("design prior with band-specific F0_NR/F0_ER passes both tables", True)
msg = error_of(ng.PriorNotCoveredError, lambda: make(prior_bounds=dict(prior, F0_ER=(0.05, 0.3))))
check("ER F0 below the ER box fails, naming the ER table", msg is not None and "ER table" in msg and "F0" in msg)
msg = error_of(ng.PriorNotCoveredError, lambda: make(prior_bounds=dict(prior, V=(2.5, 4.0))))
check("shared V too wide fails", msg is not None and "V: prior reaches" in msg)
check("prior_bounds without a table is an error",
      error_of(ValueError, lambda: Ppq_compiled(*REGION, [10.0], [5.0], prior_bounds=prior)) is not None)

pdf = make()
tb = pdf.table_bounds()
check("table_bounds: flat dict with F0_NR/F0_ER", "k" in tb and "F0_NR" in tb and "F0_ER" in tb
      and np.allclose(tb["F0_ER"], (0.1, 0.35)))
make(prior_bounds=tb)
check("table_bounds used directly as prior_bounds passes (no reshaping)", True)

rng = np.random.default_rng(1)
n = 32
walkers = {p: rng.uniform(*shared[p], n) for p in ("V", "p0")}
# shared["q0"] = (0.2, 1.8) * Q0_MEAN is the *design prior*'s range; the table's
# actual q0 box is narrower (capped at max q10 = 0.4, see table_bounds() above),
# so draw q0 from there directly rather than from shared["q0"] and clamping --
# clamping against q10 doesn't help when the q0 draw itself already exceeds 0.4.
# tb["q0"] is already the NR/ER intersection, so this is inside both tables' boxes.
walkers["q0"] = rng.uniform(*tb["q0"], n)
walkers["p10"] = rng.uniform(0.3, 0.6, n)
walkers["q10"] = np.maximum(rng.uniform(0.2, 0.4, n), walkers["q0"])
walkers["k"] = rng.uniform(0.13, 0.22, n)
walkers["F0_NR"] = rng.uniform(1e-5, 1.0, n)
walkers["F0_ER"] = rng.uniform(0.15, 0.3, n)
pdf.check_points(walkers)
check("walkers inside both tables pass", True)
walkers["F0_ER"][5] = 0.5
msg = error_of(ng.PriorNotCoveredError, lambda: pdf.check_points(walkers))
check("one walker outside the ER F0 range fails, naming it", msg is not None and "1 of 32" in msg and "point 5" in msg)

msg = error_of(KeyError, lambda: Ppq_compiled(*REGION, [10.0], [5.0], Ppq_table_NR="no_such_table"))
check("a table that is neither a file nor registered: KeyError", msg is not None and "registered" in msg)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
