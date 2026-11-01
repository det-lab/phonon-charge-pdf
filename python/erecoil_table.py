"""phonon_charge_pdf.erecoil_table: a tabulated true-recoil-energy spectrum
(PErecoil), PCHIP-interpolated, as a drop-in replacement for Ppq_py.PErecoil_NR/
PErecoil_ER's closed-form biexponentials when the real spectrum is only known
from counts at a set of Erecoil values.

Reuses normgrid.OutOfBoxError (python/cli/normgrid.py) rather than inventing a
parallel exception type: same "never extrapolate, raise on a query outside the
table" contract NormInterpolator already uses for the (Ep, Eq) normalization
tables, imported the same lazy way Ppq_fort.py does (so this module, like
Ppq_py.py, has no hard dependency on cli/ or the compiled library at import
time).

Prototype stage: this is validated against Ppq_py.PErecoil_NR/PErecoil_ER
(see test/python/test_erecoil_table.py) but not yet wired into
pq_given_Erecoil_NR/ER's quad integration -- that needs the integration
window clipped to erecoil_range first, tracked as follow-up work.
"""

import os
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

_PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
_CLI_DIR = os.path.join(_PYTHON_DIR, "cli")

__all__ = ["PErecoilTable", "load_erecoil_table"]


def load_erecoil_table(path):
    """Load a two-column (Erecoil_keV, counts) text table -- whitespace or
    comma separated, blank lines and lines starting with '#' ignored.
    Returns (erecoil, counts), sorted ascending by Erecoil."""
    data = np.loadtxt(path, comments="#", delimiter=None, ndmin=2)
    erecoil, counts = data[:, 0], data[:, 1]
    order = np.argsort(erecoil)
    return erecoil[order], counts[order]


class PErecoilTable:
    """A true-recoil-energy PDF built from tabulated (Erecoil, counts):
    monotone cubic (PCHIP) interpolated between the tabulated points and
    normalized to integrate to 1 over the table's own range. Call with a
    single Erecoil (keV):

        p_erecoil = PErecoilTable(erecoil, counts)
        p_erecoil(12.3)

    Erecoil < 0 returns 0 -- matches PErecoil_NR/PErecoil_ER's own convention,
    since the integration/peak-scan machinery that calls this can transiently
    probe just below zero. Any other Erecoil outside the table's own range
    (erecoil_range) raises normgrid.OutOfBoxError rather than ever
    extrapolating: the table's range is data, not a modeling choice, so
    there's nothing honest to extrapolate with. A caller integrating over
    Erecoil (quad, a peak scan) must clip its window to erecoil_range first --
    this never extrapolates to make that unnecessary.
    """

    def __init__(self, erecoil, counts, *, source=""):
        erecoil = np.asarray(erecoil, dtype=float)
        counts = np.asarray(counts, dtype=float)
        if erecoil.ndim != 1 or counts.shape != erecoil.shape:
            raise ValueError("erecoil and counts must be 1-D arrays of the same length")
        if erecoil.size < 2:
            raise ValueError("need at least 2 table points to interpolate")
        if np.any(np.diff(erecoil) <= 0):
            raise ValueError("erecoil must be strictly increasing (sort/dedupe first)")
        if np.any(counts < 0):
            raise ValueError("counts must be non-negative")
        self.source = source
        self._lo, self._hi = float(erecoil[0]), float(erecoil[-1])
        self._interp = PchipInterpolator(erecoil, counts, extrapolate=False)
        # The exact integral of the piecewise cubic Hermite spline itself --
        # not scipy.integrate.quad, which can underestimate this badly: its
        # adaptive sampling can miss a feature much narrower than the table's
        # full range (e.g. PErecoil_ER's ~0.17 keV fast component against a
        # multi-hundred-keV table) before it ever decides it has converged.
        raw_norm = float(self._interp.integrate(self._lo, self._hi))
        if not raw_norm > 0:
            raise ValueError(f"table integrates to {raw_norm!r}; cannot normalize")
        self._norm = raw_norm

    @classmethod
    def from_file(cls, path):
        erecoil, counts = load_erecoil_table(path)
        return cls(erecoil, counts, source=str(path))

    @property
    def erecoil_range(self):
        """(lo, hi): the table's own Erecoil range in keV -- what a caller's
        integration window must be clipped to before this is ever called."""
        return self._lo, self._hi

    @property
    def fortran_table(self):
        """(x, c0, c1, c2, c3): breakpoints and per-interval power-basis
        cubic coefficients -- scipy's own PPoly convention (value at Erecoil
        in [x[i], x[i+1]) is ((c0[i]*dx + c1[i])*dx + c2[i])*dx + c3[i],
        dx = Erecoil - x[i]), already divided by this table's own
        normalization. Exactly what Ppq_fort.py needs to load this table
        into the compiled library via PErecoil_set_table_NR/ER, whose
        eval_cubic_table (src/Ppq_fort_s.f90) evaluates this identical form."""
        c = self._interp.c / self._norm
        return self._interp.x, c[0], c[1], c[2], c[3]

    def __call__(self, Erecoil):
        Erecoil = float(Erecoil)
        if Erecoil < 0:
            return 0.0
        sys.path.insert(0, _CLI_DIR)
        from normgrid import OutOfBoxError
        tol = 1e-12 * (self._hi - self._lo)
        if not (self._lo - tol <= Erecoil <= self._hi + tol):
            raise OutOfBoxError(
                f"Erecoil={Erecoil!r} outside the range [{self._lo:.6g}, {self._hi:.6g}] of the "
                f"PErecoil table {self.source or '(in memory)'}\n  The table covers only the "
                "recoil-energy range it was built from, so a query past it means the integration "
                "window wasn't clipped to erecoil_range before quad/the peak scan ran, or a NaN/bug.",
                axis="Erecoil", value=Erecoil, box=(self._lo, self._hi), source=self.source)
        clipped = min(max(Erecoil, self._lo), self._hi)
        return float(self._interp(clipped)) / self._norm
