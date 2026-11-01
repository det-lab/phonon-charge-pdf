"""phonon_charge_pdf.erecoil_table: a tabulated true-recoil-energy spectrum
(PErecoil), PCHIP-interpolated, as a drop-in replacement for Ppq_py.PErecoil_NR/
PErecoil_ER's closed-form biexponentials when the real spectrum is only known
from counts at a set of Erecoil values.

One file format only: HDF5, with detector/band/provenance metadata --
mirrors normgrid.py's own table format (format_version, band, etc.) rather
than a second, parallel bare-numbers text format. h5py is imported lazily
(_h5py() below, only from save_erecoil_table/PErecoilTable.from_hdf5), the
same way normgrid.py's own _h5py() is, so importing this module -- or using
PErecoilTable built from in-memory arrays -- never requires it; that keeps
Ppq_py.py's "pure numpy/scipy, no extra dependency" promise intact even
though this module's one real file format needs h5py to read or write.

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
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

_PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
_CLI_DIR = os.path.join(_PYTHON_DIR, "cli")

# Table format version: bump this (and add a migration, mirroring
# normgrid.migrate_table) if the HDF5 schema below ever changes
# incompatibly. A file newer than this code understands fails loudly at
# load time instead of being silently misread.
TABLE_FORMAT_VERSION = 1

__all__ = ["PErecoilTable", "save_erecoil_table"]


def _h5py():
    import h5py
    return h5py


def save_erecoil_table(path, erecoil, counts, *, detector, band, calibration_source="",
                       source_file="", provenance=""):
    """Write a PErecoilTable-compatible HDF5 file: erecoil/counts datasets
    plus format_version/detector/band/calibration_source/source_file/
    provenance/created attrs. The one writer for the one file format
    PErecoilTable.from_hdf5 reads -- used by test fixtures and any real
    per-detector extraction script alike, so there is exactly one schema
    to keep in sync, not two.

    detector : str
        Short identifier (e.g. "X17") -- becomes part of this table's own
        metadata, checked against what's expected wherever this is loaded
        for a specific detector/band.
    band : "NR" or "ER"
    calibration_source : str
        What produced the raw events this table was built from (e.g.
        "Cf-252", "Ba-133") -- structured, short, meant to be read back
        programmatically (unlike provenance below). A detector can have
        more than one table for the same band from different calibration
        sources (e.g. a Cf-252-derived and a Ba-133-derived ER table), so
        this -- not just detector/band -- is what keeps them apart; see
        this project's own filename convention,
        erecoil_tables/<detector>_<band>_<short source tag>.h5.
    source_file : str
        The raw input data file this table was built from (e.g.
        "Cf_ptqi_402.csv") -- so a reader can find the original data
        without parsing provenance's free text.
    provenance : str
        Free text: anything else about where this table's numbers came
        from (e.g. an NRFano notebook path/commit, the cut/estimator
        used) -- not checked by this code, but the only place this is
        recorded, so write something a future reader can trace back from.
    """
    h5py = _h5py()
    if band not in ("NR", "ER"):
        raise ValueError(f"band must be 'NR' or 'ER', got {band!r}")
    erecoil = np.asarray(erecoil, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    with h5py.File(path, "w") as f:
        f.create_dataset("erecoil", data=erecoil)
        f.create_dataset("counts", data=counts)
        f.attrs["format_version"] = TABLE_FORMAT_VERSION
        f.attrs["detector"] = detector
        f.attrs["band"] = band
        f.attrs["calibration_source"] = calibration_source
        f.attrs["source_file"] = source_file
        f.attrs["provenance"] = provenance
        f.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")


class PErecoilTable:
    """A true-recoil-energy PDF built from tabulated (Erecoil, counts):
    monotone cubic (PCHIP) interpolated between the tabulated points and
    normalized to integrate to 1 over the table's own range. Call with a
    single Erecoil (keV):

        p_erecoil = PErecoilTable.from_hdf5("erecoil_tables/X17_NR.h5")
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

    def __init__(self, erecoil, counts, *, detector="", band="", calibration_source="",
                 source_file="", provenance="", source=""):
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
        self.detector = detector
        self.band = band
        self.calibration_source = calibration_source
        self.source_file = source_file
        self.provenance = provenance
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
    def from_hdf5(cls, path):
        h5py = _h5py()
        with h5py.File(path, "r") as f:
            version = f.attrs.get("format_version", 1)
            if version > TABLE_FORMAT_VERSION:
                raise ValueError(f"{path}: table format_version={version}, this code reads "
                                 f"{TABLE_FORMAT_VERSION} or older -- update phonon_charge_pdf")
            erecoil = f["erecoil"][:]
            counts = f["counts"][:]
            detector = str(f.attrs.get("detector", ""))
            band = str(f.attrs.get("band", ""))
            calibration_source = str(f.attrs.get("calibration_source", ""))
            source_file = str(f.attrs.get("source_file", ""))
            provenance = str(f.attrs.get("provenance", ""))
        return cls(erecoil, counts, detector=detector, band=band, calibration_source=calibration_source,
                   source_file=source_file, provenance=provenance, source=str(path))

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
