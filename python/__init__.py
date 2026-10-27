"""Neganov-Luke phonon / ionization-charge band PDF and event simulator for
cryogenic detector nuclear-recoil (NR) and electron-recoil (ER) bands.

See the "Two ways to use this" page for which functions need the compiled
Fortran library (usually via a published container) and which are pure
Python.
"""

from .event_generator import generate_ER_events, generate_events, generate_NR_events
from .Ppq_fort import Ppq_compiled
from .Ppq_py import __all__ as _Ppq_py_all
from .Ppq_py import *  # noqa: F401,F403 -- Y, Nbar, sigp, sigq, PEr_NR, PEr_ER, ... ; see Ppq_py.__all__

__all__ = ["Ppq_compiled", "generate_events", "generate_NR_events", "generate_ER_events"] + _Ppq_py_all

try:
    from importlib.metadata import PackageNotFoundError, version
    # importlib.metadata needs the PyPI distribution name
    # ("phonon-charge-pdf", hyphenated by PyPI convention), not this
    # package's import name (phonon_charge_pdf, underscored) -- so it has
    # to be hardcoded here rather than derived from __name__.
    __version__ = version("phonon-charge-pdf")
except PackageNotFoundError:
    __version__ = "0+unknown"
