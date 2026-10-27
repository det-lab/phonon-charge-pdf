"""Neganov-Luke phonon / ionization-charge band PDF and event simulator for
cryogenic detector nuclear-recoil (NR) and electron-recoil (ER) bands.

See the "Two ways to use this" page for which functions need the compiled
Fortran library (usually via a published container) and which are pure
Python.
"""

from . import goodness_of_fit
from .event_generator import generate_events, generate_events_ER, generate_events_NR
from .Ppq_fort import Ppq_compiled
from .Ppq_py import __all__ as _Ppq_py_all
from .Ppq_py import *  # noqa: F401,F403 -- Y, Nbar, sigp, sigq, PEr_NR, PEr_ER, ... ; see Ppq_py.__all__

# run_chisquare_test/expected_counts_from_pdf/debug_bin/make_ridge_breakpoints
# and make_Ppq_densities_NR/ER are deliberately NOT flattened here -- call
# them as goodness_of_fit.<name>, not pdf.<name> (see that module's docstring).
__all__ = (["Ppq_compiled", "generate_events", "generate_events_NR", "generate_events_ER"] + _Ppq_py_all
           + ["goodness_of_fit"])

try:
    from importlib.metadata import PackageNotFoundError, version
    # importlib.metadata needs the PyPI distribution name
    # ("phonon-charge-pdf", hyphenated by PyPI convention), not this
    # package's import name (phonon_charge_pdf, underscored) -- so it has
    # to be hardcoded here rather than derived from __name__.
    __version__ = version("phonon-charge-pdf")
except PackageNotFoundError:
    __version__ = "0+unknown"
