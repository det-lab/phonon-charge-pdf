"""phonon_charge_pdf: Neganov-Luke phonon / ionization-charge band PDF and
event simulator for cryogenic detector nuclear-recoil (NR) and
electron-recoil (ER) bands.

Three public entrypoints -- see their own docstrings:
* Ppq and generate_events/generate_NR_events/generate_ER_events
  (Ppq_fort.py, event_generator.py).
* Ppq_py.py's functions (Y, Nbar, sigp, sigq, PEr_NR, PEr_ER, aN, bN, cN,
  N_exponent, pq_given_Er_NR, pq_given_Er_ER, Ppq_inspect_NR(_vec),
  Ppq_inspect_ER(_vec), integrate_inspect): the pure-Python reference
  implementation the other two are cross-checked against -- also
  re-exported here, but see Ppq_py.py's own docstring before using
  these for anything beyond a cross-check: they're a slow, brute-force
  grid scan, not meant for a fit/MCMC loop.

Everything under internal/ and cli/ is either what the above are built
from or infrastructure for maintaining this library, and carries no
stability promise.

Ppq has a hard, unconditional dependency on the compiled Fortran
library; event_generator.py and Ppq_py.py have none. Importing this
package never touches the library -- Ppq only requires it be loadable
(see LD_LIBRARY_PATH/DYLD_LIBRARY_PATH in internal/_Ppq_fort_bindings.py)
once you construct one, and fails there, not on the first likelihood call
deep into a fit.
"""

from .event_generator import generate_ER_events, generate_events, generate_NR_events
from .Ppq_fort import Ppq
from .Ppq_py import __all__ as _Ppq_py_all
from .Ppq_py import *  # noqa: F401,F403 -- Y, Nbar, sigp, sigq, PEr_NR, PEr_ER, ... ; see Ppq_py.__all__

__all__ = ["Ppq", "generate_events", "generate_NR_events", "generate_ER_events"] + _Ppq_py_all

try:
    from importlib.metadata import PackageNotFoundError, version
    # importlib.metadata needs the PyPI distribution name
    # ("phonon-charge-pdf", hyphenated by PyPI convention), not this
    # package's import name (phonon_charge_pdf, underscored) -- so it has
    # to be hardcoded here rather than derived from __name__.
    __version__ = version("phonon-charge-pdf")
except PackageNotFoundError:
    __version__ = "0+unknown"
