"""phonon_charge_pdf: Neganov-Luke phonon / ionization-charge band PDF and
event simulator for cryogenic detector nuclear-recoil (NR) and
electron-recoil (ER) bands.

Three public entrypoints -- see their own docstrings:
* PpqPDF and generate_events/generate_NR_events/generate_ER_events
  (ppq_pdf.py, event_generator.py).
* ppq_dist.py's functions (Y, Nbar, sigp, sigq, PErN, PErG, aN, bN, cN,
  PpqExp, PpqFullN, PpqFullG, PpqN_safe_inspect(_vec),
  PpqG_safe_inspect(_vec), integrate_g_safe_inspect, consecutive,
  gaussian): the pure-Python reference implementation the other two are
  cross-checked against -- also re-exported here, but see ppq_dist.py's
  own docstring before using these for anything beyond a cross-check:
  they're a slow, brute-force grid scan, not meant for a fit/MCMC loop.

Everything under internal/ and cli/ is either what the above are built
from or infrastructure for maintaining this library, and carries no
stability promise.

PpqPDF has a hard, unconditional dependency on the compiled Fortran
library; event_generator.py and ppq_dist.py have none. Importing this
package never touches the library -- PpqPDF only requires it be loadable
(see LD_LIBRARY_PATH/DYLD_LIBRARY_PATH in internal/_ppqfort_bindings.py)
once you construct one, and fails there, not on the first likelihood call
deep into a fit.
"""

from .event_generator import generate_ER_events, generate_events, generate_NR_events
from .ppq_dist import __all__ as _ppq_dist_all
from .ppq_dist import *  # noqa: F401,F403 -- Y, Nbar, sigp, sigq, PErN, PErG, ... ; see ppq_dist.__all__
from .ppq_pdf import PpqPDF

__all__ = ["PpqPDF", "generate_events", "generate_NR_events", "generate_ER_events"] + _ppq_dist_all

try:
    from importlib.metadata import PackageNotFoundError, version
    # importlib.metadata needs the PyPI distribution name
    # ("phonon-charge-pdf", hyphenated by PyPI convention), not this
    # package's import name (phonon_charge_pdf, underscored) -- so it has
    # to be hardcoded here rather than derived from __name__.
    __version__ = version("phonon-charge-pdf")
except PackageNotFoundError:
    __version__ = "0+unknown"
