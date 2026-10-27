# Miscellaneous notes

Small facts that don't yet have a better home in these docs.

## Importing never touches the compiled library

`import phonon_charge_pdf` never touches the compiled Fortran library -- only
constructing a `Ppq_compiled` does, and that fails there, at construction, rather
than on the first likelihood call deep into a fit or MCMC run. `event_generator.py`
and `Ppq_py.py` have no Fortran dependency at all, so they work in any environment
this package installs into, container or not.
