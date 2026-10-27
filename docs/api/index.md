# API reference

Everything here is called the same way: `import phonon_charge_pdf as pdf`, then `pdf.<name>`.
For a usage walkthrough rather than a bare signature list, see
[Fortran-backed PDF](../ppq-fort.md) (`Ppq_compiled`),
[Generating simulated events](../python-events.md) (`generate_events` and friends), or
[Resolution/yield model](../ppq-py.md) (`Y`, `Nbar`, `sigp`, `sigq`, and friends).

::: phonon_charge_pdf
    options:
      members:
        - Ppq_compiled
        - generate_events
        - generate_NR_events
        - generate_ER_events
        - Y
        - Nbar
        - sigp
        - sigq
        - PEr_NR
        - PEr_ER
        - Ppq_inspect_NR
        - Ppq_inspect_ER
