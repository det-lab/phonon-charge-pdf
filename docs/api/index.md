# Python API reference

Everything below `phonon_charge_pdf` itself is called the same way: `import phonon_charge_pdf
as pdf`, then `pdf.<name>`. `goodness_of_fit` and `erecoil_fit` are the two exceptions —
namespaced submodules (`pdf.goodness_of_fit.<name>`, `phonon_charge_pdf.erecoil_fit.<name>`),
not flattened to the top level, since each is a distinct workflow step from PDF
evaluation/event generation. For a usage walkthrough rather than a bare signature list, see
[Fitting](../fitting.md) and [Using MCMC to get errors](../mcmc.md) (`Ppq_compiled`),
[Generating simulated events](../python-events.md) (`generate_events` and friends), or
[Testing goodness of fit](../goodness-of-fit.md) (`goodness_of_fit.run_chisquare_test` and
friends, including `goodness_of_fit.make_Ppq_densities_NR`/`ER`, which don't have a page of
their own since nothing outside goodness-of-fit testing needs them). For
`Ppq_inspect_NR`/`ER`, see [Inspecting the integrand](../inspecting-the-integrand.md).

`Ppq_compiled`'s `PErecoil_fit_NR`/`PErecoil_fit_ER` arguments are required and take a
`phonon_charge_pdf.erecoil_fit.PErecoilFit` — either one you fit yourself
(`fit_sum_of_exponentials`, checked against real events with `check_fit_against_events`), or
the two ready-made closed-form constants, `CF252_BIEXPONENTIAL_NR`/`CF252_BIEXPONENTIAL_ER`,
used throughout the examples on this site. See
`adr/0012-perecoil-handles-register-a-fit-not-a-table.md` for why a handle registers a fit
and not an arbitrary table.

::: phonon_charge_pdf
    options:
      members:
        - Ppq_compiled
        - generate_events
        - Y
        - Nbar_NR
        - Nbar_ER
        - sigp
        - sigq
        - PErecoil_NR
        - PErecoil_ER
        - Ppq_inspect_NR
        - Ppq_inspect_ER

::: phonon_charge_pdf.goodness_of_fit
    options:
      heading_level: 3
      members:
        - run_chisquare_test
        - expected_counts_from_pdf
        - debug_bin
        - make_ridge_breakpoints_NR
        - make_ridge_breakpoints_ER
        - make_Ppq_densities_NR
        - make_Ppq_densities_ER

::: phonon_charge_pdf.erecoil_fit
    options:
      heading_level: 3
      members:
        - PErecoilFit
        - fit_sum_of_exponentials
        - check_fit_against_events
        - CF252_BIEXPONENTIAL_NR
        - CF252_BIEXPONENTIAL_ER
