# Python API reference

Everything below `phonon_charge_pdf` itself is called the same way: `import phonon_charge_pdf
as pdf`, then `pdf.<name>`. `goodness_of_fit` is the one exception — it's a namespaced
submodule (`pdf.goodness_of_fit.<name>`), not flattened to the top level, since
goodness-of-fit testing is a distinct workflow step from PDF evaluation/event generation.
For a usage walkthrough rather than a bare signature list, see [Fitting](../fitting.md) and
[Using MCMC to get errors](../mcmc.md) (`Ppq_compiled`),
[Generating simulated events](../python-events.md) (`generate_events` and friends), or
[Testing goodness of fit](../goodness-of-fit.md) (`goodness_of_fit.run_chisquare_test` and
friends, including `goodness_of_fit.make_Ppq_densities_NR`/`ER`, which don't have a page of
their own since nothing outside goodness-of-fit testing needs them). For
`Ppq_inspect_NR`/`ER`, see [Inspecting the integrand](../inspecting-the-integrand.md).

::: phonon_charge_pdf
    options:
      members:
        - Ppq_compiled
        - generate_events
        - Y
        - Nbar
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
        - make_ridge_breakpoints
        - make_Ppq_densities_NR
        - make_Ppq_densities_ER
