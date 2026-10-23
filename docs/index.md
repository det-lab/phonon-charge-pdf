# phonon-charge-pdf

This code is useful for dark matter searches where detector output is Ep (total phonon
energy) and Eq (total charge energy). The point of the code is to provide the probability
of an (Ep, Eq) pair given a set of detector parameters for both electron recoils (`PpqG`,
where the G is for gamma because gammas are the cause of most electron recoils) and
neutron recoils (`PpqN`, where the N is for neutron).

## Quickstart

```bash
pip install phonon-charge-pdf
```

```python
import phonon_charge_pdf as pdf

events = pdf.generate_events(
    target_size=1000, ER_frac=0.6, seed_value=1,
    ep_min=2.5, ep_max=350.0, eq_min=0.75, eq_max=200.0,
    k=0.18, Z=32.0, F0_ER=0.2, F0_NR=0.122, eps=3.0e-3, V=3.0,
    p0=0.06421907, p10=0.48998486, q0=0.23718488, q10=0.27093151,
    PNa=0.53693208, PNb=6.41515782, PNd=23.71789286,
    PGa=0.573211975, PGb=0.169520023, PGd=279.552394,
)
Ep, Eq = events["EP"].to_numpy(), events["EQ"].to_numpy()
```

This works with **no Fortran build** — pure Python. For a fit/MCMC you need the compiled
library and `PpqPDF`; see [Two ways to use this](two-ways.md).

Published on PyPI as `phonon-charge-pdf`; `import phonon_charge_pdf` (hyphen becomes
underscore, the usual PyPI convention).

## Citing

Citation metadata for the code is in `CITATION.cff` (GitHub's "Cite this repository"
button); each release is archived on Zenodo. The precomputed normalization tables are a
separate dataset, [doi:10.5281/zenodo.23048215](https://doi.org/10.5281/zenodo.23048215)
(all versions; one version per added fit window), which `python/cli/normtables.py` fetches
by name.

## Where to go next

| I want to... | See |
|---|---|
| Generate simulated (Ep, Eq) events, no Fortran build needed | [Generating simulated events](python-events.md) |
| Use the resolution/yield model functions directly (`sigp`, `sigq`, `Y`, `Nbar`, ...) | [Resolution/yield model (ppq_dist)](ppq-dist.md) |
| Get normalized PDF values or a likelihood for a fit/MCMC | [Fortran-backed PDF (PpqPDF)](fortran-pdf.md) |
| Run a fit/MCMC fast, without recomputing the normalization integral every step | [Precomputed normalization tables](normalization-tables.md) |
| Build this project from source, or build/run its containers | [Building from source](building-from-source.md) |
| Check that the physics is actually correct | [Verification and validation](testing.md) |
