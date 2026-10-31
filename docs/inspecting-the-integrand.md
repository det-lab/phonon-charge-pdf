# Inspecting the integrand

`Ppq_inspect_NR`/`Ppq_inspect_ER` are a slower, Python-only way to get `Ppq_NR`/`Ppq_ER`
values directly: instead of the Fortran side's analytic peak location and windowed
quadrature, they scan the *whole* plausible Erecoil range on a uniform grid, locate the
integrand's peak(s) on that scan, and integrate each with `scipy.integrate.quad` — more
exhaustive, and much slower:

```python
import phonon_charge_pdf as pdf

result = pdf.Ppq_inspect_NR(
    Ep=10.0, Eq=5.0,
    k=0.18, Z=32.0, F0=0.122, eps=3.0e-3, V=3.0,
    p0=0.06421907, p10=0.48998486, q0=0.23718488, q10=0.27093151,
    erecoil_step=0.05,   # Erecoil scan step, keV -- see "choosing erecoil_step" below
)
Ppq_NR, quad_error = result["value"], result["value_err"]
erecoil_scan, integrand_scan = result["erecoil_scan_grid"], result["erecoil_scan_integrand"]
peaks = result["peaks"]
```

!!! warning "Deliberately the slow path"
    Never use this in a fit/MCMC inner loop. Use [`Ppq_compiled`](fitting.md) for a
    fit/MCMC; use this to independently verify `Ppq_compiled`/`event_generator.py` agree with
    the underlying physics, or when you want the scan grid and per-peak width/location
    diagnostics (`peaks` above — one dict per located peak) that `Ppq_compiled` doesn't expose.

## Plotting the integrand

`erecoil_scan_grid`/`erecoil_scan_integrand` are the raw scan the peak search ran over, and `peaks`
has each located peak's center (`Erecoil_peak`), width, and integration window (`minx`/`maxx`) —
plotting them together is a quick way to confirm the peak search found the right thing, and
that `erecoil_step` actually resolves it:

```python
import matplotlib.pyplot as plt

fig, (ax_full, ax_peak) = plt.subplots(1, 2, figsize=(10, 4))

ax_full.plot(erecoil_scan, integrand_scan, ".", ms=2)
for peak in peaks:
    ax_full.axvspan(peak["minx"], peak["maxx"], color="C1", alpha=0.2, label="integration window")
ax_full.set_yscale("log")
ax_full.set_xlabel("Erecoil (keV)")
ax_full.set_ylabel("integrand (unnormalized)")
ax_full.set_title("full Erecoil scan")
ax_full.legend(fontsize=8)

peak = peaks[0]
zoom = (erecoil_scan > peak["Erecoil_peak"] - 5 * peak["width"]) & (erecoil_scan < peak["Erecoil_peak"] + 5 * peak["width"])
ax_peak.plot(erecoil_scan[zoom], integrand_scan[zoom], ".-", ms=4)
ax_peak.axvline(peak["Erecoil_peak"], color="C1", linestyle="--", label="located peak (Erecoil_peak)")
ax_peak.set_xlabel("Erecoil (keV)")
ax_peak.set_title("zoomed near the peak")
ax_peak.legend(fontsize=8)

fig.suptitle("Ppq_NR(Ep=10.0, Eq=5.0) integrand vs Erecoil")
fig.tight_layout()
fig.savefig("ppq_nr_integrand.png", dpi=150)
```

![The Ppq_NR integrand over Erecoil at Ep=10.0, Eq=5.0: a log-scale full scan on the left, and a
linear-scale zoom around the located peak on the right.](images/ppq-nr-integrand-example.png)

The left panel's log scale shows just how sharply this integrand falls off away from the
peak — down by hundreds of orders of magnitude within a few keV — which is why the scan has
to cover the *whole* plausible Erecoil range rather than guessing a narrower one, and why the
Fortran side's windowed quadrature only integrates near the peak instead of everywhere. The
right panel confirms `erecoil_step=0.05` comfortably resolves this peak's ~0.24 keV width.

## Choosing `erecoil_step`

`erecoil_step` (the Erecoil scan step, in keV) has to be finer than the narrowest peak you expect: ER
peaks can be as narrow as the zero-energy ionization resolution `q0` (~0.06 keV for typical
CDMS parameters), so e.g. `erecoil_step=0.005` for `Ppq_inspect_ER` is finer than the 0.01 keV
grid the Fortran `Ppq_ER` itself integrates on.
