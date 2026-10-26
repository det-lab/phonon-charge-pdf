"""Container smoke test: confirm a pulled/built image can actually do a real
region-integral evaluation for both bands, before committing a cluster batch
or an HPC pull to it.

Deliberately goes through normgrid.py's own public interface (make_grid_spec/
coords_at/physical_params/_evaluate) rather than checking for a specific
Fortran-binding symbol name directly -- that way a future internal rename
(like Ppq_region_NR/Ppq_region_ER's own naming) doesn't require editing this
script, only normgrid.py's own interface has to keep working.

Shared between slurm/pull_container.job (pulls a published image from GHCR)
and slurm/verify_local_image.sh (builds and checks a local image) -- run
inside the container via `python - < container_smoke_test.py`, not copied
in, so it always reflects the version on the host, not whatever was baked
into the image at build time.
"""

import sys

sys.path.insert(0, "/app/python/internal")
sys.path.insert(0, "/app/python/cli")

import _Ppq_fort_bindings
import normgrid

print("library version", ".".join(map(str, _Ppq_fort_bindings.version())))

for band in ("NR", "ER"):
    spec = normgrid.make_grid_spec(band, {a: 1 for a in normgrid.BAND_AXES[band]})
    params = normgrid.physical_params(spec, normgrid.coords_at(spec, 0))
    val = normgrid._evaluate(spec, params, epsrel=1e-4)
    assert val == val and val > 0, f"{band} region integral looks wrong: {val}"
    print(f"{band} region integral OK: {val:.6g}")

print("normgrid axes", normgrid.AXES)
