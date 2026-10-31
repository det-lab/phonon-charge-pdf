"""Container smoke test: confirm a pulled/built image can actually do a real
region-integral evaluation for both bands, before committing a cluster batch
or an HPC pull to it.

Deliberately goes only through normgrid.py's own public interface
(make_grid_spec/coords_at/physical_params/_evaluate), never importing the
Fortran-binding module directly by name -- that way a future internal
rename (like Ppq_region_NR/Ppq_region_ER's own naming, or the binding
module's own name) doesn't require editing this script, only normgrid.py's
own interface has to keep working. normgrid.py (baked into whichever
container this runs against) reaches the binding module under whatever
name was correct for *its own* version internally -- this script never
needs to know what that name is, since it never imports it directly.

Shared between slurm/pull_container.job (pulls a published image from GHCR)
and slurm/verify_local_image.sh (builds and checks a local image) -- run
inside the container via `python - < container_smoke_test.py`, not copied
in, so it always reflects the version on the host, not whatever was baked
into the image at build time.
"""

import sys

sys.path.insert(0, "/app/python/internal")
sys.path.insert(0, "/app/python/cli")

import normgrid

for band in ("NR", "ER"):
    spec = normgrid.make_grid_spec(band, {a: 1 for a in normgrid.BAND_AXES[band]})
    params = normgrid.physical_params(spec, normgrid.coords_at(spec, 0))
    val = normgrid._evaluate(spec, params, epsrel=1e-4)
    assert val == val and val > 0, f"{band} region integral looks wrong: {val}"
    print(f"{band} region integral OK: {val:.6g}")

print("normgrid axes", normgrid.AXES)
