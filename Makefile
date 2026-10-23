# Convenience wrappers around the fpm build/test/install commands
# documented in the README ("Building from source").  Each target
# regenerates src/version.f90.inc first (not committed -- the build
# fails loudly without it rather than silently reporting a stale
# version).  The manual commands in the README remain the source of
# truth for the exact flags; this just saves retyping them.

GFORTRAN_FLAGS := -march=native -fopenmp -ftree-parallelize-loops=4 -fcoarray=single -fPIC
IFX_FLAGS      := -fpp -O3 -qopenmp -DHAVE_MULTI_IMAGE_SUPPORT=0
FLANG_FLAGS    := -O3 -fopenmp -fdo-concurrent-to-openmp=host

.PHONY: version test-gfortran test-ifx test-flang \
        install-gfortran install-ifx install-flang clean

version:
	python scripts/generate_version_include.py

test-gfortran: version
	fpm test --compiler gfortran --profile release --flag "$(GFORTRAN_FLAGS)"

test-ifx: version
	fpm test --compiler ifx --profile release --flag "$(IFX_FLAGS)"

test-flang: version
	fpm test --compiler flang --profile release --flag "$(FLANG_FLAGS)"

# Builds lib/libphonon_charge_pdf.so for the python ctypes interface.
install-gfortran: version
	fpm install --prefix=. --compiler gfortran --profile release --flag "$(GFORTRAN_FLAGS)"

install-ifx: version
	FPM_LDFLAGS="-liomp5" fpm install --prefix=. --compiler ifx --profile release --flag "$(IFX_FLAGS)"

install-flang: version
	FPM_LDFLAGS="-fopenmp" fpm install --prefix=. --compiler flang --profile release --flag "$(FLANG_FLAGS)"

clean:
	rm -rf build lib src/version.f90.inc
