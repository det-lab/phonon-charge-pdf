#!/bin/bash
# Build the LLVM image from the current local checkout (uncommitted changes
# included) and run container_smoke_test.py against it -- a fast way to
# confirm a container built from this exact tree actually works, without
# needing a cluster, a GHCR publish, or a cut release.
#
# Unlike slurm/pull_container.job (which pulls an already-published image
# by tag), this always builds fresh from whatever is on disk right now.
# Goes through `docker save` + `docker-archive://` rather than
# `docker-daemon://`: the latter needs Apptainer to reach the same Docker
# socket that built the image, which fails whenever the two run in separate
# VMs (e.g. Lima-based Apptainer alongside Docker Desktop on macOS) --
# a saved tarball sidesteps that regardless of how Docker/Apptainer happen
# to be wired up on a given machine.
#
#   ./slurm/verify_local_image.sh            # build + verify
#   ./slurm/verify_local_image.sh keep        # also keep the .sif afterward
#
# Requires Docker and Apptainer/Singularity locally. On macOS with Lima-based
# Apptainer, start the VM first if needed: limactl start apptainer
#
# Lima-hosted Apptainer notes (the macOS default -- see
# /opt/homebrew/bin/apptainer.lima): its `singularity`/`apptainer` shim
# only forwards APPTAINER_BINDPATH into the VM, so plain `export`s of
# APPTAINER_TMPDIR etc. are invisible to it; and the VM's home-directory
# mount (where $SCRATCH below lives) is read-only by default, so nothing
# the build itself writes -- its temp-extraction scratch space or its
# output .sif -- can live there either. Both are worked around below by
# routing the build through `lima` directly (it's just `limactl shell`,
# so env vars can be forwarded inline the same way the shim forwards
# APPTAINER_BINDPATH) and writing to /var/tmp, which is on the VM's own
# writable disk -- then `limactl copy` pulls the finished .sif back out
# to the host if `keep` was requested.

set -euo pipefail
cd "$(dirname "$0")/.."

TAG=${TAG:-pdf-local-verify}
# mktemp -d, then fixed basenames inside it: BSD mktemp (macOS) only
# substitutes a template's trailing Xs when nothing follows them, so
# "XXXXXX.tar"-style suffixed templates silently fail to substitute there
# (unlike GNU mktemp) -- a scratch directory sidesteps that difference.
# Lives under $HOME, not /tmp: Lima only mounts the home directory into
# its VM by default, so anything under /tmp on the host would simply be
# invisible to Apptainer running there.
SCRATCH=$(mktemp -d "$HOME/.pdf_local_verify.XXXXXX")
TARBALL="$SCRATCH/image.tar"
KEEP_SIF=${1:-}
USING_LIMA=0
VM_SIF=""

cleanup() {
    rm -rf "$SCRATCH"
    if [[ "$USING_LIMA" == 1 && -n "$VM_SIF" ]]; then
        limactl shell "$LIMA_INSTANCE" -- rm -f "$VM_SIF" 2>/dev/null || true
    fi
}
trap cleanup EXIT

mkdir -p "$SCRATCH/tmp"
export SINGULARITY_TMPDIR="$SCRATCH/tmp"
export APPTAINER_TMPDIR="$SCRATCH/tmp"
export TMPDIR="$SCRATCH/tmp"

echo "== docker build -f Dockerfile_llvm -t $TAG . =="
docker build -f Dockerfile_llvm -t "$TAG" .

echo
echo "== docker save $TAG:latest -o $TARBALL =="
docker save "$TAG:latest" -o "$TARBALL"

echo
if command -v apptainer.lima > /dev/null 2>&1; then
    USING_LIMA=1
    LIMA_INSTANCE=${LIMA_INSTANCE:-apptainer}
    SIF="/var/tmp/pdf_local_verify_$$.sif"
    VM_SIF="$SIF"
    echo "== singularity build --force $SIF docker-archive://$TARBALL (inside the $LIMA_INSTANCE VM) =="
    LIMA_INSTANCE="$LIMA_INSTANCE" lima APPTAINER_TMPDIR=/var/tmp APPTAINER_BINDPATH="$HOME" \
        apptainer build --force "$SIF" "docker-archive://$TARBALL"
else
    SIF=${SIF:-$SCRATCH/pdf_local_verify.sif}
    echo "== singularity build --force $SIF docker-archive://$TARBALL =="
    singularity build --force "$SIF" "docker-archive://$TARBALL"
fi

echo
echo "== smoke test =="
singularity exec "$SIF" /opt/anaconda/envs/pdf/bin/python - < "$(dirname "$0")/container_smoke_test.py"

if [[ "$KEEP_SIF" == "keep" ]]; then
    FINAL_SIF=/tmp/pdf_local_verify.sif
    if [[ "$USING_LIMA" == 1 ]]; then
        limactl copy "$LIMA_INSTANCE:$SIF" "$FINAL_SIF"
    else
        case "$SIF" in
            "$SCRATCH"/*) mv "$SIF" "$FINAL_SIF" ;;
            *) FINAL_SIF="$SIF" ;;
        esac
    fi
    echo
    echo "kept: $FINAL_SIF"
fi

echo
echo "local image verified OK."
