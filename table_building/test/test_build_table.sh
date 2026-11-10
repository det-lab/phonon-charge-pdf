#!/bin/bash
# End-to-end test of table_building/slurm/build_table.sh with no cluster and no Fortran: a mock
# sbatch runs the array tasks locally, and normgrid's analytic stand-in evaluator
# (NORMGRID_FAKE=smooth: exp(3 u_V + 6 u_q0), a known convergence rate) replaces
# the integrals.  Needs h5py (merge writes HDF5), e.g. inside the container.
#
#   bash table_building/test/test_build_table.sh          (from the repository root)
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$PWD/table_building/test:$PATH" PDF_NATIVE=1 PYTHON=${PYTHON:-python}
export PDF_PYTHON=$PYTHON
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
export MOCK_SLURM_DIR=$TMP
fail=0
check() { if [[ $2 -eq 0 ]]; then echo "PASS  $1"; else echo "FAIL  $1"; fail=1; fi; }
common="BAND=ER GRID_CHUNK=5000 STUDY_CHUNK=100 HELD_CHUNK=50 HELDOUT=60"

# 1. the smooth stand-in: plan, build, and the table passes validation
NORMGRID_FAKE=smooth WORKDIR=$TMP/w OUT_TABLE=$TMP/t.h5 table_building/slurm/build_table.sh ER 2.5 350 0.75 200 > $TMP/ok.log 2>&1
check "full chain succeeds on a smooth problem" $?
grep -q "PASS: worst-case error" $TMP/ok.log; check "validate reports PASS" $?
[[ -f $TMP/t.h5 ]]; check "table written" $?
python - <<PY
import json, sys
r = json.load(open("$TMP/w/tol1e-7/plan.json"))["axes"]
n = {a: v["n"] for a, v in r.items()}
sys.exit(0 if n["q0"] > n["V"] > 3 and all(n[a] == 3 for a in n if a not in ("V", "q0")) else 1)
PY
check "planned nodes: q0 > V > others (=3)" $?

# 2. re-running reuses everything (finished points are skipped)
NORMGRID_FAKE=smooth WORKDIR=$TMP/w OUT_TABLE=$TMP/t.h5 table_building/slurm/build_table.sh ER 2.5 350 0.75 200 > $TMP/again.log 2>&1
check "re-run succeeds" $?

# 3. an impossible acceptance level fails the check and says how to retry
NORMGRID_FAKE=smooth WORKDIR=$TMP/w OUT_TABLE=$TMP/t2.h5 ACCEPT=1e-30 table_building/slurm/build_table.sh ER 2.5 350 0.75 200 > $TMP/fail.log 2>&1
[[ $? -eq 1 ]]; check "ACCEPT too strict => exit status 1" $?
grep -q "TOL=2e-08 sbatch table_building/slurm/build_table.job ER" $TMP/fail.log; check "failure message suggests a tighter TOL" $?

# 4. an axis that never converges stops the chain before the big grid is built
NORMGRID_FAKE=kink WORKDIR=$TMP/k OUT_TABLE=$TMP/t3.h5 table_building/slurm/build_table.sh NR 2.5 350 0.75 200 > $TMP/kink.log 2>&1
[[ $? -ne 0 ]]; check "kink: chain stops with an error" $?
grep -q "not resolved by 33 nodes" $TMP/kink.log; check "kink: says which axis and why" $?
[[ ! -e $TMP/k/tol1e-7/spec.json ]]; check "kink: no grid spec was written" $?

# 5. a failed array task stops the chain, although (as on Alderaan) sbatch --wait exits 0
NORMGRID_FAKE=smooth PDF_PYTHON=false WORKDIR=$TMP/f OUT_TABLE=$TMP/t4.h5 table_building/slurm/build_table.sh ER 2.5 350 0.75 200 > $TMP/taskfail.log 2>&1
[[ $? -ne 0 ]]; check "failed task: chain stops with an error" $?
grep -q "not every one of its .* tasks COMPLETED" $TMP/taskfail.log; check "failed task: says which array job" $?
[[ ! -e $TMP/f/tol1e-7/plan.json ]]; check "failed task: stopped before the plan" $?

# 6. a custom box: tagged with its hash, and used by every spec
NORMGRID_FAKE=smooth BOX="V=2.6:3.4 dq=0:0.4" WORKDIR=$TMP/b OUT_TABLE=$TMP/b.h5 table_building/slurm/build_table.sh ER 2.5 350 0.75 200 > $TMP/box.log 2>&1
check "custom box: chain succeeds" $?
grep -q "(ER_ep2.5-350_eq0.75-200_box[0-9a-f]\{8\})" $TMP/box.log; check "custom box: tag carries the box hash" $?
python - <<PY
import json, sys
boxes = [json.load(open(f"$TMP/b/{f}"))["box"] for f in ("study.json", "held.json", "tol1e-7/spec.json")]
sys.exit(0 if all(b["V"] == [2.6, 3.4] and b["dq"] == [0.0, 0.4] and b["F0"] == [0.1, 0.35] for b in boxes) else 1)
PY
check "custom box: study, held-out and grid specs all use it" $?
jobs_before=$(cat $MOCK_SLURM_DIR/last_job)
BOX="k=0.1:0.2" WORKDIR=$TMP/bad table_building/slurm/build_table.sh ER 2.5 350 0.75 200 > $TMP/badbox.log 2>&1
[[ $? -ne 0 ]] && grep -q "not an axis of the ER table" $TMP/badbox.log && [[ $(cat $MOCK_SLURM_DIR/last_job) == "$jobs_before" ]]
check "bad box: fails before anything is submitted" $?

# 7. a PErecoil table: tagged with its name, embedded in every spec and the table
PE=erecoil_tables/402_ER_Ba.h5
NORMGRID_FAKE=smooth PERECOIL_TABLE=$PE OUT_TABLE=$TMP/pe.h5 WORKDIR=$TMP/pe table_building/slurm/build_table.sh ER 2.5 350 0.75 200 > $TMP/pe.log 2>&1
check "PErecoil table: chain succeeds" $?
grep -q "(ER_ep2.5-350_eq0.75-200_402_ER_Ba)" $TMP/pe.log; check "PErecoil table: tag carries its name" $?
$PYTHON - <<PY
import json, sys, h5py
sys.path.insert(0, "python")
from erecoil_table import PErecoilTable
sha = PErecoilTable.from_hdf5("$PE").sha256
pes = [json.load(open(f"$TMP/pe/{f}"))["perecoil"] for f in ("study.json", "held.json", "tol1e-7/spec.json")]
pes.append(json.loads(h5py.File("$TMP/pe.h5", "r").attrs["perecoil_json"]))
sys.exit(0 if all(p["kind"] == "table" and p["sha256"] == sha for p in pes) else 1)
PY
check "PErecoil table: study, held-out, grid spec and table all record it" $?
jobs_before=$(cat $MOCK_SLURM_DIR/last_job)
PERECOIL_TABLE=erecoil_tables/402_NR_Cf.h5 WORKDIR=$TMP/peb table_building/slurm/build_table.sh ER 2.5 350 0.75 200 > $TMP/peband.log 2>&1
[[ $? -eq 2 ]] && grep -q "is a band 'NR' spectrum, not ER" $TMP/peband.log && [[ $(cat $MOCK_SLURM_DIR/last_job) == "$jobs_before" ]]
check "PErecoil table of the wrong band: fails before anything is submitted" $?
# validate refuses held-out points computed with another spectrum
$PYTHON python/cli/normgrid.py validate --table $TMP/pe.h5 --heldout-spec $TMP/w/held.json \
    --results "$TMP/w/held_results/res_held_*.txt" > $TMP/pemix.log 2>&1
[[ $? -ne 0 ]] && grep -q "rebuild the held-out spec with the same --perecoil-table" $TMP/pemix.log
check "validate: held-out spectrum must match the table's" $?

# 8. bad usage
table_building/slurm/build_table.sh ER 2.5 350 0.75 > /dev/null 2>&1; [[ $? -eq 2 ]]; check "wrong argument count is a usage error" $?
exit $fail
