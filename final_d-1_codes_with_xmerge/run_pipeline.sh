#!/bin/bash
# Annotated x-merge circuit -> MWPM-decodable circuit, with a diagnostic at every
# stage.  Stages are skipped when their output already exists, so an interrupted
# or OOM-killed run resumes instead of restarting.
#
#   ./run_pipeline.sh x-merge-bell-d7_annotated.stim
#
# Everything lands in logs/<basename>/.  Set PY to your interpreter if needed.
set -u
PY=${PY:-python3}
IN=${1:?usage: $0 ANNOTATED.stim}
BASE=$(basename "$IN" .stim); BASE=${BASE%_annotated}
GL=${BASE}_graphlike.stim
EL=${BASE}_graphlike_elementary.stim
FX=${BASE}_fixed.stim
LOG=logs/$BASE
mkdir -p "$LOG"

stage () {  # name, output, command...
  local name=$1 out=$2; shift 2
  if [ -s "$out" ]; then echo ">>> $name: $out exists, skipping"; return 0; fi
  echo ">>> $name  ($(date +%H:%M:%S))"
  "$@" 2>&1 | tee "$LOG/$name.log"
  local rc=${PIPESTATUS[0]}
  # graphlike_basis.py writes its output BEFORE its own validation, and that
  # validation gets OOM-killed (rc=137) at d7+.  A written, loadable output after
  # such a death is still good -- the reduction itself finished.  Say so.
  if [ "$rc" -ne 0 ]; then
    if [ -s "$out" ] && $PY -c "import stim,sys; stim.Circuit.from_file(sys.argv[1])" "$out" 2>/dev/null; then
      echo "!!! $name exited $rc (137 = OOM-killed, expected in its validation step)"
      echo "    but $out was written and loads -- continuing."
    else
      echo "!!! $name FAILED rc=$rc, no usable $out (see $LOG/$name.log)"; return 1
    fi
  elif [ ! -s "$out" ]; then
    echo "!!! $name produced no $out"; return 1
  fi
  echo ">>> $name done ($(date +%H:%M:%S))"
}

echo "######## BEFORE ########"
$PY -u xmerge_diagnose.py "$IN" --decompose 2>&1 | tee "$LOG/diagnose_before.log"

stage graphlike  "$GL" $PY -u graphlike_basis.py      "$IN" -o "$GL" || exit 1
stage elementary "$EL" $PY -u elementary_basis.py     "$GL" -o "$EL" || exit 1

echo "######## AFTER graphlike+elementary ########"
$PY -u xmerge_diagnose.py "$EL" --decompose 2>&1 | tee "$LOG/diagnose_mid.log"

stage logicalfix "$FX" $PY -u logical_detector_fix.py "$EL" -o "$FX" || exit 1

echo "######## AFTER ########"
$PY -u xmerge_diagnose.py "$FX" --decompose 2>&1 | tee "$LOG/diagnose_after.log"
echo
echo "=== pipeline complete -> $FX ==="
echo "=== send me: $LOG/*.log ==="
