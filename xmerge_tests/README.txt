x-merge pipeline bundle    (Windows / macOS / Linux -- no shell needed)
=======================================================================
deps: stim 1.16.0, numpy 2.2.6, pymatching 2.4.0

STEP 1 -- check your hand-built observables match the d5 construction:

    python add_seam_observables.py x-merge-bell-d7_annotated.stim --check

  Want: "observable 0 matches the rule exactly : PASS"
        "existing L1 matches the rule: PASS"   (same for L2)
  If L1/L2 FAIL, STOP and send that output -- your strings differ from d5's,
  and everything downstream would measure a different logical.
  (If a file has only L0, drop --check and use  -o OUT.stim  to generate L1/L2.)

STEP 2 -- run the pipeline:

    python run_pipeline.py x-merge-bell-d7_annotated.stim

  Stages: diagnose -> graphlike_basis -> elementary_basis -> diagnose
          -> logical_detector_fix -> diagnose
  A stage whose output already exists and loads is SKIPPED, so an interrupted
  or out-of-memory run resumes instead of restarting.

STEP 3 -- send back every .log in  logs/x-merge-bell-d7/

READING THE DIAGNOSTIC -- three numbers decide everything:
  DEPENDENCIES  0 = healthy.  >0 = observable-coset detectors present, AND the
                reported distance is INFLATED (d5 read 5; truth was 4).
  NON-GRAPHLIKE 0 = MWPM-ready.
  outlier cov   100% = logical_detector_fix.py can fix it.
                <100% = ordinary basis artifacts remain.

MEASURED TIMING (d5: 840 detectors, 4018 faults) -- 63 s total:
    diagnose 6s | graphlike 31s | elementary 7s | diagnose 6s
    logicalfix 7s | diagnose 6s
  graphlike/elementary are O(detectors^2 x faults) single-threaded Python.
  d7 (2235 det, 10882 faults) is ~19x d5 on those stages -> expect ~10-15 min.
  d9 (~4000 det) is ~6x d7 again -> expect ~1-1.5 h.  RAM does not help; these
  stages are CPU-bound.  Only logical_detector_fix's combination search is
  memory-hungry, and it is capped by --max-order (default 2).

FOR d9: identical commands, just the d9 filename.

PATCHES TO YOUR ORIGINALS (both in graphlike_basis.py; .bak kept locally):
  * it kept only the LAST OBSERVABLE_INCLUDE, silently destroying L0 and L1 of
    a 3-observable circuit.  Now keeps all.
  * new --no-validate skips its built-in validation, which took 30+ min at d5
    and OOM-killed d7.  xmerge_diagnose.py covers it properly instead; the
    driver passes --no-validate automatically.
