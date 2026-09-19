#!/usr/bin/env python3
"""run_pipeline.py -- annotated x-merge circuit -> MWPM-decodable circuit.

Cross-platform (Windows / macOS / Linux): no shell, no env vars.  Uses the same
interpreter it was launched with, so `python run_pipeline.py ...` just works.

    python run_pipeline.py x-merge-bell-d7_3obs_annotated.stim

Stages: diagnose -> graphlike_basis -> elementary_basis -> diagnose
        -> logical_detector_fix -> diagnose

A stage whose output file already exists is SKIPPED, so an interrupted or
out-of-memory run resumes instead of restarting from the top.

graphlike_basis.py writes its output BEFORE running its own validation, so a
nonzero exit can still leave a perfectly good file.  Two benign cases:
  exit 1    its validation reported FAILED -- EXPECTED here, because the
            logical-coset detectors it cannot fix are what stage 3 fixes.
  killed    out-of-memory inside search_for_undetectable_logical_errors
            (exit 137 on Unix; on Windows a large negative / 0xC... code).
Either way, if the output exists and loads, the pipeline says so and continues.
Anything else is a real failure and stops the run.

Logs land in logs/<basename>/ .  Send those back.
"""
import argparse
import datetime
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def now():
    return datetime.datetime.now().strftime("%H:%M:%S")


def loadable(path):
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        import stim
        stim.Circuit.from_file(path)
        return True
    except Exception:
        return False


def run(cmd, log_path):
    """Run cmd, echoing output live and tee-ing it to log_path.  -> exit code."""
    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, bufsize=1,
                             universal_newlines=True,
                             encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()          # a killed stage must still leave a readable log
        p.wait()
    return p.returncode


def stage(name, out_path, script, args, logdir):
    if out_path and loadable(out_path):
        print(f">>> {name}: {out_path} exists, skipping")
        return True
    print(f">>> {name}  ({now()})")
    cmd = [sys.executable, "-u", os.path.join(HERE, script)] + args
    rc = run(cmd, os.path.join(logdir, name + ".log"))
    if rc != 0:
        if rc == 1:
            why = "exit 1 (its own validation reported FAILED -- expected before stage 3)"
        elif rc in (137, -9):
            why = "killed (out of memory in its validation step)"
        else:
            why = f"exit {rc}"
        if loadable(out_path):
            print(f"!!! {name} {why}")
            print(f"    {out_path} was still written and loads -- continuing; "
                  f"later stages re-validate.")
        else:
            print(f"!!! {name} FAILED ({why}) with no usable {out_path}")
            print(f"    see {os.path.join(logdir, name + '.log')}")
            return False
    elif out_path and not loadable(out_path):
        print(f"!!! {name} produced no usable {out_path}")
        return False
    print(f">>> {name} done ({now()})")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="annotated .stim circuit")
    ap.add_argument("--noise", type=float, default=1e-3)
    ap.add_argument("--max-order", type=int, default=2,
                    help="largest XOR-combination size logical_detector_fix "
                         "enumerates (default 2)")
    a = ap.parse_args()

    if not os.path.exists(a.input):
        print(f"no such file: {a.input}")
        return 1
    base = os.path.basename(a.input)
    if base.endswith(".stim"):
        base = base[:-5]
    if base.endswith("_annotated"):
        base = base[:-len("_annotated")]
    gl = base + "_graphlike.stim"
    el = base + "_graphlike_elementary.stim"
    fx = base + "_fixed.stim"
    logdir = os.path.join("logs", base)
    os.makedirs(logdir, exist_ok=True)
    noise = ["--noise", repr(a.noise)]

    print(f"### input {a.input}")
    print(f"### logs  {logdir}\n")

    print("######## BEFORE ########")
    stage("diagnose_before", None, "xmerge_diagnose.py",
          [a.input, "--decompose"] + noise, logdir)

    if not stage("graphlike", gl, "graphlike_basis.py",
                 [a.input, "-o", gl, "--no-validate"] + noise, logdir):
        return 1
    if not stage("elementary", el, "elementary_basis.py",
                 [gl, "-o", el] + noise, logdir):
        return 1

    print("\n######## AFTER graphlike+elementary ########")
    stage("diagnose_mid", None, "xmerge_diagnose.py",
          [el, "--decompose"] + noise, logdir)

    if not stage("logicalfix", fx, "logical_detector_fix.py",
                 [el, "-o", fx, "--max-order", str(a.max_order)] + noise, logdir):
        return 1

    print("\n######## AFTER ########")
    stage("diagnose_after", None, "xmerge_diagnose.py",
          [fx, "--decompose"] + noise, logdir)

    print(f"\n=== pipeline complete -> {fx} ===")
    print(f"=== send back every .log in {logdir} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
