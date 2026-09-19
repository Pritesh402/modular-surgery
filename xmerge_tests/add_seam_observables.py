#!/usr/bin/env python3
"""add_seam_observables.py -- add the two companion seam observables to an
x-merge circuit that only carries L0.

The x-merge-bell circuits are measured with THREE logical observables, all
final-tick data strings sharing a corner (xc, yc) where the seam column meets
the observable row:

    L0 = {row y=yc, x <  xc}  U  {col x=xc, y <= yc}     (left patch + seam)
    L1 = {col x=xc, all y}                               (the full seam column)
    L2 = {col x=xc, y >= yc}  U  {row y=yc, x >  xc}     (seam + right patch)

d5 was authored by hand in Crumble; this reproduces that construction exactly
(verify with --check) and applies the same rule at any distance, so d7/d9 do not
have to be hand-edited.

(xc, yc) are inferred from the existing L0: xc is the x carrying L0's vertical
arm, yc the y carrying its horizontal arm.  Pass --corner X Y to override.

USAGE
  python add_seam_observables.py IN.stim -o OUT.stim
  python add_seam_observables.py IN.stim --check      # verify, write nothing
"""
import argparse
import sys
from collections import Counter

import stim


def final_readouts(circ):
    """-> ({(x,y): record index} for the last measurement tick, last tick)."""
    q_of, t_of, ridx, tick = {}, {}, 0, 0
    for ins in circ.flattened():
        if ins.name == "TICK":
            tick += 1
        elif ins.name in ("M", "MX", "MY", "MZ"):
            for t in ins.targets_copy():
                if t.is_qubit_target:
                    q_of[ridx] = t.value
                    t_of[ridx] = tick
                    ridx += 1
    last = max(t_of.values())
    qc = circ.get_final_qubit_coordinates()
    pt = {}
    for k, q in q_of.items():
        if t_of[k] == last:
            pt[tuple(qc[q][:2])] = k
    return pt, last


def read_obs(circ):
    N = circ.num_measurements
    obs = {}
    for ins in circ.flattened():
        if ins.name == "OBSERVABLE_INCLUDE":
            oi = int(ins.gate_args_copy()[0]) if ins.gate_args_copy() else 0
            s = obs.setdefault(oi, set())
            for t in ins.targets_copy():
                if t.is_measurement_record_target:
                    s ^= {N + t.value}
    return obs


def build(path, out_path, corner, check):
    circ = stim.Circuit.from_file(path)
    pt, last = final_readouts(circ)
    inv = {v: k for k, v in pt.items()}
    obs = read_obs(circ)
    print(f"  {path}: {circ.num_detectors} detectors, {len(obs)} observable(s), "
          f"final tick {last}, {len(pt)} data readouts")

    if 0 not in obs:
        print("  ERROR: no observable 0 to infer the corner from")
        return False
    l0 = sorted(inv[r] for r in obs[0] if r in inv)
    if len(l0) != len(obs[0]):
        print(f"  ERROR: observable 0 has {len(obs[0]) - len(l0)} record(s) outside "
              f"the final tick; this rule only handles pure final-tick strings")
        return False

    if corner:
        xc, yc = corner
    else:
        xc = Counter(x for x, y in l0).most_common(1)[0][0]
        yc = Counter(y for x, y in l0).most_common(1)[0][0]
    print(f"  inferred corner (xc, yc) = ({xc}, {yc})")

    col = sorted(y for (x, y) in pt if x == xc)
    row = sorted(x for (x, y) in pt if y == yc)
    print(f"  seam column x={xc}: {len(col)} qubits;  row y={yc}: {len(row)} qubits")

    want0 = {(x, yc) for x in row if x < xc} | {(xc, y) for y in col if y <= yc}
    want1 = {(xc, y) for y in col}
    want2 = {(xc, y) for y in col if y >= yc} | {(x, yc) for x in row if x > xc}

    if set(l0) != want0:
        print(f"  WARNING: observable 0 does not match the rule "
              f"(have {len(l0)}, rule gives {len(want0)}, "
              f"differ on {len(set(l0) ^ want0)} point(s))")
        print("  -> the corner inference or the layout differs; pass --corner, or")
        print("     author L1/L2 by hand as you did for d5.")
        if not check:
            return False
    else:
        print(f"  observable 0 matches the rule exactly ({len(l0)} qubits)  : PASS")

    for name, want in (("L1", want1), ("L2", want2)):
        print(f"  {name}: {len(want)} qubits")

    if check:
        for oi, want in ((1, want1), (2, want2)):
            if oi in obs:
                have = {inv[r] for r in obs[oi] if r in inv}
                ok = have == want
                print(f"  existing L{oi} matches the rule: "
                      f"{'PASS' if ok else 'FAIL (differ on %d)' % len(have ^ want)}")
            else:
                print(f"  L{oi} not present in this file")
        return True

    N = circ.num_measurements
    out = stim.Circuit()
    for ins in circ.flattened():
        out.append(ins)
    for oi, want in ((1, want1), (2, want2)):
        if oi in obs:
            print(f"  L{oi} already present; leaving it alone")
            continue
        recs = sorted(pt[p] for p in want)
        out.append("OBSERVABLE_INCLUDE",
                   [stim.target_rec(r - N) for r in recs], [float(oi)])
    out.to_file(out_path)
    print(f"  wrote {out_path}: {out.num_detectors} detectors, "
          f"{out.num_observables} observables")
    try:
        out.detector_error_model(allow_gauge_detectors=True, decompose_errors=False)
        print("  DEM builds with the new observables            : PASS")
    except Exception as e:
        print("  DEM builds                                    : FAIL",
              str(e).splitlines()[0][:60])
        return False
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--corner", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--check", action="store_true",
                    help="verify an existing file against the rule; write nothing")
    a = ap.parse_args()
    if not a.check and not a.output:
        ap.error("need -o OUTPUT (or --check)")
    sys.exit(0 if build(a.input, a.output, a.corner, a.check) else 1)


if __name__ == "__main__":
    main()
