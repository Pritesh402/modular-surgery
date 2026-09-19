#!/usr/bin/env python3
"""xmerge_diagnose.py -- read-only structural report on an annotated circuit.

Answers "what is going on" without changing anything.  Run it on EVERY stage
(annotated / graphlike / elementary / fixed) to see where the problem enters.

The three numbers that matter:

  DEPENDENCIES = rank(det) + num_observables - rank(det + obs)
      0  -> healthy: the observables are outside the detector span.
     >0  -> observables are NOT independent of the detector span.  Some
            "detector" is an observable-coset parity: a logical string that no
            basis change can localise, because its coset minimum IS the code
            distance.  This is the x-merge-bell-d5 failure, and it also INFLATES
            the reported distance (a chain flipping the observable fires that
            detector, so stim scores it "detected").

  SLACK = dim(deterministic parities) - rank(det + obs)
      0  -> annotation is exactly maximal.
     >0  -> annotation is incomplete; deterministic parities were missed.

  non-graphlike single-Pauli faults
      elementary X/Z faults flipping >2 detectors.  0 means MWPM-ready.

  outlier coverage
      what fraction of those faults touch a column-weight outlier.  100% means
      the whole problem is the outliers (fixable by logical_detector_fix.py).
      Well under 100% means ordinary basis artifacts remain -- run
      graphlike_basis.py / elementary_basis.py FIRST.

USAGE
  python xmerge_diagnose.py CIRCUIT.stim [CIRCUIT2.stim ...] [--noise 1e-3]
                            [--shots 4000] [--decompose] [--top 25]
"""
import argparse
import sys
from collections import Counter

import numpy as np
import stim

pc = int.bit_count


def single_pauli_noise(circuit, p):
    allq = set(circuit.get_final_qubit_coordinates())
    out = stim.Circuit()
    two_q = ("CX", "CZ", "XCX", "CY", "ZCX", "CXX", "CZZ")
    meas = ("M", "MX", "MY", "MZ")
    single = ("R", "RX", "H", "X", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG")

    def flush(buf):
        acted = set()
        for ins in buf:
            if ins.name in two_q + meas + single:
                acted.update(t.value for t in ins.targets_copy() if t.is_qubit_target)
        for ins in buf:
            n = ins.name
            tv = [t.value for t in ins.targets_copy() if t.is_qubit_target]
            if n in two_q or n in single:
                out.append(ins); out.append("X_ERROR", tv, p); out.append("Z_ERROR", tv, p)
            elif n in meas:
                out.append("Z_ERROR" if n == "MX" else "X_ERROR", tv, p); out.append(ins)
            else:
                out.append(ins)
        idle = sorted(allq - acted)
        if buf and idle:
            out.append("X_ERROR", idle, p); out.append("Z_ERROR", idle, p)

    buf = []
    for ins in circuit.flattened():
        if ins.name == "TICK":
            flush(buf); out.append(ins); buf = []
        else:
            buf.append(ins)
    flush(buf)
    return out


def dep2_noise(circuit, p):
    allq = set(circuit.get_final_qubit_coordinates())
    out = stim.Circuit()

    def flush(buf):
        acted = set()
        for ins in buf:
            if ins.name not in ("QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE",
                                "SHIFT_COORDS", "TICK"):
                acted.update(t.value for t in ins.targets_copy() if t.is_qubit_target)
        for ins in buf:
            n = ins.name
            tv = [t.value for t in ins.targets_copy() if t.is_qubit_target]
            if n in ("CX", "CZ", "XCX", "CY", "ZCX"):
                t = ins.targets_copy()
                for k in range(0, len(t), 2):
                    out.append(n, [t[k], t[k + 1]])
                    out.append("DEPOLARIZE2", [t[k], t[k + 1]], p)
            elif n in ("M", "MX", "MY", "MZ"):
                out.append(n, ins.targets_copy(), p)
            elif n in ("R", "RX", "H", "X"):
                out.append(ins); out.append("DEPOLARIZE1", tv, p)
            else:
                out.append(ins)
        idle = sorted(allq - acted)
        if buf and idle:
            out.append("DEPOLARIZE1", idle, p)

    buf = []
    for ins in circuit.flattened():
        if ins.name == "TICK":
            flush(buf); out.append(ins); buf = []
        else:
            buf.append(ins)
    flush(buf)
    return out


def read_masks(circ):
    N = circ.num_measurements
    det, obs = [], {}
    for ins in circ.flattened():
        if ins.name == "DETECTOR":
            m = 0
            for t in ins.targets_copy():
                if t.is_measurement_record_target:
                    m ^= 1 << (N + t.value)
            det.append(m)
        elif ins.name == "OBSERVABLE_INCLUDE":
            oi = int(ins.gate_args_copy()[0]) if ins.gate_args_copy() else 0
            m = obs.get(oi, 0)
            for t in ins.targets_copy():
                if t.is_measurement_record_target:
                    m ^= 1 << (N + t.value)
            obs[oi] = m
    return det, obs, N


def gf2_rank(vectors):
    piv = {}
    for v in vectors:
        while v:
            b = v.bit_length() - 1
            if b in piv:
                v ^= piv[b]
            else:
                piv[b] = v
                break
    return len(piv)


def deterministic_dim(circ, shots):
    s = circ.compile_sampler().sample(shots=shots).astype(np.uint8)
    a = (s ^ s[0])[1:]
    rows, cols = a.shape
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if a[i, c]), None)
        if piv is None:
            continue
        a[[r, piv]] = a[[piv, r]]
        sel = a[:, c] == 1
        sel[r] = False
        a[sel] ^= a[r]
        r += 1
        if r == rows:
            break
    if r == rows:
        print(f"    WARNING: sample rank hit the shot count ({shots}); "
              f"re-run with a larger --shots, D may be understated")
    return circ.num_measurements - r


def record_maps(circ):
    q_of, t_of, ridx, tick = {}, {}, 0, 0
    for ins in circ.flattened():
        if ins.name == "TICK":
            tick += 1
        elif ins.name in ("M", "MX", "MY", "MZ"):
            for t in ins.targets_copy():
                if t.is_qubit_target:
                    q_of[ridx] = t.value; t_of[ridx] = tick; ridx += 1
    return q_of, t_of


def report(path, noise, shots, decompose, top):
    print(f"\n{'='*72}\n{path}\n{'='*72}")
    circ = stim.Circuit.from_file(path)
    det_recs, obs, N = read_masks(circ)
    nd = len(det_recs)
    obsv = [obs[k] for k in sorted(obs)]
    print(f"  detectors {nd}   observables {len(obsv)}   measurements {N}   "
          f"qubits {circ.num_qubits}")

    # --- determinism -------------------------------------------------------
    try:
        circ.detector_error_model(allow_gauge_detectors=False, decompose_errors=False)
        print("  noiseless determinism (no gauge)  : PASS")
    except Exception as e:
        print("  noiseless determinism (no gauge)  : FAIL",
              str(e).splitlines()[0][:60])

    # --- the slack invariant ----------------------------------------------
    rd = gf2_rank(det_recs)
    ra = gf2_rank(det_recs + obsv)
    D = deterministic_dim(circ, shots)
    slack = D - ra
    deps = rd + len(obsv) - ra
    print(f"  rank(det) {rd}/{nd}   rank(det+obs) {ra}   deterministic dim D {D}")
    print(f"  DEPENDENCIES = rank(det)+n_obs-rank(det+obs) = {deps}   "
          f"{'(healthy)' if deps == 0 else '<-- OBSERVABLE-COSET DETECTORS; distance is inflated'}")
    print(f"  SLACK = D - rank(det+obs) = {slack}   "
          f"{'(maximal)' if slack == 0 else '(annotation incomplete: missed parities)'}")
    if rd < nd:
        print(f"  NOTE: {nd - rd} detector(s) are linearly dependent on the others")

    # --- graphlike-ness ----------------------------------------------------
    dem = single_pauli_noise(circ, noise).detector_error_model(decompose_errors=False)
    syn = {frozenset(t.val for t in i.targets_copy() if t.is_relative_detector_id())
           for i in dem.flattened() if i.type == "error"}
    errs = [s for s in syn if s]
    ne = len(errs)
    col = [0] * nd
    for ei, s in enumerate(errs):
        for d in s:
            col[d] |= 1 << ei
    nongraph = [s for s in errs if len(s) > 2]
    print(f"  single-Pauli faults {ne}   degree hist "
          f"{dict(sorted(Counter(len(s) for s in errs).items()))}")
    print(f"  NON-GRAPHLIKE faults : {len(nongraph)}")

    if nongraph:
        w = [pc(c) for c in col]
        med = sorted(w)[nd // 2]
        flagged = sorted((d for d in range(nd) if w[d] > 3 * med),
                         key=lambda d: -w[d])
        cov = sum(1 for s in nongraph if s & set(flagged))
        print(f"  median column weight {med};  OUTLIERS (>3x median): {len(flagged)}")
        print(f"  outlier coverage of non-graphlike faults: {cov}/{len(nongraph)}"
              f"  ({100*cov//max(1,len(nongraph))}%)")
        if cov < len(nongraph):
            print("    -> residual basis artifacts remain; run graphlike_basis.py /")
            print("       elementary_basis.py BEFORE logical_detector_fix.py")
        q_of, t_of = record_maps(circ)
        qc = circ.get_final_qubit_coordinates()
        print(f"  top outliers (detector, col weight, rec weight, span, coord, tick):")
        for d in flagged[:top]:
            idx = [i for i in range(N) if (det_recs[d] >> i) & 1]
            if not idx:
                continue
            span = idx[-1] - idx[0]
            hi = idx[-1]
            print(f"    D{d:<6} col={w[d]:<5} rec={len(idx):<4} span={span:<6} "
                  f"{tuple(qc[q_of[hi]][:2])} t={t_of[hi]}")

    # --- decodability ------------------------------------------------------
    if decompose:
        noisy = dep2_noise(circ, noise)
        try:
            m = noisy.detector_error_model(decompose_errors=True,
                                           approximate_disjoint_errors=True)
            print(f"  decompose_errors=True             : PASS ({m.num_errors} errors)")
        except Exception as e:
            L = str(e).splitlines()
            print("  decompose_errors=True             : FAIL",
                  (L[1] if len(L) > 1 else L[0])[:60])
        try:
            print(f"  shortest_graphlike_error          : "
                  f"{len(noisy.shortest_graphlike_error())}")
        except Exception as e:
            print("  shortest_graphlike_error          : n/a",
                  str(e).splitlines()[0][:50])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--noise", type=float, default=1e-3)
    ap.add_argument("--shots", type=int, default=4000,
                    help="shots for the deterministic-dimension estimate")
    ap.add_argument("--decompose", action="store_true",
                    help="also try decompose_errors=True and the graphlike distance")
    ap.add_argument("--top", type=int, default=25,
                    help="how many outlier detectors to list")
    a = ap.parse_args()
    for p in a.inputs:
        try:
            report(p, a.noise, a.shots, a.decompose, a.top)
        except Exception as e:
            print(f"\n{p}: ERROR {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
