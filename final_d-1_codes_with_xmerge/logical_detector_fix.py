#!/usr/bin/env python3
"""logical_detector_fix.py -- remove observable-coset detectors so the DEM decomposes.

WHY THIS EXISTS
---------------
graphlike_basis.py / elementary_basis.py both assume the annotation's non-graphlike
structure is a BASIS artifact: the right XOR of detectors makes every fault flip
<= 2 of them.  That assumption fails when the annotator emits a parity that is not a
stabilizer check at all but a LOGICAL STRING -- a boundary-to-boundary data-qubit
chain closed with ancilla history.  Such a parity is deterministic (so the annotator
accepts it) but it lives in an observable coset, so the lightest fault that flips it
has weight ~d, not O(1).  No column operation can localise it: its coset minimum IS
the code distance.  Symptoms:

  * one or two detectors with error-incidence column weight ~10x the median
  * every non-graphlike single-Pauli fault touches one of them
  * stim: "Failed to decompose errors into graphlike components with at most two
    symptoms ... 'D7, D8, D836, L0'"  (note the L0 -- the observable rides along)
  * rank(detectors) + num_observables > dim(deterministic parity space):
    the observables are NOT independent of the detector span
  * the reported distance is INFLATED, because a fault chain that flips the
    observable also fires one of these detectors and is scored as "detected"

WHAT THIS DOES
--------------
  1. Build the single-Pauli (X_ERROR + Z_ERROR, gates + idle) incidence matrix.
  2. Flag outlier detectors: error-column weight > outlier_factor * median.
  3. Enumerate XOR combinations of the flagged set.  A combination that greedily
     reduces (against the non-flagged columns) to normal column weight is a GENUINE
     local check that the annotator split across several global strings -- keep the
     reduced form.  Combinations that stay heavy are observable-coset parities.
  4. Keep a maximal independent set of the recovered local checks; drop the rest.
  5. Validate: noiseless determinism, rank/slack against the deterministic-parity
     dimension, observable independence, decompose_errors=True, and distance
     agreement between shortest_graphlike_error and search_for_undetectable.

Run it AFTER the annotator.  It replaces the graphlike/elementary stages for the
failure they cannot fix; running them first is harmless but unnecessary.

USAGE
  python logical_detector_fix.py ANNOTATED.stim -o FIXED.stim [--noise 1e-3]
"""
import argparse
import itertools
import sys

import numpy as np
import stim

pc = int.bit_count


# ------------------------------------------------------------------- probe ---
def single_pauli_noise(circuit, p):
    """Independent X_ERROR + Z_ERROR on every gate, reset, measurement and idle
    qubit.  These are the elementary faults whose syndromes must be graphlike;
    if they all are, every DEPOLARIZE2 composite decomposes."""
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
                out.append(ins)
                out.append("X_ERROR", tv, p)
                out.append("Z_ERROR", tv, p)
            elif n in meas:
                out.append("Z_ERROR" if n == "MX" else "X_ERROR", tv, p)
                out.append(ins)
            else:
                out.append(ins)
        idle = sorted(allq - acted)
        if buf and idle:
            out.append("X_ERROR", idle, p)
            out.append("Z_ERROR", idle, p)

    buf = []
    for ins in circuit.flattened():
        if ins.name == "TICK":
            flush(buf)
            out.append(ins)
            buf = []
        else:
            buf.append(ins)
    flush(buf)
    return out


def dep2_noise(circuit, p):
    """Uniform circuit-level DEPOLARIZE model, for the decode/distance checks."""
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
                out.append(ins)
                out.append("DEPOLARIZE1", tv, p)
            else:
                out.append(ins)
        idle = sorted(allq - acted)
        if buf and idle:
            out.append("DEPOLARIZE1", idle, p)

    buf = []
    for ins in circuit.flattened():
        if ins.name == "TICK":
            flush(buf)
            out.append(ins)
            buf = []
        else:
            buf.append(ins)
    flush(buf)
    return out


# ------------------------------------------------------------------ algebra --
def read_masks(circ):
    """-> (detector record masks, {obs index: mask}, num_measurements)."""
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


def deterministic_dim(circ, shots=4000):
    """dim of the space of deterministic measurement parities, by sampling."""
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
    return circ.num_measurements - r


def record_maps(circ):
    """record index -> qubit, record index -> tick."""
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
    return q_of, t_of


# ------------------------------------------------------------------- driver --
def fix(in_path, out_path, noise=1e-3, outlier_factor=3.0, max_flagged=40,
        max_order=2, hyperedge=False):
    circ = stim.Circuit.from_file(in_path)
    det_recs, obs, N = read_masks(circ)
    nd = len(det_recs)
    print(f"  input: {nd} detectors, {len(obs)} observables, {N} measurements")

    dem = single_pauli_noise(circ, noise).detector_error_model(decompose_errors=False)
    syn = {frozenset(t.val for t in ins.targets_copy() if t.is_relative_detector_id())
           for ins in dem.flattened() if ins.type == "error"}
    errs = [s for s in syn if s]
    ne = len(errs)
    col = [0] * nd
    for ei, s in enumerate(errs):
        for d in s:
            col[d] |= 1 << ei
    nongraph = [s for s in errs if len(s) > 2]
    print(f"  single-Pauli faults {ne};  non-graphlike {len(nongraph)}")
    if not nongraph:
        print("  already graphlike -- nothing to do")
        return True

    w = [pc(c) for c in col]
    med = sorted(w)[nd // 2]
    flagged = sorted((d for d in range(nd) if w[d] > outlier_factor * med),
                     key=lambda d: -w[d])[:max_flagged]
    print(f"  median column weight {med}; flagged outliers "
          f"{[(d, w[d]) for d in flagged]}")
    if not flagged:
        print("  no outlier detectors -- this is not the logical-coset failure;")
        print("  try graphlike_basis.py / elementary_basis.py instead")
        return False
    covered = sum(1 for s in nongraph if s & set(flagged))
    print(f"  non-graphlike faults touching a flagged detector: "
          f"{covered}/{len(nongraph)}")
    if covered < len(nongraph):
        print(f"  WARNING: {len(nongraph) - covered} non-graphlike fault(s) touch NO "
              f"outlier.  Those are ordinary basis artifacts, not logical cosets --")
        print(f"  run graphlike_basis.py / elementary_basis.py FIRST, then re-run this.")

    keep_cols = [d for d in range(nd) if d not in set(flagged)]

    # Inverted index: error index -> detectors whose column carries that fault.
    # Only these can lower a target's weight, so the descent scans dozens of
    # candidates instead of all nd.
    touch = [[] for _ in range(ne)]
    for d in keep_cols:
        cm = col[d]
        while cm:
            b = cm & -cm
            touch[b.bit_length() - 1].append(d)
            cm ^= b

    def reduce_pair(vec, rec):
        """greedily XOR with unflagged columns to minimise error-column weight"""
        for _ in range(4 * nd):
            cand = set()
            cm = vec
            while cm:
                b = cm & -cm
                cand.update(touch[b.bit_length() - 1])
                cm ^= b
            best, gain = None, 0
            wv = pc(vec)
            for b in cand:
                g = wv - pc(vec ^ col[b])
                if g > gain:
                    gain, best = g, b
            if best is None:
                return vec, rec
            vec ^= col[best]
            rec ^= det_recs[best]
        return vec, rec

    # Enumerate XOR combinations of the flagged set.  NOTE: the winning move can
    # RAISE column weight before the descent pays off -- in x-merge-bell-d5,
    # D837^D839 goes 112 -> 141 -> 8.  That barrier is exactly why plain greedy
    # descent (graphlike_basis.py) cannot find these, so the combinations have to
    # be enumerated rather than descended into.  Cost is combinatorial in the
    # flagged-set size, hence max_order.
    threshold = max(w[d] for d in keep_cols) if keep_cols else med
    n_combos = sum(len(list(itertools.combinations(flagged, r)))
                   for r in range(1, max_order + 1))
    print(f"  enumerating {n_combos} combination(s) up to order {max_order}")
    recovered = []
    for r in range(1, max_order + 1):
        for combo in itertools.combinations(flagged, r):
            v = rec = 0
            for d in combo:
                v ^= col[d]
                rec ^= det_recs[d]
            rv, rr = reduce_pair(v, rec)
            if 0 < pc(rv) <= threshold:
                recovered.append((combo, rv, rr))
                print(f"    {'^'.join('D%d' % d for d in combo)} -> local check "
                      f"(column weight {pc(rv)}, {pc(rr)} records)")

    # maximal independent set of recovered checks, modulo the kept detectors
    piv = {}
    for d in keep_cols:
        v = det_recs[d]
        while v:
            b = v.bit_length() - 1
            if b in piv:
                v ^= piv[b]
            else:
                piv[b] = v
                break
    kept_new = []
    for combo, rv, rr in recovered:
        v = rr
        while v:
            b = v.bit_length() - 1
            if b in piv:
                v ^= piv[b]
            else:
                piv[b] = v
                kept_new.append(rr)
                break
    dropped = len(flagged) - len(kept_new)
    print(f"  recovered {len(kept_new)} genuine local check(s); "
          f"dropping {dropped} observable-coset detector(s)")

    # ---------------------------------------------------------------- emit ---
    q_of, t_of = record_maps(circ)
    qc = circ.get_final_qubit_coordinates()
    out = stim.Circuit()
    di = 0
    obs_ins = []
    for ins in circ.flattened():
        if ins.name == "DETECTOR":
            if di not in set(flagged):
                out.append(ins)
            di += 1
        elif ins.name == "OBSERVABLE_INCLUDE":
            obs_ins.append(ins)
        else:
            out.append(ins)
    for rec in kept_new:
        idx = [i for i in range(N) if (rec >> i) & 1]
        hi = idx[-1]
        x, y = qc[q_of[hi]][:2]
        out.append("DETECTOR", [stim.target_rec(i - N) for i in idx],
                   [float(x), float(y), float(t_of[hi])])
    for ins in obs_ins:
        out.append(ins)
    out.to_file(out_path)
    print(f"  wrote {out_path}: {out.num_detectors} detectors, "
          f"{out.num_observables} observables")

    return validate(out, noise, hyperedge)


def validate(out, noise, hyperedge=False):
    ok = True
    try:
        out.detector_error_model(allow_gauge_detectors=False, decompose_errors=False)
        print("  noiseless determinism (no gauge)              : PASS")
    except Exception as e:
        ok = False
        print("  noiseless determinism                         : FAIL",
              str(e).splitlines()[0][:60])

    det, obs, _ = read_masks(out)
    obsv = [obs[k] for k in sorted(obs)]
    rd, ra = gf2_rank(det), gf2_rank(det + obsv)
    D = deterministic_dim(out)
    indep = ra == rd + len(obsv)
    slack = D - ra
    print(f"  rank(det)={rd}/{len(det)}  rank(det+obs)={ra}  "
          f"deterministic dim={D}  slack={slack}")
    print(f"  observables independent of detector span      : "
          f"{'PASS' if indep else 'FAIL'}")
    ok &= indep

    noisy = dep2_noise(out, noise)
    try:
        d = noisy.detector_error_model(decompose_errors=True,
                                       approximate_disjoint_errors=True)
        print(f"  decompose_errors=True (MWPM-ready)            : PASS "
              f"({d.num_errors} errors)")
    except Exception as e:
        ok = False
        L = str(e).splitlines()
        print("  decompose_errors=True                         : FAIL",
              (L[1] if len(L) > 1 else L[0])[:60])

    gl = len(noisy.shortest_graphlike_error())
    su = "skipped"
    if hyperedge:
        try:
            su = len(noisy.search_for_undetectable_logical_errors(
                dont_explore_detection_event_sets_with_size_above=gl + 2,
                dont_explore_edges_with_degree_above=gl + 2,
                dont_explore_edges_increasing_symptom_degree=False))
        except Exception as e:
            su = "n/a (" + str(e).splitlines()[0][:40] + ")"
    print(f"  distance: shortest_graphlike={gl}  search_for_undetectable={su}")
    if isinstance(su, int) and su != gl:
        print("  NOTE: the two distance measures disagree -- inspect before trusting")
    if su == "skipped":
        print("  (pass --hyperedge for the undetectable-error cross-check; it is "
              "memory-hungry and OOMs at d7+)")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--noise", type=float, default=1e-3)
    ap.add_argument("--outlier-factor", type=float, default=3.0,
                    help="flag detectors whose error-column weight exceeds this "
                         "multiple of the median (default 3)")
    ap.add_argument("--max-flagged", type=int, default=40,
                    help="cap on flagged detectors (default 40)")
    ap.add_argument("--max-order", type=int, default=2,
                    help="largest XOR-combination size to enumerate (default 2). "
                         "Cost grows as C(flagged, order); raise only if order 2 "
                         "recovers nothing")
    ap.add_argument("--hyperedge", action="store_true",
                    help="also run search_for_undetectable_logical_errors. OFF by "
                         "default: memory-hungry, OOMs at d7+")
    a = ap.parse_args()
    ok = fix(a.input, a.output, a.noise, a.outlier_factor, a.max_flagged,
             a.max_order, a.hyperedge)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
