#!/usr/bin/env python3
"""elementary_basis.py -- make a DISCOVERED detector set decodable.

THE PROBLEM THIS SOLVES
-----------------------
`detector_discovery`-style GF(2) discovery returns *a* basis of the space of
deterministic parities.  It does not return *the elementary* basis.  On a
Bell-measurement seam the basis it happens to find is CUMULATIVE:

    D'_5 = S_5
    D'_6 = S_5 ^ S_6
    D'_7 = S_5 ^ S_6 ^ S_7        <- weight grows 2,4,6,8,10,...,20
    ...

The span is right, so k, gauge, determinism, observable-independence and the
TRUE (hyperedge) distance all pass.  But a single seam measurement error now
flips every cumulative detector from its round onward -> a degree-5..13
hyperedge instead of a degree-2 time edge.  MWPM sees an error model that does
not describe the circuit and mis-corrects, so the logical error rate flattens
to ~p^1 even though the distance is unchanged.

THE FIX
-------
Change basis.  Every step is  d_i <- d_i XOR d_j , an elementary column
operation over GF(2), so the detector SPAN is untouched:
    * same k, same rank, same determinism, same gauge count
    * observable stays outside the span (it is the same span)
    * same search_for_undetectable_logical_errors
Only the decoding graph changes -- which is exactly the thing that was broken.

The objective is *not* record count.  Minimising record weight would happily
turn a time chain {m_r ^ m_{r-1}} into singletons {m_r}, which is worse.  The
objective is the number of ERROR MECHANISMS that flip each detector (the column
weight of the error->detector incidence matrix).  Sum of column weights = sum
of row weights, so driving it down drives every error toward flipping <= 2
detectors, i.e. toward graphlike.

USAGE
    python elementary_basis.py ANNOTATED.stim -o ANNOTATED_elem.stim [--noise 1e-3]

Run it as the last step, after bellmeas_annotator.py.
"""
import argparse
import collections
import sys

import numpy as np
import stim


# ------------------------------------------------------------------ noise ----
def uniform_noise(circuit, p):
    """Same uniform circuit-level model used by bellmeas_annotator.py."""
    allq = set(circuit.get_final_qubit_coordinates())
    out = stim.Circuit()

    def flush(buf):
        acted = set()
        for inst in buf:
            if inst.name in ("CX", "CZ", "XCX", "CY", "ZCX", "H", "R", "RX",
                             "M", "MX", "MY", "MZ"):
                for t in inst.targets_copy():
                    if t.is_qubit_target:
                        acted.add(t.value)
        for inst in buf:
            n = inst.name
            if n in ("CX", "CZ", "XCX", "CY", "ZCX"):
                out.append(inst)
                out.append("DEPOLARIZE2", [t.value for t in inst.targets_copy()], p)
            elif n == "H":
                out.append(inst)
                out.append("DEPOLARIZE1", [t.value for t in inst.targets_copy()], p)
            elif n in ("R", "RX"):
                out.append(inst)
                out.append("Z_ERROR" if n == "RX" else "X_ERROR",
                           [t.value for t in inst.targets_copy()], p)
            elif n in ("M", "MX", "MY", "MZ"):
                out.append("Z_ERROR" if n == "MX" else "X_ERROR",
                           [t.value for t in inst.targets_copy() if t.is_qubit_target], p)
                out.append(inst)
            else:
                out.append(inst)
        idle = sorted(allq - acted)
        if buf and idle:
            out.append("DEPOLARIZE1", idle, p)

    buf = []
    for inst in circuit:
        if inst.name == "TICK":
            flush(buf)
            out.append(inst)
            buf = []
        else:
            buf.append(inst)
    flush(buf)
    return out


# ------------------------------------------------------------------ model ----
def read_annotations(circ):
    """-> (record masks per detector, coord args per detector, {obs: mask}, N)."""
    N = circ.num_measurements
    rec, coords, obs = [], [], {}
    for inst in circ.flattened():
        if inst.name == "DETECTOR":
            m = 0
            for t in inst.targets_copy():
                if t.is_measurement_record_target:
                    m ^= 1 << (N + t.value)
            rec.append(m)
            coords.append(list(inst.gate_args_copy()))
        elif inst.name == "OBSERVABLE_INCLUDE":
            oi = int(inst.gate_args_copy()[0]) if inst.gate_args_copy() else 0
            m = obs.get(oi, 0)
            for t in inst.targets_copy():
                if t.is_measurement_record_target:
                    m ^= 1 << (N + t.value)
            obs[oi] = m
    return rec, coords, obs, N


def incidence_columns(circ, p):
    """col[d] = bitmask over DEM error mechanisms that flip detector d."""
    dem = uniform_noise(circ, p).detector_error_model(
        allow_gauge_detectors=False, decompose_errors=False)
    col = [0] * circ.num_detectors
    e = 0
    for inst in dem.flattened():
        if inst.type == "error":
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    col[t.val] ^= 1 << e
            e += 1
    return col, e


# --------------------------------------------------------------- sparsify ----
def sparsify(rec, col, max_passes=30, verbose=True):
    """Greedy pairwise column reduction.  Candidates for d_i are restricted to
    detectors that share at least one error mechanism with it -- XOR-ing with a
    disjoint column can only increase weight."""
    D = len(col)
    for p in range(max_passes):
        # error -> detectors touching it (rebuilt each pass; supports shrink)
        touch = collections.defaultdict(list)
        for d in range(D):
            c = col[d]
            while c:
                b = c & -c
                touch[b.bit_length() - 1].append(d)
                c ^= b
        changed = 0
        order = sorted(range(D), key=lambda i: col[i].bit_count(), reverse=True)
        for i in order:
            wi = col[i].bit_count()
            if wi == 0:
                continue
            cand = set()
            c = col[i]
            while c:
                b = c & -c
                cand.update(touch[b.bit_length() - 1])
                c ^= b
            cand.discard(i)
            cand = sorted(cand, key=lambda j: col[j].bit_count())
            improved = True
            while improved:
                improved = False
                for j in cand:
                    wj = col[j].bit_count()
                    if wj == 0:
                        continue
                    if wj >= wi:
                        break
                    w = (col[i] ^ col[j]).bit_count()
                    if 0 < w < wi:            # never let a column die
                        col[i] ^= col[j]
                        rec[i] ^= rec[j]
                        wi = w
                        improved = True
                        changed += 1
        if verbose:
            print(f"  pass {p + 1}: {changed} reductions, "
                  f"max column weight {max(x.bit_count() for x in col)}")
        if not changed:
            break
    return rec, col


# ------------------------------------------------------------------ emit -----
def rewrite(circ, rec, coords, N, out_path):
    out = stim.Circuit()
    tail = []
    for inst in circ.flattened():
        if inst.name == "DETECTOR":
            continue
        if inst.name == "OBSERVABLE_INCLUDE":
            tail.append(inst)
            continue
        out.append(inst)
    for m, co in zip(rec, coords):
        idxs = [i for i in range(N) if (m >> i) & 1]
        out.append("DETECTOR", [stim.target_rec(i - N) for i in idxs], co)
    for inst in tail:
        out.append(inst)
    out.to_file(out_path)
    return out


# ------------------------------------------------------------------ gate -----
# NOTE: these use a pivot dictionary keyed by the highest set bit.  The earlier
# version kept a sorted basis list and re-sorted on every insertion, which is
# O(n^2 log n) big-integer comparisons -- fine at d5 (1076 detectors), effectively
# a hang at d7 (3018) and d9.  This version is O(n * rank) XORs: instant.
def gf2_rank(vecs):
    piv, r = {}, 0
    for x in vecs:
        while x:
            h = x.bit_length() - 1
            if h in piv:
                x ^= piv[h]
            else:
                piv[h] = x
                r += 1
                break
    return r


def in_span(vecs, target):
    piv = {}
    for x in vecs:
        while x:
            h = x.bit_length() - 1
            if h in piv:
                x ^= piv[h]
            else:
                piv[h] = x
                break
    while target:
        h = target.bit_length() - 1
        if h in piv:
            target ^= piv[h]
        else:
            return False
    return True


def explain_shortest(noisy, coords_of):
    """Print where the minimal graphlike fault path actually lives."""
    try:
        expl = noisy.shortest_graphlike_error(canonicalize_circuit_errors=True)
    except Exception as e:
        print(f"  shortest_graphlike_error: n/a ({str(e).splitlines()[0][:60]})")
        return None
    print(f"  shortest_graphlike_error = {len(expl)}  -- fault path:")
    for k, e in enumerate(expl):
        loc = e.circuit_error_locations[0]
        bits = []
        for t in loc.flipped_pauli_product:
            q = t.gate_target.value
            bits.append(f"{t.gate_target.pauli_type}(q{q} @{tuple(coords_of.get(q, [])[:2])})")
        if loc.flipped_measurement is not None:
            bits.append(f"MEASFLIP rec#{loc.flipped_measurement.record_index}")
        print(f"    [{k}] tick={loc.tick_offset:<5d} {loc.instruction_targets.gate:<12s} "
              + "  ".join(bits))
    return len(expl)


def degree_hist(circ, p):
    dem = uniform_noise(circ, p).detector_error_model(
        allow_gauge_detectors=False, decompose_errors=False)
    h = collections.Counter()
    for inst in dem.flattened():
        if inst.type == "error":
            h[len([t for t in inst.targets_copy() if t.is_relative_detector_id()])] += 1
    return dict(sorted(h.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--noise", type=float, default=1e-3,
                    help="p used to build the incidence matrix (basis choice is "
                         "insensitive to this; it only names the error mechanisms)")
    ap.add_argument("--hyperedge", action="store_true",
                    help="also run search_for_undetectable_logical_errors. OFF by "
                         "default: it is memory-hungry and gets OOM-killed at d7+. "
                         "It is also unnecessary -- the span check below already "
                         "proves the true distance is unchanged.")
    ap.add_argument("--no-distance", action="store_true",
                    help="skip the graphlike distance / fault-path readout")
    args = ap.parse_args()
    out = args.output or args.input.rsplit(".", 1)[0] + "_elem.stim"

    circ = stim.Circuit.from_file(args.input)
    rec0, coords, obs, N = read_annotations(circ)
    print(f"=== elementary_basis.py  {args.input} -> {out} ===")
    print(f"  detectors {len(rec0)}   measurements {N}")
    print(f"  BEFORE record-weight hist : "
          f"{dict(sorted(collections.Counter(x.bit_count() for x in rec0).items()))}")
    print(f"  BEFORE error-degree hist  : {degree_hist(circ, args.noise)}")

    col, nerr = incidence_columns(circ, args.noise)
    print(f"  DEM error mechanisms      : {nerr}")
    rec = list(rec0)
    sparsify(rec, col)

    ann = rewrite(circ, rec, coords, N, out)
    print(f"  AFTER  record-weight hist : "
          f"{dict(sorted(collections.Counter(x.bit_count() for x in rec).items()))}")
    print(f"  AFTER  error-degree hist  : {degree_hist(ann, args.noise)}")
    changed = sum(1 for a, b in zip(rec0, rec) if a != b)
    print(f"  detectors rewritten       : {changed}")

    # ---- the gate: everything that mattered before must still hold -----------
    ra, rb, ru = gf2_rank(rec0), gf2_rank(rec), gf2_rank(rec0 + rec)
    ok_span = ra == rb == ru
    ok_obs = all(not in_span(rec, m) for m in obs.values())
    ok_det = not ann.compile_detector_sampler(seed=7).sample(2000).any()
    try:
        ann.detector_error_model(allow_gauge_detectors=False)
        ok_strict = True
    except Exception:
        ok_strict = False
    print(f"  span preserved (rank {ra}/{rb}/{ru})   : {'PASS' if ok_span else 'FAIL'}")
    if ok_span:
        print("     -> an error is undetectable in the NEW annotation iff it was")
        print("        undetectable in the OLD one; the TRUE distance is unchanged")
        print("        by construction.  Any change in shortest_graphlike_error is a")
        print("        change in how tightly the graphlike search can BOUND it.")
    print(f"  observable outside span              : {'PASS' if ok_obs else 'FAIL'}")
    print(f"  detectors deterministic (noiseless)  : {'PASS' if ok_det else 'FAIL'}")
    print(f"  strict DEM (no gauge detectors)      : {'PASS' if ok_strict else 'FAIL'}")

    if not args.no_distance:
        noisy = uniform_noise(ann, args.noise)
        explain_shortest(noisy, ann.get_final_qubit_coordinates())
        if args.hyperedge:
            try:
                se = len(noisy.search_for_undetectable_logical_errors(
                    dont_explore_detection_event_sets_with_size_above=4,
                    dont_explore_edges_with_degree_above=4,
                    dont_explore_edges_increasing_symptom_degree=False))
                print(f"  search_for_undetectable_logical_errors = {se}")
            except Exception as e:
                print(f"  search_for_undetectable_logical_errors: n/a "
                      f"({str(e).splitlines()[0][:60]})")

    ok = ok_span and ok_obs and ok_det and ok_strict
    print(f"  VERDICT: {'ELEMENTARY + VALID' if ok else 'FAIL (see above)'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
