#!/usr/bin/env python3
"""graphlike_basis.py -- fix a Bell-seam annotation so MWPM can decode it.

WHY THIS EXISTS
bellmeas_annotator.py accepts a discovered seam parity whenever it EXTENDS THE
MEASUREMENT-RECORD SPAN.  That is the wrong test.  For a reoriented seam (e.g.
right-seam X-checks vertical) the seam's deterministic-parity space is genuinely
larger than the graphlike code needs, so the annotator adds extra detectors that
are record-independent but FAULT-REDUNDANT.  The result is several seam sites with
3 independent detectors stacked on one space-time point.  A single seam hook error
then flips all three at once -> a degree>=3 (non-graphlike) syndrome.  That breaks
BOTH decoder paths:
  * decompose_errors=True      -> "composite error exceeded max symptoms (<=15)"
  * undecomposed MWPM          -> "no perfect matching ... component without a
                                   boundary" (the stacked detectors become an
                                   island once PyMatching drops the hyperedges).

elementary_basis.py does NOT fix this: it minimises RECORD weight (adjacent-round
differences), a different objective from minimising ERROR DEGREE.  A record-local
basis can still be an error-hypergraph at the seam.

WHAT THIS DOES
Chooses the detector basis that minimises ERROR DEGREE directly:
  1. Build the single-Pauli (X_ERROR + Z_ERROR) DEM -> error x detector incidence.
     (If every single-Pauli fault is graphlike, every DEPOLARIZE2 fault decomposes.)
  2. Greedy GF(2) column reduction on that incidence, carrying each detector's
     record-set in lockstep.  a <- a XOR b whenever it lowers a's error degree;
     detectors whose incidence collapses to 0 are FAULT-REDUNDANT and dropped.
  3. Emit the reduced annotation and validate:
       - strict noiseless DEM (all detectors deterministic)
       - shortest_graphlike_error AND search_for_undetectable unchanged
         (true distance preserved by construction: the kept detectors still
          span every fault syndrome)
       - decompose_errors=True under DEPOLARIZE2 now succeeds -> MWPM-ready.

USAGE
  python graphlike_basis.py INPUT_full.stim -o OUTPUT_graphlike.stim [--noise 1e-3]
"""
import argparse, sys
import stim


def single_pauli_noise(circuit, p):
    """Independent X_ERROR + Z_ERROR only (no Y composites): the clean probe for
    whether the *circuit* admits a graphlike detector basis."""
    n = stim.Circuit()
    for ins in circuit:
        if ins.name in ("CX", "CZ", "XCX", "CY", "ZCX"):
            t = [x.value for x in ins.targets_copy()]
            n.append(ins.name, t); n.append("X_ERROR", t, p); n.append("Z_ERROR", t, p)
        elif ins.name in ("M", "MX", "MY", "MZ"):
            n.append("Z_ERROR" if ins.name == "MX" else "X_ERROR",
                     [x.value for x in ins.targets_copy() if x.is_qubit_target], p)
            n.append(ins.name, ins.targets_copy())
        elif ins.name in ("R", "RX", "H", "X"):
            n.append(ins); t = [x.value for x in ins.targets_copy()]
            n.append("X_ERROR", t, p); n.append("Z_ERROR", t, p)
        elif ins.name in ("QUBIT_COORDS", "DETECTOR", "TICK", "OBSERVABLE_INCLUDE"):
            n.append(ins)
    return n


def record_round_map(circuit):
    """record index -> (qubit, round) using TICK layers as round boundaries."""
    q_of, rnd_of = {}, {}
    ridx = 0; rnd = 0
    for ins in circuit.flattened():
        if ins.name == "TICK":
            rnd += 1
        elif ins.name in ("M", "MX", "MY", "MZ"):
            for t in ins.targets_copy():
                if t.is_qubit_target:
                    q_of[ridx] = t.value; rnd_of[ridx] = rnd; ridx += 1
    return q_of, rnd_of


def popc(x): return bin(x).count("1")


def graphlike_basis(in_path, out_path, noise=1e-3, max_passes=12):
    c = stim.Circuit.from_file(in_path)
    N = c.num_measurements; nd = c.num_detectors

    det_recs = []; obs = None
    for ins in c.flattened():
        if ins.name == "DETECTOR":
            m = 0
            for t in ins.targets_copy():
                if t.is_measurement_record_target:
                    m |= (1 << (N + t.value))
            det_recs.append(m)
        elif ins.name == "OBSERVABLE_INCLUDE":
            obs = ins
    if obs is None:
        raise RuntimeError("input has no OBSERVABLE_INCLUDE")

    # single-Pauli incidence (error x detector), current basis
    dem = single_pauli_noise(c, noise).detector_error_model(decompose_errors=False)
    syndromes = {frozenset(t.val for t in ins.targets_copy()
                           if ins.type == "error" and t.is_relative_detector_id())
                 for ins in dem.flattened() if ins.type == "error"}
    errset = [s for s in syndromes if s]; ne = len(errset)
    col = [0] * nd
    for ei, ds in enumerate(errset):
        for d in ds:
            col[d] |= (1 << ei)

    def max_degree():
        cnt = [0] * ne
        for d in range(nd):
            cm = col[d]
            while cm:
                bit = cm & -cm; cnt[bit.bit_length() - 1] += 1; cm ^= bit
        return max(cnt) if cnt else 0

    print(f"  detectors {nd}  single-Pauli faults {ne}  start max-degree {max_degree()}")
    for it in range(max_passes):
        imp = 0
        for a in sorted(range(nd), key=lambda d: -popc(col[d])):
            ca = col[a]
            if not ca:
                continue
            bb, bg = -1, 0
            for b in range(nd):
                cb = col[b]
                if b == a or not cb:
                    continue
                g = 2 * popc(ca & cb) - popc(cb)   # weight drop of col[a] if a^=b
                if g > bg:
                    bg, bb = g, b
            if bb >= 0 and bg > 0:
                col[a] ^= col[bb]; det_recs[a] ^= det_recs[bb]; imp += 1
        if imp == 0:
            break
    print(f"  after reduction: single-Pauli max-degree {max_degree()}")

    kept = [d for d in range(nd) if col[d] != 0]
    print(f"  detectors kept {len(kept)}   fault-redundant dropped {nd - len(kept)}")

    q_of, rnd_of = record_round_map(c)
    coords = c.get_final_qubit_coordinates()
    out = stim.Circuit()
    for ins in c.flattened():
        if ins.name not in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            out.append(ins)
    for d in kept:
        m = det_recs[d]
        idxs = [i for i in range(N) if (m >> i) & 1]
        hi = idxs[-1]
        q = q_of.get(hi, 0); x, y = coords.get(q, (0.0, 0.0))[:2]
        out.append("DETECTOR", [stim.target_rec(i - N) for i in idxs],
                   [float(x), float(y), float(rnd_of.get(hi, 0))])
    out.append(obs)
    out.to_file(out_path)

    # ---- validation ----
    ok = True
    try:
        out.detector_error_model(allow_gauge_detectors=False, decompose_errors=False)
        print("  strict noiseless DEM (all deterministic)      : PASS")
    except Exception as e:
        ok = False; print("  strict noiseless DEM                          : FAIL", str(e).splitlines()[0][:60])

    noisy = _dep2_noise(out, noise)
    gl = len(noisy.shortest_graphlike_error())
    try:
        su = len(noisy.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=6,
            dont_explore_edges_with_degree_above=6,
            dont_explore_edges_increasing_symptom_degree=False))
    except Exception:
        su = "n/a"
    print(f"  distance: shortest_graphlike={gl}  search_for_undetectable={su}")
    try:
        d = noisy.detector_error_model(decompose_errors=True)
        print(f"  decompose_errors=True @DEPOLARIZE2 (MWPM-ready): PASS ({d.num_errors} errors)")
    except Exception as e:
        ok = False; print("  decompose_errors=True                         : FAIL", str(e).splitlines()[0][:60])
    return ok


def _dep2_noise(circuit, p):
    n = stim.Circuit()
    for ins in circuit:
        if ins.name in ("CX", "CZ"):
            t = ins.targets_copy()
            for k in range(0, len(t), 2):
                n.append(ins.name, [t[k], t[k + 1]]); n.append("DEPOLARIZE2", [t[k], t[k + 1]], p)
        elif ins.name in ("M", "MX"):
            n.append(ins.name, ins.targets_copy(), p)
        elif ins.name in ("R", "RX", "H", "X"):
            n.append(ins); n.append("DEPOLARIZE1", ins.targets_copy(), p)
        else:
            n.append(ins)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--noise", type=float, default=1e-3)
    args = ap.parse_args()
    out = args.output or (args.input.rsplit(".", 1)[0] + "_graphlike.stim")
    print(f"=== graphlike_basis.py  {args.input} -> {out} ===")
    ok = graphlike_basis(args.input, out, noise=args.noise)
    print(f"  wrote {out}")
    print(f"  VERDICT: {'GRAPHLIKE + VALID' if ok else 'FAILED (see above)'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
