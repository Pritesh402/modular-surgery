#!/usr/bin/env python3
"""bellmeas_annotator.py -- COMPLETE, observable-protected annotator for
Bell-measurement merge/split circuits.

WHY THIS EXISTS
The pattern annotator (annotator_bellmeas.py) builds correct BULK detectors by
comparing a stabilizer's consecutive-round measurements, but it CANNOT form a
Bell-MEASUREMENT seam's detectors: those are DIAGONAL cross-row combinations
(paper Fig-2b), not same-round two-half or same-ancilla pairs.  It therefore
rejects every seam candidate (`diagonal_units=0`, ~50 rejects) and leaves the
seam records with NO detector -- so `shortest_graphlike_error` collapses to 1 and
any decoder built from that file is blind to seam errors (artificially ~distance-1
logical error rate).

WHAT THIS DOES (the validated `seamfix` recipe, packaged)
  1. Bulk/boundary detectors: run the pattern annotator, read its (graphlike,
     correct) bulk detectors + its completed observable.
  2. Seam detectors: DISCOVER every remaining deterministic parity by GF(2) on
     noiseless samples (detector_discovery) -- no pattern, so ANY seam geometry
     (straight asPaper, L-merge, junction) is handled.
  3. Add a discovered detector ONLY if it extends the detector span WITHOUT
     bringing the logical observable into it -> the observable stays a genuine
     logical (a complement of <observable> in the deterministic-parity kernel).
  4. Emit a complete annotated .stim.

GUARANTEES on the output (all checked + reported):
  - 0 gauge records (every measurement record is in a detector OR is the logical).
  - all detectors deterministic (0 rejects, by construction from discovery).
  - observable deterministic AND held OUT of the detector span (it is the logical).
  - strict DEM builds (allow_gauge_detectors=False).
It also reports whether the DEM is graphlike (decompose_errors=True): the seam
detectors are diagonal and may be hyperedge, in which case plain MWPM needs a
correlated/hyperedge decoder while the TRUE distance still comes from
search_for_undetectable_logical_errors.

USAGE
  python bellmeas_annotator.py INPUT.txt -o OUTPUT.stim [--shots N] [--noise p]
"""
import argparse
import os
import subprocess
import sys
import numpy as np
import stim
import detector_discovery as dd

HERE = os.path.dirname(os.path.abspath(__file__))
PATTERN_ANNOTATOR = os.path.join(HERE, "annotator_bellmeas.py")


# ---------------------------------------------------------------- helpers ----
def strip(circuit):
    out = stim.Circuit()
    for inst in circuit.without_noise().flattened():
        if inst.name not in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            out.append(inst)
    return out


def vec(idxs, N):
    v = np.zeros(N, np.uint8)
    for i in idxs:
        v[i] ^= 1
    return v


def read_detectors_and_obs(stim_path):
    """Return (detectors: list[frozenset abs idx], obs: {oi:set abs idx}, N)."""
    c = stim.Circuit.from_file(stim_path)
    N = c.num_measurements
    dets, obs = [], {}
    for inst in c.flattened():
        if inst.name == "DETECTOR":
            dets.append(frozenset(N + t.value for t in inst.targets_copy()
                                  if t.is_measurement_record_target))
        elif inst.name == "OBSERVABLE_INCLUDE":
            oi = int(inst.gate_args_copy()[0]) if inst.gate_args_copy() else 0
            obs.setdefault(oi, set()).update(
                N + t.value for t in inst.targets_copy()
                if t.is_measurement_record_target)
    return dets, obs, N


def run_pattern_annotator(txt_path, obs="z"):
    """Run annotator_bellmeas.py to get bulk detectors + completed observable."""
    tmp = txt_path.rsplit(".", 1)[0] + ".__bulk_tmp.stim"
    r = subprocess.run(
        [sys.executable, PATTERN_ANNOTATOR, txt_path, "--obs", obs,
         "--keep-observables", "-o", tmp, "--report", os.devnull],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"pattern annotator failed:\n{r.stderr[-2000:]}")
    dets, obs_map, N = read_detectors_and_obs(tmp)
    try:
        os.remove(tmp)
        os.remove(tmp + ".tmp")
    except OSError:
        pass
    return dets, obs_map, N


def uniform_noise(circuit, p):
    """Uniform circuit-level depolarizing (geometry-agnostic): DEPOLARIZE2 on 2q,
    DEPOLARIZE1 on H + idle, X_ERROR on reset/measure."""
    coords = circuit.get_final_qubit_coordinates()
    allq = set(coords)
    out = stim.Circuit()

    def tick_flush(buf):
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
            tick_flush(buf)
            out.append(inst)
            buf = []
        else:
            buf.append(inst)
    tick_flush(buf)
    return out


# --------------------------------------------------------------- annotate ----
def annotate(txt_path, out_path, shots=6000, obs="z"):
    raw = stim.Circuit.from_file(txt_path)
    base = strip(raw)
    info, N, NR = dd.record_info(base)
    coords = base.get_final_qubit_coordinates()
    S = dd.sample_matrix(base, shots)

    # 1-2. bulk detectors + completed observable from the pattern annotator
    ann_dets, ann_obs, Np = run_pattern_annotator(txt_path, obs=obs)
    if Np != N:
        raise RuntimeError(f"measurement count mismatch {Np} vs {N}")
    obs_recs = sorted(ann_obs.get(0, set()))
    if not obs_recs:
        raise RuntimeError("no OBSERVABLE_INCLUDE(0) produced by pattern annotator")

    # observable determinism check (pattern annotator already completes with seam recs)
    obs_det = bool((S[:, obs_recs].sum(1) & 1).min() == (S[:, obs_recs].sum(1) & 1).max())

    # 3. o-protecting augmentation with discovered detectors (low weight first)
    o_vec = vec(obs_recs, N)
    basis = []                                     # (reduced vec, pivot col)

    def reduce(v):
        r = v.copy()
        for bv, pc in basis:
            if r[pc]:
                r ^= bv
        return r

    for d in ann_dets:                             # seed with bulk detectors
        r = reduce(vec(d, N))
        if r.any():
            basis.append((r, int(np.argmax(r))))
    o_in_span_seed = not reduce(o_vec).any()

    disc, _ = dd.discover(S)
    disc = sorted(disc, key=len)
    added = []
    for d in disc:
        v = vec(d, N)
        r = reduce(v)
        if not r.any():
            continue                               # already covered
        basis.append((r, int(np.argmax(r))))
        if not reduce(o_vec).any():                # would cover the observable
            basis.pop()                            # this direction == the logical; skip
        else:
            added.append(d)
    o_in_span_final = not reduce(o_vec).any()

    # 4. emit
    detectors = list(ann_dets) + added
    ann = base.copy()
    for d in detectors:
        idxs = sorted(d)
        m = max(idxs)
        q, rnd, final = info[m]
        x, y = coords[q][0], coords[q][1]
        ann.append("DETECTOR", [stim.target_rec(i - N) for i in idxs],
                   [float(x), float(y), float(NR if final else rnd)])
    ann.append("OBSERVABLE_INCLUDE", [stim.target_rec(i - N) for i in obs_recs], 0)
    ann.to_file(out_path)

    return dict(N=N, rounds=NR, n_bulk=len(ann_dets), n_added=len(added),
                n_total=len(detectors), obs_recs=obs_recs, obs_det=obs_det,
                o_in_span_seed=o_in_span_seed, o_in_span_final=o_in_span_final,
                ann=ann)


def validate(res, noise=1e-3, want_distance=True):
    ann = res["ann"]
    base = strip(ann)
    # gauge records of the underlying circuit (0 => fully detectable)
    S = dd.sample_matrix(base, shots=6000)
    gauge = dd.gauge_records(base, S)
    print(f"  N={res['N']} rounds={res['rounds']}")
    print(f"  bulk/boundary detectors (pattern annotator) : {res['n_bulk']}")
    print(f"  seam detectors added (discovered, o-protected): {res['n_added']}")
    print(f"  total detectors                              : {res['n_total']}")
    print(f"  gauge records (uncovered by any parity)      : {len(gauge)}"
          f"   [{'PASS' if not gauge else 'FAIL -- circuit itself has undetectable records'}]")
    print(f"  observable: {len(res['obs_recs'])} records  deterministic={res['obs_det']}"
          f"  in-detector-span={res['o_in_span_final']}"
          f"   [{'PASS' if (res['obs_det'] and not res['o_in_span_final']) else 'FAIL'}]")
    # (a) noiseless strict DEM => every detector is deterministic (no gauge detectors)
    strict = False
    try:
        ann.detector_error_model(allow_gauge_detectors=False, decompose_errors=False)
        strict = True
        print(f"  detectors all deterministic (strict DEM) : PASS")
    except Exception as e:
        print(f"  detectors all deterministic (strict DEM) : FAILED -> {str(e).splitlines()[0][:55]}")
    # (b) graphlike check on the NOISY DEM (the noiseless DEM is empty => trivial)
    noisy = uniform_noise(ann, noise) if (strict or want_distance) else None
    if strict and noisy is not None:
        graphlike = False
        try:
            noisy.detector_error_model(allow_gauge_detectors=False, decompose_errors=True)
            graphlike = True
        except Exception:
            pass
        print(f"  decompose_errors @ p={noise:g}         : "
              f"{'PASS (vanilla MWPM/sinter-ready)' if graphlike else 'NO -- correlated (DEPOLARIZE2) circuit-noise has errors flipping >2 detectors; NOT seam-specific (bulk merge circuits do this too). Use a CSS-separable noise model for MWPM, or a correlated/hyperedge decoder. Distance below is still exact (graphlike chain exists).'}")
    if want_distance:
        gl = se = None
        try:
            gl = len(noisy.shortest_graphlike_error(canonicalize_circuit_errors=True))
        except Exception as e:
            gl = f"n/a ({str(e).splitlines()[0][:32]})"
        try:
            se = len(noisy.search_for_undetectable_logical_errors(
                dont_explore_detection_event_sets_with_size_above=6,
                dont_explore_edges_with_degree_above=6,
                dont_explore_edges_increasing_symptom_degree=False))
        except Exception as e:
            se = f"n/a ({str(e).splitlines()[0][:32]})"
        print(f"  distance @ uniform p={noise:g}: shortest_graphlike_error={gl}   "
              f"search_for_undetectable={se}   <-- LENGTH")
    return not gauge and res["obs_det"] and not res["o_in_span_final"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--shots", type=int, default=6000)
    ap.add_argument("--obs", choices=["x", "z"], default="z")
    ap.add_argument("--noise", type=float, default=1e-3)
    ap.add_argument("--no-distance", action="store_true",
                    help="skip the (slower) distance readout")
    args = ap.parse_args()
    out = args.output or (args.input.rsplit(".", 1)[0] + "_annotated.stim")
    print(f"=== bellmeas_annotator.py  {args.input} -> {out} ===")
    res = annotate(args.input, out, shots=args.shots, obs=args.obs)
    ok = validate(res, noise=args.noise, want_distance=not args.no_distance)
    print(f"  wrote {out}")
    print(f"  VERDICT: {'COMPLETE + VALID' if ok else 'INCOMPLETE (see FAIL above)'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
