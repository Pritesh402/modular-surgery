#!/usr/bin/env python3
"""Deterministic-parity detector discovery (PHASE 2B).

Time-ordered Gaussian elimination over the noiseless measurement-sample matrix.
Each measurement record either CLOSES a deterministic parity (-> emit a detector
anchored at that record, built from the records that XOR to a
deterministically-0 combination) or introduces a fresh independent random bit
(-> a new "live" generator). Records that never close are GAUGE (no detector);
we report them. This is partition-agnostic: no hardcoded seam rule, no
round-parity special-casing -- it discovers whatever local (or non-local)
deterministic combinations the circuit actually has.
"""
import numpy as np
import stim


def sample_matrix(circuit, shots=4000, seed=None):
    stripped = stim.Circuit()
    for inst in circuit.without_noise().flattened():
        if inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            continue
        stripped.append(inst)
    s = stripped.compile_sampler(seed=seed) if seed is not None \
        else stripped.compile_sampler()
    S = s.sample(shots).astype(np.uint8)
    return S


def record_info(circuit):
    """abs -> (qubit, round or None if final, is_final)."""
    layers = [[t.value for t in inst.targets_copy() if t.is_qubit_target]
              for inst in circuit.flattened()
              if inst.name in ("M", "MX", "MY", "MZ")]
    info = {}
    absidx = 0
    r = 0
    for li, tgt in enumerate(layers):
        final = (li == len(layers) - 1)
        for q in tgt:
            info[absidx] = (q, None if final else r, final)
            absidx += 1
        if not final:
            r += 1
    return info, absidx, r


def discover(S):
    """Return (detectors, gauge). detectors = list of frozenset(abs record idx)
    each XOR-ing to deterministic 0; gauge = list of abs idx never closed.

    Time-ordered elimination pivoted on SHOT rows: `live` holds independent
    random generators, each pivoted on a distinct shot index. A record closes
    iff its (reduced) sample column becomes all-zero."""
    n_shots, N = S.shape
    live = []          # list of dict: {'red': np.uint8[n_shots], 'combo': set, 'piv': int}
    detectors = []
    gauge = []
    for m in range(N):
        red = S[:, m].copy()
        combo = {m}
        for g in live:
            if red[g['piv']]:
                red ^= g['red']
                combo ^= g['combo']
        if not red.any():
            detectors.append(frozenset(combo))
        elif red.all():
            # deterministically 1 -> still a (odd) deterministic parity; emit,
            # but flag: with proper resets this is unusual.
            detectors.append(frozenset(combo))
        else:
            piv = int(np.argmax(red))     # first shot where red==1
            live.append({'red': red, 'combo': combo, 'piv': piv})
    gauge = [g for g in live]             # generators that never closed
    return detectors, live


def gauge_records(circuit, S=None):
    """Return the set of abs record indices that are in NO deterministic parity
    (truly random / gauge). A record is gauge iff e_m is NOT in the row space of
    S, i.e. it never gets 'covered' by any kernel vector."""
    if S is None:
        S = sample_matrix(circuit)
    n_shots, N = S.shape
    # Build kernel-covered set: reduce S to find pivots; free columns + their
    # pivot dependencies are 'covered'. A column is gauge iff it is an
    # INDEPENDENT pivot that no free column depends on.
    A = S.copy()
    rr = 0
    pivrow = {}
    where = []
    for cc in range(N):
        piv = None
        for k in range(rr, n_shots):
            if A[k, cc]:
                piv = k
                break
        if piv is None:
            continue
        A[[rr, piv]] = A[[piv, rr]]
        for k in range(n_shots):
            if k != rr and A[k, cc]:
                A[k] ^= A[rr]
        pivrow[cc] = rr
        where.append(cc)
        rr += 1
        if rr == n_shots:
            break
    pivots = set(where)
    free = [cc for cc in range(N) if cc not in pivots]
    covered = set(free)
    for f in free:
        for cc in where:
            if A[pivrow[cc], f]:
                covered.add(cc)
    return set(range(N)) - covered


if __name__ == "__main__":
    import sys
    from collections import defaultdict
    path = sys.argv[1] if len(sys.argv) > 1 else "d5_alt_asPaper.txt"
    c = stim.Circuit.from_file(path)
    coords = c.get_final_qubit_coordinates()
    info, N, NR = record_info(c)
    S = sample_matrix(c, shots=4000)
    dets, live = discover(S)
    g = gauge_records(c, S)
    # weight distribution
    wd = defaultdict(int)
    for d in dets:
        wd[len(d)] += 1
    print(f"{path}: N={N} rounds={NR} #detectors={len(dets)} #live(random)={len(live)}")
    print(f"  detector weight distribution: {dict(sorted(wd.items()))}")
    print(f"  #gauge records = {len(g)}: "
          f"{sorted((info[a][0], info[a][1]) for a in g)}")
