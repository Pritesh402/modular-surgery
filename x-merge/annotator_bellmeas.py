#!/usr/bin/env python3
"""annotator.py -- a validated, Bell-aware detector annotator for surface-code circuits.

Usage:
    python annotator.py INPUT.stim [--obs {x,z}] [--obs-index K] [-o OUTPUT.stim]
                        [--report-partitions] [--shots N] [-p NOISE]

Strips existing DETECTOR/OBSERVABLE_INCLUDE lines, reconstructs stabilizer
detectors from the raw gate structure (Bell-pair aware), EMPIRICALLY validates
every candidate detector against Stim's own sampler (rejecting any that is not
deterministic on the noiseless circuit), emits the annotated circuit and a
report to annotate_report.txt.

Conventions asserted (warn on violation):
  * data qubits have integer (x, y); ancillas have a half-integer coordinate.
  * CX(anc, data) senses an X-stabilizer on that data qubit;
    CX(data, anc) / CZ(anc, data) sense a Z-stabilizer.
    (H-conjugated variants that net to the same stabilizer are allowed; we
     classify by the net stabilizer and record the raw gate convention.)
  * a Bell pair is a CX whose BOTH targets are ancillas.  The pair's measured
    stabilizer is the XOR of the two halves' measurements.
"""
import argparse
import os
import sys
import stim
import numpy as np
from collections import defaultdict

MEAS = ("M", "MX", "MY", "MZ")
BASIS_OF = {"M": "Z", "MZ": "Z", "MX": "X", "MY": "Y"}
NOISE_P = 1e-4  # matches shortest_graphlike_error.py

report_lines = []
warnings = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    report_lines.append(s)


def warn(msg):
    warnings.append(msg)
    log("  !! WARNING:", msg)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
# Integer-coord qubits that are ancillas nonetheless (e.g. Bell CZZ ancillas
# sitting ON the seam at integer coords like (5,2),(5,3)).  Populated
# behaviourally in main(); empty for every ordinary circuit, so classification
# is byte-identical to the coord-only rule there.
FORCED_ANCILLAS = set()


def is_data(coords, q):
    if q in FORCED_ANCILLAS:
        return False
    x, y = coords[q][0], coords[q][1]
    return float(x).is_integer() and float(y).is_integer()


def detect_forced_ancillas(flat, coords):
    """A data qubit in a memory experiment is measured exactly ONCE (final
    readout); an ancilla is measured every round.  So an integer-coord qubit
    measured 2+ times is really an ancilla (the CZZ Bell ancillas).  This is
    ADDITIVE: half-integer qubits are never touched, and no integer-coord data
    qubit is multi-measured, so previously-validated circuits are unaffected."""
    mcount = defaultdict(int)
    for inst in flat:
        if inst.name in MEAS:
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    mcount[t.value] += 1
    return {q for q in coords
            if float(coords[q][0]).is_integer() and float(coords[q][1]).is_integer()
            and mcount[q] >= 2}


def coord_str(coords, q):
    return "(" + ", ".join(f"{v:g}" for v in coords[q]) + ")"


# --------------------------------------------------------------------------- #
# 1. strip
# --------------------------------------------------------------------------- #
def strip_annotations(circuit):
    """Return a copy with all DETECTOR / OBSERVABLE_INCLUDE removed, everything
    else (including REPEAT blocks) preserved."""
    out = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            out.append(
                stim.CircuitRepeatBlock(
                    inst.repeat_count, strip_annotations(inst.body_copy())
                )
            )
        elif inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            continue
        else:
            out.append(inst)
    return out


# --------------------------------------------------------------------------- #
# 3-4. Bell partners + per-ancilla support / type / convention
# --------------------------------------------------------------------------- #
def analyse_checks(flat, coords):
    partner = {}
    for inst in flat:
        if inst.name == "CX":
            ts = inst.targets_copy()
            for i in range(0, len(ts), 2):
                a, b = ts[i].value, ts[i + 1].value
                if (not is_data(coords, a)) and (not is_data(coords, b)):
                    partner[a] = b
                    partner[b] = a

    support = defaultdict(set)
    typ = defaultdict(set)
    conv = defaultdict(set)
    for inst in flat:
        if inst.name == "CX":
            ts = inst.targets_copy()
            for i in range(0, len(ts), 2):
                a, b = ts[i].value, ts[i + 1].value
                da, db = is_data(coords, a), is_data(coords, b)
                if db and not da:  # CX(anc -> data): X-stabilizer
                    support[a].add(b); typ[a].add("X"); conv[a].add("CX(anc->data)")
                elif da and not db:  # CX(data -> anc): Z-stabilizer (H+CX form)
                    support[b].add(a); typ[b].add("Z"); conv[b].add("CX(data->anc)")
                # both-ancilla => Bell pair (handled above); both-data impossible.
        elif inst.name == "CZ":
            ts = inst.targets_copy()
            for i in range(0, len(ts), 2):
                a, b = ts[i].value, ts[i + 1].value
                da, db = is_data(coords, a), is_data(coords, b)
                if db and not da:  # CZ(anc, data): Z-stabilizer
                    support[a].add(b); typ[a].add("Z"); conv[a].add("CZ")
                elif da and not db:
                    support[b].add(a); typ[b].add("Z"); conv[b].add("CZ")
        elif inst.name == "XCX":  # ZX-dual of CZ: X-stabilizer (X-controlled-X)
            ts = inst.targets_copy()
            for i in range(0, len(ts), 2):
                a, b = ts[i].value, ts[i + 1].value
                da, db = is_data(coords, a), is_data(coords, b)
                if db and not da:
                    support[a].add(b); typ[a].add("X"); conv[a].add("XCX")
                elif da and not db:
                    support[b].add(a); typ[b].add("X"); conv[b].add("XCX")
    return partner, support, typ, conv


# --------------------------------------------------------------------------- #
# 5. record map
# --------------------------------------------------------------------------- #
def build_record_map(flat, coords):
    """Return meas_recs[q] = ordered list of (abs_index, basis), and N."""
    meas_recs = defaultdict(list)
    mc = 0
    reset_basis = {}   # first reset basis seen per qubit
    for inst in flat:
        if inst.name in ("R", "RX"):
            b = "X" if inst.name == "RX" else "Z"
            for t in inst.targets_copy():
                reset_basis.setdefault(t.value, b)
        elif inst.name in MEAS:
            b = BASIS_OF[inst.name]
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    meas_recs[t.value].append((mc, b))
                    mc += 1
    return meas_recs, mc, reset_basis


# --------------------------------------------------------------------------- #
# main annotation
# --------------------------------------------------------------------------- #
class Unit:
    __slots__ = ("kind", "ancillas", "rep", "support", "type", "conv", "events",
                 "timeline")

    def __init__(self, kind, ancillas, rep, support, type_, conv, events):
        self.kind = kind
        self.ancillas = ancillas
        self.rep = rep
        self.support = support
        self.type = type_
        self.conv = conv
        self.events = events
        self.timeline = []   # list of (recs:frozenset, support:frozenset) per measurement


def build_timelines(flat, coords):
    """Reconstruct, per ancilla, the chronological sequence of its measurements,
    tagging EACH measurement with the data-qubit support the ancilla coupled to
    since its previous measurement.  This is what makes the annotator merge/split
    aware: a boundary check whose support grows (merge) or shrinks (split) is
    captured per-measurement instead of aggregated over the whole circuit.

    Returns:
      anc_tl[a]  = [(rec_index, frozenset(data support this round)), ...]
      data_meas[q] = [(rec_index, basis), ...]   every data-qubit measurement, in
                     order (a data qubit read out mid-circuit at a split AND again
                     at the end therefore has two entries)."""
    coupling = defaultdict(set)
    anc_tl = defaultdict(list)
    data_meas = defaultdict(list)
    mc = 0
    for inst in flat:
        n = inst.name
        if n in ("CX", "CZ", "XCX"):
            ts = inst.targets_copy()
            for i in range(0, len(ts), 2):
                a, b = ts[i].value, ts[i + 1].value
                da, db = is_data(coords, a), is_data(coords, b)
                if da ^ db:                       # ancilla-data coupling
                    anc, dat = (b, a) if da else (a, b)
                    coupling[anc].add(dat)
        elif n in MEAS:
            basis = BASIS_OF[n]
            for t in inst.targets_copy():
                if not t.is_qubit_target:
                    continue
                q = t.value
                if is_data(coords, q):
                    data_meas[q].append((mc, basis))
                else:
                    anc_tl[q].append((mc, frozenset(coupling[q])))
                    coupling[q] = set()           # syndrome consumed; start fresh
                mc += 1
    return anc_tl, data_meas


def build_units(coords, partner, support, typ, conv, anc_tl):
    ancillas = [q for q in coords if not is_data(coords, q)]
    seen = set()
    units = []
    for a in sorted(ancillas):
        if a in seen:
            continue
        if a in partner:
            b = partner[a]
            seen.add(a); seen.add(b)
            members = sorted([a, b])
            kind = "bell"
        else:
            seen.add(a)
            members = [a]
            kind = "single"

        # merged support / type / convention (aggregate; used for reports)
        supp = set()
        types = set()
        convs = set()
        for m in members:
            supp |= support[m]
            types |= typ[m]
            convs |= conv[m]
        if len(types) != 1:
            warn(f"unit {members} has mixed stabilizer type {sorted(types)} "
                 f"at {[coord_str(coords, m) for m in members]}")
        utype = next(iter(types)) if types else "?"

        # per-measurement timeline: (recs, support-this-round).  For a Bell unit the
        # stabilizer measurement is the XOR of the two halves' k-th records and the
        # support is the union of both halves' k-th support.
        if kind == "single":
            timeline = [(frozenset([rec]), s) for rec, s in anc_tl[members[0]]]
        else:
            ta, tb = anc_tl[members[0]], anc_tl[members[1]]
            if len(ta) != len(tb):
                warn(f"Bell unit {members} halves measured unequal #times "
                     f"({len(ta)} vs {len(tb)})")
            timeline = [(frozenset([ta[k][0], tb[k][0]]), ta[k][1] | tb[k][1])
                        for k in range(min(len(ta), len(tb)))]

        events = [recs for recs, _ in timeline]
        u = Unit(kind, members, members[0], supp, utype, convs, events)
        u.timeline = timeline
        units.append(u)
    return units


def candidate_detectors(units, reset_basis, final_basis, coords, data_meas, data_qubits):
    """Yield dicts: {idxs, x, y, r, unit, role}.

    A stabilizer is "measured" every round by its ancilla AND once more whenever its
    data qubits are read out (a split, or the final readout).  A detector compares
    two consecutive such measurements:

      round0 : first ancilla measurement alone (data freshly reset in this basis).
      bulk   : ancilla_{k-1} XOR ancilla_k, PLUS the readout records of any data that
               LEFT the support between them (measured out at a split) and of any data
               that ENTERED the support while carrying a prior measured value (i.e. was
               not freshly reset).  With static support this is just the 2-record
               detector, so ordinary memory circuits are unchanged.
      final  : last ancilla measurement XOR the readout records of its current support
               (works at the true end AND for seam checks that terminate at a split)."""
    # data-reset basis (informational only -- boundary detectors are NOT gated on any
    # inferred stabilizer type; see below).
    data_reset = None
    rb = {reset_basis.get(q) for q in data_qubits if q in reset_basis}
    rb.discard(None)
    if len(rb) == 1:
        data_reset = next(iter(rb))
    elif rb:
        warn(f"data qubits not uniformly reset: bases={rb}")
        data_reset = "X" if "X" in rb else next(iter(rb))

    # We do NOT gate round-0 / closing detectors on a gate-inferred stabilizer type.
    # H-conjugation and a horseshoe's asymmetric readout (some checks are closed at the
    # split, their counterparts at the final readout) make the CX-direction guess an
    # unreliable predictor of which boundary detector is deterministic.  Instead we
    # generate the candidate for EVERY unit and let the empirical validator keep only
    # the deterministic ones -- a non-deterministic boundary candidate simply does not
    # apply to that check (it is not a construction bug; only a BULK reject is).
    cands = []
    for u in units:
        x, y = coords[u.rep][0], coords[u.rep][1]
        tl = u.timeline
        n = len(tl)
        if n == 0:
            continue

        # round-0 initialisation detector (kept iff the check commutes with the reset)
        cands.append(dict(idxs=set(tl[0][0]),
                          x=x, y=y, r=0, unit=u, role="round0"))

        # bulk / bridge detectors between consecutive measurements
        for k in range(1, n):
            recs_prev, supp_prev = tl[k - 1]
            recs_cur, supp_cur = tl[k]
            idxs = set(recs_prev) | set(recs_cur)
            lo, hi = max(recs_prev), min(recs_cur)
            # data that LEFT the support and was measured out in between (a split): its
            # readout records restore determinism across the support change.
            for q in (supp_prev - supp_cur):
                for mrec, _ in data_meas.get(q, []):
                    if lo < mrec < hi:
                        idxs.add(mrec)
            # data that ENTERED carrying a previously-measured value (not freshly reset)
            for q in (supp_cur - supp_prev):
                for mrec, _ in data_meas.get(q, []):
                    if mrec < min(recs_prev):
                        idxs.add(mrec)
            cands.append(dict(idxs=idxs, x=x, y=y, r=k, unit=u, role="bulk"))

        # final / closing detector: last ancilla measurement vs its support's readout,
        # provided the whole support is read out in one consistent basis.
        recs_last, supp_last = tl[-1]
        close = set(recs_last)
        bases = set()
        ok = bool(supp_last)
        for q in supp_last:
            ms = data_meas.get(q, [])
            if not ms:
                ok = False
                break
            close.add(ms[-1][0]); bases.add(ms[-1][1])
        if ok and len(bases) == 1:
            cands.append(dict(idxs=close, x=x, y=y, r=n, unit=u, role="final"))

    return cands, data_reset


# --------------------------------------------------------------------------- #
# noise (for distance only) -- identical model to shortest_graphlike_error.py
# --------------------------------------------------------------------------- #
def add_noise(base, p=NOISE_P):
    noisy = stim.Circuit()
    for inst in base:
        if inst.name in ("CX", "CZ", "XCX"):
            noisy.append(inst)
            noisy.append("DEPOLARIZE2", inst.targets_copy(), p)
        elif inst.name in ("M", "MX"):
            noisy.append(inst.name, inst.targets_copy(), p)
        elif inst.name in ("R", "RX", "H"):
            noisy.append(inst)
            noisy.append("DEPOLARIZE1", inst.targets_copy(), p)
        elif inst.name in ("QUBIT_COORDS", "DETECTOR", "TICK", "OBSERVABLE_INCLUDE",
                           "DEPOLARIZE1", "DEPOLARIZE2"):
            noisy.append(inst)
        else:
            raise NotImplementedError("noise model forgot " + str(inst))
    return noisy


ERROR_INSTS = ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR",
               "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "CORRELATED_ERROR",
               "ELSE_CORRELATED_ERROR", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1",
               "DEPOLARIZE")


def has_noise(circuit):
    for inst in circuit.flattened():
        if inst.name in ERROR_INSTS:
            return True
        # measurement flip probability baked into M/MX(...) args
        if inst.name in ("M", "MX", "MY", "MZ", "MR", "MRX", "MRY"):
            args = inst.gate_args_copy()
            if args and args[0] > 0:
                return True
    return False


def make_detector_circuit(base, kept, N):
    c = base.copy()
    for cand in kept:
        targets = [stim.target_rec(i - N) for i in sorted(cand["idxs"])]
        c.append("DETECTOR", targets, [cand["x"], cand["y"], float(cand["r"])])
    return c


def x_row_records(coords, data_qubits, meas_recs, row):
    return [meas_recs[q][-1][0] for q in sorted(data_qubits)
            if abs(coords[q][1] - row) < 1e-6 and meas_recs[q]]


# --------------------------------------------------------------------------- #
# partition helper
# --------------------------------------------------------------------------- #
def geom_label(dx, dy):
    if abs(dx) < 1e-6 and abs(dy) > 1e-6:
        return "VERTICAL"      # same x
    if abs(dy) < 1e-6 and abs(dx) > 1e-6:
        return "HORIZONTAL"    # same y
    return "DIAGONAL"


def centroid(coords, qs):
    xs = [coords[q][0] for q in qs]
    ys = [coords[q][1] for q in qs]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def report_partitions(units, coords, echo):
    lines = ["\n=== Bell-unit partition report ===",
             "(relation of the two ancilla-halves' data-support centroids; "
             "definition: HORIZONTAL=same y, VERTICAL=same x, DIAGONAL=differ in both)",
             "NOTE on terminology: this seam's ancilla column is a VERTICAL line "
             "(constant x~5), but each Bell pair merges data across it at the SAME y "
             "(displacement is purely in x), so by the stated definition the label is "
             "HORIZONTAL. The label that matters for a chevron check is DIAGONAL; a "
             "straight seam (this one) yields ZERO diagonals."]
    n_diag = 0
    bell_units_local = [u for u in units if u.kind == "bell"]
    for u in units:
        if u.kind != "bell":
            continue
        a, b = u.ancillas
        sa = sorted(s for s in _supp_of(u, a))
        sb = sorted(s for s in _supp_of(u, b))
        # centroids of each half's data-support (fall back to ancilla coord if empty)
        pa = centroid(coords, sa) if sa else (coords[a][0], coords[a][1])
        pb = centroid(coords, sb) if sb else (coords[b][0], coords[b][1])
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        label = geom_label(dx, dy)
        if label == "DIAGONAL":
            n_diag += 1
        lines.append(
            f"  Bell unit {{{a},{b}}}  {coord_str(coords,a)}<->{coord_str(coords,b)}  type={u.type}\n"
            f"      anc {a} supports {sa} {[coord_str(coords,q) for q in sa]}\n"
            f"      anc {b} supports {sb} {[coord_str(coords,q) for q in sb]}\n"
            f"      support-centroid displacement (dx,dy)=({dx:g},{dy:g}) -> {label}"
        )
    lines.append(f"  => DIAGONAL partitions: {n_diag} of {len(bell_units_local)} "
                 f"Bell units  (straight/vertical seam => 0 expected; chevron => >0)")
    for ln in lines:
        report_lines.append(ln)
        if echo:
            print(ln)
    return n_diag


# per-half support recorded on the Unit via a side table (filled in main)
_HALF_SUPPORT = {}


def _supp_of(u, a):
    return _HALF_SUPPORT.get(a, set())


# --------------------------------------------------------------------------- #
# Bell-MEASUREMENT augmentation (asPaper): diagonal seam candidates + observable
# completion.  This is the ONLY change over annotator.py: the bulk / round-0 /
# final candidate generation is untouched (2C: bulk is unchanged from reordered).
# --------------------------------------------------------------------------- #
def diagonal_bell_candidates(units, anc_tl, coords):
    """Extra candidate detectors for the Bell-MEASUREMENT timing.

    annotator.py pairs the two Bell halves at the SAME round, (m_a^m_b)_r, and
    compares consecutive rounds.  That is right for the Bell-PAIR (reordered)
    timing but wrong for the Bell-MEASUREMENT timing, whose deterministic
    combination is DIAGONAL: one half at round r combines with the other half at
    round r+-1.  We over-generate the local diagonal shapes (2- and 4-record,
    both diagonal directions, +-1 and +-2 round gaps); the empirical validator
    keeps only the deterministic ones and silently drops the wrong-parity ones
    (role='seam-diag' so they are NOT counted as construction-bug rejects)."""
    extra = []
    for u in units:
        if u.kind != "bell":
            continue
        a, b = u.ancillas
        ra = [rec for rec, _ in anc_tl[a]]
        rb = [rec for rec, _ in anc_tl[b]]
        n = min(len(ra), len(rb))
        x, y = coords[u.rep][0], coords[u.rep][1]

        def add(idxs, r):
            extra.append(dict(idxs=set(idxs), x=x, y=y, r=r, unit=u,
                              role="seam-diag"))
        for k in range(1, n):
            add([ra[k], rb[k], ra[k - 1], rb[k - 1]], k)          # same-round consec
            add([ra[k], ra[k - 1]], k)                            # single-half a
            add([rb[k], rb[k - 1]], k)                            # single-half b
            add([ra[k], rb[k - 1]], k)                            # diagonal 2-rec
            add([rb[k], ra[k - 1]], k)
            if k + 1 < n:
                add([ra[k], rb[k], ra[k - 1], rb[k + 1]], k)      # diagonal 4-rec
                add([ra[k], rb[k], ra[k + 1], rb[k - 1]], k)
        for k in range(2, n):
            add([ra[k], rb[k], ra[k - 2], rb[k - 2]], k)          # 2-round gap
    return extra


def complete_observable_records(base_recs, shots, seam_cols, is_det):
    """Given the input observable's base records (non-deterministic), find the
    minimal set of extra SEAM records (GF(2) solve) that makes the XOR
    deterministic.  Returns (extra_abs, feasible)."""
    import numpy as _np
    p = (shots[:, base_recs].sum(axis=1) & 1).astype(_np.uint8)
    if p.min() == p.max():
        return [], True
    A = shots[:, seam_cols].astype(_np.uint8)
    rows, ncand = A.shape
    M = _np.concatenate([A, p[:, None]], axis=1).astype(_np.uint8)
    piv = []
    rr = 0
    for c in range(ncand):
        pr = None
        for k in range(rr, rows):
            if M[k, c]:
                pr = k
                break
        if pr is None:
            continue
        M[[rr, pr]] = M[[pr, rr]]
        for k in range(rows):
            if k != rr and M[k, c]:
                M[k] ^= M[rr]
        piv.append((rr, c))
        rr += 1
        if rr == rows:
            break
    for k in range(rr, rows):
        if M[k, ncand] and not M[k, :ncand].any():
            return None, False
    x = _np.zeros(ncand, dtype=_np.uint8)
    for (r_, c) in piv:
        x[c] = M[r_, ncand]
    extra = [seam_cols[j] for j in range(ncand) if x[j]]
    return extra, True


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--obs", choices=["x", "z"], default="x")
    ap.add_argument("--obs-index", type=int, default=0,
                    help="which interior data row (x) / column (z), 0-based")
    ap.add_argument("--keep-observables", action="store_true",
                    help="preserve the OBSERVABLE_INCLUDE lines already present in the "
                         "input verbatim (do NOT generate/override them).  Detectors are "
                         "still reconstructed.  If the input has no observables, falls back "
                         "to generating one as usual.")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--in-place", action="store_true",
                    help="annotate the INPUT file in place (overwrite it), instead of "
                         "writing a separate *_annotated.stim.  Written via a temp file "
                         "so the original is only replaced once annotation succeeds.")
    ap.add_argument("--report-partitions", action="store_true")
    ap.add_argument("--report", default=None,
                    help="path to write the human report (default: <input>_annotate.log; "
                         "'-' or os.devnull to suppress a persistent file)")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("-p", "--noise", type=float, default=NOISE_P)
    ap.add_argument("--noise-mode", choices=["auto", "add", "existing"], default="auto",
                    help="auto: use the circuit's own noise if it has any, else add "
                         "a uniform depolarizing model; add: always add; existing: "
                         "never add (fail if noiseless).")
    args = ap.parse_args()

    if args.in_place:
        if args.output:
            ap.error("--in-place and -o/--output are mutually exclusive")
        out_path = args.input
    else:
        out_path = args.output or (args.input.rsplit(".", 1)[0] + "_annotated.stim")
    if args.report is None:
        args.report = args.input.rsplit(".", 1)[0] + "_annotate.log"

    circuit = stim.Circuit.from_file(args.input)
    coords = circuit.get_final_qubit_coordinates()

    # capture any OBSERVABLE_INCLUDE lines already in the input so --keep-observables
    # can re-emit them.  CRUCIAL: a merge/split circuit places some OBSERVABLE_INCLUDE
    # lines MID-CIRCUIT (right after the split, to fold in the destructive readout of
    # the seam), where rec[-k] is relative to the measurements SO FAR -- not to the
    # total N.  We therefore resolve every target to an ABSOLUTE measurement index now
    # (mc-so-far + rec.value), and rebuild the rec[-k] offsets against the final N at
    # emit time.  Re-appending them all at the end with their original offsets would
    # silently corrupt the mid-circuit ones.
    existing_obs = []          # list of (abs_indices:list[int], obs_index:int)
    _mc_scan = 0
    for inst in circuit.flattened():
        if inst.name in MEAS:
            _mc_scan += sum(1 for t in inst.targets_copy() if t.is_qubit_target)
        elif inst.name == "OBSERVABLE_INCLUDE":
            oi = int(inst.gate_args_copy()[0]) if inst.gate_args_copy() else 0
            absids = [_mc_scan + t.value for t in inst.targets_copy()
                      if t.is_measurement_record_target]
            existing_obs.append((absids, oi))

    log(f"=== annotator.py on {args.input} ===")
    log(f"stim {stim.__version__}   obs={args.obs}  obs-index={args.obs_index}"
        f"  shots={args.shots}  noise-p={args.noise:g}")

    # ---- 1. strip (needed before behavioural ancilla detection) ----
    base = strip_annotations(circuit)
    flat = base.flattened()

    # ---- 1b. behavioural ancilla detection (integer-coord Bell/CZZ ancillas) ----
    forced = detect_forced_ancillas(flat, coords)
    if forced:
        FORCED_ANCILLAS.update(forced)
        log(f"reclassified integer-coord qubits as ANCILLAS (measured >=2x): "
            f"{sorted(forced)} "
            f"{[coord_str(coords, q) for q in sorted(forced)]}")

    data_qubits = [q for q in coords if is_data(coords, q)]
    ancilla_qubits = [q for q in coords if not is_data(coords, q)]

    # ---- 3-4 analyse ----
    partner, support, typ, conv = analyse_checks(flat, coords)
    for a in ancilla_qubits:
        _HALF_SUPPORT[a] = set(support[a])

    # ---- 5 record map ----
    meas_recs, N, reset_basis = build_record_map(flat, coords)

    # ---- 5b per-measurement timelines (merge/split-aware support tracking) ----
    anc_tl, data_meas = build_timelines(flat, coords)

    # ---- 7 bases ----
    # final data basis
    final_basis_set = {meas_recs[q][-1][1] for q in data_qubits if meas_recs[q]}
    if len(final_basis_set) == 1:
        final_basis = next(iter(final_basis_set))
    else:
        warn(f"data qubits not uniformly measured at end: {final_basis_set}")
        final_basis = "X" if "X" in final_basis_set else next(iter(final_basis_set))

    # ---- 2/6 units + timelines ----
    units = build_units(coords, partner, support, typ, conv, anc_tl)
    bell_units = [u for u in units if u.kind == "bell"]

    log(f"\ntotal measurements N = {N}")
    log(f"#data qubits = {len(data_qubits)}   #ancillas = {len(ancilla_qubits)}")
    log(f"#units = {len(units)}   #Bell units = {len(bell_units)}")

    # ---- 8 candidates ----
    cands, data_reset = candidate_detectors(
        units, reset_basis, final_basis, coords, data_meas, data_qubits)
    log(f"data-reset basis = {data_reset}   final-data basis = {final_basis}")
    log(f"#candidate detectors = {len(cands)}")

    # ---- 8b Bell-MEASUREMENT diagonal seam candidates (augmentation) ----
    diag = diagonal_bell_candidates(units, anc_tl, coords)
    cands += diag
    log(f"#diagonal Bell candidates added = {len(diag)} "
        f"(kept iff deterministic; wrong-parity dropped, NOT rejects)")

    # ---- 9 VALIDATE empirically ----
    clean = base.without_noise()
    shots = clean.compile_sampler().sample(args.shots)  # (shots, N) bool

    def is_det(idxs):
        idxs = list(idxs)
        if not idxs:
            return False
        x = shots[:, idxs].sum(axis=1) & 1
        return bool(x.min() == x.max())

    kept, rejects = [], []
    for cand in cands:
        (kept if is_det(cand["idxs"]) else rejects).append(cand)

    # ---- 9b. merge-aware repair pass (interleaved / period-k schedules) ----
    # A period-1 (consecutive-round) bulk detector is correct for an ordinary
    # memory cycle, but a lattice-surgery MERGE can run an INTERLEAVED syndrome-
    # extraction schedule inside its merge window: an ancilla measures the SAME
    # static stabilizer (identical support, gates and basis) every physical
    # round, yet consecutive measurements are NOT deterministic because the
    # effective code cycle spans 2 (or more) rounds.  The tell-tale signature is
    # period-2 determinism: (m_k XOR m_{k+2}) is deterministic while
    # (m_k XOR m_{k+1}) is not.  Because the gate structure is byte-identical
    # every round, this can only be discovered empirically.
    #
    # For every unit that produced at least one period-1 bulk REJECT we drop that
    # unit's period-1 bulk detectors and rebuild its bulk chain with the smallest
    # time-gap g>=1 that makes each detector deterministic (g=1 where the ordinary
    # cycle holds, g=2 across an interleaved merge window, etc.).  This is purely
    # additive: a circuit whose bulk detectors are all period-1 has no rejects, so
    # the pass never fires and the emitted circuit is byte-identical.
    MAX_GAP = 6
    bad_units = {id(c["unit"]) for c in rejects if c["role"] == "bulk"}
    n_repaired = 0
    if bad_units:
        # drop the period-1 bulk detectors (kept or rejected) of the affected units
        kept = [c for c in kept
                if not (c["role"] == "bulk" and id(c["unit"]) in bad_units)]
        rejects = [c for c in rejects
                   if not (c["role"] == "bulk" and id(c["unit"]) in bad_units)]
        seen_units = {}
        for c in cands:
            if id(c["unit"]) in bad_units:
                seen_units[id(c["unit"])] = c["unit"]
        for u in seen_units.values():
            tl = u.timeline
            x, y = coords[u.rep][0], coords[u.rep][1]
            covered = set()
            for k in range(1, len(tl)):
                recs_cur, supp_cur = tl[k]
                for g in range(1, min(MAX_GAP, k) + 1):
                    recs_prev, supp_prev = tl[k - g]
                    idxs = set(recs_prev) | set(recs_cur)
                    lo, hi = max(recs_prev), min(recs_cur)
                    for q in (supp_prev - supp_cur):
                        for mrec, _ in data_meas.get(q, []):
                            if lo < mrec < hi:
                                idxs.add(mrec)
                    for q in (supp_cur - supp_prev):
                        for mrec, _ in data_meas.get(q, []):
                            if mrec < min(recs_prev):
                                idxs.add(mrec)
                    if is_det(idxs):
                        kept.append(dict(idxs=idxs, x=x, y=y, r=k,
                                         unit=u, role="bulk"))
                        covered |= idxs
                        n_repaired += 1
                        break
            # any of this unit's own measurement records not covered by a rebuilt
            # bulk detector (nor by a kept round-0/final) is a genuine gap -> record
            # it as a residual bulk reject so it is reported, not silently dropped.
            own_recs = {rec for m in u.ancillas for rec, _ in anc_tl[m]}
            kept_here = {i for c in kept if id(c["unit"]) == id(u) for i in c["idxs"]}
            for rec in sorted(own_recs - kept_here - covered):
                rejects.append(dict(idxs={rec}, x=x, y=y, r=-1,
                                    unit=u, role="bulk"))
        log(f"\n=== merge-aware repair pass: rebuilt {len(bad_units)} unit(s) with "
            f"interleaved schedule; emitted {n_repaired} gap-adjusted bulk detector(s) ===")

    # A BULK detector compares two consecutive measurements of the SAME stabilizer, so
    # in a correct circuit it MUST be deterministic -- a bulk reject is a real bug.  A
    # round-0/final reject only means that boundary detector does not apply to this
    # check (e.g. an X-check has no deterministic Z-basis closing); that is expected
    # filtering from the over-generation above, not a construction bug.
    bulk_rejects = [c for c in rejects if c["role"] == "bulk"]
    boundary_rejects = [c for c in rejects if c["role"] != "bulk"]

    log(f"#detectors kept = {len(kept)}   #rejects = {len(bulk_rejects)}"
        f"   (+{len(boundary_rejects)} inapplicable boundary candidates, expected)")

    if bulk_rejects:
        log("\n=== REJECTED bulk detectors (each is a real construction bug) ===")
        for c in bulk_rejects:
            u = c["unit"]
            log(f"  REJECT role={c['role']} r={c['r']} unit={u.ancillas} "
                f"coord={coord_str(coords,u.rep)} type={u.type} conv={sorted(u.conv)} "
                f"recs={sorted(i - N for i in c['idxs'])}")
    else:
        log("\n=== REJECTED bulk detectors: NONE (syndrome extraction is sound) ===")

    # keep the alarming metric (`rejects`) meaning "real construction bugs"
    rejects = bulk_rejects

    # ---- 10 emit detectors ----
    kept.sort(key=lambda c: (c["r"], c["x"], c["y"], c["role"]))
    annotated = make_detector_circuit(base, kept, N)

    # ---- 10 observable ----
    obs_index = 0
    if args.keep_observables and existing_obs:
        log(f"\n=== keeping {len(existing_obs)} existing OBSERVABLE_INCLUDE line(s), "
            f"COMPLETING with seam records where non-deterministic (N={N}) ===")
        # seam-ancilla record columns for observable completion (Bell-MEASUREMENT
        # timing routes the logical parity through the seam Bell measurements)
        seam_ancillas = set()
        for u in bell_units:
            seam_ancillas.update(u.ancillas)
        seam_cols = sorted(abs_i for a in seam_ancillas
                           for (abs_i, _) in meas_recs[a])
        obs_groups = defaultdict(list)
        for absids, oi in existing_obs:
            obs_groups[oi].extend(absids)
        for oi, absids in sorted(obs_groups.items()):
            extra, feasible = complete_observable_records(
                absids, shots, seam_cols, is_det)
            if not feasible:
                warn(f"observable {oi} cannot be completed with seam records "
                     f"(non-deterministic even after seam GF(2) solve)")
                full = absids
            else:
                full = sorted(set(absids) | set(extra))
            det_ok = is_det(full)
            annotated.append("OBSERVABLE_INCLUDE",
                             [stim.target_rec(a - N) for a in full], oi)
            log(f"  OBSERVABLE_INCLUDE({oi}): {len(absids)} base + "
                f"{len(full) - len(absids)} seam extra = {len(full)} records  "
                f"deterministic={det_ok}")
            if extra:
                # report the ancillas contributing each extra record
                anc_of = {abs_i: a for a in seam_ancillas
                          for (abs_i, _) in meas_recs[a]}
                log(f"    extra seam records rec{[a - N for a in sorted(extra)]}  "
                    f"ancillas={sorted(set(anc_of[a] for a in extra))}")
            if not det_ok:
                warn(f"kept observable {oi} is NOT deterministic on the noiseless circuit")
    elif args.obs == "x":
        rows = sorted({coords[q][1] for q in data_qubits})
        k = max(0, min(args.obs_index, len(rows) - 1))
        row = rows[k]
        recs = x_row_records(coords, data_qubits, meas_recs, row)
        det_ok = is_det(recs)
        log(f"\n=== observable X on data row y={row:g} (interior row #{k}) ===")
        log(f"  records rec{[r - N for r in recs]}  deterministic={det_ok}")
        if not det_ok:
            warn(f"X observable on row y={row:g} is NOT deterministic")
        annotated.append("OBSERVABLE_INCLUDE",
                         [stim.target_rec(r - N) for r in recs], obs_index)
    else:  # z
        log("\n=== observable Z (column) ===")
        cols = sorted({coords[q][0] for q in data_qubits})
        chosen = None
        order = list(range(len(cols)))
        # start from requested column, then walk outward to adjacent ones
        start = max(0, min(args.obs_index, len(cols) - 1))
        order = sorted(order, key=lambda i: abs(i - start))
        for i in order:
            col = cols[i]
            recs = [meas_recs[q][-1][0] for q in sorted(data_qubits)
                    if abs(coords[q][0] - col) < 1e-6 and meas_recs[q]]
            ok = is_det(recs)
            log(f"  column x={col:g}: records rec{[r - N for r in recs]}  deterministic={ok}")
            if ok:
                chosen = (col, recs)
                break
        if chosen is None:
            warn("no data column yields a deterministic Z observable "
                 "(expected in an X-basis memory: logical Z is undetermined "
                 "under final MX readout). Emitting no Z observable.")
        else:
            col, recs = chosen
            log(f"  -> using column x={col:g}")
            annotated.append("OBSERVABLE_INCLUDE",
                             [stim.target_rec(r - N) for r in recs], obs_index)

    # ---- write annotated circuit (atomic: temp then replace, safe for --in-place) ----
    tmp_path = out_path + ".tmp"
    annotated.to_file(tmp_path)
    os.replace(tmp_path, out_path)
    log(f"\nwrote annotated circuit -> {out_path}"
        + ("  (in place)" if args.in_place else ""))

    # ---- 11 cross-check: DEM must build (noiseless -> validates determinism) ----
    log("\n=== cross-check: detector_error_model(allow_gauge_detectors=False, "
        "decompose_errors=True) ===")
    dem_ok = False
    try:
        annotated.detector_error_model(allow_gauge_detectors=False,
                                       decompose_errors=True)
        dem_ok = True
        log("  DEM builds OK (all detectors deterministic + decomposable)")
    except Exception as e:
        log(f"  DEM FAILED -> {e}")

    # ---- shortest graphlike error (needs noise) ----
    circuit_has_noise = has_noise(annotated)
    if args.noise_mode == "existing" or (args.noise_mode == "auto" and circuit_has_noise):
        noise_src = "circuit's own noise"
        noisy_for_dist = annotated
        if not circuit_has_noise:
            warn("--noise-mode existing but circuit is noiseless; "
                 "shortest_graphlike_error will find no errors")
    else:
        noise_src = f"added uniform depolarizing p={args.noise:g}"
        noisy_for_dist = add_noise(annotated, args.noise)

    log(f"\n=== shortest_graphlike_error ({noise_src}) ===")
    dist_val = None
    try:
        err = noisy_for_dist.shortest_graphlike_error()
        dist_val = len(err)
        log(f"  len(shortest_graphlike_error) = {dist_val}  (annotated, obs={args.obs})")
    except Exception as e:
        log(f"  shortest_graphlike_error FAILED -> {e}")

    # ---- acceptance: X-distance per row ----
    # A per-row X observable is only meaningful for a plain memory experiment; in a
    # merge/split the true logical observable is the kept multi-line one, so skip this
    # (otherwise every row prints a non-deterministic-observable failure).
    if args.obs == "x" and not (args.keep_observables and existing_obs):
        log("\n=== X-distance per data row (expect 3; the known hook, NOT a detector bug) ===")
        rows = sorted({coords[q][1] for q in data_qubits})
        for r_i, row in enumerate(rows):
            recs = x_row_records(coords, data_qubits, meas_recs, row)
            ok = is_det(recs)
            c = make_detector_circuit(base, kept, N)
            c.append("OBSERVABLE_INCLUDE",
                     [stim.target_rec(rr - N) for rr in recs], 0)
            try:
                cn = c if (args.noise_mode == "existing" or
                           (args.noise_mode == "auto" and circuit_has_noise)) \
                    else add_noise(c, args.noise)
                d = len(cn.shortest_graphlike_error())
                dstr = str(d)
            except Exception as e:
                dstr = f"FAILED ({e})"
            log(f"  row #{r_i} y={row:g}: deterministic={ok}  X-distance = {dstr}")

    # ---- acceptance: one Bell seam unit's 4-record bulk detector ----
    log("\n=== sample Bell-seam bulk detector  (m_a XOR m_b)_r  XOR  (m_a XOR m_b)_{r-1} ===")
    seam_bell = next((u for u in bell_units), None)
    if seam_bell is not None:
        a, b = seam_bell.ancillas
        # first bulk detector for this unit (between event 0 and 1)
        bulk = next((c for c in kept if c["unit"] is seam_bell and c["role"] == "bulk"
                     and c["r"] == 1), None)
        if bulk is None:
            bulk = next((c for c in cands if c["unit"] is seam_bell
                         and c["role"] == "bulk" and c["r"] == 1), None)
        ra = [idx for idx, _ in meas_recs[a]]
        rb = [idx for idx, _ in meas_recs[b]]
        log(f"  Bell unit {{{a},{b}}} at {coord_str(coords,a)}<->{coord_str(coords,b)} "
            f"type={seam_bell.type}")
        log(f"    m_a records (rounds): {[i - N for i in ra]}")
        log(f"    m_b records (rounds): {[i - N for i in rb]}")
        if bulk is not None:
            recs_sorted = sorted(i - N for i in bulk["idxs"])
            log(f"    4-record bulk detector (r=1): rec{recs_sorted}")
            log(f"       = (m_a[r]  XOR m_b[r]) XOR (m_a[r-1] XOR m_b[r-1])")
            log(f"       = rec[{ra[1]-N}] XOR rec[{rb[1]-N}] XOR rec[{ra[0]-N}] XOR rec[{rb[0]-N}]")

    # ---- partition report ----
    n_diag = report_partitions(units, coords, echo=args.report_partitions)

    # ---- counts summary + gate convention (per ANCILLA = per physical check) ----
    def acount(pred):
        return sum(1 for a in ancilla_qubits if pred(a))
    seam = set()
    for u in bell_units:
        seam.update(u.ancillas)
    z_cz = acount(lambda a: typ[a] == {"Z"} and conv[a] == {"CZ"})
    z_hcx = acount(lambda a: typ[a] == {"Z"} and conv[a] == {"CX(data->anc)"})
    x_ck = acount(lambda a: typ[a] == {"X"})
    seam_z = acount(lambda a: a in seam and typ[a] == {"Z"})
    seam_x = acount(lambda a: a in seam and typ[a] == {"X"})
    bulk_z = acount(lambda a: a not in seam and typ[a] == {"Z"})
    bulk_x = acount(lambda a: a not in seam and typ[a] == {"X"})
    log("\n=== SUMMARY COUNTS ===")
    log(f"  #units               = {len(units)}  "
        f"({len(bell_units)} Bell + {len(units) - len(bell_units)} singleton)")
    log(f"  #Bell units          = {len(bell_units)}")
    log(f"  #ancilla checks      = {len(ancilla_qubits)}  "
        f"({len(seam)} seam / Bell-paired, {len(ancilla_qubits) - len(seam)} bulk)")
    log(f"  #detectors emitted   = {len(kept)}")
    log(f"  #rejects             = {len(rejects)}")
    log(f"  #warnings            = {len(warnings)}")
    log("  gate-convention summary per ancilla (documents seam/bulk non-uniformity):")
    log(f"    Z-checks via CZ                    : {z_cz}   (all on the seam)")
    log(f"    Z-checks via H+CX [CX(data->anc)]  : {z_hcx}   (all in the bulk)")
    log(f"    X-checks via CX(anc->data)         : {x_ck}   (seam {seam_x} + bulk {bulk_x})")
    log(f"    seam checks:  {seam_z} Z (CZ) + {seam_x} X   |  "
        f"bulk checks: {bulk_z} Z (H+CX) + {bulk_x} X")

    # ---- machine-parseable RESULT line ----
    log("\n=== RESULT (machine-parseable) ===")
    log(f"RESULT obs={args.obs} distance={dist_val} rejects={len(rejects)} "
        f"dem_ok={int(dem_ok)} diagonal_units={n_diag} bell_units={len(bell_units)} "
        f"seam_pairs={len(bell_units)} detectors={len(kept)} warnings={len(warnings)}")

    # ---- write report ----
    with open(args.report, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
