from multiprocessing import Pool, cpu_count, freeze_support
# from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

import argparse
import collections
import csv
import itertools
import os
import re
import pymatching
import numpy as np
import stim
import matplotlib.pyplot as plt
# import lightstim
from multiprocessing import Pool, cpu_count, freeze_support
# from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

# =========================
# Circuits we use for comparisons
# =========================
#
# Each circuit is stored in its own .stim text file next to this script.
# Just give the filename here; load_circuit() reads it and builds the
# stim.Circuit for you.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_circuit(filename):
    return stim.Circuit.from_file(os.path.join(SCRIPT_DIR, filename))


# circuit_d5_bell = load_circuit("pauliXX_d5_bell_graphlike.stim")
# circuit_d5_interleaving = load_circuit("pauliXX_d5_interleaving.stim")
# circuit_d5_alternating = load_circuit("pauliXX_d5_alternating.stim")

# The d7 circuits are loaded on demand (see SERIES / run_series below), so
# --plot-only works even if a .stim file is missing or being edited.

# =========================
# Shared noise settings
# =========================

p1 = 0.0001
default_p2 = 0.0001
p_idle = p1  # idle DEPOLARIZE1, same rate as the gates (see build_noisy_circuit)
# Sampling budget.  Each sweep point draws shots in BATCHES until it has
# collected max_errors logical errors or spent max_shots shots, whichever comes
# first, and reports errors / shots-actually-taken.
#
#   max_shots   ceiling on work per point.  Sets the smallest rate measurable
#               at all: below ~10/max_shots a point reads as zero.
#   max_errors  precision target.  Relative standard error is 1/sqrt(errors),
#               so 1000 errors is +-3.2%.  The cap only ever saves time; it
#               binds when the true rate exceeds max_errors/max_shots (1e-5
#               here), so it fires on high-p2 points and not on low ones.
#   batch_size  shots held in memory at once.  THIS is what keeps the run
#               alive: one unbatched 1e8-shot sample() on a d9 circuit (6564
#               detectors) asks the OS for 656 GB and is SIGKILLed with no
#               traceback.  Bit-packed at 100k shots/batch the peak is ~82 MB.
max_shots = 100_000_000
max_errors = 1000
batch_size = 100_000

shots = max_shots  # kept so anything still referencing `shots` resolves

# Sweep points are independent, so running them in parallel costs nothing
# statistically.  Each worker holds its own circuit, DEM and one batch, so
# memory is roughly workers * (a few hundred MB); 8 is comfortable in 64 GB.
DEFAULT_WORKERS = max(1, min(8, (cpu_count() or 1) // 2))
override_p2_values = np.linspace(0.01, 0.001, 10)

# Make the over_ride_pairs that will provide the CX gates across the inter-QPU boundary with different error rates
def make_override_pairs(number_string):
    numbers = list(map(int, number_string.split()))
    return [(numbers[i], numbers[i + 1]) for i in range(len(numbers) - 1)]

# d5, left boundary only
# override_pairs_d5_bell_LS_on_different_QPU = make_override_pairs("154 155 152 160 156 157 153 161 158 159 147 164 165 148 149 166 150 167 151 168 169")
# override_pairs_d5_bell_LS_on_different_QPU_only_seam_override = make_override_pairs("154 147 154 164 148 155 155 165 149 156 156 166 150 157 157 167 151 158 158 168 159 169 154 155 156 157 158 159 152 160 153 161")
# override_pairs_d5_interleaving_LS_on_different_QPU = make_override_pairs("215 204 214 203 213 202 212 201 211 200 210")
# override_pairs_d5_alternating_LS_on_different_QPU = make_override_pairs("215 204 214 203 213 202 212 201 211 200 210")

# d7, left boundary only
# override_pairs_d7_bell_LS_on_different_QPU = make_override_pairs("290 300 300 314 300 301 291 301 301 315 292 302 302 316 302 303 293 303 303 317 294 304 304 318 304 305 295 305 305 319 296 306 306 320 307 306 307 321 297 308 298 309 299 310")
# override_pairs_d7_interleaving_LS_on_different_QPU = make_override_pairs("406 392 407 393 408 394 409 395 410 396 411 397 412 398 413")
# override_pairs_d7_alternating_LS_on_different_QPU = make_override_pairs("406 392 407 393 408 394 409 395 410 396 411 397 412 398 413")


# # d3, left boundary only
# override_pairs_d3_bell_LS_on_different_QPU = make_override_pairs("52 56 56 62 56 57 53 57 57 63 55 60 54 58 58 64 58 59 59 65")
# override_pairs_d3_interleaving_LS_on_different_QPU = make_override_pairs("78 72 79 73 80 74 81")
# override_pairs_d3_alternating_LS_on_different_QPU = make_override_pairs("78 72 79 73 80 74 81")

# d9, left boundary only
# override_pairs_d9_bell_LS_on_different_QPU = make_override_pairs("481 494 494 512 494 495 482 495 495 513  483 496 496 514 496 497 484 497 497 515 485 498 498 516 498 499 486 499 499 517 487 500 500 518 500 501 488 501 501 519 489 502 502 520 502 503 503 521 490 504 491 505 492 506 493 507")
# override_pairs_d9_interleaving_LS_on_different_QPU = make_override_pairs("666 648 667 649 668 650 669 651 670 652 671 653 672 654 673 655 674 656 675")
# override_pairs_d9_alternating_LS_on_different_QPU = make_override_pairs("666 648 667 649 668 650 669 651 670 652 671 653 672 654 673 655 674 656 675")

# xmerge left, d5
override_pairs_d5_bell_LS_on_different_QPU = make_override_pairs("290 300 300 314 300 301 291 301 301 315 292 302 302 316 302 303 293 303 303 317 294 304 304 318 304 305 295 305 305 319 296 306 306 320 307 306 307 321 297 308 298 309 299 310")
override_pairs_d5_interleaving_LS_on_different_QPU = make_override_pairs("406 392 407 393 408 394 409 395 410 396 411 397 412 398 413")
override_pairs_d5_alternating_LS_on_different_QPU = make_override_pairs("406 392 407 393 408 394 409 395 410 396 411 397 412 398 413")

def make_override_edges(override_pairs_raw):
    override_edges = set()
    for a, b in override_pairs_raw:
        override_edges.add((a, b))
        override_edges.add((b, a))
    return override_edges

def make_noisy_circuit(circuit, override_edges, override_p2, default_p2=default_p2, p1=p1):
    noisy_circuit = stim.Circuit()
    for instruction in circuit:
        if instruction.name in ["CX", "CZ"]:
            targets = instruction.targets_copy()
            if len(targets) % 2 != 0:
                raise ValueError(f"Odd number of targets in {instruction}")

            for k in range(0, len(targets), 2):
                t1 = targets[k]
                t2 = targets[k + 1]
                q1 = t1.value
                q2 = t2.value

                noisy_circuit.append(instruction.name, [t1, t2])

                if (q1, q2) in override_edges:
                    noisy_circuit.append("DEPOLARIZE2", [t1, t2], override_p2)
                else:
                    noisy_circuit.append("DEPOLARIZE2", [t1, t2], default_p2)

        elif instruction.name in ["M", "MX"]:
            noisy_circuit.append(instruction.name, instruction.targets_copy(), p1)

        elif instruction.name in ["R", "RX", "H", "X"]:
            noisy_circuit.append(instruction)
            noisy_circuit.append("DEPOLARIZE1", instruction.targets_copy(), p1)

        elif instruction.name in [
            "QUBIT_COORDS", "DETECTOR", "TICK", "OBSERVABLE_INCLUDE",
            "DEPOLARIZE1", "DEPOLARIZE2"
        ]:
            noisy_circuit.append(instruction)

        else:
            raise NotImplementedError("forgot " + str(instruction))

    return noisy_circuit


# =========================
# Noise model: CXX gates + idle noise
# =========================
#
# The bell merges apply a three-qubit CXX (one control, two targets).  stim has
# no CXX instruction, so the circuit spells it out as CX(c, t1) on one tick and
# CX(c, t2) on the following tick.  In hardware that is one gate in one tick, so
# build_noisy_circuit
#   * charges one DEPOLARIZE2 per CXX, on (c, t1), and none on (c, t2);
#   * drops the TICK between the two halves, so qubits outside the CXX idle once
#     for it, and c, t1, t2 do not idle at all.
# A split tick can also carry a lone CX from a seam qubit at the patch edge that
# has only one neighbour (e.g. (7,12) -> (8,12) in x-merge-bell-d5); it is merged
# into the CXX tick and charged like any other CX.
#
# Idle noise: in every tick that has an operation, each live qubit that is not
# acted on gets DEPOLARIZE1(p_idle).  A qubit is live from its first operation
# to its last, so nothing is charged before the first reset or after the final
# measurement.

QUANTUM_OPS = {"CX", "CZ", "H", "X", "R", "RX", "M", "MX"}
PASS_THROUGH = {"QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE",
                "DEPOLARIZE1", "DEPOLARIZE2"}


def split_ticks(circuit):
    layers = [[]]
    for instruction in circuit:
        if instruction.name == "TICK":
            layers.append([])
        else:
            layers[-1].append(instruction)
    return layers


def cx_pairs(layer):
    pairs = []
    for instruction in layer:
        if instruction.name == "CX":
            targets = [t.value for t in instruction.targets_copy()]
            pairs += list(zip(targets[0::2], targets[1::2]))
    return pairs


def busy_qubits(layer):
    return {t.value for instruction in layer if instruction.name in QUANTUM_OPS
            for t in instruction.targets_copy()}


def find_cxx_halves(circuit):
    """Locate the CX gates that are the second half of a CXX.

    Returns (second_halves, first_halves, lone, extra_tick_layers):
      second_halves / first_halves : dicts keyed by (layer_index, control,
          target), mapping each half to the other half's key;
      lone : (layer_index, control, target) of single CXs sitting in a split
          tick whose control had no first half (edge seam qubits);
      extra_tick_layers : layer indices whose trailing TICK separates a CXX
          from its second half.

    A second-half layer is a tick layer that (a) contains only CX gates, (b) is
    strictly smaller than the preceding layer, (c) has at least one control that
    was also a control in the preceding layer, and (d) can be merged into the
    preceding layer without any qubit other than those controls acting twice.
    """
    layers = split_ticks(circuit)
    second_halves, first_halves, lone = {}, {}, set()
    extra_tick_layers = set()

    for i in range(1, len(layers)):
        layer = layers[i]
        if (i - 2) in extra_tick_layers:
            continue  # the preceding layer is itself a second half
        if not layer or any(instruction.name != "CX" for instruction in layer):
            continue
        pairs = cx_pairs(layer)
        prev_pairs = cx_pairs(layers[i - 1])
        if not pairs or len(pairs) >= len(prev_pairs):
            continue
        prev_by_control = dict(prev_pairs)
        shared_controls = {c for c, _ in pairs if c in prev_by_control}
        if not shared_controls:
            continue
        used = {q for pair in pairs for q in pair}
        if (used & busy_qubits(layers[i - 1])) - shared_controls:
            continue

        for c, t in pairs:
            if c in prev_by_control:
                first = (i - 1, c, prev_by_control[c])
                second = (i, c, t)
                second_halves[second] = first
                first_halves[first] = second
            else:
                lone.add((i, c, t))
        extra_tick_layers.add(i - 1)

    return second_halves, first_halves, lone, extra_tick_layers


# 63 non-identity 3-qubit Pauli strings, fixed order (III dropped).
_PAULI3 = [''.join(s) for s in itertools.product('IXYZ', repeat=3)][1:]


def append_depolarize3(circuit, q0, q1, q2, p):
    """Uniform 3-qubit depolarizing channel on (q0, q1, q2): each of the 63
    non-identity 3-qubit Paulis applied with marginal probability p/63.

    Stim has no DEPOLARIZE3, so we emit a mutually-exclusive
    CORRELATED_ERROR / ELSE_CORRELATED_ERROR chain. In that chain, item k
    fires only if none before it fired, so to make each Pauli's *marginal*
    probability equal p/63 we use the conditional probability
    p_cond = (p/63) / (1 - sum_of_marginals_emitted_so_far), i.e. Eq. C6 of
    arXiv:2506.09028.
    """
    if p <= 0:
        return
    qubits = (q0, q1, q2)
    per = p / 63.0
    remaining = 1.0  # = 1 - (sum of marginals already emitted)
    for k, s in enumerate(_PAULI3):
        targets = []
        for pauli, q in zip(s, qubits):
            if pauli == 'X':
                targets.append(stim.target_x(q))
            elif pauli == 'Y':
                targets.append(stim.target_y(q))
            elif pauli == 'Z':
                targets.append(stim.target_z(q))
            # 'I' contributes no target
        p_cond = per / remaining
        circuit.append(
            "CORRELATED_ERROR" if k == 0 else "ELSE_CORRELATED_ERROR",
            targets, p_cond,
        )
        remaining -= per


def build_noisy_circuit(circuit, override_edges, override_p2,
                        default_p2=default_p2, p1=p1, p_idle=p_idle,
                        mark_idle=False, verbose=False,
                        default_p3=None, override_p3=None):
    """Gate noise + idle noise, with each CXX merged into one tick.

    Every CX/CZ line is kept intact and followed by its DEPOLARIZE2 line(s)
    (default rate, then override rate for inter-QPU pairs).  Each CXX instead
    gets a uniform 3-qubit depolarizing channel on (control, t1, t2), at
    default_p3 / override_p3 (which default to the 2-qubit rates).  At the end
    of each tick the idle qubits get DEPOLARIZE1(p_idle).

    mark_idle : also put an I gate on the idle qubits just before their
                DEPOLARIZE1, so the idle locations are visible in crumble.
    """
    if default_p3 is None:
        default_p3 = default_p2
    if override_p3 is None:
        override_p3 = override_p2

    layers = split_ticks(circuit)
    second_halves, first_halves, lone, extra_tick_layers = find_cxx_halves(circuit)
    if verbose:
        print(f"found {len(second_halves)} CXX gates, {len(lone)} lone seam CX, "
              f"{len(extra_tick_layers)} merged ticks:")
        for (i, c, t2), (_, _, t1) in sorted(second_halves.items()):
            print(f"  layer {i - 1}/{i}: CXX control {c} targets {t1}, {t2}")
        for i, c, t in sorted(lone):
            print(f"  layer {i}: lone CX {c} -> {t}")

    groups = []
    i = 0
    while i < len(layers):
        if i in extra_tick_layers:
            groups.append([i, i + 1])
            i += 2
        else:
            groups.append([i])
            i += 1

    group_busy = [set().union(*(busy_qubits(layers[j]) for j in g)) for g in groups]
    first_group, last_group = {}, {}
    for k, busy in enumerate(group_busy):
        for q in busy:
            first_group.setdefault(q, k)
            last_group[q] = k

    noisy_circuit = stim.Circuit()
    for k, group in enumerate(groups):
        cxx_channels = []  # (control, t1, t2, p3) flushed after this group's gate lines
        for j in group:
            for instruction in layers[j]:
                name = instruction.name
                if name in ["CX", "CZ"]:
                    targets = instruction.targets_copy()
                    if len(targets) % 2 != 0:
                        raise ValueError(f"Odd number of targets in {instruction}")
                    noisy_circuit.append(instruction)

                    default_pairs, override_pairs = [], []
                    for t1, t2 in zip(targets[0::2], targets[1::2]):
                        key = (j, t1.value, t2.value)
                        if key in second_halves:
                            continue  # noise is charged with the first half of the CXX

                        if key in first_halves:
                            # This CX is the first half of a CXX; charge a single
                            # 3-qubit depolarizing channel on (control, t1, other).
                            _, c, other = first_halves[key]
                            inter_qpu = (
                                (t1.value, t2.value) in override_edges
                                or (c, other) in override_edges
                            )
                            p3 = override_p3 if inter_qpu else default_p3
                            cxx_channels.append((t1.value, t2.value, other, p3))
                            continue  # 3-qubit channel replaces the DEPOLARIZE2

                        # Genuine 2-qubit gate: plain CZ, or a lone seam CX.
                        if (t1.value, t2.value) in override_edges:
                            override_pairs += [t1, t2]
                        else:
                            default_pairs += [t1, t2]

                    if default_pairs:
                        noisy_circuit.append("DEPOLARIZE2", default_pairs, default_p2)
                    if override_pairs:
                        noisy_circuit.append("DEPOLARIZE2", override_pairs, override_p2)

                elif name in ["M", "MX"]:
                    noisy_circuit.append(name, instruction.targets_copy(), p1)

                elif name == "R":
                    # preparation error: |1> instead of |0>
                    noisy_circuit.append(instruction)
                    noisy_circuit.append("X_ERROR", instruction.targets_copy(), p1)

                elif name == "RX":
                    # preparation error: |-> instead of |+> (X_ERROR would do nothing)
                    noisy_circuit.append(instruction)
                    noisy_circuit.append("Z_ERROR", instruction.targets_copy(), p1)

                elif name in ["H", "X"]:
                    noisy_circuit.append(instruction)
                    noisy_circuit.append("DEPOLARIZE1", instruction.targets_copy(), p1)

                elif name in PASS_THROUGH:
                    noisy_circuit.append(instruction)

                else:
                    raise NotImplementedError("forgot " + str(instruction))

        for c, ta, tb, p3 in cxx_channels:
            append_depolarize3(noisy_circuit, c, ta, tb, p3)

        busy = group_busy[k]
        idle = sorted(q for q in first_group
                      if first_group[q] < k < last_group[q] and q not in busy)
        if busy and idle and p_idle > 0:
            if mark_idle:
                noisy_circuit.append("I", idle)
            noisy_circuit.append("DEPOLARIZE1", idle, p_idle)

        if k < len(groups) - 1:
            noisy_circuit.append("TICK")

    return noisy_circuit


def save_noisy_circuit(circuit, override_pairs_raw, override_p2, filename,
                       **noise_kwargs):
    """Write build_noisy_circuit's output to a .stim file for manual checking."""
    noisy_circuit = build_noisy_circuit(
        circuit, make_override_edges(override_pairs_raw), override_p2,
        **noise_kwargs,
    )
    path = os.path.join(SCRIPT_DIR, filename)
    noisy_circuit.to_file(path)
    print(f"Saved noisy circuit to {path}")
    return path


def logical_error_rate(noisy_circuit, max_shots=max_shots, max_errors=max_errors,
                       batch_size=batch_size, progress_prefix=None,
                       progress_every_shots=5_000_000):
    """Sample in batches until max_errors errors or max_shots shots.

    Returns (num_errors, shots_done).  The caller forms the rate as
    num_errors / shots_done -- always the shots actually taken, never
    max_shots, because the error cap makes those differ by orders of magnitude
    on the high-p2 points.

    The estimator is unchanged from the old one-shot version: k/n is k/n
    whether the shots arrive in one array or a thousand.  Batching is
    bookkeeping and bit packing is a storage format; neither touches the
    statistics, so d3/d5/d7/d9 stay directly comparable.
    """
    model = noisy_circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)
    sampler = noisy_circuit.compile_detector_sampler()

    shots_done, num_errors, next_report = 0, 0, progress_every_shots
    while shots_done < max_shots and num_errors < max_errors:
        n = min(batch_size, max_shots - shots_done)
        syndrome, actual = sampler.sample(
            shots=n, separate_observables=True, bit_packed=True)
        predicted = matching.decode_batch(
            syndrome, bit_packed_shots=True, bit_packed_predictions=True)
        # Both sides zero the padding bits of the last byte, so comparing the
        # packed bytes is exactly the same test as comparing the raw bits.
        num_errors += int(np.count_nonzero(np.any(predicted != actual, axis=1)))
        shots_done += n

        if progress_prefix is not None and shots_done >= next_report:
            print(f"{progress_prefix} {shots_done:,}/{max_shots:,} shots, "
                  f"{num_errors} errors", flush=True)
            next_report += progress_every_shots

    return num_errors, shots_done


def _simulate_point(job):
    """Run one sweep point, in this process or a worker.  Plain data in/out.

    Takes the circuit FILENAME rather than a stim.Circuit so nothing exotic has
    to survive pickling when this runs under multiprocessing.
    """
    (circuit_file, override_pairs_raw, override_p2, p_idle_,
     max_shots_, max_errors_, batch_size_) = job
    noisy_circuit = build_noisy_circuit(
        load_circuit(circuit_file), make_override_edges(override_pairs_raw),
        override_p2, p_idle=p_idle_)
    dem = noisy_circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True)
    distance = len(dem.shortest_graphlike_error())
    num_errors, shots_used = logical_error_rate(
        noisy_circuit, max_shots=max_shots_, max_errors=max_errors_,
        batch_size=batch_size_, progress_prefix=f"  [p2={override_p2:.4e}]")
    return override_p2, num_errors, shots_used, distance

# def logical_error_rate(noisy_circuit, shots=shots):
#     pipeline = SimulationPipeline(
#         decoder_config=DecoderConfig("bposd"),
#         max_errors=500,
#         max_shots=shots,
#         batch_size=min(1_000, shots),
#         num_workers=6,  # change up to number of cores
#         print_progress=False,
#     )
    # stats = pipeline.run(noisy_circuit)
    # return stats.logical_error_rate

# =========================
# Saving and reloading sweep results
# =========================
#
# Every sweep point is written to disk the moment it is computed, so an
# interrupted run never costs more than the point in flight.  Re-running the
# script reloads whatever is already saved and only simulates the missing
# override_p2 values, and --plot-only draws the figure from the saved values
# without touching the simulator at all.
#
# Layout (next to this script):
#   data/pauliXX_d7_bell.csv          one series, appended point by point
#   data/pauliXX_d7_interleaving.csv
#   data/pauliXX_d7_alternating.csv
#   data/pauliXX_d7_all_series.csv    all series in one file, as blocks:
#
#       pauliXX_d7_bell
#       x,y
#       x,y
#
#       pauliXX_d7_interleaving
#       x,y
#       ...

DATA_DIR = os.path.join(SCRIPT_DIR, "data")
COMBINED_CSV = os.path.join(DATA_DIR, "xmerge_d5_all_series.csv")

# Saved points are matched to requested ones by rounded x, so the tiny float
# wobble from np.linspace never causes a redundant re-run.
X_DECIMALS = 12

# Each entry is one curve: where its circuit comes from, which CSV holds its
# numbers, and how it is drawn.  Change "label"/"marker" freely -- the plot is
# rebuilt from the CSVs, no simulation needed.
SERIES = [
    {
        "name": "bell",
        "key": "xmerge_d5_bell",
        "label": "bell (ours)",
        "title": "d5 108-Bell LS across different QPUs",
        "circuit_file": "x_merge_d5_bell.stim",
        "override_pairs": override_pairs_d5_bell_LS_on_different_QPU,
        "marker": "^",
    },
    {
        "name": "interleaving",
        "key": "xmerge_d5_interleaving",
        "label": "interleaving",
        "title": "d5 108-Interleaving LS across different QPUs",
        "circuit_file": "x_merge_d5_interleaving.stim",
        "override_pairs": override_pairs_d5_interleaving_LS_on_different_QPU,
        "marker": "v",
    },
    {
        "name": "alternating",
        "key": "xmerge_d5_alternating",
        "label": "alternating",
        "title": "d5-Alternating LS across different QPUs",
        "circuit_file": "x_merge_d5_alternating.stim",
        "override_pairs": override_pairs_d5_alternating_LS_on_different_QPU,
        "marker": "<",
    },
]

PLOT_TITLE = r"Cross shaped junnction merge across different QPUs ($d=5$)"
X_LABEL = "Inter-QPU link error rate"
Y_LABEL = "Logical error rate per shot"


def series_path(key):
    return os.path.join(DATA_DIR, f"{key}.csv")


def load_series(key):
    """Return {override_p2: logical_error_rate} for whatever is saved so far.

    A point written twice (e.g. a --rerun on top of an old file) keeps the last
    value in the file.
    """
    path = series_path(key)
    if not os.path.exists(path):
        return {}
    points = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                x = float(row["override_p2"])
                y = float(row["logical_error_rate"])
            except (KeyError, TypeError, ValueError):
                continue  # skip a header repeat or a half-written final line
            points[round(x, X_DECIMALS)] = y
    return points


def load_series_full(key):
    """Like load_series, but keeps each point's shots and error count too.

    Returns {override_p2: (rate, shots, errors)}.  Files written before the
    `errors` column existed get their count back as round(rate * shots), which
    is exact -- rate was computed as errors/shots.
    """
    path = series_path(key)
    if not os.path.exists(path):
        return {}
    points = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                x = float(row["override_p2"])
                y = float(row["logical_error_rate"])
                n = int(float(row["shots"]))
            except (KeyError, TypeError, ValueError):
                continue  # header repeat, half-written line, or no shots column
            try:
                k = int(float(row["errors"]))
            except (KeyError, TypeError, ValueError):
                k = int(round(y * n))
            points[round(x, X_DECIMALS)] = (y, n, k)
    return points


def point_is_complete(entry, max_shots, max_errors):
    """Has a saved point already met the stopping rule this run is using?

    The rule is the same one logical_error_rate applies: a point is done when
    it has max_errors errors or has spent max_shots shots.  Judging saved data
    by that rule means raising the budget automatically re-simulates the points
    that no longer qualify -- e.g. the old 1e6-shot d9 points, which have
    neither 1000 errors nor 1e8 shots -- while leaving genuinely finished ones
    alone.  It also keeps every saved point, old or new, consistent with the
    same stopping rule across d3/d5/d7/d9.
    """
    _, shots_used, errors = entry
    return errors >= max_errors or shots_used >= max_shots


def append_point(key, x, y, distance=None, shots=None, errors=None):
    """Append one (x, y) point to a series CSV and flush it to disk.

    `shots` is the number actually taken for this point, which under the error
    cap varies from point to point -- it is what y was divided by, and what any
    later error bar has to be computed from.  `errors` is the raw count, kept
    so 1/sqrt(errors) is recoverable without re-deriving it from y*shots.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = series_path(key)
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if fresh:
            writer.writerow(["override_p2", "logical_error_rate", "distance",
                             "shots", "errors"])
        writer.writerow([repr(float(x)), repr(float(y)),
                         "" if distance is None else distance,
                         "" if shots is None else shots,
                         "" if errors is None else errors])
        f.flush()
        os.fsync(f.fileno())
    return path


def clear_series(key):
    """Drop a series' saved points (kept as a .bak) so it is simulated afresh."""
    path = series_path(key)
    if os.path.exists(path):
        backup = path + ".bak"
        os.replace(path, backup)
        print(f"Moved previous {key} data to {backup}")


def write_combined_csv(path=COMBINED_CSV):
    """Collect every saved series into one file, one labelled block each."""
    os.makedirs(DATA_DIR, exist_ok=True)
    written = []
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for spec in SERIES:
            points = load_series(spec["key"])
            if not points:
                continue
            writer.writerow([spec["key"]])
            for x in sorted(points):
                writer.writerow([repr(x), repr(points[x])])
            writer.writerow([])
            written.append(f"{spec['key']} ({len(points)} points)")
    if written:
        print(f"Saved combined data to {path}: " + ", ".join(written))
    else:
        print(f"No saved data yet; wrote empty {path}")
    return path


def read_combined_csv(path=COMBINED_CSV):
    """Read back what write_combined_csv wrote: {series key: {x: y}}."""
    series = {}
    current = None
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                current = None
            elif len(row) == 1:
                current = series.setdefault(row[0].strip(), {})
            elif current is not None:
                current[round(float(row[0]), X_DECIMALS)] = float(row[1])
    return series


def _record_point(key, results, outcome):
    """Save one finished point and print it.  All CSV writing happens here, in
    the parent process, so parallel workers never touch the file."""
    override_p2, num_errors, shots_used, distance = outcome
    x = round(float(override_p2), X_DECIMALS)
    ler = num_errors / shots_used          # shots_used, never max_shots
    results[x] = ler
    append_point(key, x, ler, distance=distance, shots=shots_used,
                 errors=num_errors)

    if num_errors:
        precision = f"+-{100.0 / num_errors ** 0.5:.1f}%"
    else:
        # No errors seen: quote the 95% upper bound instead of pretending 0.
        precision = f"< {3.0 / shots_used:.2e} (95% CL)"
    print(f"override_p2 = {override_p2:.8e}")
    print(f"distance = {distance}")
    print(f"logical error rate = {ler:.8e}   "
          f"({num_errors} errors / {shots_used:,} shots, {precision})  (saved)",
          flush=True)


def sweep_dataset(circuit_file, override_pairs_raw, label, override_p2_values,
                  key, max_shots=max_shots, max_errors=max_errors,
                  batch_size=batch_size, p_idle=p_idle, workers=1):
    """Simulate the missing points of one series, saving each as it lands."""
    saved = load_series_full(key)
    results = {x: entry[0] for x, entry in saved.items()}

    print(f"\n=== {label} ===")
    if saved:
        print(f"{len(saved)} point(s) already in {series_path(key)}")

    todo = []
    for override_p2 in override_p2_values:
        x = round(float(override_p2), X_DECIMALS)
        entry = saved.get(x)
        if entry is not None and point_is_complete(entry, max_shots, max_errors):
            rate, shots_used, errors = entry
            print(f"override_p2 = {override_p2:.8e}  (saved) "
                  f"logical error rate = {rate:.8e}  "
                  f"({errors} errors / {shots_used:,} shots)")
            continue
        if entry is not None:
            rate, shots_used, errors = entry
            print(f"override_p2 = {override_p2:.8e}  (re-running: saved point "
                  f"has {errors} errors in {shots_used:,} shots, short of "
                  f"{max_errors} errors or {max_shots:,} shots)")
            results.pop(x, None)
        todo.append((circuit_file, override_pairs_raw, float(override_p2),
                     p_idle, max_shots, max_errors, batch_size))

    if todo:
        n_proc = min(workers, len(todo))
        if n_proc > 1:
            print(f"simulating {len(todo)} point(s) on {n_proc} worker(s)",
                  flush=True)
            # imap_unordered so a point is saved the moment it finishes rather
            # than waiting on the slowest one in the batch.
            with Pool(processes=n_proc) as pool:
                for outcome in pool.imap_unordered(_simulate_point, todo):
                    _record_point(key, results, outcome)
        else:
            for job in todo:
                _record_point(key, results, _simulate_point(job))

    return np.array([results[round(float(v), X_DECIMALS)] for v in override_p2_values])


def run_series(spec, override_p2_values=override_p2_values,
               max_shots=max_shots, max_errors=max_errors,
               batch_size=batch_size, workers=1):
    return sweep_dataset(
        circuit_file=spec["circuit_file"],
        override_pairs_raw=spec["override_pairs"],
        label=spec["title"],
        override_p2_values=override_p2_values,
        key=spec["key"],
        max_shots=max_shots,
        max_errors=max_errors,
        batch_size=batch_size,
        workers=workers,
    )


# =========================
# Plotting (from the saved CSVs -- never re-simulates)
# =========================

def plot_saved(keys=None, save_path=None, show=True):
    plt.figure(figsize=(7, 5))
    plotted = 0
    for spec in SERIES:
        if keys is not None and spec["key"] not in keys:
            continue
        points = load_series(spec["key"])
        if not points:
            print(f"No saved data for {spec['key']}; skipping it in the plot")
            continue
        xs = sorted(points)
        plt.plot(xs, [points[x] for x in xs], marker=spec["marker"],
                 label=spec["label"])
        plotted += 1

    if not plotted:
        print("Nothing to plot.")
        plt.close()
        return

    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.title(PLOT_TITLE)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()


def main():
    names = [s["name"] for s in SERIES] + [s["key"] for s in SERIES]
    parser = argparse.ArgumentParser(
        description="Sweep the inter-QPU link error rate. Each point is saved to "
                    "data/<series>.csv as soon as it is computed, so an "
                    "interrupted run resumes instead of starting over, and the "
                    "plot can be redrawn later without re-simulating.")
    parser.add_argument("--only", nargs="+", metavar="SERIES", choices=names,
                        help="run only these series (%s)" %
                             ", ".join(s["name"] for s in SERIES))
    parser.add_argument("--plot-only", action="store_true",
                        help="skip all simulation and plot the saved values")
    parser.add_argument("--rerun", action="store_true",
                        help="discard the saved points of the selected series "
                             "(kept as .bak) and simulate them again")
    parser.add_argument("--max-shots", type=int, default=max_shots,
                        help=f"ceiling on shots per point (default {max_shots:,})")
    parser.add_argument("--max-errors", type=int, default=max_errors,
                        help="stop a point once it has this many logical "
                             f"errors (default {max_errors}; 1/sqrt(n) "
                             "relative precision). Use a huge number to "
                             "disable the cap.")
    parser.add_argument("--batch-size", type=int, default=batch_size,
                        help=f"shots per batch (default {batch_size:,}). "
                             "Peak memory is one batch; do not remove the "
                             "batching to chase speed.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"parallel worker processes (default "
                             f"{DEFAULT_WORKERS}). Points are independent, so "
                             "this scales nearly linearly.")
    parser.add_argument("--save-plot", metavar="PATH",
                        help="also write the figure to this file")
    parser.add_argument("--no-plot", action="store_true",
                        help="only simulate and save; do not draw anything")
    args = parser.parse_args()

    selected = [s for s in SERIES
                if args.only is None or s["name"] in args.only or s["key"] in args.only]

    if not args.plot_only:
        if args.rerun:
            for spec in selected:
                clear_series(spec["key"])
        for spec in selected:
            run_series(spec, max_shots=args.max_shots,
                       max_errors=args.max_errors,
                       batch_size=args.batch_size,
                       workers=args.workers)
        write_combined_csv()

    if not args.no_plot:
        # Always draw every series that has saved data, even the ones this run
        # did not touch.
        plot_saved(save_path=args.save_plot)


if __name__ == "__main__":
    freeze_support()
    main()
