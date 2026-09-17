from multiprocessing import freeze_support
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

import csv
import os
import re
import pymatching
import numpy as np
import stim
import matplotlib.pyplot as plt
import lightstim
from multiprocessing import freeze_support
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

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

# circuit_d3_d3_d3 = load_circuit("circuit_d3_d3_d3.stim")
# circuit_d5_bell_graphlike = load_circuit("pauliXX_d5_bell_graphlike.stim")
circuit_d5_bell = load_circuit("pauliXX_d5_bell_graphlike.stim")
circuit_d5_bell_singleRound = load_circuit("pauliXX_d5_bell_singleRound_beforeafter.stim")
circuit_d5_interleaving = load_circuit("pauliXX_d5_interleaving.stim")
circuit_d5_alternating = load_circuit("pauliXX_d5_alternating.stim")


# d7 codes
# circuit_d7_bell = load_circuit("pauliXX_d7_bell.stim")
# circuit_d7_alternating = load_circuit("pauliXX_d7_alternating.stim")
# circuit_d7_interleaving = load_circuit("pauliXX_d7_interleaving.stim")

# =========================
# Shared noise settings
# =========================

p1 = 0.0001
default_p2 = 0.0001
shots = 1_000_000
override_p2_values = np.linspace(0.01, 0.001, 10)

# Make the over_ride_pairs that will provide the CX gates across the inter-QPU boundary with different error rates
def make_override_pairs(number_string):
    numbers = list(map(int, number_string.split()))
    return [(numbers[i], numbers[i + 1]) for i in range(len(numbers) - 1)]

# Examples

# Logical CNOT perfomance of the d3-d3-d3 lattice surgery
# override_pairs_d3_d3_d3_LS_on_different_QPU = make_override_pairs("23 19 24 20 25 21 26")
# override_pairs_d5_bell_graphlike_LS_on_different_QPU = make_override_pairs("65 83 97 66 84 98 75 92 67 85 99 68 86 100 76 93 77 94 69 70 87 101 78 88 79 89 80 90 81 91 82 147 154 155 148 165 164 152 160 149 156 166 157 150 167 153 161 151 158 168 159 169")
# override_pairs_d5_bell_LS_on_different_QPU = make_override_pairs("83 84 75 92 85 86 76 93 77 94 87 78 88 79 89 80 90 81 91 82 152 157 153 158 154 159 155 160 156 161")
# override_pairs_d5_interleaving_LS_on_different_QPU = make_override_pairs("84 95 85 96 86 97 87 98 88 99 89 100 90 101 91 102 92 103 93 104 94 215 204 214 203 213 202 212 201 211 200 210")
# override_pairs_d5_alternating_LS_on_different_QPU = make_override_pairs("84 74 85 75 86 76 87 77 88 78 89 79 90 80 91 81 92 82 93 83 94 189 200 190 201 191 202 192 203 193 204")

# d5, left boundary only
override_pairs_d5_bell_LS_on_different_QPU = make_override_pairs("154 155 152 160 156 157 153 161 158 159 147 164 165 148 149 166 150 167 151 168 169")
override_pairs_d5_bell_LS_on_different_QPU_only_seam_override = make_override_pairs("154 155 152 160 156 157 153 161 158 159")
override_pairs_d5_bell_singleRound_LS_on_different_QPU_only_seam_override = make_override_pairs("154 155 152 160 156 157 153 161 158 159")
override_pairs_d5_interleaving_LS_on_different_QPU = make_override_pairs("215 204 214 203 213 202 212 201 211 200 210")
override_pairs_d5_alternating_LS_on_different_QPU = make_override_pairs("215 204 214 203 213 202 212 201 211 200 210")

# d7 overrides
# override_pairs_d7_bell_LS_on_different_QPU = make_override_pairs("172 173 161 185 174 175 162 186 176 177 163 187 164 188 165 178 179 166 180 167 181 168 182 169 183 170 184 171 290 300 314 291 301 315 297 308 292 302 316 293 303 317 298 309 294 304 318 305 295 319 299 310 296 306 320 307")
# override_pairs_d7_alternating_LS_on_different_QPU = make_override_pairs("188 202 187 201 186 200 185 199 184 198 183 197 182 196 181 195 180 194 179 193 178 192 177 191 176 190 175 189 174 322 314 321 313 320 312 319 311 318 310 317 309 316 308 315")
# override_pairs_d7_interleaving_LS_on_different_QPU = make_override_pairs("188 202 187 201 186 200 185 199 184 198 183 197 182 196 181 195 180 194 179 193 178 192 177 191 176 190 175 189 174 413 398 412 397 411 396 410 395 409 394 408 393 407 392 406")

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
# Bell-specific noise model: CXX gates
# =========================
#
# The bell merge applies a three-qubit CXX (one control, two targets).  stim has
# no CXX instruction, so the circuit spells it out as CX(c, t1) on one tick and
# CX(c, t2) on the following tick.  make_noisy_circuit above sees two CX gates
# and charges a full DEPOLARIZE2 to each of them, i.e. it bills the bell circuit
# for two two-qubit gates where the hardware only runs one.  Over 5 merge rounds
# x 2 CXX gates that is 10 extra two-qubit error locations that the alternating
# and interleaved circuits never pay.
#
# Note that neither make_noisy_circuit nor this function inserts any idling
# noise: unused qubits are completely error free, so the extra tick by itself
# costs the bell circuit nothing right now.  merge_cxx_tick=True still collapses
# the two halves into a single tick so that stays true if an idle-noise model is
# added later.

def find_cxx_halves(circuit):
    """Locate the CX gates that are the second half of a CXX.

    Returns (second_halves, first_halves, extra_tick_layers) where the first two
    are dicts keyed by (layer_index, control, target) mapping each half to the
    other half's (layer_index, control, target), and extra_tick_layers is the set
    of layer indices whose trailing TICK separates a CXX from its second half.

    A second-half layer is a tick layer that (a) contains only CX gates, (b) is
    strictly smaller than the preceding layer, and (c) has every one of its
    controls also acting as a control in the preceding layer.  In the d5 bell
    circuit this matches exactly the 5 merge rounds and nothing else.
    """
    layers = []
    current = []
    for instruction in circuit:
        if instruction.name == "TICK":
            layers.append(current)
            current = []
        else:
            current.append(instruction)
    layers.append(current)

    def cx_pairs(layer):
        pairs = []
        for instruction in layer:
            if instruction.name == "CX":
                targets = [t.value for t in instruction.targets_copy()]
                pairs += list(zip(targets[0::2], targets[1::2]))
        return pairs

    second_halves = {}
    first_halves = {}
    extra_tick_layers = set()

    for i in range(1, len(layers)):
        layer = layers[i]
        if not layer or any(instruction.name != "CX" for instruction in layer):
            continue
        pairs = cx_pairs(layer)
        prev_pairs = cx_pairs(layers[i - 1])
        if not pairs or len(pairs) >= len(prev_pairs):
            continue
        prev_by_control = {c: t for c, t in prev_pairs}
        if not all(c in prev_by_control for c, _ in pairs):
            continue

        for c, t in pairs:
            first = (i - 1, c, prev_by_control[c])
            second = (i, c, t)
            second_halves[second] = first
            first_halves[first] = second
        extra_tick_layers.add(i - 1)

    return second_halves, first_halves, extra_tick_layers


def make_bell_noisy_circuit(circuit, override_edges, override_p2,
                            default_p2=default_p2, p1=p1,
                            merge_cxx_tick=True, cxx_extra_p1=0.0,
                            verbose=False):
    """make_noisy_circuit, but each CX/CX pair forming a CXX is charged once.

    merge_cxx_tick : drop the TICK between the two halves so the CXX occupies a
                     single tick, matching the hardware gate.
    cxx_extra_p1   : optional DEPOLARIZE1 on the second target, to model a
                     three-qubit gate being somewhat noisier than a two-qubit
                     one.  0.0 means a CXX costs exactly one CX worth of noise.
    """
    second_halves, first_halves, extra_tick_layers = find_cxx_halves(circuit)
    if not second_halves:
        raise ValueError(
            "no CXX gates found; use make_noisy_circuit for this circuit"
        )
    if verbose:
        print(f"found {len(second_halves)} CXX gates:")
        for (i, c, t2), (_, _, t1) in sorted(second_halves.items()):
            print(f"  layer {i - 1}/{i}: CXX control {c} targets {t1}, {t2}")

    noisy_circuit = stim.Circuit()
    layer = 0
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

                key = (layer, q1, q2)
                if key in second_halves:
                    # Second half of a CXX: the noise was already charged on the
                    # first half, so this gate is free.
                    if cxx_extra_p1 > 0:
                        noisy_circuit.append("DEPOLARIZE1", [t2], cxx_extra_p1)
                    continue

                inter_qpu = (q1, q2) in override_edges
                if key in first_halves:
                    # Charge the whole CXX at the inter-QPU rate if either half
                    # crosses the boundary.
                    _, c, other = first_halves[key]
                    inter_qpu = inter_qpu or (c, other) in override_edges

                noisy_circuit.append(
                    "DEPOLARIZE2", [t1, t2],
                    override_p2 if inter_qpu else default_p2,
                )

        elif instruction.name in ["M", "MX"]:
            noisy_circuit.append(instruction.name, instruction.targets_copy(), p1)

        elif instruction.name in ["R", "RX", "H", "X"]:
            noisy_circuit.append(instruction)
            noisy_circuit.append("DEPOLARIZE1", instruction.targets_copy(), p1)

        elif instruction.name == "TICK":
            if not (merge_cxx_tick and layer in extra_tick_layers):
                noisy_circuit.append(instruction)
            layer += 1

        elif instruction.name in [
            "QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE",
            "DEPOLARIZE1", "DEPOLARIZE2"
        ]:
            noisy_circuit.append(instruction)

        else:
            raise NotImplementedError("forgot " + str(instruction))

    return noisy_circuit


def logical_error_rate(noisy_circuit, shots=shots):
    model = noisy_circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)
    sampler = noisy_circuit.compile_detector_sampler()
    syndrome, actual_observables = sampler.sample(shots=shots, separate_observables=True)
    predicted_observables = matching.decode_batch(syndrome)
    num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))
    return num_errors / shots

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

# def label_to_filename(label, suffix=".csv"):
#     slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
#     return f"{slug}{suffix}"

# def save_xy_csv(x_values, y_values, label, out_dir=SCRIPT_DIR):
#     """Save the (x, y) pairs behind a plotted series to a CSV named after its label."""
#     path = os.path.join(out_dir, label_to_filename(label))
#     with open(path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["override_p2", "logical_error_rate"])
#         writer.writerows(zip(x_values, y_values))
#     print(f"Saved {label!r} data to {path}")
#     return path

def sweep_dataset(circuit, override_pairs_raw, label, override_p2_values, shots=shots):
    override_edges = make_override_edges(override_pairs_raw)
    logical_error_rates = []

    print(f"\n=== {label} ===")
    for override_p2 in override_p2_values:
        noisy_circuit = make_noisy_circuit(circuit, override_edges, override_p2)
        distance = len(noisy_circuit.shortest_graphlike_error())
        ler = logical_error_rate(noisy_circuit, shots=shots)
        logical_error_rates.append(ler)

        print(f"override_p2 = {override_p2:.8e}")
        print(f"distance = {distance}")
        print(f"logical error rate = {ler:.8e}")

    logical_error_rates = np.array(logical_error_rates)

    order = np.argsort(override_p2_values)
    # save_xy_csv(np.asarray(override_p2_values)[order], logical_error_rates[order], label)

    return logical_error_rates

def main():
    # =========================
    # Run sweeps for both cases
    # =========================

    '''save_two_qubit_noise_map(
        circuit=circuit_d3_d3,
        override_edges=make_override_edges(override_pairs_d3_d3_LS_on_different_QPU),
        override_p2=0.01,
        default_p2=0.0001,
        filename="d3_d3_two_qubit_noise_map.txt",
    )'''

    # Compare logical CNOT perfomance of the 7x5 lattice (with traveling stabilizers) to the d3-d3-d3 and d5-d5-d5 lattices 
    # ler_d3_d3_d3_different_QPU = sweep_dataset(
    #     circuit=circuit_d3_d3_d3,
    #     override_pairs_raw=override_pairs_d3_d3_d3_LS_on_different_QPU,
    #     label="D3-D3-D3 LS across different QPUs",
    #     override_p2_values=override_p2_values,
    #     shots=shots,
    # )
    # ler_d5_bell_graphlike_different_QPU = sweep_dataset(
    #     circuit=circuit_d5_bell_graphlike,
    #     override_pairs_raw=override_pairs_d5_bell_graphlike_LS_on_different_QPU,
    #     label="D5-Bell Graphlike LS across different QPUs",
    #     override_p2_values=override_p2_values,
    #     shots=shots,
    # )
    # ler_d5_bell_different_QPU = sweep_dataset(
    #         circuit=circuit_d5_bell,
    #         override_pairs_raw=override_pairs_d5_bell_LS_on_different_QPU,
    #         label="D5-Bell LS across different QPUs",
    #         override_p2_values=override_p2_values,
    #         shots=shots,
    # )
    ler_d5_bell_different_QPU_only_seam_override = sweep_dataset(
            circuit=circuit_d5_bell,
            override_pairs_raw=override_pairs_d5_bell_LS_on_different_QPU_only_seam_override,
            label="D5-Bell LS across different QPUs",
            override_p2_values=override_p2_values,
            shots=shots,
    )
    ler_d5_bell_singleRound_different_QPU_only_seam_override = sweep_dataset(
            circuit=circuit_d5_bell,
            override_pairs_raw=override_pairs_d5_bell_singleRound_LS_on_different_QPU_only_seam_override,
            label="D5-Bell Single Round Before/After surgery LS across different QPUs",
            override_p2_values=override_p2_values,
            shots=shots,
    )
    ler_d5_interleaving_different_QPU = sweep_dataset(
        circuit=circuit_d5_interleaving,
        override_pairs_raw=override_pairs_d5_interleaving_LS_on_different_QPU,
        label="D5-Interleaving LS across different QPUs",
        override_p2_values=override_p2_values,
        shots=shots,
    )
    ler_d5_alternating_different_QPU = sweep_dataset(
        circuit=circuit_d5_alternating,
        override_pairs_raw=override_pairs_d5_alternating_LS_on_different_QPU,
        label="D5-Alternating LS across different QPUs",
        override_p2_values=override_p2_values,
        shots=shots,
    )
    
    # d7 runners
    # ler_d7_interleaving_different_QPU = sweep_dataset(
    #     circuit=circuit_d7_interleaving,
    #     override_pairs_raw=override_pairs_d7_interleaving_LS_on_different_QPU,
    #     label="D7-Interleaving LS across different QPUs",
    #     override_p2_values=override_p2_values,
    #     shots=shots,
    # )
    # ler_d7_alternating_different_QPU = sweep_dataset(
    #     circuit=circuit_d7_alternating,
    #     override_pairs_raw=override_pairs_d7_alternating_LS_on_different_QPU,
    #     label="D7-Alternating LS across different QPUs",
    #     override_p2_values=override_p2_values,
    #     shots=shots,
    # )
    # ler_d7_bell_different_QPU = sweep_dataset(
    #         circuit=circuit_d7_bell,
    #         override_pairs_raw=override_pairs_d7_bell_LS_on_different_QPU,
    #         label="D7-Bell LS across different QPUs",
    #         override_p2_values=override_p2_values,
    #         shots=shots,
    # )

    # =========================
    # Plot results together
    # =========================
    markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p', 'h', '*', 'x', '+']
    order = np.argsort(override_p2_values)
    x = override_p2_values[order]

    plt.figure(figsize=(7, 5))

    # Compare logical CNOT perfomance of the 7x5 lattice (with traveling stabilizers) to the d3-d3-d3 and d5-d5-d5 lattices
    # plt.plot(x, ler_d5_bell_different_QPU[order], marker=markers[1], label='D5-Bell LS across different QPUs')
    plt.plot(x, ler_d5_bell_different_QPU_only_seam_override[order], marker=markers[2], label='D5-Bell LS across different QPUs')
    plt.plot(x, ler_d5_bell_singleRound_different_QPU_only_seam_override[order], marker=markers[2], label='D5-Bell Single Round Before/After surgery LS across different QPUs')
    plt.plot(x, ler_d5_interleaving_different_QPU[order], marker=markers[3], label='D5-Interleaving LS across different QPUs')
    plt.plot(x, ler_d5_alternating_different_QPU[order], marker=markers[4], label='D5-Alternating LS across different QPUs')

    # plt.plot(x, ler_d7_bell_different_QPU[order], marker=markers[0], label='D7-Bell LS across different QPUs')
    # plt.plot(x, ler_d7_interleaving_different_QPU[order], marker=markers[3], label='D7-Interleaving LS across different QPUs')
    # plt.plot(x, ler_d7_alternating_different_QPU[order], marker=markers[4], label='D7-Alternating LS across different QPUs')
    
    plt.xlabel("Inter-QPU error probability")
    plt.ylabel("Logical error rate")
    plt.title("Logical error rate vs inter-QPU error probability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    freeze_support()
    main()