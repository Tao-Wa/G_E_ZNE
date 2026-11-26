"""
Clifford road sampling for Pauli propagation and noise estimation.

This module performs path sampling through quantum circuits by propagating Pauli observables
backwards through gates, randomly selecting branches for non-Clifford gates (RZ, RX, RY)
and deterministically propagating through Clifford gates. The sampled paths are then
converted to Clifford circuits, run on a noisy simulator, and the resulting expectation
values are used to compute noise weights.
"""

import os
import json
import ast
import gc
import time
import re
import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Operator, Pauli
from rand_seed import create_restartable_random_system
from tools import twirl_circuit
from Noise_model import create_fixed_noise_model


# ========================== Helper Functions ==========================
def natural_sort_key(filename):
    """Natural sort by numeric part in filename."""
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else filename


def save_array_to_file(array, filename, array_name):
    """Save a Python list/array to a text file with a variable assignment."""
    with open(filename, 'w') as f:
        f.write(f"{array_name} = {array}")


def clifford_list_to_circuit(clifford_qasm, num_qubits):
    """Convert a list of QASM gate strings into a QuantumCircuit."""
    qasm_str = "OPENQASM 2.0;\n"
    qasm_str += "include \"qelib1.inc\";\n"
    qasm_str += f"qreg q[{num_qubits}];\n"
    for line in clifford_qasm:
        qasm_str += line + "\n"
    return QuantumCircuit.from_qasm_str(qasm_str)


def uniform_gate_folding(circuit, scale_factor):
    """
    Noise amplification via uniform gate folding.
    Each gate is folded the same number of times.

    Args:
        circuit: Original quantum circuit
        scale_factor: Noise amplification factor (>=1)

    Returns:
        folded_circuit: Folded circuit
    """
    if scale_factor < 1:
        raise ValueError("Scale factor must be >= 1")
    if scale_factor == 1:
        return circuit.copy()

    folded_circuit = QuantumCircuit(circuit.num_qubits)
    original_gates = circuit.data.copy()
    k = (scale_factor - 1) / 2
    k_int = int(k)
    fractional_part = k - k_int

    for gate in original_gates:
        folded_circuit.append(gate.operation, gate.qubits)
        for _ in range(k_int):
            folded_circuit.barrier(gate.qubits)
            inverted_op = gate.operation.inverse()
            folded_circuit.append(inverted_op, gate.qubits)
            folded_circuit.barrier(gate.qubits)
            folded_circuit.append(gate.operation, gate.qubits)
            folded_circuit.barrier(gate.qubits)
        if fractional_part > 0 and np.random.random() < fractional_part:
            folded_circuit.barrier(gate.qubits)
            inverted_op = gate.operation.inverse()
            folded_circuit.append(inverted_op, gate.qubits)
            folded_circuit.barrier(gate.qubits)
            folded_circuit.append(gate.operation, gate.qubits)
    return folded_circuit


# ========================== Pauli Propagation ==========================
def propagate_pauli_through_gate(P, gate, qubits):
    """
    Propagate a Pauli string through a Clifford gate (deterministic).

    Args:
        P: Pauli string (e.g., "XIZX")
        gate: Gate name (e.g., "x", "h", "s", "sdg", "sx", "cx")
        qubits: List of qubit indices the gate acts on

    Returns:
        (new_P, sign): Propagated Pauli string and sign factor (±1)
    """
    P = list(P)
    tgt = qubits[0]
    sign = 1

    if gate == "x":
        if P[tgt] == "Z":
            P[tgt] = "Z"
            sign = -1
        elif P[tgt] == "Y":
            P[tgt] = "Y"
            sign = -1
        return "".join(P), sign

    elif gate == "h":
        if P[tgt] == "X":
            P[tgt] = "Z"
            sign = 1
        elif P[tgt] == "Z":
            P[tgt] = "X"
            sign = 1
        elif P[tgt] == "I":
            sign = 1
        return "".join(P), sign

    elif gate == "s":
        if P[tgt] == "X":
            P[tgt] = "Y"
            sign = -1
        elif P[tgt] == "Y":
            P[tgt] = "X"
        return "".join(P), sign

    elif gate == "sdg":
        if P[tgt] == "X":
            P[tgt] = "Y"
        elif P[tgt] == "Y":
            P[tgt] = "X"
            sign = -1
        return "".join(P), sign

    elif gate == "sx":
        if P[tgt] == "Z":
            P[tgt] = "Y"
        elif P[tgt] == "Y":
            P[tgt] = "Z"
            sign = -1
        return "".join(P), sign

    elif gate == "cx":
        ctrl, tgt = qubits
        P = list(P)
        # Standard Pauli propagation rules for CNOT
        if P[ctrl] == "X":
            if P[tgt] == "I":
                P[tgt] = "X"
            elif P[tgt] == "X":
                P[tgt] = "I"
            elif P[tgt] == "Y":
                P[ctrl] = "Y"
                P[tgt] = "Z"
            elif P[tgt] == "Z":
                P[ctrl] = "Y"
                P[tgt] = "Y"
                sign = -1
        elif P[ctrl] == "Y":
            if P[tgt] == "I":
                P[tgt] = "X"
            elif P[tgt] == "X":
                P[tgt] = "I"
            elif P[tgt] == "Y":
                P[ctrl] = "X"
                P[tgt] = "Z"
                sign = -1
            elif P[tgt] == "Z":
                P[ctrl] = "X"
                P[tgt] = "Y"
        elif P[ctrl] == "I":
            if P[tgt] == "Y":
                P[ctrl] = "Z"
            elif P[tgt] == "Z":
                P[ctrl] = "Z"
        elif P[ctrl] == "Z":
            if P[tgt] == "Y":
                P[ctrl] = "I"
                P[tgt] = "Y"
            elif P[tgt] == "Z":
                P[ctrl] = "I"
        return "".join(P), sign

    return "".join(P), sign


def sample_rz(theta, P, qubits, rng):
    """
    Sample a branch for an RZ gate and return the new Pauli string and a replacement Clifford gate.

    Args:
        theta: Rotation angle
        P: Current Pauli string
        qubits: List of qubits the gate acts on
        rng: Random number generator

    Returns:
        (new_P, chosen_gate, prob, p2): New Pauli string, replacement gate, amplitude, probability factor
    """
    tgt = qubits[0]
    pauli = P[tgt]
    p2 = 0.5

    if pauli == "X":
        branches = [("X", np.cos(theta), 1), ("Y", np.sin(theta), -1)]
    elif pauli == "Y":
        branches = [("Y", np.cos(theta), 1), ("X", np.sin(theta), 1)]
    else:
        return P, ("rz", [0], qubits), 1.0, 1.0

    amps = [abs(a) for (_, a, _) in branches]
    s = sum(amps)
    probs = [a / s for a in amps]
    num = rng.uniform(0, 1)

    if abs(probs[0]) < 1e-4 or abs(probs[1]) < 1e-4:
        idx = 0 if num < probs[0] else 1
        p2 = 1.0
    else:
        idx = 0 if num < 0.5 else 1

    new_letter, amp, sign = branches[idx]
    new_P = list(P)
    new_P[tgt] = new_letter
    new_P = "".join(new_P)

    if new_letter == pauli:
        chosen_gate = ("rz", [0], qubits)
    elif new_letter == "Y":   # X -> Y
        chosen_gate = ("rz", ["pi/2"], qubits)
    else:                     # Y -> X
        chosen_gate = ("rz", ["-pi/2"], qubits)

    return new_P, chosen_gate, amp * sign, p2


def sample_rx(theta, P, qubits, rng):
    """Sample branch for RX gate (similar to RZ)."""
    tgt = qubits[0]
    pauli = P[tgt]
    p2 = 0.5

    if pauli == "Y":
        branches = [("Y", np.cos(theta), 1), ("Z", np.sin(theta), -1)]
    elif pauli == "Z":
        branches = [("Z", np.cos(theta), 1), ("Y", np.sin(theta), 1)]
    else:
        return P, ("rx", [0], qubits), 1.0, 1.0

    amps = [abs(a) for (_, a, _) in branches]
    s = sum(amps)
    probs = [a / s for a in amps]
    num = rng.uniform(0, 1)

    if abs(probs[0]) < 1e-4 or abs(probs[1]) < 1e-4:
        idx = 0 if num < probs[0] else 1
        p2 = 1.0
    else:
        idx = 0 if num < 0.5 else 1

    new_letter, amp, sign = branches[idx]
    new_P = list(P)
    new_P[tgt] = new_letter
    new_P = "".join(new_P)

    if new_letter == pauli:
        chosen_gate = ("rx", [0], qubits)
    elif new_letter == "Y":   # Z -> Y
        chosen_gate = ("rx", ["pi/2"], qubits)
    else:                     # Y -> Z
        chosen_gate = ("rx", ["-pi/2"], qubits)

    return new_P, chosen_gate, amp * sign, p2


def sample_ry(theta, P, qubits, rng):
    """Sample branch for RY gate (similar to RZ)."""
    tgt = qubits[0]
    pauli = P[tgt]
    p2 = 0.5

    if pauli == "X":
        branches = [("X", np.cos(theta), 1), ("Z", np.sin(theta), 1)]
    elif pauli == "Z":
        branches = [("Z", np.cos(theta), 1), ("X", np.sin(theta), -1)]
    else:
        return P, ("ry", [0], qubits), 1.0, 1.0

    amps = [abs(a) for (_, a, _) in branches]
    s = sum(amps)
    probs = [a / s for a in amps]
    num = rng.uniform(0, 1)

    if abs(probs[0]) < 1e-4 or abs(probs[1]) < 1e-4:
        idx = 0 if num < probs[0] else 1
        p2 = 1.0
    else:
        idx = 0 if num < 0.5 else 1

    new_letter, amp, sign = branches[idx]
    new_P = list(P)
    new_P[tgt] = new_letter
    new_P = "".join(new_P)

    if new_letter == pauli:
        chosen_gate = ("ry", [0], qubits)
    elif new_letter == "Z":   # X -> Z
        chosen_gate = ("ry", ["pi/2"], qubits)
    else:                     # Z -> X
        chosen_gate = ("ry", ["-pi/2"], qubits)

    return new_P, chosen_gate, amp * sign, p2


# ========================== Path Sampling ==========================
def sample_path_with_observable(qc_0, observable, samples=1, max_attempts=300, rng=None):
    """
    Sample paths by backward propagation of the observable.

    Args:
        qc_0: Original quantum circuit
        observable: Pauli observable to propagate
        samples: Number of desired paths
        max_attempts: Maximum attempts to generate a valid path (starting from identity)
        rng: Random number generator (if None, uses np.random)

    Returns:
        List of dictionaries, each containing:
            - "path": list of Pauli strings at each step
            - "weight": combined branching probability factor
            - "probability": amplitude product
            - "clifford_circuit": list of QASM lines for the replacement Clifford circuit
    """
    if rng is None:
        rng = np.random

    qc = qc_0.copy()
    n_qubits = qc.num_qubits
    obs_str = observable.to_label()
    ops = qc.data
    results = []
    seen_paths = set()   # Avoid duplicate paths

    for _ in range(samples):
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            weight = 1.0
            P = obs_str
            path = [P]
            clifford_ops = []
            path_prob = 1.0

            # Backward pass over instructions
            for inst, qargs, _ in reversed(ops):
                name = inst.name.lower()
                qubits = [q.index for q in qargs]

                if name == "measure":
                    continue
                if name == "barrier":
                    clifford_ops.append(("barrier", [], qubits))
                    path.append(P)
                    continue

                # Non-Clifford: random branch
                if name == "rz":
                    theta = float(inst.params[0])
                    P, chosen_gate, prob, p2 = sample_rz(theta, P, qubits, rng)
                    weight *= p2
                    path_prob *= prob
                    clifford_ops.append(chosen_gate)
                    path.append(P)
                    continue
                if name == "rx":
                    theta = float(inst.params[0])
                    P, chosen_gate, prob, p2 = sample_rx(theta, P, qubits, rng)
                    weight *= p2
                    path_prob *= prob
                    clifford_ops.append(chosen_gate)
                    path.append(P)
                    continue
                if name == "ry":
                    theta = float(inst.params[0])
                    P, chosen_gate, prob, p2 = sample_ry(theta, P, qubits, rng)
                    weight *= p2
                    path_prob *= prob
                    clifford_ops.append(chosen_gate)
                    path.append(P)
                    continue

                # Clifford: deterministic propagation
                prev_P, sign = propagate_pauli_through_gate(P, name, qubits)
                path_prob *= sign
                path.append(prev_P)
                clifford_ops.append((name, inst.params, qubits))
                P = prev_P

            # Check if the initial state (final Pauli after full backward) is identity-like
            init_P = path[-1]
            offset = 0
            for i in range(n_qubits):
                if init_P[i] in ("X", "Y"):
                    offset = 1
                    break
            if offset == 0:
                # Valid path: accept
                path = path[::-1]           # reverse to forward order
                path_key = tuple(path)
                if path_key in seen_paths:
                    continue                # duplicate, retry
                seen_paths.add(path_key)
                break
            # else invalid, continue loop

        if attempt == max_attempts:
            # Force skip this sample
            continue

        # Build forward Clifford circuit (reverse order of ops)
        clifford_ops = list(reversed(clifford_ops))
        clifford_qasm = []
        for g, params, qubits in clifford_ops:
            if len(params) > 0:
                p = ",".join(str(x) for x in params)
                clifford_qasm.append(f"{g}({p}) q[{qubits[0]}];")
            else:
                if len(qubits) == 1:
                    clifford_qasm.append(f"{g} q[{qubits[0]}];")
                else:
                    clifford_qasm.append(f"{g} q[{qubits[0]}], q[{qubits[1]}];")

        results.append({
            "path": path,
            "weight": weight,
            "probability": path_prob,
            "clifford_circuit": clifford_qasm
        })

    return results


# ========================== Main Processing ==========================
def process_circuit_batch(circuit_dir, observable_str, target_dir, record=True):
    """
    Main function: process all circuits in circuit_dir, perform path sampling,
    compute noise weights via noisy simulation, and save results.
    """
    observable = Pauli(observable_str)
    observable_op = Operator(observable)
    circuit_files = sorted(
        [f for f in os.listdir(circuit_dir) if f.endswith('.qasm')],
        key=natural_sort_key
    )
    if not circuit_files:
        print(f"Error: No .qasm files found in '{circuit_dir}'!")
        return

    print(f"Found {len(circuit_files)} quantum circuit files")
    print(f"Calculating observable: {observable}\n")

    seed = 123
    rng = create_restartable_random_system(seed)
    simulator = Aer.get_backend("aer_simulator_density_matrix")
    noise_model = create_fixed_noise_model()
    T = 1   # Pauli twirling repetitions (kept as original)

    # Directory for storing sampled paths
    os.makedirs("result", exist_ok=True)

    for i, filename in enumerate(circuit_files):
        print(f'circuits_{i}:')
        rng.restart(seed)
        filepath = os.path.join(circuit_dir, filename)
        qc = QuantumCircuit.from_qasm_file(filepath)
        qc.remove_final_measurements()

        # ---- Step 1: Path sampling (if record=True) ----
        if record:
            # Original code: for each circuit, run 400 batches of 5000 samples
            for batch in range(400):
                print(f"  Sampling batch {batch}")
                results = sample_path_with_observable(
                    qc, observable=observable, samples=5000, rng=rng
                )
                with open(f"result/result_{batch}.txt", "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        # ---- Step 2: Process all saved result files to compute noise ----
        weigh = []      # "probability" values
        weight = []     # "weight" values
        noise_all = []  # raw noise values from each run

        base_dir = "result"
        files = sorted(os.listdir(base_dir))
        for file_idx, fname in enumerate(files):
            print(f"\n===== Processing {fname} =====")
            file_path1 = os.path.join(base_dir, fname)
            with open(file_path1, "r", encoding="utf-8") as f:
                try:
                    results = json.load(f)
                except:
                    results = ast.literal_eval(f.read())

            all_circuits = []
            noise = []
            rng1 = np.random.default_rng(42)  # fixed seed for reproducibility

            for r in results:
                weight.append(r["weight"])
                weigh.append(r["probability"])
                qc_clifford_0 = clifford_list_to_circuit(r["clifford_circuit"], qc.num_qubits)

                for j in range(T):
                    seed1 = rng1.integers(1 << 30)
                    if T == 1:
                        qc_clifford1 = qc_clifford_0.copy()
                    else:
                        qc_clifford1 = twirl_circuit(qc_clifford_0, seed=seed1)
                    qc_clifford = uniform_gate_folding(qc_clifford1, 1)
                    qc_clifford.save_density_matrix()
                    all_circuits.append(qc_clifford)

            result_batch = simulator.run(all_circuits, noise_model=noise_model).result()
            for idx in range(len(all_circuits)):
                rho = np.array(result_batch.data(all_circuits[idx])["density_matrix"], dtype=complex)
                exp_val = np.real(np.trace(rho @ observable_op.data))
                noise.append(np.abs(exp_val))
            noise_all.extend(noise)
            print(f"  Batch {file_idx} done")
            # Cleanup to free memory
            del results, all_circuits, result_batch, noise
            gc.collect()

        # Average over twirling repetitions (T=1 so this is just list)
        noise = [sum(noise_all[i:i+T]) / T for i in range(0, len(noise_all), T)]

        # Save arrays to target directory
        os.makedirs(target_dir, exist_ok=True)
        weigh_filename = "weigh_random_10_1_0.txt"
        noise_filename = "noise_random_10_1_0.txt"
        weight_filename = "pauli_weight_random_10_1_0.txt"

        save_array_to_file(weigh, os.path.join(target_dir, weigh_filename), "weigh")
        save_array_to_file(noise, os.path.join(target_dir, noise_filename), "noise")
        save_array_to_file(weight, os.path.join(target_dir, weight_filename), "pauli_weight")

        print(f"Finished circuit {i}, saved to {target_dir}")


if __name__ == '__main__':
    start_time = time.time()
    circuit_dir = "random_circuits"
    observable_str = "ZZZZ"
    target_dir = "./random_circuits/weigh_data"
    process_circuit_batch(circuit_dir, observable_str, target_dir, record=True)
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")