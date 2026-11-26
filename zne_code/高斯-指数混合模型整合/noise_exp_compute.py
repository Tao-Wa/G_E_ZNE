"""Exact and noisy expectation values of every exported circuit.

For each circuit this step computes

* the exact expectation value from a statevector simulation, and
* the noisy expectation value from a density-matrix simulation under the
  ``FakeQuito`` noise model,

and stores both plus the relative noise ratio in
``<family>_circuits_<n>/results/expectation_values.txt``.

This is a *producer* step: it always re-simulates, it is not gated by the
``record`` switch, and its output is the reference ("Exact") column used by all
downstream error analysis.
"""

import json
import pickle
import os
import re
import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Pauli, Statevector, Operator, SparsePauliOp
from Noise_model import create_fixed_noise_model

# ---------------------------- Utility Functions ----------------------------
def natural_sort_key(filename):
    """Natural sort by numbers in the filename."""
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else filename

def load_config(config_path="config.json"):
    """Load configuration file."""
    with open(config_path, "r") as f:
        return json.load(f)

def build_circuit_dir(cir_type, entry):
    """Build circuit directory based on circuit type and entry number."""
    if cir_type == 'random':
        return os.path.join("random_circuits", f"random_circuits_{entry}")
    elif cir_type == 'ising_trotter':
        return os.path.join("ising_trotter_circuits", f"ising_trotter_circuits_{entry}")
    elif cir_type == 'grover':
        return os.path.join("grover_circuits", f"grover_circuits_{entry}")
    else:
        raise ValueError(f"Unknown circuit type: {cir_type}")

def build_observable(cir_type, config):
    """Build observable based on circuit type."""
    if cir_type == "grover":
        n = 4          # Number of system qubits
        m = 1          # Number of ancilla qubits
        total_qubits = n + m

        # Project onto |00⟩⟨00| on the system
        proj_search = np.zeros((2 ** n, 2 ** n), dtype=complex)
        proj_search[0, 0] = 1.0

        I_anc = np.eye(2 ** m, dtype=complex)
        Pi_full = np.kron(I_anc, proj_search)

        observable_operator = Pi_full  # The -I_total/16 part was commented out in original code
        observable = Operator(observable_operator)
        observable_pauli = SparsePauliOp.from_operator(observable)
        print(observable_pauli)
        return observable
    else:
        # For random or ising_trotter, use the Pauli string
        observable_str = config["observable"]
        return Pauli(observable_str)

# ---------------------------- Single Circuit Processing ----------------------------
def process_circuit(qc, observable, noise_model, simulator, q_num):
    """
    Process a single circuit, return exact and noisy expectation values.
    """
    # Exact expectation value (noiseless)
    state = Statevector.from_instruction(qc)
    exact_exp = state.expectation_value(observable).real

    # Noisy expectation value
    noisy_qc = qc.copy()
    noisy_qc.save_density_matrix()
    result = simulator.run(
        noisy_qc,
        noise_model=noise_model,
        shots=2 ** q_num + 1,
        optimization_level=0
    ).result()
    density_matrix = result.data()['density_matrix']
    noisy_exp = density_matrix.expectation_value(observable).real

    return exact_exp, noisy_exp

# ---------------------------- Main Processing Flow ----------------------------
def main():
    # Load configuration
    config = load_config()
    cir_type = config["cir_type"]
    entries = list(range(1, 20))
    # entries = [6]
    if cir_type != "grover":
        q_num = config["num_qubits"]
    else:
        q_num = 5    # Used to set shots

    observable = build_observable(cir_type, config)

    # Noise model and simulator
    noise_model = create_fixed_noise_model()
    simulator = Aer.get_backend('qasm_simulator')

    # Iterate over each entry
    for entry in entries:
        circuit_dir = build_circuit_dir(cir_type, entry)
        results_dir = os.path.join(circuit_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Get all .qasm files, sorted naturally
        circuit_files = sorted(
            [f for f in os.listdir(circuit_dir) if f.endswith('.qasm')],
            key=natural_sort_key
        )
        if not circuit_files:
            print(f"Error: No .qasm files found in '{circuit_dir}'!")
            continue

        print(f"Found {len(circuit_files)} quantum circuit files in {circuit_dir}")

        exact_expectations = []
        noisy_expectations = []
        noise_ratios = []

        # Process each circuit file
        for idx, filename in enumerate(circuit_files):
            filepath = os.path.join(circuit_dir, filename)
            try:
                qc = QuantumCircuit.from_qasm_file(filepath)
                qc.remove_final_measurements()   # Ensure no measurement operations
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

            print(f"Circuit #{idx} ({filename}):")
            exact_exp, noisy_exp = process_circuit(qc, observable, noise_model, simulator, q_num)

            # Calculate noise ratio
            if abs(exact_exp) > 1e-6:
                noise_ratio = abs(exact_exp - noisy_exp) / abs(exact_exp)
            else:
                noise_ratio = abs(exact_exp - noisy_exp)

            exact_expectations.append(exact_exp)
            noisy_expectations.append(noisy_exp)
            noise_ratios.append(noise_ratio)

            print(f"- Exact expectation value: {exact_exp:.6f}")
            print(f"- Noisy expectation value: {noisy_exp:.6f}")
            print(f"- Noise ratio: {noise_ratio:.2%}")
            print("-" * 60)

        # Save results to file
        result_file = os.path.join(results_dir, "expectation_values.txt")
        with open(result_file, 'w') as f:
            f.write(f"Observable: {observable}\n\n")
            f.write("Circuit\tExact\t\tNoisy\t\tNoise Ratio\n")
            for i, (exact, noisy, ratio) in enumerate(zip(exact_expectations, noisy_expectations, noise_ratios)):
                f.write(f"{i}\t{exact:.6f}\t{noisy:.6f}\t{ratio:.4f}\n")

        # Console output summary
        print("\nAll expectation values:")
        print("Circuit\tExact\t\tNoisy\t\tNoise Ratio")
        for i, (exact, noisy, ratio) in enumerate(zip(exact_expectations, noisy_expectations, noise_ratios)):
            print(f"{i}\t{exact:.6f}\t{noisy:.6f}\t{ratio:.4f}")
        print(f"\nDetailed results saved to: {result_file}\n")

if __name__ == "__main__":
    main()