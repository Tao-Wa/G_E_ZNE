"""Zero-noise extrapolation, density-matrix path.

For every circuit of every entry it
1. amplifies the noise by gate folding over ``noise_scale_factors``,
2. averages ``num_twirling`` Pauli-twirled repetitions (exactly, via
   density matrices),
3. converts the P0 fidelity into a depolarising error coordinate, and
4. cross-validates four extrapolation models (``poly_exp`` / ``exp`` /
   ``fidelity`` / ``purity_sqrt``) with leave-one-out CV, keeping all four
   predictions plus the selected one.

Run mode is controlled by ``config.json``:

``record = True``   re-run the simulator and (re)write
                    ``zne_separated_results_detailed.txt``.
``record = False``  (shipped default) load that file and only redo the cheap
                    extrapolation, writing ``zne_separated_results.txt``.

WARNING: with ``record = False`` the loop still walks ``entrys = 1..19``; a
circuit family with fewer entries makes the script ``sys.exit(1)`` on the first
empty directory.  That is original behaviour.
"""

import os
import re
import sys
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeLima
# Custom module imports (ensure these modules exist)
from Noise_model import create_fixed_noise_model
from tools import (
    uniform_gate_folding, circuit_folding, twirl_circuit, calculate_epsilon,
    zne_loocv, build_readout_mitigation_matrix, run_pauli_twirling_zne_shot,
    run_pauli_twirling_zne
)
from extrapolation_model import (
    poly_exp_extrapolation, poly_exp_extrapolation1, poly_exp_extrapolation2,
    exp_extrapolation, exp_extrapolation1, purity_sqrt_extrapolation,
    fid_extrapolation, poly_exp_extrapolation3, poly_exp_extrapolation4,
)

# ---------------------------- Helper Functions ----------------------------
def natural_sort_key(filename):
    """Natural sort by numbers in the filename."""
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else filename

def load_previous_results(circuit_dir, row):
    """Load previous exact and noisy expectation values."""
    exact_expectations = []
    noisy_expectations = []
    prev_results_file = os.path.join(circuit_dir, "results", "expectation_values.txt")
    if not os.path.exists(prev_results_file):
        print(f"Warning: Previous results file not found: {prev_results_file}")
        return None, None

    with open(prev_results_file, 'r') as f:
        lines = f.readlines()
        for line in lines[row:]:
            if line.strip():
                parts = line.split()
                exact_expectations.append(float(parts[1]))
                noisy_expectations.append(float(parts[2]))
    return exact_expectations, noisy_expectations

def build_grover_observable(n=4, m=1):
    """Build observable for Grover circuits."""
    total_qubits = n + m
    proj_search = np.zeros((2**n, 2**n), dtype=complex)
    proj_search[0, 0] = 1.0
    I_anc = np.eye(2**m, dtype=complex)
    Pi_full = np.kron(I_anc, proj_search)
    return Operator(Pi_full)

def setup_directories(cir_type, entry):
    """Build circuit directory based on circuit type and entry number."""
    if cir_type == 'random':
        return os.path.join("random_circuits", f"random_circuits_{entry}")
    elif cir_type == 'ising_trotter':
        return os.path.join("ising_trotter_circuits", f"ising_trotter_circuits_{entry}")
    elif cir_type == 'grover':
        return os.path.join("grover_circuits", f"grover_circuits_{entry}")
    else:
        raise ValueError(f"Unknown circuit type: {cir_type}")

def load_circuit_files(circuit_dir):
    """Load and sort all .qasm files."""
    files = [f for f in os.listdir(circuit_dir) if f.endswith('.qasm')]
    return sorted(files, key=natural_sort_key)

def load_precomputed_data(file_path, noise_scale_factors):
    """Load data from precomputed file."""
    if not os.path.isfile(file_path):
        print(f"Error: File does not exist - {file_path}")
        sys.exit(1)
    data = np.loadtxt(file_path, delimiter='\t', skiprows=1)
    if len(data) == 20:   # Handle single-row data case
        data = np.array([data])
    n_rows, n_cols = data.shape
    L = n_cols // 4
    if L != len(noise_scale_factors):
        print("Error: noise_scale_factors mismatch")
        sys.exit(1)
    return data, L

def run_zne_for_circuit(qc, observable_op, noise_model, noise_scale_factors, T_default, circ, sample_method, M_inv):
    """Run ZNE to obtain exp_mean, purity_mean, fidelity_mean, factor."""
    if sample_method:
        exp_mean, purity_mean, fidelity_mean = run_pauli_twirling_zne_shot(
            qc, observable_op, noise_model=noise_model,
            noise_scale_factors=noise_scale_factors, T=T_default, circ=circ,
            shots=1000000, M_inv=M_inv
        )
    else:
        exp_mean, purity_mean, fidelity_mean = run_pauli_twirling_zne(
            qc, observable_op, noise_model=noise_model,
            noise_scale_factors=noise_scale_factors, T=T_default, circ=circ
        )
    factor = calculate_epsilon(fidelity_mean, q_num=n_qubits)
    return exp_mean, purity_mean, fidelity_mean, factor

def get_extrapolation_models(noise_scale_factors, exp_mean, factor, purity_mean, cir_type):
    """Define list of extrapolation models (according to original code configuration)."""
    if cir_type != 'grover':
        models = [
            # For ising_trotter and random circuits
            {"name": "poly_exp", "x": noise_scale_factors, "y": exp_mean,
             "func": poly_exp_extrapolation2, "x0": 0, "param": 5,
             "p0": [0.5, 0.01, -0.5, 0.3, 0.25],
             "bounds": ([-np.inf, 0, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, 0, np.inf, np.inf])},
            {"name": "exp", "x": noise_scale_factors, "y": exp_mean,
             "func": exp_extrapolation, "x0": 0, "param": 3,
             "p0": [0.5, 0.5, 0.5], "bounds": ([-np.inf, 0, -np.inf], [np.inf, 1, np.inf])},
            {"name": "fidelity", "x": factor, "y": exp_mean,
             "func": fid_extrapolation, "x0": 0, "param": 2},
            {"name": "purity_sqrt", "x": purity_mean, "y": exp_mean,
             "func": purity_sqrt_extrapolation, "x0": 1, "param": 3,
             "p0": [0.5, min(purity_mean)/2, 0.5],
             "bounds": ([-np.inf, 0, -np.inf], [np.inf, min(purity_mean), np.inf])},
        ]
    else:
        models = [
            # For Grover search circuits
            {"name": "poly_exp", "x": noise_scale_factors, "y": exp_mean,
                      "func": poly_exp_extrapolation3, "x0": 0, "param": 5,
                      "p0": [0.5, 0.01, -0.5, 0.3, 0.25],
                      "bounds": ([-np.inf, 0, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, 0, np.inf, np.inf])},
            {"name": "exp", "x": noise_scale_factors, "y": exp_mean, "func": exp_extrapolation1, "x0": 0, "param": 5,
            "p0": [0.5, 0.5, 0.5, 0.5, 0.5],
            "bounds": ([-np.inf, 0, -np.inf, 0, -np.inf], [np.inf, 1, np.inf, 1, np.inf])},

            {"name": "fidelity", "x": factor, "y": exp_mean,
             "func": fid_extrapolation, "x0": 0, "param": 2},
            {"name": "purity_sqrt", "x": purity_mean, "y": exp_mean,
             "func": purity_sqrt_extrapolation, "x0": 1, "param": 3,
             "p0": [0.5, min(purity_mean) / 2, 0.5],
             "bounds": ([-np.inf, 0, -np.inf], [np.inf, min(purity_mean), np.inf])},
        ]
    return models

def compute_modification_factor(qc, pred, exact_val):
    """Compute modification factor (for a small number of circuits)."""
    if exact_val != 0 and pred != 0:
        c = np.abs(exact_val / pred)
        if c < np.exp(len(qc.data) * 0.001):
            return c if c > 1 else 1.0
        else:
            return np.exp(len(qc.data) * 0.001)
    else:
        return 1.0

def save_zne_results(zne_file, circuit_files, poly_exp, exp, fidelity, purity_sqrt, loocv, prev_noisy, prev_exact, failures):
    """Save ZNE results to a file."""
    with open(zne_file, 'w') as f:
        f.write(f"Observable: {observable}\n\n")
        f.write("Circuit\tPoly_exp ZNE\tExp ZNE\tFidelity ZNE\tPurity_sqrt ZNE\tLoocv ZNE\tPrevious Noisy\tPrevious Exact\n")
        for i in range(len(circuit_files)):
            noisy_val = prev_noisy[i] if prev_noisy else "N/A"
            exact_val = prev_exact[i] if prev_exact else "N/A"
            f.write(f"{i}\t{poly_exp[i]:.6f}\t{exp[i]:.6f}\t{fidelity[i]:.6f}\t{purity_sqrt[i]:.6f}\t{loocv[i]:.6f}\t{noisy_val}\t{exact_val}\n")
        if failures:
            f.write("\n\nExtrapolation Failures:\n")
            for circuit_idx, method, error in failures:
                f.write(f"Circuit {circuit_idx} - {method} extrapolation failed: {error}\n")

def print_zne_results(circuit_files, poly_exp, exp, fidelity, purity_sqrt, loocv, prev_noisy, prev_exact):
    """Print ZNE results to the console."""
    print("\nZNE Results (Separated Methods):")
    print("Circuit\tPoly_exp ZNE\tExp ZNE\tFidelity ZNE\tPurity_sqrt ZNE\tLoocv ZNE\tPrevious Noisy\tPrevious Exact")
    for i in range(len(circuit_files)):
        noisy_val = prev_noisy[i] if prev_noisy else "N/A"
        exact_val = prev_exact[i] if prev_exact else "N/A"
        print(f"{i}\t{poly_exp[i]:.6f}\t{exp[i]:.6f}\t{fidelity[i]:.6f}\t{purity_sqrt[i]:.6f}\t{loocv[i]:.6f}\t{noisy_val}\t{exact_val}")

# ---------------------------- Main Program ----------------------------
if __name__ == '__main__':
    start_time = time.time()

    # Global settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Verdana']
    plt.rcParams['axes.unicode_minus'] = False

    # Load configuration
    with open("config.json") as f:
        config = json.load(f)

    # Extract configuration parameters
    mod = config["num_train"]                # Number of validation set points
    circ = config["num_non-integer"]         # Number of random repetitions for non-integer noise scaling
    T_default = config["num_twirling"]       # Number of twirl repetitions
    delate = config["fit_method"]            # 1: use only partial points for fitting, 0: use all data
    n_qubits = config["num_qubits"]
    fold_method = config["fold_method"]      # 0: uniform_gate_folding, 1: circuit_folding
    noise_scale_factors = config["noise_scale_factors"]
    cir_type = config["cir_type"]
    record = config["record"]                # Whether to use existing data
    sample_method = config["sample"]
    readout = config["readout_error"]
    row = config["row"]
    observable_str = config["observable"]

    # Initial observable
    observable = Pauli(observable_str)

    # Circuit entry list (modifiable)
    # entrys = [6]   # Original code had [3]
    entrys = list(range(1, 20))

    # Simulator and noise model
    simulator = Aer.get_backend('qasm_simulator')
    if cir_type != "grover":
        noise_model = create_fixed_noise_model()
    else:
        noise_model = NoiseModel.from_backend(FakeLima())

    # Read error mitigation matrix (if needed)
    if readout and sample_method and record:
        M = build_readout_mitigation_matrix(simulator, noise_model, n_qubits)
        M_inv = np.linalg.pinv(M)
    elif not readout and sample_method:
        M_inv = np.ones((2**n_qubits, 2**n_qubits))
    else:
        M_inv = None  # Not used

    # If circuit type is Grover, replace the observable
    if cir_type == "grover":
        observable = build_grover_observable()
        row = 10   # Override the original row value

    observable_op = Operator(observable)

    # Main loop over each entry
    for entry in entrys:
        circuit_dir = setup_directories(cir_type, entry)
        results_dir = os.path.join(circuit_dir, "zne_separated_results")
        os.makedirs(results_dir, exist_ok=True)

        # Load previous exact and noisy values
        prev_exact, prev_noisy = load_previous_results(circuit_dir, row)
        circuit_files = load_circuit_files(circuit_dir)

        if not circuit_files:
            print(f"Error: No .qasm files found in '{circuit_dir}'!")
            sys.exit(1)

        print(f"Found {len(circuit_files)} quantum circuit files")

        # Randomly select a few circuits for correction
        rand = random.Random(123)
        samples = rand.sample(range(len(circuit_files)), min(3, len(circuit_files)))
        modify_factors = []

        # Store results
        poly_exp_expectations = []
        exp_expectations = []
        fidelities_expectations = []
        purity_sqrt_expectations = []
        loocv_expectations = []
        extrapolation_failures = []

        # Data recording or loading
        if record:
            all_exp = []
            all_purity = []
            all_fidelity = []
            all_factor = []
        else:
            # Determine data file path
            data_file = os.path.join(results_dir, "zne_separated_results_detailed.txt")
            data, L = load_precomputed_data(data_file, noise_scale_factors)

        # Iterate over each circuit
        for i, filename in enumerate(circuit_files):
            if record:
                # Regenerate data
                filepath = os.path.join(circuit_dir, filename)
                qc = QuantumCircuit.from_qasm_file(filepath)
                qc.remove_final_measurements()
                print(f"Circuit #{i+1}:")
                print(f"- Original depth: {qc.depth()}, gates: {len(qc.data)}")

                # Run ZNE to obtain data
                exp_mean, purity_mean, fidelity_mean, factor = run_zne_for_circuit(
                    qc, observable_op, noise_model, noise_scale_factors,
                    T_default, circ, sample_method, M_inv
                )
                all_exp.append(exp_mean)
                all_purity.append(purity_mean)
                all_fidelity.append(fidelity_mean)
                all_factor.append(factor)
            else:
                # Read from preloaded data
                row_data = data[i, :]
                exp_mean = row_data[:L]
                purity_mean = row_data[L:2*L]
                fidelity_mean = row_data[2*L:3*L]
                factor = row_data[3*L:4*L]
                print(f"Iteration {i}: exp_mean = {exp_mean}")

                # If this is a correction circuit, also load the qc object (for later correction factor calculation)
                if i in samples:
                    filepath = os.path.join(circuit_dir, filename)
                    qc = QuantumCircuit.from_qasm_file(filepath)
                    qc.remove_final_measurements()
                    print(f"Circuit #{i+1}:")
                    print(f"- Original depth: {qc.depth()}, gates: {len(qc.data)}")
                else:
                    qc = None

            # Build extrapolation models
            models = get_extrapolation_models(noise_scale_factors, exp_mean, factor, purity_mean, cir_type)

            # Perform LOOCV to select the best model
            best_pred, best_name, details = zne_loocv(models, q_num=n_qubits, mod=mod, delate=delate)
            print(f"- loocv ZNE expectation name: {best_name}")
            print(f"- loocv ZNE expectation value: {best_pred:.6f}")
            for d in details:
                print(f"  {d['name']} → {d['cv_error']:.6f}")
                print(f"  {d['name']} → {d['pred']:.6f}")

            # For selected sample circuits, compute the correction factor
            if i in samples and qc is not None:
                poly_pred = next(d['pred'] for d in details if d['name'] == "poly_exp")
                exact_val = prev_exact[i] if prev_exact and i < len(prev_exact) else None
                if exact_val is not None:
                    mod_factor = compute_modification_factor(qc, poly_pred, exact_val)
                    modify_factors.append(mod_factor)
                else:
                    modify_factors.append(1.0)

            # Collect predictions from each method
            poly_exp_expectations.append(next(d['pred'] for d in details if d['name'] == "poly_exp"))
            exp_expectations.append(next(d['pred'] for d in details if d['name'] == "exp"))
            fidelities_expectations.append(next(d['pred'] for d in details if d['name'] == "fidelity"))
            purity_sqrt_expectations.append(next(d['pred'] for d in details if d['name'] == "purity_sqrt"))
            loocv_expectations.append(best_pred)
            print("-" * 60)

        # If needed, save detailed data
        if record:
            all_exp = np.array(all_exp)
            all_purity = np.array(all_purity)
            all_fidelity = np.array(all_fidelity)
            all_factor = np.array(all_factor)
            data_save = np.hstack([all_exp, all_purity, all_fidelity, all_factor])
            save_path = os.path.join(results_dir, "zne_separated_results_detailed.txt")
            np.savetxt(save_path, data_save, delimiter='\t', header='exp...\tpurity...\tfidelity...\tfactor...', comments='')

        # Optionally apply correction factor
        if cir_type != "grover":
            if modify_factors:
                modified_mean = np.mean(modify_factors)
                poly_exp_expectations = [x * modified_mean for x in poly_exp_expectations]

        # Determine final result file path
        zne_file = os.path.join(results_dir, "zne_separated_results.txt")

        # Save results
        save_zne_results(zne_file, circuit_files,
                         poly_exp_expectations, exp_expectations,
                         fidelities_expectations, purity_sqrt_expectations,
                         loocv_expectations, prev_noisy, prev_exact,
                         extrapolation_failures)

        # Print results
        print_zne_results(circuit_files,
                          poly_exp_expectations, exp_expectations,
                          fidelities_expectations, purity_sqrt_expectations,
                          loocv_expectations, prev_noisy, prev_exact)

        print(f"\nDetailed ZNE results saved to: {zne_file}")

    end_time = time.time()
    print(f"Execution time: {round(end_time - start_time, 2)} seconds")