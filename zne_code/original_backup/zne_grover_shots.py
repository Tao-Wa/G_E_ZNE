import os
import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Operator
from qiskit.quantum_info import Pauli
import matplotlib.pyplot as plt
import random
import re
import sys
import time
import json
from tools import (uniform_gate_folding, circuit_folding, twirl_circuit, calculate_epsilon, zne_loocv,
                   build_readout_mitigation_matrix, run_pauli_twirling_zne_shot)  # no longer import run_pauli_twirling_zne_shot
from extrapolation_model import (poly_exp_extrapolation, poly_exp_extrapolation1, poly_exp_extrapolation2,
                                 exp_extrapolation, exp_extrapolation1,
                                 purity_sqrt_extrapolation, fid_extrapolation, poly_exp_extrapolation3,
                                 poly_exp_extrapolation4, )
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeLima, FakeQuito, FakeJakarta, FakeNairobi

# ========== Override run_pauli_twirling_zne_shot to support independent random seeds ==========

# ========== Original Helper Functions ==========
def natural_sort_key(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else filename

def load_previous_results():
    """Load previous exact and noisy expectation values (keep original logic)."""
    exact_expectations = []
    noisy_expectations = []
    prev_results_file = os.path.join(circuit_dir, "results", "expectation_values.txt")
    if not os.path.exists(prev_results_file):
        print(f"Warning: Previous results file not found: {prev_results_file}")
        return None, None
    with open(prev_results_file, 'r') as f:
        lines = f.readlines()
        for line in lines[row1:]:
            if line.strip():
                parts = line.split()
                exact_expectations.append(float(parts[1]))
                noisy_expectations.append(float(parts[2]))
    return exact_expectations, noisy_expectations

def create_fixed_noise_model():
    backends = {
        "FakeLima": FakeLima(),
        "FakeQuito": FakeQuito(),
        "FakeJakarta": FakeJakarta(),
        "FakeNairobi": FakeNairobi()
    }
    selected_backend = backends["FakeLima"]
    noise_model = NoiseModel.from_backend(selected_backend)
    return noise_model

def extrapolate_experiment(exp_mean, purity_mean, fidelity_mean, factor,
                           noise_scale_factors, n_qubits, mod, delate):
    """
    Perform extrapolation for one experiment, return a dictionary of extrapolated values for each method.
    """
    models = [
        {"name": "poly_exp", "x": noise_scale_factors, "y": exp_mean, "func": poly_exp_extrapolation3,
         "x0": 0, "param": 5, "p0": [0.5, 0.01, -0.5, 0.3, 0.25],
         "bounds": ([-np.inf, 0, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, 0, np.inf, np.inf])},
        {"name": "exp", "x": noise_scale_factors, "y": exp_mean, "func": exp_extrapolation1, "x0": 0, "param": 3,
         "p0": [0.5, 0.5, 0.5, 0.5, 0.5], "bounds": ([-np.inf, 0, -np.inf, 0, -np.inf], [np.inf, 1, np.inf, 1, np.inf])},
        # {"name": "exp", "x": noise_scale_factors, "y": exp_mean, "func": exp_extrapolation, "x0": 0, "param": 3,
        #  "p0": [0.5, 0.5, 0.5], "bounds": ([-np.inf, 0, -np.inf], [np.inf, 1, np.inf])},
        {"name": "fidelity", "x": factor, "y": exp_mean, "func": fid_extrapolation, "x0": 0, "param": 2},
        {"name": "purity_sqrt", "x": purity_mean, "y": exp_mean, "func": purity_sqrt_extrapolation,
         "x0": 1, "param": 3, "p0": [0.5, min(purity_mean) / 2, 0.5],
         "bounds": ([-np.inf, 0, -np.inf], [np.inf, min(purity_mean), np.inf])},
    ]
    best_pred, best_name, details = zne_loocv(models, q_num=n_qubits, mod=mod, delate=delate)
    # Extract extrapolated values
    poly_exp_val = next(d['pred'] for d in details if d['name'] == "poly_exp")
    exp_val = next(d['pred'] for d in details if d['name'] == "exp")
    fid_val = next(d['pred'] for d in details if d['name'] == "fidelity")
    purity_sqrt_val = next(d['pred'] for d in details if d['name'] == "purity_sqrt")
    return {
        "poly_exp": poly_exp_val,
        "exp": exp_val,
        "fidelity": fid_val,
        "purity_sqrt": purity_sqrt_val,
        "loocv": best_pred
    }

def write_extrap_results(zne_file, all_extrap_rows, prev_noisy, prev_exact, observable):
    """
    Write extrapolation results to a file and print to the console.
    all_extrap_rows: list of (circuit_idx, exp_idx, poly_exp, exp, fidelity, purity_sqrt, loocv)
    """
    with open(zne_file, 'w') as f:
        f.write(f"Observable: {observable}\n\n")
        f.write("Circuit\tPoly_exp ZNE\tExp ZNE\tFidelity ZNE\tPurity_sqrt ZNE\tLoocv ZNE\tPrevious Noisy\tPrevious Exact\n")
        # Print header
        print("Circuit\tPoly_exp ZNE\tExp ZNE\tFidelity ZNE\tPurity_sqrt ZNE\tLoocv ZNE\tPrevious Noisy\tPrevious Exact")

        for idx, row in enumerate(all_extrap_rows):
            circuit_idx, exp_idx, poly_exp, exp_val, fid, purity_sqrt, loocv = row
            noisy_val = prev_noisy[circuit_idx][idx] if prev_noisy and circuit_idx < len(prev_noisy) else "N/A"
            exact_val = prev_exact[circuit_idx] if prev_exact and circuit_idx < len(prev_exact) else "N/A"
            line = f"{exp_idx}\t{poly_exp:.6f}\t{exp_val:.6f}\t{fid:.6f}\t{purity_sqrt:.6f}\t{loocv:.6f}\t{noisy_val:.6f}\t{exact_val}"
            f.write(line + "\n")
            print(line)

if __name__ == '__main__':
    start_time = time.time()
    simulator = Aer.get_backend('qasm_simulator')
    statevector_sim = Aer.get_backend('statevector_simulator')
    noise_model = create_fixed_noise_model()

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Verdana']
    plt.rcParams['axes.unicode_minus'] = False

    with open("config.json") as f:
        config = json.load(f)

    mod = config["num_train"]
    circ = config["num_non-integer"]
    T_default = config["num_twirling"]
    delate = config["fit_method"]
    n_qubits = 5
    fold_method = config["fold_method"]
    noise_scale_factors = config["noise_scale_factors"]
    cir_type = "grover"
    record = config["record"]
    sample_method = True
    readout = True
    row1 = config["row"]
    observable = Pauli(config["observable"])
    # entrys = [3]
    entrys = list(range(1, 6))
    # New: read number of experiment repetitions
    num_experiments = config.get("num_experiments", 1)

    if readout == True and sample_method == True and record == True:
        M = build_readout_mitigation_matrix(simulator, noise_model, n_qubits, shots=1000000)
        M_inv = np.linalg.pinv(M)
    elif readout == False and sample_method == True:
        M_inv = np.ones((2 ** n_qubits, 2 ** n_qubits))

    if cir_type == "grover":
        n = 4
        m = 1
        total_qubits = n + m
        row1 = 10
        proj_search = np.zeros((2 ** n, 2 ** n), dtype=complex)
        proj_search[0, 0] = 1.0
        I_anc = np.eye(2 ** m, dtype=complex)
        Pi_full = np.kron(I_anc, proj_search)
        observable = Operator(Pi_full)

    for entry in entrys:
        if cir_type == 'random':
            circuit_dir = os.path.join("random_circuits", f"random_circuits_{entry}")
        elif cir_type == 'ising_trotter':
            circuit_dir = os.path.join("ising_trotter_circuits", f"ising_trotter_circuits_{entry}")
        elif cir_type == 'grover':
            circuit_dir = os.path.join("grover_circuits", f"grover_circuits_{entry}")

        observable_op = Operator(observable)
        results_dir = os.path.join(circuit_dir, "zne_separated_results")
        os.makedirs(results_dir, exist_ok=True)
        prev_exact, prev_noisy = load_previous_results()
        print(prev_exact, prev_noisy)
        circuit_files = sorted(
            [f for f in os.listdir(circuit_dir) if f.endswith('.qasm')],
            key=natural_sort_key
        )

        if not circuit_files:
            print(f"Error: No .qasm files found in '{circuit_dir}'!")
            exit(1)

        print(f"Found {len(circuit_files)} quantum circuit files")
        zne_results = {i: [] for i in range(len(circuit_files))}

        poly_exp_expectations = []
        exp_expectations = []
        fidelities_expectations = []
        purity_sqrt_expectations = []
        loocv_expectations = []
        extrapolation_failures = []
        factor = []

        lit = list(range(len(circuit_files)))
        rand = random.Random(123)
        samples = rand.sample(lit, min(3, len(circuit_files)))
        modify_list_1 = []

        L = len(noise_scale_factors)

        if record == True:
            all_data_rows = []  # Store all experiment rows
            all_extrap_rows = []
            prev_noisies_shot = []
            for i, filename in enumerate(circuit_files):
                prev_noisy_shot = []
                # Inside circuit loop under record == True branch
                filepath = os.path.join(circuit_dir, filename)
                qc = QuantumCircuit.from_qasm_file(filepath)
                qc.remove_final_measurements()
                print(f"Circuit #{i + 1}: depth={qc.depth()}, gates={len(qc.data)}")
                fit_qc = qc.copy()

                fixed_seed = 42
                for exp_idx in range(num_experiments):
                    exp_mean, purity_mean, fidelity_mean = run_pauli_twirling_zne_shot(
                                                                                        fit_qc, observable_op,
                                                                                        noise_model=noise_model,
                                                                                        noise_scale_factors=noise_scale_factors,
                                                                                        T=T_default,
                                                                                        circ=circ,
                                                                                        shots=1000000,  # can be adjusted from config
                                                                                        M_inv=M_inv,
                                                                                        seed=fixed_seed
                                                                                    )
                    prev_noisy_shot.append(exp_mean[0])
                    factor = calculate_epsilon(fidelity_mean, q_num=n_qubits)
                    # Extrapolate
                    extrap = extrapolate_experiment(exp_mean, purity_mean, fidelity_mean, factor,
                                                    noise_scale_factors, n_qubits, mod, delate)
                    # Save raw data
                    all_data_rows.append(np.concatenate([exp_mean, purity_mean, fidelity_mean, factor]))
                    all_extrap_rows.append((i, exp_idx, extrap["poly_exp"], extrap["exp"],
                                            extrap["fidelity"], extrap["purity_sqrt"], extrap["loocv"]))
                # Save detailed file
                prev_noisies_shot.append(prev_noisy_shot)
                data_to_save = np.vstack(all_data_rows)
                suffix = "_multi" if num_experiments > 1 else ""
                if readout and cir_type == "grover" and sample_method:
                    save_path = os.path.join(results_dir, f"zne_separated_results_detailed_readout{suffix}.txt")
                elif not readout and cir_type == "grover" and sample_method:
                    save_path = os.path.join(results_dir, f"zne_separated_results_detailed_non_readout{suffix}.txt")
                else:
                    save_path = os.path.join(results_dir, f"zne_separated_results_detailed{suffix}.txt")
                header = '\t'.join([f'exp_{i}' for i in range(L)] +
                                   [f'purity_{i}' for i in range(L)] +
                                   [f'fidelity_{i}' for i in range(L)] +
                                   [f'factor_{i}' for i in range(L)])
                np.savetxt(save_path, data_to_save, delimiter='\t', header=header, comments='')
                print(f"Saved {data_to_save.shape[0]} experiment rows to {save_path}")
                # Save extrapolation results file (one line per experiment)
                if readout and cir_type == "grover":
                    zne_file = os.path.join(results_dir, "zne_separated_results_readout.txt")
                elif not readout and cir_type == "grover":
                    zne_file = os.path.join(results_dir, "zne_separated_results_non_readout.txt")
                else:
                    zne_file = os.path.join(results_dir, "zne_separated_results.txt")
                write_extrap_results(zne_file, all_extrap_rows, prev_noisies_shot, prev_exact, observable)
        else:  # record == False
            file_path = os.path.join(results_dir, "zne_separated_results_detailed_readout_multi.txt")
            data = np.loadtxt(file_path, delimiter='\t', skiprows=1)
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            n_rows, n_cols = data.shape
            L = len(noise_scale_factors)
            n_circuits = len(circuit_files)
            n_exp_per_circuit = n_rows // n_circuits
            all_extrap_rows = []
            prev_noisies_shot = []
            for i in range(n_circuits):
                prev_noisy_shot = []
                start = i * n_exp_per_circuit
                end = (i + 1) * n_exp_per_circuit
                circuit_data = data[start:end, :]
                for exp_idx in range(n_exp_per_circuit):
                    row = circuit_data[exp_idx, :]
                    exp_mean = row[:L]
                    purity_mean = row[L:2 * L]
                    fidelity_mean = row[2 * L:3 * L]
                    factor = row[3 * L:]
                    prev_noisy_shot.append(exp_mean[0])
                    # Extrapolate
                    extrap = extrapolate_experiment(exp_mean, purity_mean, fidelity_mean, factor,
                                                    noise_scale_factors, n_qubits, mod, delate)
                    all_extrap_rows.append((i, exp_idx, extrap["poly_exp"], extrap["exp"],
                                            extrap["fidelity"], extrap["purity_sqrt"], extrap["loocv"]))
            prev_noisies_shot.append(prev_noisy_shot)
            # Write extrapolation file
            if readout and cir_type == "grover":
                zne_file = os.path.join(results_dir, "zne_separated_results_readout.txt")
            elif not readout and cir_type == "grover":
                zne_file = os.path.join(results_dir, "zne_separated_results_non_readout.txt")
            else:
                zne_file = os.path.join(results_dir, "zne_separated_results.txt")
            write_extrap_results(zne_file, all_extrap_rows, prev_noisies_shot, prev_exact, observable)

    end_time = time.time()
    print(f"Execution time: {round(end_time - start_time, 2)} seconds")