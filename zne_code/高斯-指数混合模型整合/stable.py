"""Stability of the Grover extrapolation against the initial fit guess.

Repeats the single-model (``poly_exp_extrapolation3``) extrapolation of one
Grover circuit ``stable_num = 50`` times, each time with a different random
initial parameter vector drawn from ``random.Random(42)``, and histograms the 50
extrapolated values.

The draw order (init_c1, init_c2, init_a0, init_a1, init_b per iteration) and the
seed 42 are part of the reproducibility contract -- do not reorder.

Output: ``grover_circuits_3/stable_results/stability_histogram_entry_3.png``
"""

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

def extrapolate_experiment(exp_mean, purity_mean, fidelity_mean, factor,
                           noise_scale_factors, n_qubits, mod, delate, p0):
    """
    Perform extrapolation for one experiment, return a dictionary of extrapolated values for each method.
    """
    models = [
        {"name": "poly_exp", "x": noise_scale_factors, "y": exp_mean, "func": poly_exp_extrapolation3,
         "x0": 0, "param": 5, "p0": p0,
         "bounds": ([-np.inf, 0, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, 0, np.inf, np.inf])}
    ]
    best_pred, best_name, details = zne_loocv(models, q_num=n_qubits, mod=mod, delate=delate)
    # Extract extrapolated values
    poly_exp_val = next(d['pred'] for d in details if d['name'] == "poly_exp")
    return {
        "poly_exp": poly_exp_val,
    }


if __name__ == '__main__':
    start_time = time.time()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Verdana']
    plt.rcParams['axes.unicode_minus'] = False

    with open("config.json") as f:
        config = json.load(f)

    cir_type = "grover"
    noise_scale_factors = config["noise_scale_factors"]
    n_qubits = config["num_qubits"] if cir_type != "grover" else 5
    mod = config["num_train"]
    delate = config["fit_method"]

    entrys = [3]
    for entry in entrys:
        if cir_type == 'random':
            circuit_dir = os.path.join("random_circuits", f"random_circuits_{entry}")
        elif cir_type == 'ising_trotter':
            circuit_dir = os.path.join("ising_trotter_circuits", f"ising_trotter_circuits_{entry}")
        elif cir_type == 'grover':
            circuit_dir = os.path.join("grover_circuits", f"grover_circuits_{entry}")

        results_dir = os.path.join(circuit_dir, "zne_separated_results")
        os.makedirs(results_dir, exist_ok=True)

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
        seed = 42
        rand = random.Random(seed)

        L = len(noise_scale_factors)  # record == False
        file_path = os.path.join(results_dir, "zne_separated_results_detailed.txt")
        data = np.loadtxt(file_path, delimiter='\t', skiprows=1)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        n_rows, n_cols = data.shape
        L = len(noise_scale_factors)
        n_circuits = len(circuit_files)
        stable_num = 50
        n_exp_per_circuit = n_rows // n_circuits
        all_extrap_rows = []
        start = 0
        end =  n_exp_per_circuit
        circuit_data = data[start:end, :]
        row = circuit_data[0, :]
        exp_mean = row[:L]
        purity_mean = row[L:2 * L]
        fidelity_mean = row[2 * L:3 * L]
        factor = row[3 * L:]
        for i in range(stable_num):
            init_c1 = rand.uniform(-1, 0)
            init_c2 = rand.uniform(0, 0.1)
            init_a0 = rand.uniform(-1, 1)
            init_a1 = rand.uniform(-1, 1)
            init_b = rand.uniform(-1, 1)
            p0 = [init_a0, init_c2, init_c1, init_a1, init_b]
            print(p0)
            # Extrapolate
            extrap = extrapolate_experiment(exp_mean, purity_mean, fidelity_mean, factor,
                                            noise_scale_factors, n_qubits, mod, delate, p0)
            print(extrap["poly_exp"])
            all_extrap_rows.append((i, extrap["poly_exp"]))
        print(all_extrap_rows)
        # 绘制初始值稳定性图
        # ========== Stability Visualization (Boxplot + Statistics) ==========
        exact = 0.961319
        values = [v for _, v in all_extrap_rows]
        mean_val = np.mean(values)
        std_val = np.std(values)
        median_val = np.median(values)
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1

        # Create figure: left boxplot, right bar with error bar
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # ========== Histogram with legend including CV ==========
        exact_val = 0.961319
        cv_val = std_val / abs(mean_val) * 100  # 相对方差百分比

        fig2, ax = plt.subplots(figsize=(6, 4))
        ax.hist(values, bins=20, density=False, alpha=0.6, color='lightgreen', edgecolor='black')
        ax.axvline(mean_val, color='blue', linestyle='--', label=f'Mean: {mean_val:.5f}')
        ax.axvline(median_val, color='orange', linestyle='-.', label=f'Median: {median_val:.5f}')
        ax.axvline(exact_val, color='black', linestyle='-', linewidth=2, label=f'Exact: {exact_val}')

        # 添加一个空的代理曲线来显示 CV（不可见）
        from matplotlib.lines import Line2D

        handles, labels = ax.get_legend_handles_labels()
        cv_handle = Line2D([0], [0], marker='', color='none', linestyle='none', label=f'CV: {cv_val:.2f}%')
        handles.append(cv_handle)
        labels.append(cv_handle.get_label())

        ax.set_xlabel('Extrapolated Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram of Extrapolation Values')

        # 图例：缩小字体，减小边框，半透明
        ax.legend(handles, labels, fontsize=6, framealpha=0.8, edgecolor='gray',
                  handlelength=1.5, loc='upper right', bbox_to_anchor=(0.98, 0.98))

        ax.grid(alpha=0.3)
        save_dir = os.path.join(circuit_dir, "stable_results")
        os.makedirs(save_dir, exist_ok=True)
        hist_path = os.path.join(save_dir, f"stability_histogram_entry_{entry}.png")
        plt.savefig(hist_path, dpi=150)
        plt.show()


    end_time = time.time()
    print(f"Execution time: {round(end_time - start_time, 2)} seconds")