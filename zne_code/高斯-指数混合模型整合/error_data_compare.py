"""Aggregate per-step error curves into the paper's error-vs-step figure.

Walks ``<family>_circuits/`` , reads every
``<family>_circuits_<n>/error_analysis_results/*.txt`` written by ``compare.py``
and plots the average absolute error of each ZNE method against the step index
(Trotter step for Ising, 2-qubit depth for random circuits, circuit index for Grover).

Output: ``<family>_circuits/compare_plot/average_error_vs_step_<cir_type>.png``
"""

import os
import re
import json
import matplotlib.pyplot as plt

# ---------------------------- Configuration Loading ----------------------------
def load_config(config_path="config.json"):
    """Load configuration file."""
    with open(config_path, "r") as f:
        return json.load(f)

# ---------------------------- Directory Setup ----------------------------
def setup_directories(cir_type):
    """Create and return base directory and image save directory based on circuit type."""
    base_dir = os.getcwd()
    if cir_type == 'random':
        base_dir = os.path.join(os.getcwd(), "random_circuits")
    elif cir_type == 'ising_trotter':
        base_dir = os.path.join(os.getcwd(), "ising_trotter_circuits")
    elif cir_type == 'grover':
        base_dir = os.path.join(os.getcwd(), "grover_circuits")
    else:
        raise ValueError(f"Unknown circuit type: {cir_type}")

    os.makedirs(base_dir, exist_ok=True)
    save_dir = os.path.join(base_dir, "compare_plot")
    os.makedirs(save_dir, exist_ok=True)
    return base_dir, save_dir

# ---------------------------- Data Extraction ----------------------------
def extract_error_data(base_dir, cir_type):
    """Traverse all circuit folders, extract average errors, return steps and errors dictionary."""
    methods = [
        "Noisy Error",
        "Gaussian_exp ZNE Error",
        "Exp ZNE Error",
        "IC-ZNE Error",
        "pZNE Error",
        "Loocv ZNE Error"
    ]
    errors = {m: [] for m in methods}
    steps = []
    pattern = re.compile(r"Average (.+?): ([0-9.]+)")

    # Traverse all subfolders under the base directory
    for folder in sorted(os.listdir(base_dir)):
        # Only process folders that match the naming convention
        if not (folder.startswith("random_circuits_") or
                folder.startswith("ising_trotter_circuits_") or
                folder.startswith("grover_circuits_")):
            continue

        # Extract step number
        try:
            if cir_type == 'grover':
                step = int(folder.split("_")[-1])
            else:
                step = int(folder.split("_")[-1]) - 1
        except ValueError:
            continue

        # Locate the error_analysis_results directory
        subdir = os.path.join(base_dir, folder, "error_analysis_results")
        if not os.path.isdir(subdir):
            continue

        # Find the first .txt file
        txt_files = [f for f in os.listdir(subdir) if f.endswith(".txt")]
        if not txt_files:
            continue

        file_path = os.path.join(subdir, txt_files[0])
        with open(file_path, "r") as f:
            text = f.read()

        # Extract average errors
        matches = pattern.findall(text)
        if not matches:
            continue

        steps.append(step)
        found = {k.strip(): float(v) for k, v in matches}
        for m in methods:
            errors[m].append(found.get(m, None))

    # Sort by step
    order = sorted(range(len(steps)), key=lambda i: steps[i])
    steps = [steps[i] for i in order]
    if cir_type == 'random':
        steps = [2 * s for s in steps]

    for m in methods:
        errors[m] = [errors[m][i] for i in order]

    return steps, errors, methods

# ---------------------------- Plotting ----------------------------
def plot_results(steps, errors, methods, cir_type, save_dir):
    """Plot line chart of average error vs step/depth and save."""
    plt.figure(figsize=(9, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#17becf']  # 蓝-橙-绿-紫-青
    # Plot only the first five methods (original logic)
    for i, m in enumerate(methods[:5]):
        # if m != 'Noisy Error' and m != 'Fid ZNE Error':
        plt.plot(steps, errors[m], marker='o',color = colors[i], label=m if m != 'Gaussian_exp ZNE Error' else 'G-E ZNE Error')

    plt.ylabel("Average Error", fontsize=12)

    if cir_type == 'ising_trotter':
        plt.xlabel("Trotter step (i)", fontsize=12)
        #plt.title("Average Error vs Step across Ising Circuits", fontsize=14)
    elif cir_type == 'random':
        plt.xlabel("2-qubits depth", fontsize=12)
        #plt.title("Average Error vs depth across Random Circuits", fontsize=14)
    elif cir_type == 'grover':
        plt.xlabel("Number of qubits", fontsize=12)
        #plt.title("Average Error vs Number of qubits across Grover Circuits", fontsize=14)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"average_error_vs_step_{cir_type}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    print(f"✅ Figure saved to: {save_path}")

# ---------------------------- Main Program ----------------------------
def main():
    config = load_config()
    cir_type = config["cir_type"]

    base_dir, save_dir = setup_directories(cir_type)
    steps, errors, methods = extract_error_data(base_dir, cir_type)

    if not steps:
        print("Warning: No valid data found, cannot plot.")
        return

    plot_results(steps, errors, methods, cir_type, save_dir)

if __name__ == "__main__":
    main()