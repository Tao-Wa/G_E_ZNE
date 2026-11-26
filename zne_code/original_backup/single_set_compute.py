import os
import json
import matplotlib.pyplot as plt

# ---------------------------- Global Configuration ----------------------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Verdana']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------- Data Loading ----------------------------
def load_results(file_path):
    """
    Load results file, supporting two formats:
    1. ZNE results file (contains Poly_exp ZNE, Exp ZNE, Fidelity ZNE, Purity_sqrt ZNE, Loocv ZNE, Previous Noisy, Previous Exact)
    2. Ordinary noisy results file (contains Exact, Noisy, Noise Ratio)
    """
    if not os.path.exists(file_path):
        print(f"Error: Results file not found: {file_path}")
        return None

    exact_values = []
    noisy_values = []
    poly_exp_zne = []
    exp_zne = []
    fidelity_zne = []
    purity_sqrt_zne = []
    loocv_zne = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        start_reading = False
        for line in lines:
            # Identify header line
            if "Circuit" in line and ("Poly_exp" in line or "Exact" in line):
                start_reading = True
                continue
            if not start_reading or not line.strip():
                continue

            parts = line.split()
            try:
                int(parts[0])  # Validate circuit index
                if "zne_separated_results" in file_path:
                    # ZNE file format
                    poly_exp_zne.append(float(parts[1]))
                    exp_zne.append(float(parts[2]))
                    fidelity_zne.append(float(parts[3]))
                    purity_sqrt_zne.append(float(parts[4]))
                    loocv_zne.append(float(parts[5]))
                    noisy_values.append(float(parts[6]))
                    exact_values.append(float(parts[7]))
                else:
                    # Ordinary noisy file format
                    exact_values.append(float(parts[1]))
                    noisy_values.append(float(parts[2]))
            except (ValueError, IndexError) as e:
                print(f"Warning: Error parsing line: {line.strip()} - {e}")

    if "zne_separated_results" in file_path:
        return {
            "exact": exact_values,
            "noisy": noisy_values,
            "poly_exp_zne": poly_exp_zne,
            "exp_zne": exp_zne,
            "fidelity_zne": fidelity_zne,
            "purity_sqrt_zne": purity_sqrt_zne,
            "loocv_zne": loocv_zne
        }
    else:
        return {"exact": exact_values, "noisy": noisy_values}

# ---------------------------- Error Calculation ----------------------------
def calculate_errors(results):
    """Calculate absolute errors for each method."""
    errors = {
        "Noisy": [],
        "Gaussian_exp ZNE": [],
        "Exp ZNE": [],
        "IC-ZNE": [],
        "pZNE": [],
        "LOOVC": []
    }
    n_circuits = len(results["exact"])
    for i in range(n_circuits):
        exact = results["exact"][i]
        errors["Noisy"].append(abs(results["noisy"][i] - exact))
        errors["Gaussian_exp ZNE"].append(abs(results["poly_exp_zne"][i] - exact))
        errors["Exp ZNE"].append(abs(results["exp_zne"][i] - exact))
        errors["IC-ZNE"].append(abs(results["fidelity_zne"][i] - exact))
        errors["pZNE"].append(abs(results["purity_sqrt_zne"][i] - exact))
        errors["LOOVC"].append(abs(results["loocv_zne"][i] - exact))
    return errors

# ---------------------------- Plotting ----------------------------
def plot_error_distribution(errors, methods, save_path):
    """
    Draw boxplot of absolute error distribution for each method.
    errors: dict, keys are method names, values are error lists
    methods: list, names of methods to plot (in order)
    save_path: str, path to save the figure
    """
    data = [errors[m] for m in methods]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#17becf']  # 蓝-橙-绿-紫-青
    output_methods = output_methods = ['Noisy', 'G-E ZNE', 'Dual_exp ZNE', 'IC-ZNE', 'pZNE']
    plt.figure(figsize=(9, 6))
    bp = plt.boxplot(
        data,
        patch_artist=True,
        flierprops=dict(marker='.', markersize=4, alpha=0.5, color='black'),
        medianprops=dict(color='black', linewidth=1)
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.xticks(range(1, len(methods) + 1), output_methods, rotation=0)
    plt.ylabel("Error distribution")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# ---------------------------- Paths and Directories ----------------------------
def get_base_dir(cir_type):
    """Get base directory based on circuit type."""
    base_dir = os.path.join(os.getcwd(), f"{cir_type}_circuits")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def get_circuit_dir(cir_type, entry):
    """Get circuit directory for a given entry."""
    return os.path.join(f"{cir_type}_circuits", f"{cir_type}_circuits_{entry}")

def get_result_file_paths(circuit_dir, cir_type, readout):
    """Get paths to noisy and ZNE result files."""
    noisy_file = os.path.join(circuit_dir, "results", "expectation_values.txt")
    zne_file = os.path.join(circuit_dir, "zne_separated_results", "zne_separated_results.txt")
    return noisy_file, zne_file

def get_save_path(save_dir, cir_type, readout):
    """Generate the figure save path."""
    return os.path.join(save_dir, f"Comparison of Error Distributions for ZNE Methods_{cir_type}.png")

# ---------------------------- Main Program ----------------------------
def main():
    # Load configuration
    with open("config.json") as f:
        config = json.load(f)

    cir_type = config["cir_type"]
    readout = config.get("readout_error", False)   # Ensure default value
    # entrys = list(range(1, 21))   # Commented out in original code, kept for flexibility
    if cir_type == "random":
        entrys = [19]   # Process only one entry
    elif cir_type == "ising_trotter":
        entrys = [16]
    elif cir_type == "grover":
        entrys = [3]

    # Base directory and save directory
    base_dir = get_base_dir(cir_type)
    save_dir = os.path.join(base_dir, f"{cir_type}_circuits_{entrys[0]}", "comparison_results")
    os.makedirs(save_dir, exist_ok=True)

    # Define methods to plot (only the first five, as in the original code using methods[:5])
    methods_to_plot = ['Noisy', 'Gaussian_exp ZNE', 'Exp ZNE', 'IC-ZNE', 'pZNE']

    for entry in entrys:
        print(f"Processing entry: {entry}")
        circuit_dir = get_circuit_dir(cir_type, entry)
        results_dir = os.path.join(circuit_dir, "error_analysis_results")
        os.makedirs(results_dir, exist_ok=True)

        # Get file paths
        noisy_file, zne_file = get_result_file_paths(circuit_dir, cir_type, readout)

        # Load data
        noisy_results = load_results(noisy_file)
        zne_results = load_results(zne_file)

        if not noisy_results or not zne_results:
            print("Error: Unable to load required results data.")
            return

        # Calculate errors
        errors = calculate_errors(zne_results)

        # Plot and save
        save_path = get_save_path(save_dir, cir_type, readout)
        plot_error_distribution(errors, methods_to_plot, save_path)
        print(f"Figure saved to: {save_path}")

if __name__ == '__main__':
    main()