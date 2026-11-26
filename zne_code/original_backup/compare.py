import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------- Configuration & Paths ----------------------------
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

# ---------------------------- Data Loading ----------------------------
def load_noisy_results(file_path):
    """Load noisy results file (expectation_values.txt)."""
    if not os.path.exists(file_path):
        print(f"Error: Results file not found: {file_path}")
        return None

    exact_values = []
    noisy_values = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        start_reading = False
        for line in lines:
            if "Circuit" in line and "Exact" in line:
                start_reading = True
                continue
            if start_reading and line.strip():
                parts = line.split()
                try:
                    exact_values.append(float(parts[1]))
                    noisy_values.append(float(parts[2]))
                except Exception as e:
                    print(f"Warning: Error parsing line: {line.strip()} - {e}")
    return {"exact": exact_values, "noisy": noisy_values}

def load_zne_results(file_path):
    """Load ZNE results file (zne_separated_results.txt)."""
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
            if "Circuit" in line and "Poly_exp" in line:
                start_reading = True
                continue
            if start_reading and line.strip():
                parts = line.split()
                try:
                    poly_exp_zne.append(float(parts[1]))
                    exp_zne.append(float(parts[2]))
                    fidelity_zne.append(float(parts[3]))
                    purity_sqrt_zne.append(float(parts[4]))
                    loocv_zne.append(float(parts[5]))
                    noisy_values.append(float(parts[6]))
                    exact_values.append(float(parts[7]))
                except Exception as e:
                    print(f"Warning: Error parsing line: {line.strip()} - {e}")

    return {
        "exact": exact_values,
        "noisy": noisy_values,
        "poly_exp_zne": poly_exp_zne,
        "exp_zne": exp_zne,
        "fidelity_zne": fidelity_zne,
        "purity_sqrt_zne": purity_sqrt_zne,
        "loocv_zne": loocv_zne
    }

# ---------------------------- Error Calculation ----------------------------
def calculate_absolute_errors(results):
    """Calculate absolute errors."""
    errors = {
        "noisy_abs": [],
        "poly_exp_zne_abs": [],
        "exp_zne_abs": [],
        "fidelity_zne_abs": [],
        "purity_sqrt_zne_abs": [],
        "loocv_zne_abs": []
    }
    n = len(results["exact"])
    for i in range(n):
        exact = results["exact"][i]
        errors["noisy_abs"].append(abs(results["noisy"][i] - exact))
        errors["poly_exp_zne_abs"].append(abs(results["poly_exp_zne"][i] - exact))
        errors["exp_zne_abs"].append(abs(results["exp_zne"][i] - exact))
        errors["fidelity_zne_abs"].append(abs(results["fidelity_zne"][i] - exact))
        errors["purity_sqrt_zne_abs"].append(abs(results["purity_sqrt_zne"][i] - exact))
        errors["loocv_zne_abs"].append(abs(results["loocv_zne"][i] - exact))
    return errors

def calculate_reduction_rates(errors):
    """Calculate error reduction rates of ZNE methods relative to raw noisy errors."""
    reduction_rates = {
        "poly_exp_reduction": [],
        "exp_reduction": [],
        "fidelity_reduction": [],
        "purity_sqrt_reduction": [],
        "loocv_reduction": []
    }
    n = len(errors["noisy_abs"])
    for i in range(n):
        noisy_err = errors["noisy_abs"][i]
        if noisy_err > 1e-6:
            reduction_rates["poly_exp_reduction"].append((noisy_err - errors["poly_exp_zne_abs"][i]) / noisy_err)
            reduction_rates["exp_reduction"].append((noisy_err - errors["exp_zne_abs"][i]) / noisy_err)
            reduction_rates["fidelity_reduction"].append((noisy_err - errors["fidelity_zne_abs"][i]) / noisy_err)
            reduction_rates["purity_sqrt_reduction"].append((noisy_err - errors["purity_sqrt_zne_abs"][i]) / noisy_err)
            reduction_rates["loocv_reduction"].append((noisy_err - errors["loocv_zne_abs"][i]) / noisy_err)
        else:
            reduction_rates["poly_exp_reduction"].append(0.0)
            reduction_rates["exp_reduction"].append(0.0)
            reduction_rates["fidelity_reduction"].append(0.0)
            reduction_rates["purity_sqrt_reduction"].append(0.0)
            reduction_rates["loocv_reduction"].append(0.0)
    return reduction_rates

# ---------------------------- Output and Visualization ----------------------------
def print_average_errors(errors):
    """Print average absolute errors."""
    avg_noisy = np.mean(errors["noisy_abs"])
    avg_poly = np.mean(errors["poly_exp_zne_abs"])
    avg_exp = np.mean(errors["exp_zne_abs"])
    avg_fid = np.mean(errors["fidelity_zne_abs"])
    avg_pur = np.mean(errors["purity_sqrt_zne_abs"])
    avg_loocv = np.mean(errors["loocv_zne_abs"])

    print("\nAverage Error Rates:")
    print(f"- Noisy: {avg_noisy:.6f}")
    print(f"- Gaussian_exp ZNE: {avg_poly:.6f}")
    print(f"- Exp ZNE: {avg_exp:.6f}")
    print(f"- IC-ZNE: {avg_fid:.6f}")
    print(f"- pZNE: {avg_pur:.6f}")
    print(f"- Loocv ZNE: {avg_loocv:.6f}")

    return avg_noisy, avg_poly, avg_exp, avg_fid, avg_pur, avg_loocv

def print_average_reductions(reduction_rates):
    """Print average error reduction rates."""
    avg_poly = np.mean(reduction_rates["poly_exp_reduction"])
    avg_exp = np.mean(reduction_rates["exp_reduction"])
    avg_fid = np.mean(reduction_rates["fidelity_reduction"])
    avg_pur = np.mean(reduction_rates["purity_sqrt_reduction"])
    avg_loocv = np.mean(reduction_rates["loocv_reduction"])

    print("\nAverage Error Reduction Rates:")
    print(f"- Poly_exp ZNE: {avg_poly:.2%}")
    print(f"- Exp ZNE: {avg_exp:.2%}")
    print(f"- Fidelity ZNE: {avg_fid:.2%}")
    print(f"- Purity_sqrt ZNE: {avg_pur:.2%}")
    print(f"- Loocv ZNE: {avg_loocv:.2%}")

def save_detailed_analysis(errors, avg, results_dir, circuit_files):
    """Save per-circuit detailed error analysis to a text file."""
    detailed_file = os.path.join(results_dir, "detailed_error_analysis.txt")
    with open(detailed_file, 'w') as f:
        f.write("Detailed Error Analysis for Quantum Circuits\n")
        f.write("=" * 70 + "\n")
        f.write(f"Average Noisy Error: {avg[0]:.6f}\n")
        f.write(f"Average Gaussian_exp ZNE Error: {avg[1]:.6f}\n")
        f.write(f"Average Exp ZNE Error: {avg[2]:.6f}\n")
        f.write(f"Average IC-ZNE Error: {avg[3]:.6f}\n")
        f.write(f"Average pZNE Error: {avg[4]:.6f}\n")
        f.write(f"Average Loocv ZNE Error: {avg[5]:.6f}\n")

        f.write(f"Number of circuits: {len(circuit_files)}\n\n")

        f.write("Per-Circuit Absolute Errors:\n")
        f.write("Circuit\tNoisy\tPoly_exp\tExp\tFidelity\tPurity_sqrt\tLoocv\n")
        for i in range(len(circuit_files)):
            f.write(f"{i+1}\t{errors['noisy_abs'][i]:.6f}\t"
                    f"{errors['poly_exp_zne_abs'][i]:.6f}\t"
                    f"{errors['exp_zne_abs'][i]:.6f}\t"
                    f"{errors['fidelity_zne_abs'][i]:.6f}\t"
                    f"{errors['purity_sqrt_zne_abs'][i]:.6f}\t"
                    f"{errors['loocv_zne_abs'][i]:.6f}\n")
    print(f"\nDetailed error analysis saved to: {detailed_file}")

def plot_error_comparison(avg_errors, methods, colors, num_circuits):
    """Plot bar chart of average errors."""
    plt.figure(figsize=(20, 16))

    # Subplot 1: Bar chart of average errors
    plt.subplot(3, 1, 1)
    bars = plt.bar(methods, avg_errors, color=colors)
    plt.ylabel('Average Absolute Error')
    plt.title(f'Average Error Rates Comparison ({num_circuits} Circuits)')
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    # Subplot 2: Placeholder for per-circuit error curve (original structure preserved)
    plt.subplot(3, 1, 2)
    plt.text(0.5, 0.5, "Per-circuit error plot (optional)", ha='center', va='center', transform=plt.gca().transAxes)
    plt.axis('off')

    # Subplot 3: Placeholder for additional analysis (original structure preserved)
    plt.subplot(3, 1, 3)
    plt.text(0.5, 0.5, "Additional analysis (placeholder)", ha='center', va='center', transform=plt.gca().transAxes)
    plt.axis('off')

    plt.tight_layout()
    #plt.show()

# ---------------------------- Main Program ----------------------------
def main():
    # Configuration
    config = load_config()
    cir_type = config["cir_type"]
    entries = list(range(1, 20))  # Process circuit sets 1 to 5

    for entry in entries:
        print(f"\nProcessing entry: {entry}")
        circuit_dir = build_circuit_dir(cir_type, entry)
        results_dir = os.path.join(circuit_dir, "error_analysis_results")
        os.makedirs(results_dir, exist_ok=True)

        # Get list of circuit files (to determine number of circuits)
        circuit_files = sorted([f for f in os.listdir(circuit_dir) if f.endswith('.qasm')])
        num_circuits = len(circuit_files)

        # Define result file paths
        noisy_file = os.path.join(circuit_dir, "results", "expectation_values.txt")
        zne_file = os.path.join(circuit_dir, "zne_separated_results", "zne_separated_results.txt")

        # Load data
        noisy_results = load_noisy_results(noisy_file)
        zne_results = load_zne_results(zne_file)

        if noisy_results is None or zne_results is None:
            print("Error: Unable to load required results data. Skipping this entry.")
            continue

        # Calculate errors
        errors = calculate_absolute_errors(zne_results)

        # Print average errors
        avg_noisy, avg_poly, avg_exp, avg_fid, avg_pur, avg_loocv = print_average_errors(errors)
        avg = [avg_noisy, avg_poly, avg_exp, avg_fid, avg_pur, avg_loocv]
        # Calculate and print error reduction rates
        reduction_rates = calculate_reduction_rates(errors)
        print_average_reductions(reduction_rates)

        # Save detailed analysis
        save_detailed_analysis(errors, avg, results_dir, circuit_files)

        # Visualization
        methods = ['Noisy', 'Poly_exp ZNE', 'Exp ZNE', 'Fidelity', 'Purity_sqrt', 'LOOCV']
        avg_errors = [avg_noisy, avg_poly, avg_exp, avg_fid, avg_pur, avg_loocv]
        colors = ['salmon', 'limegreen', 'dodgerblue', 'yellow', 'purple', 'orange']
        plot_error_comparison(avg_errors, methods, colors, num_circuits)

if __name__ == "__main__":
    main()