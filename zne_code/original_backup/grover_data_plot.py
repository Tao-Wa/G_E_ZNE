import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------- Configuration ----------------------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Verdana']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------- Data Extraction (Line Chart) ----------------------------
def extract_zne_values(base_dir):
    """Extract one value from each grover_circuits_i/zne_separated_results.txt (line chart)."""
    method_names = [
        "G_E ZNE", "Dual_exp ZNE", "IC-ZNE",
        "pZNE", "Loocv ZNE", "Noisy", "Exact"
    ]
    data = {name: [] for name in method_names}
    steps = []
    pattern = re.compile(r"^\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")

    for folder in sorted(os.listdir(base_dir)):
        if not folder.startswith("grover_circuits_"):
            continue
        try:
            step = int(folder.split("_")[-1])
        except ValueError:
            continue
        txt_path = os.path.join(base_dir, folder, "zne_separated_results", "zne_separated_results.txt")
        if not os.path.isfile(txt_path):
            continue
        with open(txt_path, "r") as f:
            content = f.read()
        lines = content.splitlines()
        for line in lines:
            if line.strip().startswith("Circuit"):
                continue
            match = pattern.match(line.strip())
            if match:
                values = [float(g) for g in match.groups()]
                for idx, name in enumerate(method_names):
                    data[name].append(values[idx])
                steps.append(step)
                break
    # Sort by step
    order = sorted(range(len(steps)), key=lambda i: steps[i])
    steps = [steps[i] for i in order]
    for name in method_names:
        data[name] = [data[name][i] for i in order]
    return steps, data, method_names

def plot_line_chart(steps, data, method_names, save_path):
    """Plot line chart (excluding Loocv ZNE)."""
    plt.figure(figsize=(9, 6))
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#17becf', '#9467bd', '#808080', '#000000']
    markers = ['o', 's', '^', 'D', 'v', 'x', '*']
    for idx, name in enumerate(method_names):
        if steps and data[name] and name != "Loocv ZNE":
            plt.plot(steps, data[name],
                     marker=markers[idx % len(markers)],
                     color=colors[idx % len(colors)],
                     label=name, linewidth=2)
    plt.xlabel("Step (i)", fontsize=12)
    plt.ylabel("Expectation value", fontsize=12)
    plt.title("ZNE Methods Comparison on Grover Circuits (Line)", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    print(f"✅ Line chart saved to: {save_path}")

# ---------------------------- Data Extraction (Boxplot) ----------------------------
def extract_boxplot_data(file_path):
    """Extract multiple experiment data from zne_separated_results_readout.txt, using output_methods as labels."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Find header line
    header_line = None
    data_start = 0
    for idx, line in enumerate(lines):
        if line.strip().startswith("Circuit"):
            header_line = line.strip()
            data_start = idx + 1
            break
    if not header_line:
        raise ValueError("Header line not found")
    headers = header_line.split('\t')

    target_methods = ['Poly_exp ZNE', 'Exp ZNE', 'Fidelity ZNE', 'Purity_sqrt ZNE', 'Loocv ZNE']
    output_methods = ['G-E ZNE', 'Dual_exp ZNE', 'IC-ZNE', 'pZNE', 'Loocv ZNE']

    # Build mapping: target_name -> output_name
    method_mapping = dict(zip(target_methods, output_methods))

    # Record column index for each output_method
    col_index = {}  # key: output_method_name, value: column index
    for idx, name in enumerate(headers):
        if name in target_methods:
            output_name = method_mapping[name]
            col_index[output_name] = idx

    # Exact value column
    exact_col = None
    if 'Previous Exact' in headers:
        exact_col = headers.index('Previous Exact')
    else:
        exact_col = -1  # last column

    # Initialize data dictionary with output_methods as keys
    methods_data = {method: [] for method in output_methods}
    exact_values = []

    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if not parts[0].isdigit():
            continue
        # Extract values for each method
        for method, idx in col_index.items():
            try:
                val = float(parts[idx])
                methods_data[method].append(val)
            except:
                pass
        # Extract exact value
        try:
            if exact_col == -1:
                exact_val = float(parts[-1])
            else:
                exact_val = float(parts[exact_col])
            exact_values.append(exact_val)
        except:
            pass

    if not methods_data[output_methods[0]]:
        raise ValueError("No valid data found")
    exact_value = np.median(exact_values) if exact_values else None
    return methods_data, exact_value

def plot_box_chart(methods_data, exact_value, save_path):
    """Plot boxplot with a horizontal dashed line for the exact value."""
    method_names = list(methods_data.keys())[:4]
    data_to_plot = [methods_data[name] for name in method_names]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#17becf']  # 蓝-橙-绿-紫-青
    plt.figure(figsize=(9, 6))
    bp = plt.boxplot(data_to_plot, patch_artist=True,
                     flierprops=dict(marker='.', markersize=4, alpha=0.5, color='black'),
                     medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    plt.xticks(range(1, len(method_names)+1), method_names, rotation=0, ha='center')
    plt.ylabel("Expectation value", fontsize=12)
    plt.title("ZNE Methods Comparison on Grover Circuits (Boxplot)", fontsize=14)
    plt.axhline(y=exact_value, color='red', linestyle='--', linewidth=1.5,
                label=f'Exact = {exact_value:.4f}')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    print(f"✅ Boxplot saved to: {save_path}")

# ---------------------------- Main Program ----------------------------
def main():
    entry = 3
    base_dir = os.path.join(os.getcwd(), "grover_circuits")
    save_dir1 = os.path.join(base_dir, "compare_plot")
    save_dir2 = os.path.join(base_dir, f"grover_circuits_{entry}", "compare_plot")
    os.makedirs(save_dir1, exist_ok=True)

    # 1. Line chart (one circuit per step)
    steps, data, method_names = extract_zne_values(base_dir)
    if steps:
        line_save_path = os.path.join(save_dir1, "zne_methods_comparison_grover.png")
        plot_line_chart(steps, data, method_names, line_save_path)
    else:
        print("Warning: No data found for line chart (zne_separated_results.txt)")

    # 2. Boxplot (based on readout multiple-experiment file)
    os.makedirs(save_dir2, exist_ok=True)
    # The file is located at grover_circuits_3/zne_separated_results/zne_separated_results_readout.txt
    # If multiple exist, take the first found; here we fix grover_circuits_3
    readout_file = os.path.join(base_dir, f"grover_circuits_{entry}", "zne_separated_results", "zne_separated_results_readout.txt")
    if not os.path.exists(readout_file):
        # Try searching all grover_circuits_i folders for this file
        found = False
        for folder in sorted(os.listdir(base_dir)):
            if folder.startswith("grover_circuits_"):
                candidate = os.path.join(base_dir, folder, "zne_separated_results", "zne_separated_results_readout.txt")
                if os.path.exists(candidate):
                    readout_file = candidate
                    found = True
                    break
        if not found:
            print("Warning: No data found for boxplot (zne_separated_results_readout.txt)")
        else:
            print(f"Using boxplot data file: {readout_file}")
    if os.path.exists(readout_file):
        try:
            methods_data, exact_value = extract_boxplot_data(readout_file)
            box_save_path = os.path.join(save_dir2, "zne_methods_boxplot.png")
            plot_box_chart(methods_data, exact_value, box_save_path)
        except Exception as e:
            print(f"Error generating boxplot: {e}")
    else:
        print("Boxplot data file not found, skipping boxplot generation.")

if __name__ == "__main__":
    main()