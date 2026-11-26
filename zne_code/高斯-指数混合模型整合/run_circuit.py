"""Render every exported circuit to a matplotlib diagram.

Note: only the *first* circuit of the directory is drawn (``if i == 0`` in the
original loop) -- the remaining circuits are only described on stdout.  The
per-circuit statistics printed here are informational output.
"""

import os
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
import re
import json
from qiskit.transpiler.passes import RemoveBarriers

def natural_sort_key(filename):
    # Match numeric part in the filename
    numbers = re.findall(r'\d+', filename)
    # If found, convert to integer; otherwise return original string
    return int(numbers[0]) if numbers else filename

with open("config.json") as f:
    config = json.load(f)

# Set directory path
cir_type = config["cir_type"]         # Circuit type
# cir_type = 'ising_trotter'  # Ising evolution circuit

if cir_type == 'random':
    circuit_dir = os.path.join("random_circuits", f"random_circuits_{2}")
elif cir_type == 'ising_trotter':
    circuit_dir = os.path.join("ising_trotter_circuits", f"ising_trotter_circuits_{2}")
elif cir_type == 'grover':
    circuit_dir = os.path.join("grover_circuits", f"grover_circuits_{1}")

picture_dir = os.path.join(circuit_dir, "pictures")  # Image save directory

# Ensure directory exists
if not os.path.exists(circuit_dir):
    print(f"Error: Directory '{circuit_dir}' does not exist!")
    print("Please run the circuit generation script first to create quantum circuit files.")
    exit(1)

# Create image save directory
os.makedirs(picture_dir, exist_ok=True)

# Get all circuit files
circuit_files = sorted(
    [f for f in os.listdir(circuit_dir) if f.endswith('.qasm')],
    key=natural_sort_key
)

if not circuit_files:
    print(f"Error: No .qasm files found in directory '{circuit_dir}'!")
    exit(1)

print(f"Found {len(circuit_files)} quantum circuit files")
print(f"Images will be saved to: {picture_dir}\n")

# Create separate image for each circuit
for i, filename in enumerate(circuit_files):
    # Load quantum circuit
    try:
        filepath = os.path.join(circuit_dir, filename)
        qc = QuantumCircuit.from_qasm_file(filepath)
        qc = RemoveBarriers()(qc)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue

    # Print basic circuit information
    print(f"Circuit #{i + 1} ({filename}):")
    print(f"- Number of qubits: {qc.num_qubits}")
    print(f"- Total number of gates: {len(qc.data)}")

    # Count gate types
    rx_count = sum(1 for op in qc.data if op.operation.name == 'rx')
    ry_count = sum(1 for op in qc.data if op.operation.name == 'ry')
    cx_count = sum(1 for op in qc.data if op.operation.name == 'cx')

    print(f"- Number of RX gates: {rx_count}")
    print(f"- Number of RY gates: {ry_count}")
    print(f"- Number of CX gates: {cx_count}")
    print(f"- Total single-qubit gates: {rx_count + ry_count}, Total two-qubit gates: {cx_count}")

    # Create image file path
    img_filename = f"{os.path.splitext(filename)[0]}.png"
    img_path = os.path.join(picture_dir, img_filename)

    # Draw circuit diagram using mpl and save to pictures subdirectory
    if i == 0:
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111)
        qc.draw('mpl', ax=ax, style='clifford', scale=0.8)
        # plt.title(f"ising_trotter", fontsize=14)
        plt.savefig(img_path, bbox_inches='tight')
        plt.close(fig)

    print(f"- Circuit diagram saved as: {img_path}")

    # Print text representation of the circuit in the terminal
    print(f"\nText representation of circuit #{i + 1}:")
    print(qc)
    print("\n" + "=" * 80 + "\n")

print("All circuits processed!")
print(f"Circuit diagrams saved to {picture_dir} directory")