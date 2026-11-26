"""
Grover search circuit generator (n=4 data qubits + 1 ancilla qubit)
Goal: search for |00000⟩, compute the probability of all data qubits being |0⟩ (ancilla arbitrary)
Save circuits as pickle files for subsequent noise simulation.
"""

import os
import math
import pickle
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import DensityMatrix

# ---------------------------- Helper Functions ----------------------------
def oracle_phase_on_zero(qc, data_qubits):
    """
    Apply phase flip (-1) to the all-zero state of data qubits.
    data_qubits: number of data qubits (first data_qubits qubits)
    """
    qc.x(range(data_qubits))                 # Flip |0...0> to |1...1>
    qc.mcx(list(range(data_qubits)), data_qubits)   # Ancilla qubit index = data_qubits
    qc.x(range(data_qubits))                 # Restore

def diffusion_operator(qc, data_qubits):
    """Diffusion operator D = H⊗n (2|0⟩⟨0| - I) H⊗n, acts on the first data_qubits qubits."""
    qc.h(range(data_qubits))
    oracle_phase_on_zero(qc, data_qubits)    # Phase flip using ancilla
    qc.h(range(data_qubits))

def build_grover_circuit(n_data, k_iter):
    """
    Build Grover circuit.
    n_data: number of data qubits (excluding ancilla)
    k_iter: number of Grover iterations
    Returns QuantumCircuit with n_data+1 qubits.
    """
    qc = QuantumCircuit(n_data + 1)          # +1 for ancilla
    # Initialize data qubits to uniform superposition
    qc.h(range(n_data))
    # Initialize ancilla to |-⟩ = (|0⟩ - |1⟩)/√2
    qc.x(n_data)
    qc.h(n_data)
    # Apply k Grover iterations
    for _ in range(k_iter):
        oracle_phase_on_zero(qc, n_data)     # Phase flip target state
        diffusion_operator(qc, n_data)       # Diffusion operator
    return qc

def compute_success_probability(qc, n_data):
    """
    Compute the probability that all data qubits are |0⟩ (ancilla arbitrary).
    Uses the same DensityMatrix + projector method as the original code.
    """
    # Build projector I_anc ⊗ |0...0⟩⟨0...0|
    dim_data = 2 ** n_data
    proj_data = np.zeros((dim_data, dim_data), dtype=complex)
    proj_data[0, 0] = 1.0
    I_anc = np.eye(2, dtype=complex)
    Pi_full = np.kron(I_anc, proj_data)

    rho = DensityMatrix.from_instruction(qc)
    prob = np.real(np.trace(rho.data @ Pi_full))
    return prob

def save_circuit(qc, k_iter, save_dir="./grover_gen/"):
    """Transpile and save the circuit as a pickle file."""
    os.makedirs(save_dir, exist_ok=True)
    basis_gates = ['rz', 'sx', 'x', 'cx', 'id']
    qc_transpiled = transpile(qc, basis_gates=basis_gates, optimization_level=1)
    file_path = os.path.join(save_dir, f"step_{k_iter}.pk")
    # Keep exactly the same save format as the original code
    data_to_save = [{'circuit': qc_transpiled}]
    with open(file_path, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"Circuit saved: {file_path}")

# ---------------------------- Main Program ----------------------------
def main():
    n_data = 4
    N = 2 ** n_data
    k_max = math.floor((math.pi / 4) * math.sqrt(N))
    print(f"n = {n_data}, N = {N}, theoretical optimal k = {k_max}")

    ks = list(range(1, k_max + 3))   # 1,2,3,4,5
    print(f"Generating circuits for k = {ks}")

    for k in ks:
        qc = build_grover_circuit(n_data, k)
        prob = compute_success_probability(qc, n_data)
        print(f"k = {k:2d} | Success probability (data qubits all |0⟩) = {prob:.8f}")
        save_circuit(qc, k)

if __name__ == "__main__":
    main()