import json

# Define parameter dictionary
params = {
    "num_train": 0,      # Number of data points for validation set
    "num_twirling": 16,  # Number of Pauli twirling repetitions
    "num_non-integer": 1, # Number of non-integer noise scaling repetitions
    "fit_method": 1,     # Whether to use only partial data points for fitting (1: yes, 0: no)
    "fold_method": 0,    # Gate folding method: 0 = uniform_gate_folding, 1 = circuit_folding
    "num_qubits": 4,     # Number of qubits in the circuit
    "row": 3,            # Number of lines to skip when reading zne_result file
    "num_experiments": 10,
    "noise_scale_factors": [1, 3, 5, 7, 9],  # Noise amplification factors
    "observable": "ZZZZ",   # Pauli observable
    "cir_type": "grover",  # Type of circuit to run: random, ising_trotter, or grover
    "readout_error": False,   # Whether to apply readout error mitigation
    "sample": False,         # Whether to obtain results via sampling or density matrix
    "record": False          # True: regenerate data; False: use saved data
}

# Write to JSON file
with open("config.json", "w") as f:
    json.dump(params, f, indent=4)  # indent parameter makes the content more readable