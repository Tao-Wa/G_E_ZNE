"""Generator for ``config.json`` -- the single source of truth for a run.

Every other script reads ``config.json``; none of them import this module.  The
module-level defaults below are written out only when this file is executed
directly, so importing it can no longer silently overwrite a hand-edited
``config.json``.

Behaviour-relevant switches
---------------------------
record
    ``False`` (default, and the value in the shipped ``config.json``) makes the
    ZNE scripts *load the raw simulator data from disk* instead of re-running the
    simulations.  ``True`` forces full recomputation.
sample
    ``False`` = density-matrix ZNE (``zne.py``).  ``True`` = shot-based ZNE for
    Grover circuits (``zne_grover_shots.py``).
readout_error
    ``True`` applies readout-error mitigation (needs a calibration matrix, which
    is only built when ``record`` is also ``True``).
cir_type
    ``random`` | ``ising_trotter`` | ``grover`` -- selects the circuit family and
    therefore which circuit directories, result files and figures are used.
"""

import json

# Default parameter dictionary.
params = {
    "num_train": 0,       # Number of data points for validation set
    "num_twirling": 16,   # Number of Pauli twirling repetitions
    "num_non-integer": 1,  # Number of non-integer noise scaling repetitions
    "fit_method": 1,      # Whether to use only partial data points for fitting (1: yes, 0: no)
    "fold_method": 0,     # Gate folding method: 0 = uniform_gate_folding, 1 = circuit_folding
    "num_qubits": 4,      # Number of qubits in the circuit
    "row": 3,             # Number of lines to skip when reading zne_result file
    "num_experiments": 10,
    "noise_scale_factors": [1, 3, 5, 7, 9],  # Noise amplification factors
    "observable": "ZZZZ",   # Pauli observable
    "cir_type": "grover",  # Type of circuit to run: random, ising_trotter, or grover
    "readout_error": False,   # Whether to apply readout error mitigation
    "sample": False,         # Whether to obtain results via sampling or density matrix
    "record": False          # True: regenerate data; False: use saved data
}


def write_config(path: str = "config.json") -> None:
    """Write ``params`` to ``path`` (indent=4 keeps the file diff-friendly)."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=4)


# Write to JSON file -- only when this script is run directly.
# Previously this executed on import, which meant that *any* accidental
# ``import config`` would reset a hand-edited config.json back to the defaults.
if __name__ == "__main__":
    write_config()
