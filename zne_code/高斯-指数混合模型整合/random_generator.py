"""Random-circuit generator (one entry per target two-qubit depth).

Synthesises random circuits and rejects/retries until the transpiled two-qubit
gate depth matches the target, then pickles a QASM collection per depth into
``random_gen/step_<2i>.pk``.

The module fixes every seed it can (Python, NumPy, PyTorch) at import time via
:func:`fix_random_seed`, which runs on import -- keep it that way, it is part of
the reproducibility contract.
"""

import json, pickle
import glob, os

import numpy as np

from qiskit.providers.fake_provider import FakeLima, FakeQuito, FakeGuadalupe
from qiskit import transpile
from qiskit import execute
from functools import partial
from qiskit.circuit.random import random_circuit
import random
import torch
import scipy
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit.exceptions import CircuitError
from tqdm.notebook import tqdm

def fix_random_seed(seed=0):
    """
    Fix all random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f'random seed fixed to {seed}')

def generate_data(be, cx_depths, num_qubits, save_path: str, n_entries_per_file: int):
    """
    Generate and save random circuits for each CX depth.
    Args:
        be: backend (used for transpilation, but not directly in this function)
        cx_depths: list of target two‑qubit gate depths after transpilation
        num_qubits: number of qubits in the circuit
        save_path: directory to save pickle files
        n_entries_per_file: number of circuits to generate per depth
    """
    for cx_depth in cx_depths:
        circuits_collection = {
            'version': '1.0',
            'description': 'Batch generated QASM circuit collection',
            'backend_used': str(be) if be else None,
            'circuit': {}
        }
        n = num_qubits
        entries = []

        for _ in tqdm(range(n_entries_per_file)):
            success = False
            while not success:
                try:
                    circuit = construct_random_circuit(cx_depth, be, n)
                    qasm_string = circuit.qasm()
                    success = True
                except (scipy.linalg.LinAlgError, TranspilerError, np.linalg.LinAlgError, CircuitError) as e:
                    print(f"Ran into an error:, {e}")
            circuit_name = f"circuit_{_}"
            circuits_collection['circuit'][circuit_name] = {
                'qasm_string': qasm_string,
                'properties': {
                    'num_qubits': num_qubits,
                    'actual_depth': circuit.depth(),
                    'size': len(circuit),
                    'id': _
                }
            }

        path = f"{save_path}/step_{cx_depth}.pk"
        with open(path, "wb") as f:
            pickle.dump(circuits_collection, f)

import qiskit
from qiskit import QuantumCircuit

def get_active_qubits_dynamics(circuit):
    """
    Return the set of qubit indices that are actually used in non‑barrier, non‑delay operations.
    """
    operations = circuit.data
    active_qubits = set()

    for op in operations:
        if isinstance(op[0], qiskit.circuit.Barrier) or isinstance(op[0], qiskit.circuit.Delay):
            continue
        for qubit in op[1]:
            active_qubits.add(qubit.index)

    return active_qubits

def remove_idle_qubits(circuit, layout):
    """
    Remove idle qubits (qubits that never appear in any non‑barrier/non‑delay operation)
    and return a new circuit with only active qubits, preserving the order in layout.
    """
    # Get active qubits, which excludes 'Delay' and 'Barrier' operations
    active_qubits = get_active_qubits_dynamics(circuit)

    # Create a new list ordered_qubits, preserving the order in layout
    ordered_qubits = [qubit for qubit in layout if qubit in active_qubits]

    # Keep the original ClassicalRegisters intact
    new_qc = QuantumCircuit(len(ordered_qubits))
    for cr in circuit.cregs:
        new_qc.add_register(cr)

    # Go through each instruction and replace the qubit indices with the indices of ordered qubits
    for instr, qargs, cargs in circuit.data:
        ordered_qarg_indices = [ordered_qubits.index(qubit.index) for qubit in qargs if qubit.index in ordered_qubits]
        # Since we're keeping the classical bits as they were, there's no need to change the indices
        if ordered_qarg_indices:  # If list is not empty, means qubits are involved in operation
            new_qc.append(instr, qargs=ordered_qarg_indices, cargs=cargs)

    return new_qc

def construct_random_circuit(two_q_gate_depth_post_trans, be, num_qubit):
    """
    Construct a random circuit with a target two‑qubit gate depth after transpilation.
    The function tries random circuits until the depth of CX gates matches the target.
    """
    layout = list(range(num_qubit))

    if num_qubit == 4:
        factor = 1
    else:
        raise NotImplementedError

    if not two_q_gate_depth_post_trans:
        # Depth 0: just a shallow random circuit
        d = np.random.randint(1, 3)
        rc_first = random_circuit(num_qubit, d, 1, measure=False)
        rc_copy = rc_first.copy()
        rc = transpile(rc_copy, basis_gates=['rz', 'sx', 'x', 'cx', 'id'], optimization_level=1, initial_layout=layout)

    elif 1 <= two_q_gate_depth_post_trans < 2:
        # Target depth between 1 and 2 (but depth is integer, so effectively 1)
        d = np.random.randint(1, 6)
        rc_first = random_circuit(num_qubit, d, 2, measure=False)
        rc_copy = rc_first.copy()
        rc = transpile(rc_copy, basis_gates=['rz', 'sx', 'x', 'cx', 'id'], optimization_level=1, initial_layout=layout)
        while rc.depth(filter_function=lambda x: x[0].num_qubits == 2) != two_q_gate_depth_post_trans:
            d = np.random.randint(1, 6)
            rc_first = random_circuit(num_qubit, d, 2, measure=False)
            rc_copy = rc_first.copy()
            rc_copy.barrier()
            rc = transpile(rc_copy, basis_gates=['rz', 'sx', 'x', 'cx', 'id'], optimization_level=1,
                           initial_layout=layout)

    elif 2 <= two_q_gate_depth_post_trans <= 60:
        # Target depth between 2 and 60
        d = np.random.randint(1, two_q_gate_depth_post_trans // factor)
        rc_first = random_circuit(num_qubit, d, 2, measure=False)
        rc_copy = rc_first.copy()
        rc = transpile(rc_copy, basis_gates=['rz', 'sx', 'x', 'cx', 'id'], optimization_level=1, initial_layout=layout)

        while rc.depth(filter_function=lambda x: x[0].num_qubits == 2) != two_q_gate_depth_post_trans:
            d = np.random.randint(1, two_q_gate_depth_post_trans // factor)
            rc_first = random_circuit(num_qubit, d, 2, measure=False)
            rc_copy = rc_first.copy()
            rc = transpile(rc_copy, basis_gates=['rz', 'sx', 'x', 'cx', 'id'], optimization_level=1, initial_layout=layout)
    else:
        raise NotImplementedError

    return rc

# Fix random seed for reproducibility
fix_random_seed(42)
num_qubit = 4

if __name__ == '__main__':
    # Generate circuits for two‑qubit gate depths 0,2,4,...,36 (even numbers up to 36)
    two_q_gate_depth_post_trans = list(range(19))
    for i in range(19):
        two_q_gate_depth_post_trans[i] *= 2

    backend = FakeQuito()
    generate_data(be=backend, cx_depths=two_q_gate_depth_post_trans, num_qubits=num_qubit,
                  save_path='./random_gen/', n_entries_per_file=200)