"""Trotterised-Ising circuit generator.

Builds ``IsingModel`` Trotter circuits with a random coupling ``J ~ U(0, h)``, a
random measurement basis and a random state-vector initialisation, and pickles
200 circuits per Trotter step into ``trotter_gen/step_<depth>.pk``.

The restartable RNG (``rand_seed.create_restartable_random_system``, seed 42) is
restarted at the beginning of every depth, so each depth file is generated from
an identical random stream.  Preserve that call order.
"""

from qiskit.quantum_info import random_statevector
from qiskit.circuit.library import RYGate, RXGate, RZGate, CXGate
from qiskit import QuantumRegister
from qiskit import transpile
from qiskit import execute
from qiskit.providers.fake_provider import FakeLima, FakeQuito
from qiskit import QuantumCircuit
from rand_seed import create_restartable_random_system
from qiskit_aer import AerSimulator, QasmSimulator

from tqdm.notebook import tqdm_notebook

import numpy as np
from qiskit.transpiler.exceptions import TranspilerError
import scipy
import pickle

class IsingModel:
    """
    Class for generating Ising model quantum circuits with Trotterization.
    """

    class Options(dict):
        """
        Configuration options for the Ising model.
        Inherits from dict to allow easy parameter updates.
        """
        def __init__(self, *args, **kwargs):
            super().__init__()
            self['nq'] = 4
            self['h'] = 1
            self['J'] = 0.15
            self['dt'] = 1/4
            self['depth'] = 15
            self.update(*args, **kwargs)

        def config_4q_paper(self):
            """Set parameters to match the 4‑qubit paper configuration."""
            self['h'] = 1
            self['J'] = 0.15
            self['dt'] = np.pi / 15  # /4*2

        def config_10q_paper(self):
            """Set parameters to match the 10‑qubit paper configuration."""
            self['h'] = 1
            self['J'] = 0.5236
            self['dt'] = 1/4

    @classmethod
    def apply_quantum_circuit_layer(cls, qc: QuantumCircuit, ops: 'IsingModel.Options'):
        """
        Apply a single Trotter step layer to the circuit.
        Implements the evolution for one time step.
        """
        allq = list(range(ops['nq']))

        qc.rx(2*ops['h']*ops['dt'], allq)

        qc.barrier()
        for q0 in allq[0::2]:
            q1 = q0+1
            qc.cx(q0,q1)
        qc.rz(-2*ops['J']*ops['dt'], allq[1::2])
        for q0 in allq[0::2]:
            q1 = q0+1
            qc.cx(q0,q1)

        qc.barrier()
        for q0 in allq[1:-2:2]:
            q1 = q0+1
            qc.cx(q0,q1)
        qc.rz(-2*ops['J']*ops['dt'], allq[2:-1:2])
        for q0 in allq[1:-2:2]:
            q1 = q0+1
            qc.cx(q0,q1)
        qc.barrier()

    @classmethod
    def make_quantum_circuit(cls, ops: 'IsingModel.Options'):
        """
        Build the full Ising model quantum circuit with Trotter steps.
        Includes final basis rotations for measurement.
        """
        qc = QuantumCircuit(ops['nq'])
        allq = list(range(ops['nq']))

        for step in range(ops['depth']):
            cls.apply_quantum_circuit_layer(qc, ops)
        for i in range(4):
            if ops['measure_basis'][i] == 'Z':
                pass
            elif ops['measure_basis'][i] == 'X':
                qc.h(allq[i])
            elif ops['measure_basis'][i] == 'Y':
                qc.sdg(allq[i])
                qc.h(allq[i])
            else:
                raise Exception("Must be X Y or Z")
        qc.measure_all()

        return qc

    @classmethod
    def make_circs_sweep(cls, ops, num_steps, measure_basis):
        """
        Create a circuit for a specific Trotter depth and measurement basis.
        """
        ops['measure_basis'] = measure_basis
        ops['depth'] = num_steps
        qc = cls.make_quantum_circuit(ops)
        qc.metadata = {}
        qc.metadata['measure_basis'] = measure_basis
        qc.metadata['depth'] = num_steps
        return qc

# Ideal simulator backend (noiseless)
backend_ideal = QasmSimulator()
run_config_ideal = {'shots': 10000, 'backend': backend_ideal, 'name': 'ideal'}

def generate_random_initial_state():
    """Generate a random initial state vector and return a circuit initializing it."""
    qc = QuantumCircuit(4)
    random_state = random_statevector(2**4).data
    qc.initialize(random_state, list(range(4)))
    return qc

def generate_data(generating_func, two_q_gate_depths, save_path: str, n_entries_per_file: int, rng, seed=42):
    """
    Generate and save circuits for each depth.
    Args:
        generating_func: function that takes (depth, rng) and returns a circuit.
        two_q_gate_depths: list of depths to generate.
        save_path: directory where pickle files will be stored.
        n_entries_per_file: number of circuits per depth.
        rng: restartable random number generator.
        seed: initial seed.
    """
    for depth in two_q_gate_depths:
        entries = []
        rng.restart(seed)
        for i in tqdm_notebook(range(n_entries_per_file)):
            success = False
            while not success:
                try:
                    circuit = generating_func(depth, rng)
                    success = True
                except (scipy.linalg.LinAlgError, TranspilerError, np.linalg.LinAlgError) as e:
                    print(f"Ran into an error:, {e}")

            to_append = {}
            to_append['circuit'] = circuit
            entries.append(to_append)

        path = f"{save_path}/step_{depth}.pk"
        with open(path, "wb") as f:
            pickle.dump(entries, f)

def generate_random_initial_num_gate(num_gates=2):
    """Generate a random circuit with a given number of gates (unused in main flow)."""
    qr = QuantumRegister(4)
    qc = QuantumCircuit(qr)

    for _ in range(num_gates):
        gate_type = np.random.choice(['rx', 'rz', 'cnot', 'id'])
        qubit_indices = np.random.choice(range(4), size=2, replace=False)

        if gate_type == 'ry':
            theta = 2 * np.pi * np.random.rand()
            qc.append(RYGate(theta), [qr[qubit_indices[0]]])
        elif gate_type == 'rx':
            theta = 2 * np.pi * np.random.rand()
            qc.append(RXGate(theta), [qr[qubit_indices[0]]])
        elif gate_type == 'rz':
            theta = 2 * np.pi * np.random.rand()
            qc.append(RZGate(theta), [qr[qubit_indices[0]]])
        elif gate_type == 'id':
            pass
        elif gate_type == 'cnot':
            theta = 2 * np.pi * np.random.rand()
            qc.append(RXGate(theta), [qr[qubit_indices[0]]])
            qc.append(CXGate(), [qr[qubit_indices[0]], qr[qubit_indices[1]]])

    return qc

def construct_ising_circuit_random_init(J, basis, num_steps):
    """
    Build an Ising circuit with random initial state (actually fixed initial state here)
    and given J, measurement basis, and number of Trotter steps.
    """
    cls = IsingModel
    ops = cls.Options()
    ops.config_4q_paper()
    ops.update({'J': J})
    qc_init = QuantumCircuit.from_qasm_str('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\n')
    qc_init.barrier()
    qc = qc_init.compose(cls.make_circs_sweep(ops, num_steps, basis), list(range(4)))
    return qc

backend_ideal = QasmSimulator()
backend_noisy = FakeQuito()
h = 1

def construct_random_ising_func_random_init(num_steps, rng):
    """
    Generate a random Ising circuit with random J and random measurement basis.
    Used as generating_func in generate_data.
    """
    J = rng.uniform(0, h, size=1)[0]
    print(J)
    basis = [rng.choice(list('XYZ')) for _ in range(4)]
    random_ising = construct_ising_circuit_random_init(J, basis, num_steps)
    return random_ising

if __name__ == '__main__':
    seed = 42
    rng = create_restartable_random_system(seed)
    steps = list(range(0,18))
    generate_data(construct_random_ising_func_random_init, steps, './trotter_gen', 200, rng, seed)