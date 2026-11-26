"""Numerical core of the Hybrid G-E ZNE pipeline.

Contents
--------
Gate folding
    ``uniform_gate_folding`` / ``circuit_folding`` implement the two noise
    amplification strategies used by the paper.
Pauli twirling
    ``twirl_circuit`` randomises the Pauli frame around every two-qubit gate and
    ``conjugate_pauli_by_gate`` propagates that frame through the gate.
Readout mitigation
    ``build_readout_mitigation_matrix`` estimates the assignment matrix ``M``.
Simulation
    ``run_pauli_twirling_zne`` (density matrix, exact) and
    ``run_pauli_twirling_zne_shot`` (shot based; used for Grover circuits).
Model selection
    ``calculate_epsilon`` and ``zne_loocv``.

WARNING for future edits -- behaviour-critical details
-----------------------------------------------------
* ``uniform_gate_folding`` consumes the *global* ``np.random`` stream whenever
  ``scale_factor`` is not an odd integer (fractional fold).  Callers invoke it
  once per (twirl, scale, repetition) **and a second, independent time** to build
  the fidelity circuit.  Do not "de-duplicate" those two calls: it would change
  how many draws are taken from ``np.random`` and therefore every later number.
* Every floating-point sum below is written as an explicit Python loop on
  purpose.  Vectorising with ``np.dot``/``np.sum`` would use pairwise or BLAS
  summation and would change the last bits of the reported values.
"""

from __future__ import annotations

import random
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Operator, SparsePauliOp
from scipy.optimize import curve_fit

# ==================== Gate Folding Functions ===================
def uniform_gate_folding(circuit, scale_factor):
    """
    Noise amplification via uniform gate folding.
    Each gate is folded the same number of times.

    Args:
        circuit: Original quantum circuit
        scale_factor: Noise amplification factor (>=1)

    Returns:
        folded_circuit: Folded circuit
    """
    if scale_factor < 1:
        raise ValueError("Scale factor must be >= 1")

    # If scale factor is 1, return the original circuit
    if scale_factor == 1:
        return circuit.copy()

    # Create folded circuit
    folded_circuit = QuantumCircuit(circuit.num_qubits)

    # Get all gates from the original circuit
    original_gates = circuit.data.copy()

    # Calculate the number of folding pairs to add per gate
    # scale_factor = 1 + 2 * k, where k is the number of (gate + inverse) pairs per gate
    k = (scale_factor - 1) / 2

    # Separate integer and fractional parts
    k_int = int(k)
    fractional_part = k - k_int

    # Apply the same folding operation to each gate
    for gate in original_gates:
        # Add the original gate
        folded_circuit.append(gate.operation, gate.qubits)

        # Add integer part folds
        for _ in range(k_int):
            folded_circuit.barrier(gate.qubits)
            inverted_op = gate.operation.inverse()
            folded_circuit.append(inverted_op, gate.qubits)
            folded_circuit.barrier(gate.qubits)
            folded_circuit.append(gate.operation, gate.qubits)
            folded_circuit.barrier(gate.qubits)

        # Add fractional part fold (probabilistic)
        if fractional_part > 0:
            if np.random.random() < fractional_part:
                folded_circuit.barrier(gate.qubits)
                inverted_op = gate.operation.inverse()
                folded_circuit.append(inverted_op, gate.qubits)
                folded_circuit.barrier(gate.qubits)
                folded_circuit.append(gate.operation, gate.qubits)

    return folded_circuit

def circuit_folding(circuit, scale_factor):
    """
    Noise amplification via circuit folding.

    Args:
        circuit: Original quantum circuit
        scale_factor: Noise amplification factor (>=1)

    Returns:
        folded_circuit: Folded circuit
    """
    if scale_factor < 1:
        raise ValueError("Scale factor must be >= 1")

    # If scale factor is 1, return the original circuit
    if scale_factor == 1:
        return circuit.copy()

    # Create folded circuit
    folded_circuit = QuantumCircuit(circuit.num_qubits)

    # Calculate integer and fractional parts
    k = (scale_factor - 1) / 2
    n = int(k)
    f = k - n

    # Add the original circuit
    folded_circuit.compose(circuit, inplace=True)

    # Add integer multiples of (original circuit + inverse) pairs
    for _ in range(n):
        folded_circuit.barrier(circuit.qubits)
        folded_circuit.compose(circuit.inverse(), inplace=True)
        folded_circuit.barrier(circuit.qubits)
        folded_circuit.compose(circuit, inplace=True)

    # Add fractional part folding
    if f > 0:
        num_gates = len(circuit.data)
        num_folded_gates = int(f * num_gates)

        if num_folded_gates > 0:
            # Create a partial circuit
            partial_circuit = QuantumCircuit(circuit.num_qubits)
            for i in range(num_folded_gates):
                gate = circuit.data[i]
                partial_circuit.append(gate.operation, gate.qubits)

            # Add the inverse of the partial circuit
            folded_circuit.barrier(partial_circuit.qubits)
            folded_circuit.compose(partial_circuit.inverse(), inplace=True)
            folded_circuit.barrier(partial_circuit.qubits)
            # Add the partial circuit
            folded_circuit.compose(partial_circuit, inplace=True)

    return folded_circuit

# =================== Pauli Twirling Tools ====================
_PAULI_LABELS: Tuple[str, ...] = ('I', 'X', 'Y', 'Z')
_PAULI_GATES = {
    'I': lambda qc, q: None,
    'X': lambda qc, q: qc.x(q),
    'Y': lambda qc, q: qc.y(q),
    'Z': lambda qc, q: qc.z(q),
}
_PAULI_MATRICES = {
    'I': np.eye(2),
    'X': np.array([[0, 1], [1, 0]]),
    'Y': np.array([[0, -1j], [1j, 0]]),
    'Z': np.array([[1, 0], [0, -1]]),
}


@lru_cache(maxsize=None)
def _pauli_basis(num_qubits: int) -> Tuple[Tuple[str, ...], Tuple[np.ndarray, ...]]:
    """Return all ``(label, matrix)`` pairs of the ``4**n`` Pauli strings.

    The label is little-endian (``label[q]`` acts on qubit ``q``) and the matrix
    is nested exactly as ``np.kron(m_q, np.kron(m_{q-1}, ...))``, matching the
    convention used by ``conjugate_pauli_by_gate``.

    PERF: this is a pure function of ``num_qubits`` and used to be rebuilt from
    scratch for every Pauli string of every twirled two-qubit gate.  Caching it
    changes no value -- the same matrices are simply constructed once.
    """
    labels: List[str] = []
    matrices: List[np.ndarray] = []
    for idx in range(4 ** num_qubits):
        label: List[str] = []
        tmp = idx
        mat: Optional[np.ndarray] = None
        for _ in range(num_qubits):
            name = _PAULI_LABELS[tmp % 4]
            tmp //= 4
            label.append(name)
            m = _PAULI_MATRICES[name]
            mat = m if mat is None else np.kron(m, mat)
        labels.append(''.join(label))
        matrices.append(mat)
    return tuple(labels), tuple(matrices)


@lru_cache(maxsize=None)
def _pauli_matrix_by_label(num_qubits: int) -> Dict[str, np.ndarray]:
    """Label -> matrix lookup for ``num_qubits`` (cached companion of the basis)."""
    labels, matrices = _pauli_basis(num_qubits)
    return dict(zip(labels, matrices))


def pauli_string_from_operator(op_matrix: np.ndarray, num_qubits: int) -> str:
    """Map a matrix back to the closest Pauli string (supports <= 2 qubits)."""
    labels, pauli_ops = _pauli_basis(num_qubits)
    overlaps = [np.abs(np.trace(np.conj(P.T) @ op_matrix)) for P in pauli_ops]
    return labels[int(np.argmax(overlaps))]


def conjugate_pauli_by_gate(pauli_label: str, gate_matrix: np.ndarray) -> str:
    """Compute ``G P G^dagger`` and map the result back to a Pauli string."""
    n = len(pauli_label)
    P = _pauli_matrix_by_label(n)[pauli_label]
    M = gate_matrix @ P @ gate_matrix.conj().T
    return pauli_string_from_operator(M, n)


def twirl_circuit(qc_in: QuantumCircuit, multi_qubit_gate_names: Sequence[str] = ('cx', 'cnot'),
                  seed: Optional[int] = None) -> QuantumCircuit:
    """Perform Pauli twirling on the specified multi-qubit gates.

    ``seed`` is forwarded verbatim to ``random.Random``; the number of draws and
    their order are part of the reproducibility contract -- do not reorder the
    loop body.
    """
    qc = QuantumCircuit(qc_in.num_qubits)
    rng = random.Random(seed)

    for inst, qargs, cargs in qc_in.data:
        if inst.name.lower() in multi_qubit_gate_names:
            # pre Pauli
            labels = [rng.choice(_PAULI_LABELS) for _ in qargs]
            for lab, q in zip(labels, qargs):
                if lab != 'I':
                    _PAULI_GATES[lab](qc, q.index)

            # gate
            qc.append(inst, qargs, cargs)

            # post Pauli = G P G†
            try:
                gate_matrix = Operator(inst).data
            except Exception:
                if inst.name.lower() in ['cx', 'cnot']:
                    gate_matrix = Operator(QuantumCircuit(2).cx(0, 1)).data
                else:
                    raise ValueError(f"Unsupported gate for twirling: {inst.name}")

            pre_label = ''.join(labels)
            post_label = conjugate_pauli_by_gate(pre_label, gate_matrix)
            for lab, q in zip(post_label, qargs):
                if lab != 'I':
                    _PAULI_GATES[lab](qc, q.index)
        else:
            qc.append(inst, qargs, cargs)

    return qc

# =================== Readout Error Mitigation Matrix ====================
def build_readout_mitigation_matrix(simulator, noise_model, n_qubits: int,
                                    shots: int = 1000000) -> np.ndarray:
    """Estimate the readout assignment matrix ``M`` (column = prepared state).

    ``M[j, i]`` is the probability of measuring bit string ``j`` when state
    ``i`` was prepared.  One simulator job is submitted for all basis states.
    """
    dim = 2 ** n_qubits
    M = np.zeros((dim, dim))

    cal_circs = []

    # Build |i⟩ → measurement
    for i in range(dim):
        qc = QuantumCircuit(n_qubits)
        bitstring = format(i, f"0{n_qubits}b")

        for q, b in enumerate(bitstring[::-1]):
            if b == '1':
                qc.x(q)

        qc.measure_all()
        cal_circs.append(qc)

    result = simulator.run(cal_circs, noise_model=noise_model, shots=shots).result()

    # Fill matrix M_{meas, ideal}
    for i, qc in enumerate(cal_circs):
        counts = result.get_counts(qc)
        for bitstring, c in counts.items():
            j = int(bitstring, 2)
            M[j, i] += c

    # Normalize columns (conditional probabilities)
    M = M / shots

    return M

# =================== Compute Circuit Outputs ====================
def _scale_index_map(noise_scale_factors: Sequence[float]) -> Dict[float, int]:
    """First-occurrence index of each scale factor.

    Replaces repeated ``list.index()`` scans inside the hot result-extraction
    loops.  ``setdefault`` keeps the *first* index, exactly like ``.index()``.
    """
    mapping: Dict[float, int] = {}
    for index, value in enumerate(noise_scale_factors):
        mapping.setdefault(value, index)
    return mapping


def run_pauli_twirling_zne(qc, observable_op, noise_model,
                           noise_scale_factors, T=6, circ=1):
    """
    Run Pauli-twirled circuits to obtain noisy expectation values, fidelity and purity.

    All circuits (main + fidelity) are submitted in a *single* simulator job and
    evaluated from the returned density matrices, i.e. exactly, without shots.

    Parameters
    ----------
    qc, observable_op, noise_model:
        Target circuit, observable and device noise model.
    noise_scale_factors:
        Circuit folding factors.
    T:
        Number of Pauli twirling repetitions (``t`` loop).  ``T == 1`` disables
        twirling entirely -- that branch is intentional.
    circ:
        Number of repetitions used for fractional (non-odd) folding factors.

    Returns
    -------
    (exp_mean, purity_mean, fidelity_mean) : tuple of np.ndarray
        Averages over the ``T`` twirls, one entry per scale factor.
    """
    simulator = Aer.get_backend("aer_simulator_density_matrix")

    all_circs = []      # All circuits to run
    circ_info = []      # Circuit labels (t, scale, 'main'/'fid', idx)
    rng = np.random.default_rng(42)

    # Batch generate main circuits and fidelity circuits
    for t in range(T):
        seed = rng.integers(1 << 30)
        if T == 1:
            twirled = qc.copy()
        else:
            twirled = twirl_circuit(qc, seed=seed)

        for scale in noise_scale_factors:
            for i in range(circ):
                # Main circuit
                folded = uniform_gate_folding(twirled, scale).copy()
                folded.save_density_matrix()
                all_circs.append(folded)
                circ_info.append((t, scale, "main", len(all_circs) - 1))

                # Fidelity circuit (U U†)
                fid_qc = uniform_gate_folding(twirled, scale).copy()
                fid_circ = QuantumCircuit(qc.num_qubits)
                fid_circ.compose(fid_qc, inplace=True)
                fid_circ.barrier()
                fid_circ.compose(fid_qc.inverse(), inplace=True)
                fid_circ.save_density_matrix()
                all_circs.append(fid_circ)
                circ_info.append((t, scale, "fid", len(all_circs) - 1))

    # Run all circuits at once
    result = simulator.run(all_circs, noise_model=noise_model).result()

    n_scales = len(noise_scale_factors)
    exp_results = np.zeros((T, n_scales))
    purity_results = np.zeros((T, n_scales))
    fidelity_results = np.zeros((T, n_scales))

    # Extract results
    scale_to_index = _scale_index_map(noise_scale_factors)
    for (t, scale, tag, idx) in circ_info:
        data_dict = result.data(all_circs[idx])
        key = "density_matrix"  # default key
        if key not in data_dict:
            raise KeyError(f"'{key}' not found in result for circuit {idx}")
        rho = np.array(data_dict["density_matrix"], dtype=complex)
        j = scale_to_index[scale]

        if tag == "main":
            exp_results[t, j] = np.real(np.trace(rho @ observable_op.data))
            purity_results[t, j] = np.real(np.trace(rho @ rho))
        elif tag == "fid":
            fidelity_results[t, j] = np.real(rho[0, 0])

    exp_mean = np.mean(exp_results, axis=0)
    purity_mean = np.mean(purity_results, axis=0)
    fidelity_mean = np.mean(fidelity_results, axis=0)
    return exp_mean, purity_mean, fidelity_mean

def run_pauli_twirling_zne_shot(qc, observable_op, noise_model,
                                noise_scale_factors, T=6, circ=1, shots=8192,
                                M_inv=None, seed=42):
    """Pauli twirling + ZNE with a hybrid shot / density-matrix backend.

    Expectation values and fidelities are estimated from measurements with
    ``shots`` shots; the purity is still taken exactly from a density-matrix
    simulation.  Arbitrary Pauli linear combinations (including X/Y terms) are
    supported, as is measurement-error mitigation through ``M_inv``.

    ``seed`` seeds ``np.random.default_rng``, which in turn seeds the per-twirl
    ``twirl_circuit`` calls.  The draw order (one ``rng.integers`` per twirl, then
    the twirl itself) is part of the reproducibility contract.
    """
    simulator_dm = Aer.get_backend("aer_simulator_density_matrix")
    simulator = Aer.get_backend("aer_simulator")

    # Convert observable to SparsePauliOp
    if not isinstance(observable_op, SparsePauliOp):
        observable_pauli = SparsePauliOp.from_operator(observable_op)
    else:
        observable_pauli = observable_op

    pauli_list = observable_pauli.paulis
    coeff_list = observable_pauli.coeffs

    n_qubits = qc.num_qubits
    rng = np.random.default_rng(seed)  # Use the provided seed

    shot_circs = []
    dm_circs = []
    circ_info = []

    def build_measurement_circuit(base_circ, pauli_str):
        circ = base_circ.copy()
        for qubit, p_char in enumerate(pauli_str):
            if p_char == 'X':
                circ.h(qubit)
            elif p_char == 'Y':
                circ.sdg(qubit)
                circ.h(qubit)
        circ.measure_all()
        return circ

    for t in range(T):
        seed_t = rng.integers(1 << 30)
        if T == 1:
            twirled = qc.copy()
        else:
            twirled = twirl_circuit(qc, seed=seed_t)

        for scale in noise_scale_factors:
            for rep in range(circ):
                folded_base = uniform_gate_folding(twirled, scale).copy()

                for pauli_idx, (pauli, coeff) in enumerate(zip(pauli_list, coeff_list)):
                    pauli_str = pauli.to_label()
                    meas_circ = build_measurement_circuit(folded_base, pauli_str)
                    shot_circs.append(meas_circ)
                    circ_info.append((t, scale, "main", pauli_idx, rep, len(shot_circs)-1))

                folded_dm = folded_base.copy()
                folded_dm.save_density_matrix()
                dm_circs.append(folded_dm)

                # NOTE: this is a *second*, independent fold of the same twirled
                # circuit -- not a duplicate of `folded_base`.  For fractional
                # scale factors `uniform_gate_folding` draws from the global
                # np.random stream, so collapsing the two calls would change
                # every subsequent random number.  Do not "optimise" this away.
                fid_base = uniform_gate_folding(twirled, scale).copy()
                fid_circ = QuantumCircuit(n_qubits)
                fid_circ.compose(fid_base, inplace=True)
                fid_circ.barrier()
                fid_circ.compose(fid_base.inverse(), inplace=True)
                fid_circ.measure_all()
                shot_circs.append(fid_circ)
                circ_info.append((t, scale, "fid", None, rep, len(shot_circs)-1))

    result_shots = simulator.run(shot_circs, noise_model=noise_model, shots=shots).result()
    result_dm = simulator_dm.run(dm_circs, noise_model=noise_model).result()

    n_scales = len(noise_scale_factors)
    exp_results = np.zeros((T, n_scales))
    purity_results = np.zeros((T, n_scales))
    fidelity_results = np.zeros((T, n_scales))

    def counts_to_prob(counts):
        p = np.zeros(2**n_qubits)
        total = sum(counts.values())
        for bitstring, cnt in counts.items():
            idx = int(bitstring, 2)
            p[idx] = cnt / total
        return p

    def compute_pauli_expectation(p, pauli_str):
        """<P> for one Pauli string.

        Kept as an explicit index-ordered summation: a vectorised rewrite would
        re-associate the floating point additions and change the last bits.
        """
        exp = 0.0
        for i in range(2**n_qubits):
            bits = format(i, f'0{n_qubits}b')
            sign = 1
            for q, p_char in enumerate(pauli_str):
                if p_char in ('X', 'Y', 'Z'):
                    sign *= 1 if bits[q] == '0' else -1
            exp += p[i] * sign
        return exp

    exp_components = defaultdict(dict)
    fid_components = defaultdict(list)

    for (t, scale, typ, pauli_idx, rep, circ_idx) in circ_info:
        counts = result_shots.get_counts(circ_idx)

        if typ == "main":
            p = counts_to_prob(counts)
            if M_inv is not None:
                p = M_inv @ p
                p = np.maximum(p, 0)
                p /= p.sum()
            pauli_str = pauli_list[pauli_idx].to_label()
            exp_val = compute_pauli_expectation(p, pauli_str)
            exp_components[(t, scale, rep)][pauli_idx] = exp_val

        elif typ == "fid":
            p = counts_to_prob(counts)
            if M_inv is not None:
                p = M_inv @ p
                p = np.maximum(p, 0)
                p /= p.sum()
            fid = p[0]  # Probability of all-zero state
            fid_components[(t, scale, rep)].append(fid)

    scale_to_index = _scale_index_map(noise_scale_factors)

    for (t, scale, rep), pauli_exps in exp_components.items():
        j = scale_to_index[scale]
        total_exp = 0.0
        for pauli_idx, exp_val in pauli_exps.items():
            total_exp += coeff_list[pauli_idx] * exp_val
        exp_results[t, j] += total_exp

    for (t, scale, rep), fid_list in fid_components.items():
        j = scale_to_index[scale]
        if fid_list:
            fidelity_results[t, j] += fid_list[0]

    for t in range(T):
        for j in range(n_scales):
            exp_results[t, j] /= circ
            fidelity_results[t, j] /= circ

    dm_idx = 0
    for t in range(T):
        for scale in noise_scale_factors:
            j = scale_to_index[scale]
            purities = []
            for rep in range(circ):
                rho = np.array(result_dm.data(dm_idx)["density_matrix"])
                purity = np.real(np.trace(rho @ rho))
                purities.append(purity)
                dm_idx += 1
            purity_results[t, j] = np.mean(purities)

    exp_mean = np.mean(exp_results, axis=0)
    purity_mean = np.mean(purity_results, axis=0)
    fidelity_mean = np.mean(fidelity_results, axis=0)

    return exp_mean, purity_mean, fidelity_mean

def calculate_epsilon(fidelities, q_num):
    """Convert a P0 fidelity to the depolarising parameter epsilon.

    Parameters
    ----------
    fidelities:
        Sequence of P0 (all-zero outcome probability) fidelity values.
    q_num:
        Number of qubits, giving the ``1 / 2**q_num`` threshold.

    Returns
    -------
    np.ndarray
        Epsilon values, one per input fidelity.

    Kept as an element-wise loop: ``np.where`` would evaluate both branches for
    every element and is not obviously bit-identical.
    """
    base = 2 ** q_num
    epsilon_arr = np.zeros_like(fidelities)
    threshold = 1 / base

    for i, p0 in enumerate(fidelities):
        p0 = max(0.0, min(1.0, p0))

        if p0 > threshold:
            numerator = 1 - np.sqrt(p0 - (1 - p0) / base)
            denominator = 1 + 1 / base
            epsilon_arr[i] = numerator / denominator
        else:
            epsilon_arr[i] = (1 - p0) / (1 + p0)
    return epsilon_arr

# =================== Extrapolation Model Results ====================
def zne_loocv(models, q_num, mod, delate, value_bound=1.2, fallback_value=None):
    """Cross-validate the extrapolation models and return the best prediction.

    Parameters
    ----------
    models:
        List of model dicts, each with ``name``, ``x``, ``y``, ``func``, ``x0``,
        ``param`` and optionally ``p0`` / ``bounds``.
    q_num:
        Number of qubits (passed through for API compatibility).
    mod:
        Validation scheme -- ``0`` = validate on the training set itself,
        ``1`` = leave-one-out (``param + 1`` points), ``2`` = two held-out points.
    delate:
        ``1`` = fit only the first ``param`` points, ``0`` = fit all points.
    value_bound:
        Predictions with ``|pred| > value_bound`` are rejected and replaced by
        ``y_data[0]``.
    fallback_value:
        Value used when a fit fails; defaults to ``y_data[0]``.

    Returns
    -------
    (best_pred, best_name, details)
    """

    n = len(models[0]["x"])
    details = []
    best_model, best_err = None, float("inf")

    for model in models:
        params = model["param"]
        x_data = np.array(model["x"])
        y_data = np.array(model["y"])
        func = model["func"]
        p0 = model.get("p0", None)
        bounds = model.get("bounds", (-np.inf, np.inf))
        target_x = model.get("x0")
        total_err, success = 0, True

        # Split data into training set (n-2 points) and validation set (2 points)
        if mod == 2:
            x_data = x_data[:params+2]
            y_data = y_data[:params+2]
            for i in range(n):
                for j in range(i):
                    x_train = np.delete(x_data, [i, j])
                    y_train = np.delete(y_data, [i, j])
                    x_test1, y_test1 = x_data[i], y_data[i]
                    x_test2, y_test2 = x_data[j], y_data[j]
                    try:
                        popt, _ = curve_fit(func, x_train, y_train, p0=p0, bounds=bounds, maxfev=300000)
                        y_pred1 = func(x_test1, *popt)
                        total_err += abs(y_pred1 - y_test1)
                        y_pred2 = func(x_test2, *popt)
                        total_err += abs(y_pred2 - y_test2)
                    except Exception:
                        success = False
                        break

        # Split data into training set (n-1 points) and validation set (1 point)
        elif mod == 1:
            num = params + 1
            x_data = x_data[:num]
            y_data = y_data[:num]
            for i in range(num):
                x_train = np.delete(x_data, i)
                y_train = np.delete(y_data, i)
                x_test, y_test = x_data[i], y_data[i]
                try:
                    popt, _ = curve_fit(func, x_train, y_train, p0=p0, bounds=bounds, maxfev=200000)
                    y_pred = func(x_test, *popt)
                    total_err += abs(y_pred - y_test)
                except Exception:
                    success = False
                    break

        # Validate only using the training set itself
        elif mod == 0:
            try:
                popt, _ = curve_fit(func, x_data, y_data, p0=p0, bounds=bounds, maxfev=300000)
                for i in range(n):
                    x_test, y_test = x_data[i], y_data[i]
                    y_pred = func(x_test, *popt)
                    total_err += abs(y_pred - y_test)
            except Exception:
                success = False
                break

        # Extrapolate for each model to obtain parameters and predictions
        if success:
            try:
                if delate == 1:   # Use only the first 'params' points for fitting
                    x_data_fit = x_data[:params]
                    y_data_fit = y_data[:params]
                else:             # Use all points for fitting
                    x_data_fit = x_data
                    y_data_fit = y_data
                popt, _ = curve_fit(func, x_data_fit, y_data_fit, p0=p0, bounds=bounds, maxfev=300000)
                if model["name"] == 'poly_exp':
                    print(model["name"])
                    print(popt)

                if model["name"] == 'linea':
                    final_pred = popt[0]
                else:
                    final_pred = func(target_x, *popt)

            except Exception:
                final_pred = fallback_value if fallback_value is not None else y_data[0]

        else:
            final_pred = fallback_value if fallback_value is not None else y_data[0]
            total_err = float("inf")

        # Basic validity check of the prediction
        if abs(final_pred) > value_bound:
            valid = False
            final_pred = y_data[0]
        else:
            valid = True

        details.append({
            "name": model["name"],
            "cv_error": total_err,
            "pred": float(final_pred),
            "success": success,
            "valid": valid
        })

        # Determine the best fitting model
        if success and valid and total_err < best_err:
            best_err = total_err
            best_model = model

    if best_model is None:
        return fallback_value if fallback_value is not None else models[0]["y"][0], None, details
    best_name = best_model["name"]
    best_pred = next(d["pred"] for d in details if d["name"] == best_name)
    return best_pred, best_name, details