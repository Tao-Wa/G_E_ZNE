import numpy as np
import random
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Operator

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
_PAULI_LABELS = ['I', 'X', 'Y', 'Z']
_PAULI_GATES = {
    'I': lambda qc, q: None,
    'X': lambda qc, q: qc.x(q),
    'Y': lambda qc, q: qc.y(q),
    'Z': lambda qc, q: qc.z(q),
}

def pauli_string_from_operator(op_matrix, num_qubits):
    """Map a matrix back to the closest Pauli string (supports ≤ 2 qubits)."""
    paulis = ['I', 'X', 'Y', 'Z']
    labels, pauli_ops = [], []

    for idx in range(4 ** num_qubits):
        label = []
        tmp = idx
        mat = 1
        for q in range(num_qubits):
            p = paulis[tmp % 4]
            tmp //= 4
            label.append(p)
            if p == 'I':
                m = np.eye(2)
            elif p == 'X':
                m = np.array([[0, 1], [1, 0]])
            elif p == 'Y':
                m = np.array([[0, -1j], [1j, 0]])
            else:
                m = np.array([[1, 0], [0, -1]])
            mat = np.kron(m, mat) if q > 0 else m
        labels.append(''.join(label))
        pauli_ops.append(mat)

    overlaps = [np.abs(np.trace(np.conj(P.T) @ op_matrix)) for P in pauli_ops]
    return labels[int(np.argmax(overlaps))]

def conjugate_pauli_by_gate(pauli_label, gate_matrix):
    """Compute G P G† and map back to a Pauli string."""
    n = len(pauli_label)
    P = None
    for i, ch in enumerate(pauli_label):
        if ch == 'I':
            m = np.eye(2)
        elif ch == 'X':
            m = np.array([[0, 1], [1, 0]])
        elif ch == 'Y':
            m = np.array([[0, -1j], [1j, 0]])
        else:
            m = np.array([[1, 0], [0, -1]])
        P = np.kron(m, P) if (i > 0 and P is not None) else (m if P is None else np.kron(m, P))
    M = gate_matrix @ P @ gate_matrix.conj().T
    return pauli_string_from_operator(M, n)

def twirl_circuit(qc_in: QuantumCircuit, multi_qubit_gate_names=('cx', 'cnot'), seed=None):
    """Perform Pauli twirling on specified multi-qubit gates."""
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
def build_readout_mitigation_matrix(simulator, noise_model, n_qubits, shots=1000000):
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
import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import SparsePauliOp

def run_pauli_twirling_zne(qc, observable_op, noise_model,
                           noise_scale_factors, T=6, circ=1):
    """
    Run Pauli-twirled circuits to obtain noisy expectation values, fidelity, and purity.

    Parameters:
        qc: Target circuit
        observable_op: Observable
        noise_model: Noise model
        noise_scale_factors: Circuit folding factors
        T: Number of Pauli twirling repetitions
        circ: Number of random repetitions for fractional folding

    Returns:
        float: Mean noisy expectation value
        float: Mean purity
        float: Mean fidelity
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
    for (t, scale, tag, idx) in circ_info:
        data_dict = result.data(all_circs[idx])
        key = "density_matrix"  # default key
        if key not in data_dict:
            raise KeyError(f"'{key}' not found in result for circuit {idx}")
        rho = np.array(data_dict["density_matrix"], dtype=complex)
        j = noise_scale_factors.index(scale)

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
    """
    Use Pauli twirling + ZNE, expectation and fidelity simulated with shots, purity with density matrix.
    Supports arbitrary Pauli linear combinations (including X, Y terms) and measurement error mitigation (M_inv).
    Added seed parameter to control randomness.
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
        exp = 0.0
        for i in range(2**n_qubits):
            bits = format(i, f'0{n_qubits}b')
            sign = 1
            for q, p_char in enumerate(pauli_str):
                if p_char in ('X', 'Y', 'Z'):
                    sign *= 1 if bits[q] == '0' else -1
            exp += p[i] * sign
        return exp

    from collections import defaultdict
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

    for (t, scale, rep), pauli_exps in exp_components.items():
        j = noise_scale_factors.index(scale)
        total_exp = 0.0
        for pauli_idx, exp_val in pauli_exps.items():
            total_exp += coeff_list[pauli_idx] * exp_val
        exp_results[t, j] += total_exp

    for (t, scale, rep), fid_list in fid_components.items():
        j = noise_scale_factors.index(scale)
        if fid_list:
            fidelity_results[t, j] += fid_list[0]

    for t in range(T):
        for j in range(n_scales):
            exp_results[t, j] /= circ
            fidelity_results[t, j] /= circ

    dm_idx = 0
    for t in range(T):
        for scale in noise_scale_factors:
            j = noise_scale_factors.index(scale)
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
    """
    Calculate epsilon values from the given fidelity array and number of qubits.

    Parameters:
        fidelities: Array of P0 fidelity values
        q_num: Number of qubits

    Returns:
        numpy array: Epsilon values corresponding to the input fidelities
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
from scipy.optimize import curve_fit

def zne_loocv(models, q_num, mod, delate, value_bound=1.2, fallback_value=None):
    """
    Obtain predictions from different extrapolation models using LOOCV.

    Parameters:
        models: List of models containing fitting functions and experimental data
        value_bound: Threshold for valid predictions; values above are considered failures
        mod: Number of validation set points
        delate: Whether to use only a subset of points for fitting (1=yes, 0=no)

    Returns:
        str: Name of the best fitting model
        float: Extrapolated result from the best model
        dict: Dictionary containing results from all models
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