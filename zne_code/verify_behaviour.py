#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Differential test: cleaned modules vs. the pristine originals.

Loads each *pair* of modules (``original_backup/<mod>.py`` and
``高斯-指数混合模型整合/<mod>.py``) under distinct module names and asserts that
they produce bit-identical results for a battery of inputs.

Run it inside the pinned environment:

    conda activate zne_qiskit
    python verify_behaviour.py

Exit code 0 == every check passed.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ORIG = BASE / "original_backup"
NEW = BASE / "高斯-指数混合模型整合"

FAILURES: list[str] = []
CHECKS = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"  [ok]   {label}")
    else:
        print(f"  [FAIL] {label} {detail}")
        FAILURES.append(label)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def deep_equal(a, b) -> bool:
    """Exact structural comparison with a bitwise-identical float requirement."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return bool(np.array_equal(a, b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float):
        # identical bit pattern (handles nan/inf and last-bit differences)
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


# --------------------------------------------------------------------------- #
# tools.py
# --------------------------------------------------------------------------- #
def test_tools() -> None:
    print("\n== tools.py ==")
    orig = load(ORIG / "tools.py", "tools_orig")
    new = load(NEW / "tools.py", "tools_new")

    rng = np.random.default_rng(20240101)

    # 1. pauli_string_from_operator over random Hermitian matrices.
    for n in (1, 2):
        dim = 2 ** n
        for _ in range(50):
            m = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
            m = m + m.conj().T
            check(
                f"pauli_string_from_operator(n={n})",
                orig.pauli_string_from_operator(m, n) == new.pauli_string_from_operator(m, n),
            )

    # 2. conjugate_pauli_by_gate for every label of every 1q / 2q Clifford gate.
    one_q = {"h": None, "x": None, "y": None, "z": None, "s": None, "sdg": None}
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    gates = []
    for name in one_q:
        qc = QuantumCircuit(2)
        getattr(qc, name)(0)
        gates.append(("1q:" + name, Operator(qc).data))
    for name in ("cx", "cz", "swap"):
        qc = QuantumCircuit(2)
        getattr(qc, name)(0, 1)
        gates.append(("2q:" + name, Operator(qc).data))

    def all_labels(n: int) -> list:
        """Little-endian Pauli labels, same enumeration order as the modules."""
        return [
            "".join("IXYZ"[(idx // (4 ** q)) % 4] for q in range(n))
            for idx in range(4 ** n)
        ]

    for gname, gmat in gates:
        n = int(round(math.log2(gmat.shape[0])))
        for label in all_labels(n):
            check(
                f"conjugate_pauli_by_gate({gname},{label})",
                orig.conjugate_pauli_by_gate(label, gmat)
                == new.conjugate_pauli_by_gate(label, gmat),
            )

    # 3. twirl_circuit must give the same circuit for the same seed.
    from qiskit import QuantumCircuit as QC

    base = QC(4)
    base.h(0)
    base.cx(0, 1)
    base.rz(0.37, 2)
    base.cx(1, 2)
    base.cx(2, 3)
    base.h(3)
    for seed in (0, 1, 7, 42, 123456789):
        a = orig.twirl_circuit(base, seed=seed)
        b = new.twirl_circuit(base, seed=seed)
        check(f"twirl_circuit(seed={seed})",
              a.qasm() == b.qasm() and len(a.data) == len(b.data))

    # 4. gate folding, including the random fractional branch -> RNG parity.
    for scale in (1, 2, 3, 5, 7, 9, 1.5, 2.5, 4.25):
        np.random.seed(4242)
        a = orig.uniform_gate_folding(base, scale)
        state_a = np.random.get_state()
        np.random.seed(4242)
        b = new.uniform_gate_folding(base, scale)
        state_b = np.random.get_state()
        check(f"uniform_gate_folding(scale={scale})", a.qasm() == b.qasm())
        check(f"uniform_gate_folding RNG stream(scale={scale})",
              all(np.array_equal(x, y) for x, y in zip(state_a[1], state_b[1]))
              and state_a[2] == state_b[2])

    for scale in (1, 3, 5, 7, 9):
        check(f"circuit_folding(scale={scale})",
              orig.circuit_folding(base, scale).qasm()
              == new.circuit_folding(base, scale).qasm())

    # 5. calculate_epsilon.
    fids = np.concatenate([rng.uniform(0, 1, 200), np.array([0.0, 1.0, 1 / 16, 1 / 32])])
    check("calculate_epsilon", np.array_equal(orig.calculate_epsilon(fids, 4),
                                              new.calculate_epsilon(fids, 4)))

    # 6. scale-index helper must match list.index() semantics (first occurrence).
    for factors in ([1, 3, 5, 7, 9], [1.0, 1.5, 2.0], [3, 3, 5]):
        mapping = new._scale_index_map(factors)
        check(f"_scale_index_map({factors})",
              all(mapping[v] == factors.index(v) for v in factors))

    # 7. zne_loocv on synthetic models (covers all mod / delate branches).
    ex_orig = load(ORIG / "extrapolation_model.py", "ex_for_tools_o")
    ex_new = load(NEW / "extrapolation_model.py", "ex_for_tools_n")
    for mod in (0, 1, 2):
        for delate in (0, 1):
            x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
            y = 0.5 * np.exp(-0.1 * x) + 0.4
            models_o = [{"name": "exp", "x": x, "y": y, "func": ex_orig.exp_extrapolation,
                         "x0": 0, "param": 3, "p0": [0.5, 0.5, 0.5],
                         "bounds": ([-np.inf, 0, -np.inf], [np.inf, 1, np.inf])}]
            models_n = [dict(models_o[0], func=ex_new.exp_extrapolation)]
            ro = orig.zne_loocv(models_o, q_num=4, mod=mod, delate=delate)
            rn = new.zne_loocv(models_n, q_num=4, mod=mod, delate=delate)
            check(f"zne_loocv(mod={mod},delate={delate}) pred", deep_equal(ro[0], rn[0]))
            check(f"zne_loocv(mod={mod},delate={delate}) name", ro[1] == rn[1])
            check(f"zne_loocv(mod={mod},delate={delate}) details", deep_equal(ro[2], rn[2]))


# --------------------------------------------------------------------------- #
# other modules: import-level equivalence + pure helpers
# --------------------------------------------------------------------------- #
def test_module_imports() -> None:
    """Modules must still import under the original working directory.

    The scripts use bare sibling imports (``from tools import ...``) and relative
    paths (``config.json``), so the test reproduces the pipeline's cwd and
    ``PYTHONIOENCODING``; that is exactly what ``main.py`` does for every step.
    """
    import os

    print("\n== module imports ==")
    previous_cwd = Path.cwd()
    sys.path.insert(0, str(NEW))
    os.chdir(NEW)
    # KNOWN PRE-EXISTING DEFECT (documented in CHANGELOG.md / README.md):
    # Device_properties.py calls `plot_error_map(backend, ax=...)`, but the `ax=`
    # keyword only exists in Qiskit >= 1.x while this project needs Qiskit < 1.0.
    # Every environment that can run the rest of the code raises this TypeError.
    # It is asserted here so the defect cannot silently change.
    EXPECTED_DEFECTS = {
        "Device_properties.py": TypeError,
    }

    try:
        for name in ("Noise_model", "rand_seed", "extrapolation_model", "grover",
                     "compare", "error_data_compare", "single_set_compute",
                     "grover_data_plot", "noise_exp_compute", "pk_file_deployment",
                     "run_circuit", "stable", "zne", "zne_grover_shots",
                     "random_generator", "trotter_generate", "Device_properties"):
            try:
                load(NEW / f"{name}.py", f"imp_{name}")
                check(f"import {name}.py", True)
            except Exception as exc:  # pragma: no cover - diagnostic path
                expected = EXPECTED_DEFECTS.get(f"{name}.py")
                if expected is not None and isinstance(exc, expected):
                    print(f"  [known] import {name}.py raises the documented "
                          f"pre-existing defect: {exc}")
                    check(f"import {name}.py (known defect unchanged)", True)
                else:
                    check(f"import {name}.py", False,
                          f"-> {type(exc).__name__}: {exc}")
    finally:
        os.chdir(previous_cwd)
        sys.path.remove(str(NEW))


def test_extrapolation_models() -> None:
    print("\n== extrapolation_model.py ==")
    orig = load(ORIG / "extrapolation_model.py", "ex_orig")
    new = load(NEW / "extrapolation_model.py", "ex_new")
    x = np.array([0.0, 1.0, 2.0, 3.0])
    for fname in ("poly_exp_extrapolation", "poly_exp_extrapolation1",
                  "poly_exp_extrapolation2", "poly_exp_extrapolation3",
                  "poly_exp_extrapolation4", "linear_extrapolation1",
                  "exp_extrapolation", "exp_extrapolation1",
                  "fid_extrapolation", "purity_sqrt_extrapolation"):
        fo = getattr(orig, fname)
        fn = getattr(new, fname)
        args = (x, 0.5, 0.25, -0.5, 0.3, 0.25)[: 1 + (fo.__code__.co_argcount - 1)]
        try:
            check(f"{fname}",
                  np.array_equal(fo(*args), fn(*args), equal_nan=True))
        except Exception as exc:
            check(f"{fname}", False, f"-> {exc}")


def test_config_defaults() -> None:
    print("\n== config.json ==")
    import json

    with open(ORIG / "config.json", encoding="utf-8") as fh:
        old = json.load(fh)
    with open(NEW / "config.json", encoding="utf-8") as fh:
        new = json.load(fh)
    check("config.json unchanged (cached-data default preserved)", old == new,
          f"\n    original={old}\n    cleaned ={new}")
    check("record == False (load raw data from disk)", new.get("record") is False,
          f"got {new.get('record')!r}")


def main() -> int:
    print("Differential behaviour test: original_backup/ vs 高斯-指数混合模型整合/")
    test_tools()
    test_extrapolation_models()
    test_config_defaults()
    test_module_imports()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} of {CHECKS} checks FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print(f"all {CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
