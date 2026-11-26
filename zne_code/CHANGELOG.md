# CHANGELOG

Every change made to the archived project, and why it cannot alter a single
reported number.

Layout produced by this task:

```
zne_code/
├── main.py                 NEW   single sequential entry point
├── verify_behaviour.py     NEW   differential test (cleaned vs. pristine)
├── requirements.txt        NEW
├── environment.yml         NEW
├── README.md               NEW
├── README_ENV.md           NEW
├── CHANGELOG.md            NEW
├── original_backup/        NEW   pristine copies taken **from the .zip**
└── 高斯-指数混合模型整合/    the working (cleaned) tree
```

`original_backup/` was populated directly from
`Hybrid_Gaussian_exponential_zero_noise_extrapolation.zip`, not by copying the
working tree, so it is guaranteed pristine (29 files: every `.py`, `config.json`
and the `Device_properties/` tables).  Bulk data (`.pk`, `.qasm`, `.qpy`, the
multi-GB `pictures/` PNGs, `example_random/weigh_data/`) was deliberately **not**
duplicated: it is untouched in the working tree and is also preserved inside the
original zip, which was not modified.

---

## 1. Behaviour-relevant edits

### 1.1 `tools.py` — Pauli-basis caching

**Change.** Added two `functools.lru_cache`d helpers

* `_pauli_basis(num_qubits) -> (labels, matrices)` — builds the `4**n` Pauli
  strings and their matrices exactly as the original inline loop did
  (`mat = m if mat is None else np.kron(m, mat)`, i.e. left-nested `np.kron`),
* `_pauli_matrix_by_label(num_qubits) -> dict` — the label → matrix view.

`pauli_string_from_operator` and `conjugate_pauli_by_gate` now read from them
instead of rebuilding the whole basis on every call.

**Why it is behaviour-preserving.** Both helpers are pure functions of
`num_qubits`; they return the *same* `np.ndarray` objects that the original code
constructed, with the same dtypes (`np.eye(2)` float, `X`/`Z` int, `Y` complex)
and the same nesting order.  The callers only read (`np.conj(P.T) @ op_matrix`,
`gate_matrix @ P @ gate_matrix.conj().T`), never mutate, so sharing the arrays is
safe.

**Verified.** `verify_behaviour.py` compares the original and cleaned
implementations for 50 random Hermitian matrices per qubit count and for all
`4**2` labels x 9 gates (Clifford 1q and 2q).  All identical.

### 1.2 `tools.py` — hoisted scale-factor index lookup

**Change.** Added `_scale_index_map(noise_scale_factors)` and replaced the three
in-loop `noise_scale_factors.index(scale)` scans in
`run_pauli_twirling_zne` / `run_pauli_twirling_zne_shot`.

**Why it is behaviour-preserving.** `dict.setdefault` keeps the *first*
occurrence index, which is exactly the semantics of `list.index()`, so the
mapping is identical even for duplicated factors.  Only an `int` changes
provenance, never a float.

**Verified.** `verify_behaviour.py` asserts `_scale_index_map(f)[v] == f.index(v)`
for every value, including a duplicated-factor list.

### 1.3 `tools.py` — import consolidation and docstrings (no semantic change)

* The mid-file duplicate imports `import numpy as np`,
  `from qiskit import QuantumCircuit, Aer` and
  `from scipy.optimize import curve_fit`, plus the in-function
  `from collections import defaultdict`, were moved to the top of the module.
  Re-importing an already-imported module is a no-op, so removing the duplicates
  cannot change behaviour.  `SparsePauliOp` was promoted to the top-level
  `qiskit.quantum_info` import because it is still used by
  `run_pauli_twirling_zne_shot`.
* English module/function docstrings, type hints on the public helpers, and two
  WARNING comments were added.

### 1.4 `config.py` — the `config.json` write is now guarded

**Change.** The module-level

```python
with open("config.json", "w") as f:
    json.dump(params, f, indent=4)
```

became a `write_config()` function called only under
`if __name__ == "__main__":`.

**Why it is behaviour-preserving.** A grep over the whole tree confirms that
**nothing imports `config.py`** (`grep -rn "import config" --include=*.py` returns
no hits), so the statement only ever ran when the file was executed directly --
which still writes `config.json` exactly as before.  The guard prevents an
accidental `import config` from silently resetting a hand-edited `config.json`
(which would have flipped `record` back and triggered a full re-simulation).

`python config.py` still produces a byte-equivalent `config.json`
(same keys, same values, `indent=4`).

### 1.5 `Noise_model.py` — docstrings and type hints only

`create_fixed_noise_model()` is unchanged: same four fake backends are
constructed and `FakeQuito` is still the selected device.  A warning was added
that the V1 fake-backend API must not be "upgraded".

### 1.6 `extrapolation_model.py` — docstrings; dead string literals -> comments

**Change.** Module docstring, English docstrings for all ten ansatz functions,
and the three module-level triple-quoted blocks (`linear_extrapolation` and the
duplicate `purity_sqrt_extrapolation`) converted from string literals to `#`
comments.

**Why it is behaviour-preserving.** A bare string literal at module level is a
no-op expression statement; it defines nothing.  Converting it to a comment
removes an unexecuted expression and leaves the module namespace identical.

**Not changed:** every function signature, every parameter order, every formula.
Parameter order matters because `zne.py`, `stable.py` and
`zne_grover_shots.py` supply `p0` / `bounds` positionally.

**Verified.** `verify_behaviour.py` evaluates all ten functions on a common input
and requires exact array equality (`equal_nan=True`).

### 1.7 Module docstrings for the remaining scripts

An English module docstring was prepended to `compare.py`,
`error_data_compare.py`, `single_set_compute.py`, `grover_data_plot.py`,
`noise_exp_compute.py`, `pk_file_deployment.py`, `run_circuit.py`, `stable.py`,
`zne.py`, `zne_grover_shots.py`, `Device_properties.py`, `random_generator.py`,
`trotter_generate.py`, `rand_seed.py`.

Nothing else in those files was touched.  Leading blank lines were stripped; the
rest of each file is byte-identical to the original.

---

## 2. Deliberate NON-changes (things that look like bugs but were kept)

| Location | Observation | Why it was kept |
|---|---|---|
| `tools.py` `run_pauli_twirling_zne_shot`, the `folded_base` / `fid_base` pair | `uniform_gate_folding(twirled, scale)` is called **twice** per (twirl, scale, rep) | The two calls are *independent*: for fractional scale factors the function draws from the global `np.random` stream. Merging them would consume a different number of draws and change every downstream number. A WARNING comment was added. |
| `tools.py` `compute_pauli_expectation`, `calculate_epsilon` | Explicit Python loops | Vectorising re-associates floating-point additions; the last bits of the reported values would change. |
| `zne_grover_shots.py` `load_previous_results()` | Reads the globals `circuit_dir` / `row1` instead of taking parameters | It works because both globals are assigned earlier in `__main__`. "Fixing" the signature risks changing the read offset. Documented in the module docstring. |
| `rand_seed.py` `test_restartable_system()` | Ends with a recursive self-call (infinite recursion) | Dead code -- the function is never called. Left untouched. |
| `zne.py` line ~220 | Builds a `NoiseModel` even when `record=False` (where it is unused) | Removing the construction would be a behaviour change; the cost is ~0.1 s. Left as is and noted as a possible future optimisation. |
| `Device_properties.py` `plot_error_map(backend, ax=...)` | `ax=` does not exist in any pre-1.0 Qiskit | **Known pre-existing defect.** It crashes on every environment compatible with the rest of the code, which is why the archive contains `fake_backends_layout.png` but no `fake_backends_error_map.png`. Left untouched; documented and surfaced by `main.py --list`. |

No random seed was added, removed, moved, or re-valued.  See
README.md §"Seed inventory" for the full list.

---

## 3. Environment fixes (not code changes)

The archived tree could not run under a current Qiskit.  Nothing in the project
code was changed for this; instead the environment was pinned to the last Qiskit
release that supports the API the code actually calls.  Details and the evidence
are in `README_ENV.md`.  Three visualisation dependencies were additionally
installed so that plotting steps work:

* `pylatexenc` (needed by `QuantumCircuit.draw('mpl')` in `run_circuit.py`)
* Graphviz binaries (needed by `plot_error_map` in `Device_properties.py`)
* `seaborn` (needed by `plot_error_map` in `Device_properties.py`)

These only affect whether a figure can be rendered; they do not enter any
numerical result.

---

## 4. Files added for reproducibility / verification

* `main.py` — orchestrator.  It shells out to the **unmodified** scripts, one
  subprocess per step, with the original working directory, so running a step
  through `main.py` is the same as running the script by hand.  It exports
  `MPLBACKEND=Agg` (the scripts call `plt.show()` *after* `plt.savefig()`; with an
  interactive backend that blocks the pipeline forever) and
  `PYTHONIOENCODING=utf-8` (the scripts print non-ASCII characters such as `⟩`,
  which raises `UnicodeEncodeError` on a GBK Windows console).
* `verify_behaviour.py` — differential test, cleaned modules vs.
  `original_backup/`.  323 assertions.
* `requirements.txt`, `environment.yml`, `README.md`, `README_ENV.md`.

---

## 5. Verification summary

| Check | Result |
|---|---|
| `verify_behaviour.py` (differential, cleaned vs pristine) | **323/323 checks pass** |
| Cached-path pipeline, `main.py --all --keep-going` (grover) | completes in **~21 s** |
| 177 archived result `.txt` regenerated | **172 byte-identical** |
| 5 grover `detailed_error_analysis.txt` | per-circuit numbers identical; the archive predates the "Average ..." block.  Pristine `compare.py` run in the new env produces a **byte-identical** file to the cleaned one. |
| `ising_trotter` step 8, 18 entries x 200 circuits | **11/18 entries byte-identical**; 23 cells out of 3600 differ (0.64 %) |
| 14 archived figures regenerated | **14/14 pixel-identical** (PNG bytes differ only in metadata/timestamps) |
| `grover_circuits_1..4/zne_separated_results/zne_separated_results_readout.txt` | **byte-identical** to the archived paper values; also identical between pristine and cleaned code |
| pristine vs cleaned `zne_loocv` on the 23 differing cells | **identical predictions** |

The 23 differing `ising_trotter` cells are a pre-existing inconsistency in the
archived data, **not** a consequence of any edit here:

* in all 23 the archive stores `y_data[0]` -- the fallback `tools.zne_loocv` uses
  when `|pred| > value_bound (1.2)` -- while the regenerated prediction is
  `|pred|` in 1.08-1.20, just under the threshold;
* the pristine and cleaned implementations return bit-identical predictions on
  those very inputs, so the edits are excluded as a cause;
* scipy 1.11.4 / 1.12.0 / 1.13.1 all give the same just-below-1.2 values, so the
  scipy version is excluded as well;
* `value_bound` is excluded: entries 1-11 contain 4 predictions inside
  `(1.0, 1.2]` and 37 above 1.2, and they still reproduce exactly at 1.2;
* the archive's `__pycache__` references three modules that no longer exist
  (`pauli_twirling.py`, `paoli_twiling.py`, `paoli_twiling_zne.py`), so the
  fitting code was refactored while these results were produced.

Full detail: README.md §"Verification".
