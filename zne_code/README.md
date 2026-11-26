# Hybrid Gaussian-exponential zero-noise extrapolation for periodic circuits

Reproducible working copy of the simulation code accompanying the paper
*"Hybrid Gaussian-exponential zero-noise extrapolation for periodic circuits"*
(submitted to **Quantum**).

```
zne_code/
├── main.py                  single sequential entry point  <-- start here
├── verify_behaviour.py      differential test: cleaned vs pristine  (323 checks)
├── requirements.txt         exact == pins
├── environment.yml          same pins for conda
├── README.md                this file
├── README_ENV.md            environment + why each pin
├── CHANGELOG.md             every change, and why it is behaviour-preserving
├── original_backup/         untouched originals (extracted from the .zip)
└── 高斯-指数混合模型整合/     the cleaned, working tree
    ├── config.json          run configuration (the only file you normally edit)
    ├── config.py            generator for config.json (only writes when run directly)
    ├── main-pipeline scripts
    │   ├── grover.py                    Grover circuit generator
    │   ├── random_generator.py          random circuit generator
    │   ├── trotter_generate.py          Trotterised-Ising circuit generator
    │   ├── pk_file_deployment.py        .pk  -> .qasm/.qpy export
    │   ├── run_circuit.py               circuit diagrams
    │   ├── noise_exp_compute.py         exact + noisy expectation values
    │   ├── zne.py                       ZNE, density-matrix path
    │   ├── zne_grover_shots.py          ZNE, shot-based path (Grover)
    │   ├── compare.py                   per-circuit error analysis
    │   ├── single_set_compute.py        error-distribution boxplot
    │   ├── error_data_compare.py        average error vs step figure
    │   ├── grover_data_plot.py          Grover summary figures
    │   ├── stable.py                    extrapolation stability histogram
    │   ├── Device_properties.py         fake-backend device tables/figures
    │   ├── Noise_model.py               shared FakeQuito noise model
    │   ├── tools.py                     folding / twirling / ZNE core
    │   ├── extrapolation_model.py       extrapolation ansatz functions
    │   └── rand_seed.py                 restartable RNG helper
    ├── example_random/      standalone path-sampling study
    │   ├── clifford_road_sample.py      (production: samples paths -> weigh_data/)
    │   └── Normality_Test.py            (analysis: normality of noise weights)
    ├── <family>_circuits/   circuits + cached results per circuit family
    ├── {random,}gen/        generated .pk circuit collections
    └── Device_properties/   device tables and figures
```

---

## Quick start

```bash
conda activate zne_qiskit          # see README_ENV.md if not created yet
cd E:/论文修改/zne_code

python main.py --list              # show the steps and the current run mode
python main.py --all               # run the cached-data pipeline (~20 s)
```

`--all` does **not** re-simulate anything: `config.json` has `record = false`, so
the ZNE steps load their raw data from disk and only redo the (cheap)
extrapolation and plotting.

---

## `main.py` usage

| Flag | Meaning |
|---|---|
| `--list` | Print every step, its group, the current pipeline mode, and which steps `--all` would run. |
| `--all` | Run the default pipeline: all `analysis` steps, in dependency order, skipping steps that do not apply to the current `cir_type`. |
| `--steps N,M,...` | Run exactly these step numbers (in registry order regardless of the order you type them). Inapplicable steps become a hard error. |
| `--from N` | Run every step from `N` up to 16. |
| `--skip N,M,...` | Remove steps from whatever was selected. |
| `--with-production` | With `--all`: also run the raw-data production steps (1-7). |
| `--with-utility` | With `--all`: also run the utility steps (16). |
| `--keep-going` | Continue with later steps after a failure instead of aborting. |
| *(no arguments)* | Interactive menu listing the steps and prompting for a selection (`all`, a list, or `q`). |

Real examples:

```bash
python main.py --list                     # inventory + mode
python main.py                            # interactive menu
python main.py --all                      # default cached-data pipeline
python main.py --steps 10,11,12           # only the three error-analysis steps
python main.py --from 8                   # steps 8..16
python main.py --all --skip 14            # pipeline without the stability step
python main.py --all --keep-going         # don't abort on the first failure
python main.py --steps 5,7                # re-export circuits, then recompute expectations
python main.py --all --with-production --with-utility   # literally everything
python main.py --steps 16                 # the device-characterisation utility
```

### Step list

| # | group | key | script | what it consumes -> produces |
|---|---|---|---|---|
| 1 | production | `gen-grover` | `grover.py` | nothing -> `grover_gen/step_*.pk` |
| 2 | production | `gen-random` | `random_generator.py` | nothing -> `random_gen/step_*.pk` |
| 3 | production | `gen-trotter` | `trotter_generate.py` | nothing -> `trotter_gen/step_*.pk` |
| 4 | production | `clifford-sample` | `example_random/clifford_road_sample.py` | `random_circuits/circuit_0.qasm` -> `weigh_data/*.txt` |
| 5 | production | `export-circuits` | `pk_file_deployment.py` | `*_gen/step_*.pk` -> `<family>_circuits_*/circuit_*.qasm/.qpy` |
| 6 | production | `draw-circuits` | `run_circuit.py` | `circuit_*.qasm` -> `pictures/circuit_0.png` |
| 7 | production | `noisy-expectation` | `noise_exp_compute.py` | `circuit_*.qasm` -> `results/expectation_values.txt` |
| 8 | analysis | `zne-density` | `zne.py` | cached `zne_separated_results_detailed.txt` -> `zne_separated_results.txt` |
| 9 | analysis | `zne-grover-shots` | `zne_grover_shots.py` | cached `..._detailed_readout_multi.txt` -> `zne_separated_results_readout.txt` |
| 10 | analysis | `error-analysis` | `compare.py` | `expectation_values.txt` + `zne_separated_results.txt` -> `error_analysis_results/detailed_error_analysis.txt` |
| 11 | analysis | `error-boxplot` | `single_set_compute.py` | same two files -> `comparison_results/*.png` |
| 12 | analysis | `error-vs-step` | `error_data_compare.py` | `error_analysis_results/*.txt` -> `compare_plot/average_error_vs_step_<cir_type>.png` |
| 13 | analysis | `grover-plots` | `grover_data_plot.py` | Grover `zne_separated_results*.txt` -> `grover_circuits*/compare_plot/*.png` |
| 14 | analysis | `stability` | `stable.py` | `grover_circuits_3/.../zne_separated_results_detailed.txt` -> `stable_results/*.png` |
| 15 | analysis | `normality-test` | `example_random/Normality_Test.py` | `weigh_data/*.txt` -> `log_norm_test/*.png` |
| 16 | utility | `device-properties` | `Device_properties.py` | FakeQuito/FakeLima -> `Device_properties/*.csv|.png` |

`production` steps are excluded from `--all` because their outputs are already in
the tree and re-running them re-simulates.  Steps 1, 2, 3, 8, 9 and 13 are
`cir_type`-specific; `--all` prints a loud `SKIPPED ... (not applicable)` line for
those, while asking for one explicitly is a hard error.

Every step prints a banner with its number, name, inputs, outputs, the effective
pipeline mode and the elapsed time; a missing input aborts with a clear message.
Steps run in their own subprocess, so `main.py` cannot leak state between them.

---

## Configuration

All run behaviour is in `高斯-指数混合模型整合/config.json`.  These are the exact
existing parameters that select "load cached data" vs "recompute".

| Parameter | `config.json` line | Shipped value | Meaning |
|---|---|---|---|
| **`record`** | 21 | **`false`** | **The cache/recompute switch.** `false` = the ZNE steps load the already-computed raw simulator data from `zne_separated_results_detailed*.txt` and only redo the extrapolation. `true` = re-run the simulators and overwrite that raw data. |
| `sample` | 20 | `false` | `false` = density-matrix ZNE (`zne.py`, step 8). `true` = shot-based ZNE with the sampled path (`zne_grover_shots.py`, step 9). |
| `readout_error` | 19 | `false` | `true` applies readout-error mitigation. The calibration matrix `M_inv` is only built when `record` is *also* `true`. |
| `cir_type` | 17 | `"grover"` | `random` \| `ising_trotter` \| `grover`. Selects the circuit family: which directory is read, which ZNE script is applicable, and which figures are written. |
| `num_twirling` | 6 | `16` | Pauli-twirling repetitions `T`. |
| `num_non-integer` | 7 | `1` | Repetitions for non-odd (fractional) fold factors. |
| `noise_scale_factors` | 13 | `[1,3,5,7,9]` | Fold factors; the count must match the column block width of the cached files. |
| `num_qubits` | 10 | `4` | Qubit count for the non-Grover families (Grover hardcodes 5). |
| `fit_method` | 8 | `1` | `1` fits only the first `param` points, `0` fits all. |
| `num_train` | 5 | `0` | LOOCV scheme: `0` = validate on the training set, `1` = leave-one-out, `2` = two held-out points. |
| `num_experiments` | 12 | `10` | Repeated experiments per circuit in the shot-based Grover path. |
| `row` | 11 | `3` | Lines to skip when reading `expectation_values.txt` (overridden to 10 for Grover). |
| `observable` | 14 | `"ZZZZ"` | Pauli observable for the non-Grover families. |
| `fold_method` | 9 | `0` | `0` = `uniform_gate_folding`, `1` = `circuit_folding`. **Note:** read from config but the ZNE scripts always call `uniform_gate_folding`; the value is effectively informational. |

### Flip back to full recomputation

```bash
# 1. edit config.json:  "record": true
# 2. re-run the producing steps as well:
cd E:/论文修改/zne_code
python main.py --with-production --all          # e.g. steps 5,6,7,8,9,...
# or, more targeted:
python main.py --steps 7,8              # recompute expectations, then ZNE (density path)
python main.py --steps 7,9              # recompute expectations, then ZNE (shot path)
```

Set `"cir_type"` to the family you want before doing so.  With
`record: true`, steps 8/9 re-simulate and **overwrite**
`zne_separated_results_detailed*.txt`; the previous contents are only preserved
in the original zip, so take a copy first if you care.

> Do **not** run `python config.py` unless you want `config.json` reset to the
> defaults -- that is what it does.  Nothing imports it any more.

---

## Seed inventory (do not disturb)

RNG call order is part of the reproducibility contract.  Full list:

| File:line | Seed / RNG | Role |
|---|---|---|
| `tools.py:271` | `np.random.default_rng(42)` | seeds the per-twirl seeds in `run_pauli_twirling_zne` |
| `tools.py:329` | `seed=42` (default arg) | default for `run_pauli_twirling_zne_shot` |
| `tools.py:348` | `np.random.default_rng(seed)` | shot-path twirl seeds |
| `tools.py:56` | global `np.random.random()` | fractional gate folding (no explicit seed -- uses whatever global state exists) |
| `tools.py:176,179` | `random.Random(seed)` | `twirl_circuit`, seeded from the draw above |
| `zne.py:255` | `random.Random(123)` | which 3 circuits get the correction factor |
| `zne_grover_shots.py:190` | `random.Random(123)` | same selection, Grover path |
| `zne_grover_shots.py:209` | `fixed_seed = 42` | per-circuit seed for the shot path |
| `stable.py:95-96` | `seed = 42`, `random.Random(42)` | the 50 initial fit guesses |
| `random_generator.py:18-30,170` | `fix_random_seed(42)` | called **at import time** |
| `trotter_generate.py:140,215-216` | `create_restartable_random_system(42)`, `rng.restart(seed)` | restarted per depth |
| `rand_seed.py:3,9` | `base_seed=42` -> `np.random.RandomState(42)` | wrapper default |
| `example_random/clifford_road_sample.py:469-470,514` | `seed = 123`, `np.random.default_rng(42)` | path sampling |
| `example_random/clifford_road_sample.py:80` | global `np.random.random()` | fractional folding (unseeded) |

Two places deliberately rely on the **global** `np.random` stream being in a
particular state (`tools.py:56` and `clifford_road_sample.py:80`).  Because those
code paths run under `np.random`'s default seeding, their output is *not*
reproducible across processes -- a pre-existing property of the original code,
not something introduced here.  It only affects the `record: true` recompute
path; the cached path never calls them.

---

## Circuit families and experiment structure

| Family | Entries | Circuits/entry | Depth parameter | ZNE script |
|---|---|---|---|---|
| `ising_trotter` | 18 (`ising_trotter_circuits_1..18`) | 200 | Trotter steps 0..17 | `zne.py` (density matrix) |
| `random` | 19 (`random_circuits_1..19`) | 200 | 2-qubit gate depth 0,2,4,...,36 | `zne.py` (density matrix) |
| `grover` | 5 (`grover_circuits_1..5`) | 1 | Grover iterations k = 1..5 | `zne_grover_shots.py` (shots) |

* **Folding**: `[1, 3, 5, 7, 9]`, uniform gate folding.
* **Twirling**: 16 Pauli-twirled repetitions averaged per fold factor
  (`T_TWIRL = 4` in the paper's later sibling script; the archive uses 16).
* **Fitting**: four models per circuit -- `poly_exp` (Grover:
  `poly_exp_extrapolation3`, others: `poly_exp_extrapolation2`), `exp`
  (`exp_extrapolation1` for Grover, `exp_extrapolation` otherwise), `fidelity`
  (linear in the error coordinate) and `purity_sqrt` -- fitted with
  `scipy.optimize.curve_fit` (`maxfev` 200k-300k), scored by LOOCV
  (`num_train = 0`), with predictions outside `|pred| > 1.2` rejected.
* **Shots**: Grover path uses 1,000,000 shots for expectation/fidelity; purity is
  always exact from the density matrix.
* **Grover observable**: `I_anc (x) |0..0><0..0|` on 4 data qubits (5 qubits total).

---

## Verification

Run the differential test (needs the pinned env):

```bash
conda activate zne_qiskit
cd E:/论文修改/zne_code
python verify_behaviour.py     # 323 checks, cleaned vs original_backup/
```

What was actually run and observed:

| Check | Result |
|---|---|
| `verify_behaviour.py` | **323 / 323 checks pass** |
| `main.py --all --keep-going` (cir_type `grover`) | completes in **20.6 s** |
| 177 archived `.txt` results, regenerated | **172 byte-identical** |
| grover `zne_separated_results_readout.txt` (paper values) | **byte-identical** to the archive, entries 1-4 |
| grover path, all steps 9-15 | all produce archive-identical text or pixel-identical figures |
| **ising_trotter** step 8 (density-matrix ZNE), 18 entries x 200 circuits | **11 / 18 entries byte-identical**; 7 entries (12-18) differ in **23 cells out of 3600** (0.64 %) -- see below |
| 5 grover `error_analysis_results/detailed_error_analysis.txt` | **differ only by an added "Average ..." block**; all per-circuit numbers identical. The archived files were written by an *older* `compare.py` that lacked that block -- confirmed by diffing against the pristine script. |
| 14 archived figures, regenerated | **14 / 14 pixel-identical** (MD5 differs only in PNG metadata) |
| pristine `compare.py` run in the new env vs cleaned | **byte-identical output** |
| pristine `zne_grover_shots.py` vs cleaned | **byte-identical output** |
| pristine `zne_loocv` vs cleaned `zne_loocv` on the 23 differing cells | **identical predictions** -> the difference is *not* caused by the code edits |

### The 23 non-reproducing cells (residual uncertainty)

For `ising_trotter` entries 12-18 the regenerated `zne_separated_results.txt`
differs from the archive in exactly 23 cells (4 circuits x 12-13, 4 x 14-19),
always in the `exp`, `fidelity` or `purity_sqrt` column.  Pattern:

* archive value = `y_data[0]`, i.e. the **fallback** `tools.zne_loocv` uses when
  the fitted prediction is rejected (`|pred| > value_bound = 1.2`) or the fit
  fails;
* regenerated value = the fitted prediction, always `|pred|` in **1.08 - 1.20**,
  i.e. just *below* the 1.2 cut.

What was ruled out:

1. **The code edits.**  Running the pristine `tools.zne_loocv` (from
   `original_backup/`) on the same inputs gives bit-identical predictions, and
   the pristine `compare.py` / `zne_grover_shots.py` give byte-identical output.
2. **The SciPy version.**  `1.11.4`, `1.12.0` and `1.13.1` all return the same
   (just-below-1.2) predictions for all 23 cells.
3. **A different `value_bound` at run time.**  Entries 1-11 contain 4 predictions
   inside `(1.0, 1.2]` and 37 above 1.2, and they reproduce *exactly* with
   `value_bound = 1.2`.  So the threshold cannot have been different.

Most likely explanation: those 7 archived files were produced by a slightly
different code revision (the archive's `__pycache__` still references modules
that no longer exist -- `pauli_twirling.py`, `paoli_twiling_zne.py` -- so the
fitting code was refactored while the results were being produced).  The values
the *archived* code produces are the regenerated ones.  All 18 entries'
`zne_separated_results_detailed.txt` **inputs** are byte-identical to the archive,
so nothing is lost by recomputing the 23 cells.

To inspect them:

```bash
conda activate zne_qiskit
cd E:/论文修改/zne_code
# compare the archive with a fresh run (take a backup first!)
python main.py --steps 8          # needs "cir_type": "ising_trotter"
```

Not verified (and why):

* The **recompute** paths (`record: true`) were never executed end-to-end: they
  take hours, and the task forbids re-running them.  Their code paths are
  covered only at the level of the differential test on the pure helpers.
* Steps 1-7 (circuit generation / export / noisy expectation values) were not
  re-run, because their outputs are already committed and regenerating them
  would re-simulate.  Their modules are import-verified only.
* The `random` family's step 8 was not re-run (same code path as
  `ising_trotter`, and its cached inputs are complete).
* `Device_properties.py` (step 16) cannot complete on *any* environment
  compatible with this code -- see below.

### Known data gap in the archive: `grover_circuits_5`

`zne_grover_shots.py` with `record: false` reads
`grover_circuits_<n>/zne_separated_results/zne_separated_results_detailed_readout_multi.txt`
for `n = 1..5`.  Entries 1-4 have that file; **entry 5 does not** (it only has the
single-experiment `zne_separated_results_detailed.txt`), so step 9 aborts at
entry 5 with `FileNotFoundError`.  This is a gap in the archived data, not a code
defect, and it reproduces identically with the pristine script.

Options:

```bash
python main.py --all --keep-going     # entries 1-4 already regenerated; continue
```

or use a family whose cache is complete (all 18 `ising_trotter` and all 19
`random` entries have their cached files), or re-run step 9 with
`"record": true` in `config.json` to regenerate the missing file.

### Known pre-existing defect: step 16 (`Device_properties.py`)

The script calls `plot_error_map(backend, ax=...)`.  The `ax=` keyword does not
exist in Qiskit < 1.0 -- the signature is
`plot_error_map(backend, figsize, show_title, qubit_coordinates)` -- while this
project *requires* Qiskit < 1.0 (`from qiskit import Aer`).  The call therefore
raises `TypeError` on every valid environment.  That is exactly why the archive
contains `Device_properties/fake_backends_layout.png` but **no**
`fake_backends_error_map.png`.  The CSV tables and the other three figures are
written before the crash.

It was **not** fixed, because fixing it changes behaviour.  To fix it yourself,
replace the two calls with separate figures, e.g.:

```python
plot_error_map(backend_quito); plt.title("FakeQuito Error Map"); plt.savefig(...)
```

`main.py` classifies it as a `utility` step, excludes it from `--all`, and prints
the explanation in `--list`.  Run it explicitly with `python main.py --steps 16`.

---

## What was optimised, and why it is safe

The analysis pipeline was already cheap (~20 s); the wins are in the recompute
path.  See `CHANGELOG.md` for the full reasoning.

1. **Cached the Pauli basis** (`tools.py`, `_pauli_basis` /
   `_pauli_matrix_by_label`, `functools.lru_cache`).
   `conjugate_pauli_by_gate` used to rebuild all `4**n` Pauli matrices with
   `np.kron` on *every* twirled two-qubit gate -- roughly `#2q_gates x T`
   rebuilds of 16 matrices per circuit, i.e. ~10^5 redundant `np.kron` calls per
   circuit set.  Pure function of `num_qubits`; identical arrays, built once.
2. **Hoisted the scale-factor index lookup** (`tools.py`, `_scale_index_map`).
   Three `noise_scale_factors.index(scale)` scans inside the result-extraction
   loops became one dict lookup.  `setdefault` reproduces `list.index()`'s
   first-match semantics exactly.
3. **Removed duplicate imports** (`tools.py`) and **an unexecuted module-level
   config write** (`config.py`, guarded behind `if __name__ == "__main__"`).
4. **Subprocess-per-step orchestration** (`main.py`): also a correctness win --
   no shared matplotlib/backend/numpy state between scripts, and `MPLBACKEND=Agg`
   prevents `plt.show()` from blocking a headless run.

Deliberately **not** done (would risk changing values): vectorising
`compute_pauli_expectation` or `calculate_epsilon` (float re-association),
merging the two `uniform_gate_folding` calls per fold (changes how many draws are
taken from the global `np.random` stream), and caching `curve_fit` results
(seeded by `p0`, not pure).
