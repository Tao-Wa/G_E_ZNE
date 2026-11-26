# `path_normality_sweep` — depth sweep of the Pauli-path log-normality test

Standalone extension of the paper's Theorem-1 normality check
(`example_random/Normality_Test.py` + `example_random/clifford_road_sample.py`)
from **one depth to several depths**.

This folder is **additive**: nothing in the original code tree
(`zne_code/高斯-指数混合模型整合/`) is modified, moved or deleted. The code here
only *imports* from it (read only).

---

## 1. What it does

The paper's Theorem 1 says that for a periodic circuit the noise amplification
factor `W` accumulated along a random Pauli transfer path is log-normal,
i.e. `ln W ~ Normal`. The paper verifies it empirically at one depth by

1. sampling many Pauli paths of one circuit (backward propagation of the
   observable, a random 50/50 branch at every non-Clifford rotation),
2. turning each path into a *Clifford* circuit, simulating it under the
   `FakeQuito` noise model and reading off `W_alpha = |Tr(rho_noisy · O)|`,
3. forming self-normalised importance weights `w ∝ 1/q̃` (`q̃` = product of the
   branching probabilities along the path, the file called `pauli_weight`),
4. fitting a Gaussian to `ln W` and reporting the **weighted quantile
   correlation coefficient `r`** on the grid `p = linspace(0.01, 0.99, 500)`.

`run_sweep.py` repeats that estimator at several depths, where the depth is the
number of equidistant **Trotter periods `m`** of the same Trotter step and the
same Hamiltonian parameters.

It also reports, per depth:

| metric | meaning |
|---|---|
| `r` | the paper's weighted quantile correlation, on the identical 500-point grid |
| `ks_weighted` | sup-distance between the **importance-weighted ECDF** of `ln W` and the fitted normal CDF (see §5 for the exact definition — it is *not* a null-hypothesis KS test) |
| `mu`, `sigma` | weighted mean and standard deviation of `ln W` (the fitted Gaussian parameters) |
| `ess` | effective sample size `1 / sum(w²)` after self-normalisation |
| `sigma2_over_m` | `sigma²/m` — Theorem 1 predicts this tends to a constant as `m` grows |

---

## 2. Layout

```
extra_exp/path_normality_sweep/
├── run_sweep.py            # standalone entry point (CLI)
├── fast_sampler.py         # NEW: batched rewrite of the path sampler (see §6)
├── README.md               # this file
├── circuits/               # Trotter circuits actually used
│   ├── trotter_m<m>.qasm
│   ├── trotter_m<m>.qpy
│   └── manifest.json       # J, dt, h, m, n_qubits, observable, basis, seed,
│                           # qasm sha256, depth, gate counts
└── results/
    ├── raw/                # RAW SAMPLES, kept: raw_m<m>.npz (one per depth)
    ├── metrics.csv         # one row per depth
    ├── figures/            # qq_m<m>.png, sweep_r_ks_vs_m.png, sigma2_over_m_vs_m.png
    ├── RESULTS.md          # auto-generated report of the last analysis
    └── paper_reference/    # (extra, outside the requested layout) the depth-18
                            # random-circuit run used as evidence in §8
```

`--outdir` (default: this folder) can point anywhere; the tree above is created
inside it on demand.

---

## 3. How to run

Use the existing conda interpreter (nothing is installed or changed):

```bash
PY="C:/Users/wangt/anaconda3/envs/zne_qiskit/python.exe"
cd "E:/论文修改/zne_code/extra_exp/path_normality_sweep"
```

**Smoke test (≈3–5 min with `--sampler original`, ≈20 s with the default vector
sampler):**

```bash
"$PY" run_sweep.py --quick
```

**Full depth sweep (default `m = 3, 6, 9, 12, 15`):**

```bash
"$PY" run_sweep.py --sampler vector --depths 3,6,9,12,15 \
    --n-samples 20000 --seed 20261008 --resume
```

**Resume / re-run only what is missing:**

```bash
"$PY" run_sweep.py --resume ...        # skips any depth whose results/raw/raw_m<m>.npz exists
```

**Recompute metrics + figures from the stored raw samples (no re-sampling):**

```bash
"$PY" run_sweep.py --analyse-only
```

**Cross-check the fast sampler against the untouched original sampler:**

```bash
"$PY" run_sweep.py --validate-sampler --depths 3 --batch-size 5000 --validate-calls 6
```

**Re-run the paper's own depth-18 random circuit with this pipeline:**

```bash
"$PY" run_sweep.py --sampler vector \
    --circuit-file "E:/论文修改/zne_code/高斯-指数混合模型整合/example_random/random_circuits/circuit_0.qasm" \
    --label 18 --n-samples 100000 --outdir results/paper_reference
```

### CLI reference

| flag | default | meaning |
|---|---|---|
| `--depths` | `3,6,9,12,15` | Trotter periods `m` (comma separated) |
| `--j` | `0.37454` | Ising coupling `J` (`0.37454 = np.random.RandomState(42).uniform(0,1)`) |
| `--dt` | `pi/15` | Trotter step; float **or** a literal such as `pi/15`, `2*pi/3` |
| `--h` | `1.0` | transverse field |
| `--n-qubits` | `4` | qubit count |
| `--observable` | `ZZZZ` | measured Pauli string |
| `--basis` | `ZZZZ` | measurement basis per qubit (`ZZZZ` ⇒ pure Trotter layers, perfectly periodic) |
| `--n-samples` | `200000` | accepted paths per depth (`--sampler vector`) / requested samples (`--sampler original`) |
| `--seed` | `20261008` | RNG seed, logged in the console, in `manifest.json`, in each `.npz` and in `RESULTS.md` |
| `--sampler` | `vector` | `vector` (fast) or `original` (untouched imported function) |
| `--batch-attempts` | `4096` | attempts per batched round (`vector`) |
| `--batch-size` | `500` | requested samples per call to the original sampler |
| `--sim-batch` | `2000` | circuits per noisy-simulator run |
| `--quick` | off | `n_samples=2000, batch_size=100, sim_batch=500, stall_batches=1` |
| `--resume` | off | skip depths whose raw file already exists |
| `--analyse-only` | off | rebuild `metrics.csv`, figures and `RESULTS.md` from `results/raw/` |
| `--saturation-stop` / `--no-saturation-stop` | on | stop early when the reachable path set is exhausted (`--sampler original` only) |
| `--circuit-file`, `--label` | – / `18` | run one external QASM circuit with a depth label |
| `--outdir` | this folder | output root |

### Seeds

* **`--seed` default `20261008`** — used only for the Pauli-path sampler
  (`np.random.RandomState(seed)`, or `rand_seed.create_restartable_random_system(seed)`
  for the original sampler). It is re-seeded identically for every depth, so the
  per-depth streams are reproducible. It is printed to stdout, stored in
  `circuits/manifest.json`, in the `meta_json` blob of every `raw_m<m>.npz`, and
  written into `results/RESULTS.md`.
* No pre-existing seed in the original tree was changed. The original
  `clifford_road_sample.process_circuit_batch` (seed 123) and
  `random_generator.fix_random_seed(42)` are untouched and unused here.

---

## 4. Raw data format

One file per depth: `results/raw/raw_m<m>.npz` (`numpy.savez_compressed`).
**The samples are kept, not thrown away** — `--analyse-only` needs nothing else.

| array | dtype | content |
|---|---|---|
| `W` | float64 | `W_alpha = |Tr(rho_noisy · O)|` for every sample |
| `qtilde` | float64 | the unnormalised path probability, i.e. the original's `pauli_weight` (product of the branching probabilities; an exact power of 2) |
| `k` | float64 | `-log2(qtilde)` = number of branching decisions along the path |
| `path_id` | int64 | index of the distinct Pauli path that produced the sample |
| `meta_json` | str | JSON: `m`, `J`, `dt`, `h`, `n_qubits`, `observable`, `basis`, `seed`, `sampler`, counts, timings |

For the `--sampler original` path the arrays are the same, but `pauli_weight`
is the original's `weight` field and `weigh` (`probability`) is not stored
(both are recomputable from the path data; only `weight` is needed by the
estimator).

---

## 5. Weighted KS statistic — exact definition

Sort the `ln W` values: `x_(1) ≤ … ≤ x_(n)`. With the self-normalised
importance weights `w_i` (including a `W > 0` mask, as in the paper's script) let

```
C_i = sum_{j<=i} w_j / sum_j w_j
```

be the importance-weighted ECDF, and `F` the CDF of the fitted Normal
`N(mu, sigma)`. We report

```
D = max_i max( |C_i - F(x_(i))| , |C_(i-1) - F(x_(i))| ),
```

i.e. the sup-distance between the weighted ECDF and the fitted normal, evaluated
at both the left and right limit of every step.

**Caveat (stated explicitly because it matters).** This is *not* the
Kolmogorov–Smirnov test in its null-hypothesis sense. The importance weights are
not uniform, so the classical KS null distribution (and its p-value) does not
apply, and the parameters `mu`, `sigma` are estimated from the same sample that
is being tested. `D` is used only as a descriptive goodness-of-fit statistic,
directly comparable across depths because it is computed the same way
everywhere. A useful scale is `1/sqrt(ESS)`: values of `D` around that scale are
consistent with sampling noise rather than genuine non-normality.

---

## 6. Reused vs reimplemented (and why)

**Imported unchanged from the paper tree** (`sys.path` is extended to
`zne_code/高斯-指数混合模型整合` and its `example_random/` subfolder):

| what | from |
|---|---|
| Trotter circuit (the paper's own generator) | `trotter_generate.IsingModel.Options` / `.make_circs_sweep` |
| Pauli path sampler | `example_random.clifford_road_sample.sample_path_with_observable` |
| path → Clifford circuit | `example_random.clifford_road_sample.clifford_list_to_circuit` |
| gate folding at scale 1 | `example_random.clifford_road_sample.uniform_gate_folding` |
| RNG wrapper | `rand_seed.create_restartable_random_system` |
| noise model | `Noise_model.create_fixed_noise_model` (FakeQuito) |
| simulator | `qiskit_aer` `AerSimulator(method="density_matrix")` |
| estimator | ported line by line from `example_random/Normality_Test.py`: same `log_w = -log(qtilde)`, same max-subtraction stabilisation, same self-normalisation, same weighted `mu`/`sigma`, same `p = linspace(0.01, 0.99, 500)`, same `np.interp(p, cumsum(w), logW)` weighted quantiles, same `np.corrcoef` |

**New file `fast_sampler.py`.** The original sampler costs ≈50 ms per sampled
path at `m = 15` (a pure-Python loop over ~240 instructions per attempt, ≈4 %
acceptance); a 5-depth sweep at a useful sample count would take many hours.
`fast_sampler.VectorSampler` therefore processes attempts in NumPy batches.

It is a **speed-only rewrite**, and this is how it is kept faithful:

* **Every lookup table is produced by calling the original functions.**  The
  Clifford propagation tables come from `propagate_pauli_through_gate`; the
  rotation branch tables (new Pauli letter, replacement gate, accumulated
  amplitude, accumulated `p2`) come from `sample_rz`/`sample_rx`/`sample_ry`
  called with a stub RNG pinned to `u = 0.0` (branch 0) and `u = 0.999999`
  (branch 1); the QASM line formatting (`fmt_line`) is copied verbatim from
  `sample_path_with_observable`. Nothing about the gate algebra, the branch
  probabilities or the replacement-gate rules is transcribed by hand.
* **What changed (all plumbing):**
  1. *RNG stream layout.* One `rng.uniform(0,1,size=B)` vector is drawn per
     non-Clifford instruction instead of one scalar per attempt. Every draw is
     still an i.i.d. `U(0,1)` used with the same thresholds
     (`idx = 0 if u < 0.5 else 1`, or `idx = 0 if u < probs[0] else 1` in the
     near-deterministic case, which is also implemented), so each attempt's path
     distribution is unchanged; only *which* uniform goes to which attempt
     changes. A draw is also taken for attempts whose Pauli letter does not
     branch (the sequential code consumes none there).
  2. *No intra-call `seen_paths` de-duplication.* The original returns distinct
     paths inside one call; the batched version returns i.i.d. draws. Since the
     multiplicity of a path is ∝ `q̃` under i.i.d. sampling and (in the
     many-call limit) also ∝ `q̃` for the de-duplicated scheme, applying
     `w ∝ 1/q̃` gives the same self-normalised weighted distribution — which is
     the only object the estimator consumes.
  3. *No `max_attempts` cap.* Rejected attempts are discarded and the batch
     continues, instead of dropping the requested sample after 300 attempts.
  4. No early pruning: validity is tested exactly once, at the end of the full
     backward pass, as in the original.
  5. The trivial per-sample work of the original is kept in the driver:
     `clifford_list_to_circuit` + `uniform_gate_folding(circ, 1)` +
     `save_density_matrix` + one `AerSimulator.run` per batch, with the same
     `W = |Re Tr(rho O)|`.

**Validation.** `run_sweep.py --validate-sampler` performs two checks per depth
(a path is identified by its QASM line sequence, the object the estimator
consumes):

1. **Exact support (ground truth).** `exact_signature()` computes the reachable
   path DAG by dynamic programming over `(instruction, Pauli string)` states —
   only `4**n_qubits` states per instruction, so it is exact and instant even
   when the number of paths is ~1e19. It returns the true number of reachable
   valid paths and the exact range of the branching count `k = -log2(qtilde)`.
2. **One-sided containment.** The original sampler is run to saturation
   (repeated calls until its distinct-path set stops growing) and the batched
   sampler is asked for many accepted paths. Every path the batched sampler
   emits must be reachable via the original sampler, and `qtilde` must agree on
   the intersection.

Note that the two samplers use *independent* random streams (documented
deviation 1), so their individual samples are different paths and a set-equality
test would be the wrong check; what must agree is the support, the `k` range and
`qtilde`.

Measured result for `m = 3`
(`--validate-sampler --depths 3 --batch-size 800 --validate-calls 12
--validate-vector-samples 30000`):

```
m=3: EXACT support |D|=504  k in [10, 18]
       original : 504 distinct paths found (344.6 s), k in [10, 18]
       vector   : 504 distinct paths found (0.7 s),  k in [10, 18]
       paths emitted by vector but NOT reachable via the original sampler: 0   (must be 0)
       qtilde mismatches on the intersection (504 paths): 0
```

Both samplers recover the *complete* reachable support, with identical `qtilde`
on every path. The batched sampler is ≈500× faster.

---

## 7. Runtime

Measured on this machine (Python 3.11 / qiskit 0.46.3 / qiskit-aer 0.14.2),
per depth, `--sampler vector --n-samples 20000`
(these are the numbers recorded in `results/RESULTS.md`):

| m | circuit instructions | sampling | noisy simulation | total |
|---|---|---|---|---|
| 3 | 47 | 0.45 s | 40.7 s | 41 s |
| 6 | 95 | 1.37 s | 72.0 s | 73 s |
| 9 | 143 | 1.41 s | 111.1 s | 113 s |
| 12 | 191 | 2.23 s | 121.1 s | 123 s |
| 15 | 239 | 4.53 s | 188.0 s | 193 s |

Full 5-depth sweep: **550 s wall (≈9 min)**.
The `--circuit-file` reference run at depth 18 with `--n-samples 100000` took
**992 s (≈16.5 min)**, of which 977 s was the noisy simulation.

The noisy simulation (`AerSimulator`, density matrix + FakeQuito noise model,
≈2.6 ms per 47-gate circuit up to ≈9 ms per 239-gate circuit) now dominates;
sampling is no longer a bottleneck. With `--sampler original` the same work
costs ≈290 ms per path at `m = 3` and ≈50 ms per path at `m = 15`, i.e. tens of
minutes per depth.

**Note on sample count.** The effective sample size saturates and does *not*
grow with `--n-samples` (see `results/RESULTS.md`): the weights `w = 2^k` have a
very wide dynamic range for the Trotter circuit, so the weighted empirical
distribution is controlled by a handful of paths. Raising `--n-samples` from
2 000 to 20 000 left `ESS` at ≈25 for `m = 15`. Increasing `N` therefore buys
almost nothing here, which is why the default `--n-samples` for the shipped run
is 20 000.

---

## 8. Relation to the paper's reported `r = 0.9992` (important)

The paper's Q-Q figure is introduced in the **"Random circuits"** subsection
(`quantum-template.tex`, Sec. `sec:level4.2`) with the caption
*"Quantile-Quantile (Q-Q) plot at two-qubit depth of 18"*, and the text says
*"we sampled the noise scaling factors over paths, obtaining 2e6 samples"*.
The archived data of that experiment is
`example_random/weigh_data/noise_random_10_1_0.txt` together with
`pauli_weight_random_10_1_0.txt` (1 999 987 samples each), and it belongs to the
**random-circuit** family (`example_random/random_circuits/circuit_0.qasm`,
produced by `random_generator.py` at two-qubit gate depth 18), *not* to the
Trotterised-Ising family.

So the single depth at which the paper reports normality is a **2-qubit gate
depth of 18 in the random-circuit family**, whereas the sweep requested here is
over the **number of Trotter periods `m`**. The two axes are different circuits.

Two things were done about this:

1. **Estimator equivalence check — DONE, PASSES.** Running the estimator of
   `run_sweep.py` on the paper's archived arrays
   (`example_random/weigh_data/noise_random_10_1_0.txt` +
   `pauli_weight_random_10_1_0.txt`, 1 999 987 samples) reproduces
   **r = 0.99917 ≈ 0.9992** exactly (same weights, same fitted `mu`/`sigma`,
   same 500-point grid). The estimator in this folder is the paper's estimator.
   This check runs automatically and is reported in `results/RESULTS.md`.
2. **The shipped circuit is NOT the source of the archived arrays.** The
   `--circuit-file` mode re-runs `example_random/random_circuits/circuit_0.qasm`
   through this pipeline; it gives `r = 0.918` at depth 18, not 0.9992. The
   reason is a provenance mismatch, not an estimator difference:

   | | archived arrays (2e6 samples) | `example_random/random_circuits/circuit_0.qasm` (100 k samples) |
   |---|---|---|
   | sampled branching count `k = -log2(qtilde)` | 20 … 36 (17 values) | 20 … 91 |
   | `sd(k)` | ≈ 1.7 (implied by `ESS/N = 0.186`) | much larger |
   | `r` | **0.99917** | 0.99089 |
   | `ESS / N` | 18.6 % | 0.05 % |
   | 2-qubit gate depth | – | 18 |

   More samples can only *widen* a sampled `k`-range, so the archived arrays —
   with 20× more samples — cannot come from `circuit_0.qasm`, whose `k` reaches
   91. An exact DAG scan (`exact_signature`, no sampling) over the whole shipped
   families `random_gen/step_0…36` and `random_circuits_1…19` also finds no
   circuit whose `k`-distribution matches. **The archived samples were produced
   from a circuit that is no longer in the repository**, so the paper's
   `r = 0.9992` point can only be verified by re-analysing the archived arrays
   (done above), not by re-sampling a shipped circuit. The
   `results/paper_reference/` folder keeps the evidence.
   Outstanding next step for the user: locate the original circuit (search
   `random_gen/*.pk`, `trotter_gen/*.pk`, `ising_trotter_circuits/`, or an
   external archive).

The Trotter sweep in `results/` is the *new* axis (20 000 samples per depth,
seed 20261008):

| m | r | KS | ESS | `sigma²/m` |
|---|---|---|---|---|
| 3 | 0.98657 | 0.1001 | 7793 | 0.00108 |
| 6 | **0.99638** | 0.0422 | 1281 | 0.00190 |
| 9 | 0.99146 | 0.0701 | 300 | 0.00259 |
| 12 | 0.96222 | 0.1300 | 41.2 | 0.00441 |
| 15 | 0.98762 | 0.1405 | 27.0 | 0.00368 |

`r` peaks at **0.9964** for `m = 6` and does not reach 0.9992. The reason is
statistical power, not a failure of Theorem 1: the branching-count spread — and
with it the dynamic range of `w = 2^k` — grows with `m`, so `ESS` collapses
(7793 → 27) and the weighted quantile estimate becomes noise-limited. At every
depth `KS` is at or below `1/sqrt(ESS)` (e.g. `m = 9`: `KS = 0.070` vs
`1/sqrt(300) = 0.058`; `m = 15`: `KS = 0.141` vs `1/sqrt(27) = 0.192`), i.e.
consistent with sampling noise rather than a genuine departure from normality.
Raising `--n-samples` from 2 000 to 20 000 left `ESS` unchanged at `m = 15`
(≈25), so more sampling does not fix this with the present estimator.

`sigma²/m` is *not* flat over `m = 3…15`: it rises from 0.00108 to 0.00441 and
then dips to 0.00368 — with only 5 points and the two largest-`m` values carrying
`ESS ≈ 41` and `27`, no convergence claim can be made at these depths.

---

## 9. Reproducibility checklist

* Python `C:/Users/wangt/anaconda3/envs/zne_qiskit/python.exe`
  (qiskit 0.46.3, qiskit-aer 0.14.2, numpy 1.26.4, scipy 1.13.1, matplotlib 3.9.4)
* Simulator and noise model unchanged: `AerSimulator(method="density_matrix")`
  + `Noise_model.create_fixed_noise_model()` (FakeQuito, V1 fake backend).
* Trotter circuits come from `trotter_generate.IsingModel` with the paper's
  `config_4q_paper()` defaults, overridden by `--j/--h/--dt/--n-qubits`.
  `circuits/manifest.json` records a SHA-256 of every QASM file.
* Seed: `--seed 20261008` (default), identical for every depth, logged
  everywhere and written into `results/RESULTS.md`.
* Raw samples are kept in `results/raw/`; `--analyse-only` rebuilds every metric
  and figure from them alone.
