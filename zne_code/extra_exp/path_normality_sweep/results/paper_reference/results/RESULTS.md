# Depth sweep of the Pauli-path log-normality test

Generated: 2026-10-08T07:32:53+00:00 (UTC)

## Configuration

- depths (Trotter periods) `m` : [3, 6, 9, 12, 15]
- J = 0.37454, h = 1.0, dt = 0.20943951023931953 (pi/15 as given), n_qubits = 4
- observable = ZZZZ, measurement basis = ZZZZ
- **seed = 20261008** (used for the path sampler only)
- sampler = `vector` (fast_sampler.VectorSampler, batched)
- samples per depth actually used = same for all depths: 100000 (CLI --n-samples = 100000); per-call batch = 500, sim batch = 2000
- saturation stop = True (stop after 2 consecutive calls with no new path)
- noise model: FakeQuito (via `Noise_model.create_fixed_noise_model`)
- simulator: `qiskit_aer` AerSimulator(method='density_matrix')

## Estimator (identical to the paper)

Self-normalised importance weights (port of `example_random/Normality_Test.py`):

```
log_w = -log(qtilde); log_w -= max(log_w); w = exp(log_w); w /= sum(w)
```

Weighted Gaussian fit on the log scale: `mu = sum(w log W)`, `sigma = sqrt(sum(w (log W - mu)^2))`.

Weighted sample quantiles by `np.interp(p, cumsum(w_sorted), logW_sorted)` on `p = linspace(0.01, 0.99, 500)`; `r` is the Pearson correlation between the theoretical normal quantiles and the weighted sample quantiles (the paper's 0.9992 metric).

**Weighted KS statistic.** With `x_(i)` the sorted `ln W` values, `C_i = sum_{j<=i} w_j / sum_j w_j` the weighted ECDF and `F` the CDF of the fitted normal, we report

```
D = max_i max( |C_i - F(x_(i))| , |C_(i-1) - F(x_(i))| )
```

This is the sup-distance between the *importance-weighted* ECDF and the fitted normal. It is **not** the standard two-sample KS test: the importance weights are not uniform, so the usual KS null distribution and p-value do not apply. `D` is reported as a descriptive goodness-of-fit statistic only.

**Reference.** The paper quotes `r = 0.9992` in the Q-Q figure. That figure is introduced in the *random-circuit* subsection (caption 'Q-Q plot at two-qubit depth of 18'), not in the Trotter subsection.

### Direct estimator check on the paper's own archived arrays

Source: `example_random\weigh_data\noise_random_10_1_0.txt + example_random\weigh_data\pauli_weight_random_10_1_0.txt` (1999987 samples). Same estimator, same weight scheme, same 500-point grid:

| quantity | value |
|---|---|
| **r (weighted quantile correlation)** | **0.99917** |
| weighted KS statistic | 0.01764 |
| mu | -0.92948 |
| sigma | 0.19126 |
| ESS | 372541.2 |
| distinct k = -log2(qtilde) | 20..36 (17 values) |

This reproduces the paper's quoted `r = 0.9992` to 4 decimals, so the estimator in this script is the paper's estimator. (The small residual difference, 0.99917 vs 0.9992, is the paper's rounding.)

**Caveat found while doing this check (please read).** The archived arrays are *not* reproducible from the circuit shipped alongside them (`example_random/random_circuits/circuit_0.qasm`): the archived `pauli_weight` values span `k = -log2(qtilde)` in [20, 36], whereas that circuit yields k in [28, 82] with the untouched original sampler (and [20, 91] with `fast_sampler`). A circuit of two-qubit depth 18 cannot produce the archived k-range, so the archived data came from a different, smaller circuit than the QASM file stored next to it. The `r = 0.9992` point is therefore only checkable by re-analysing the archived arrays (done above), not by re-sampling the shipped circuit. The `--circuit-file` run into `results/paper_reference/` demonstrates this: it gives r = 0.918 at depth 18 instead of 0.9992, precisely because the circuit is not the source of the archived arrays.


The *reachable* `k`-support (exact, from the path DAG, not sampled) of the archived data is `k_min = 20`; no shipped random circuit attains `k_min > 13` (a scan over all 19 depths gives 0,7,6,4,5,8,2,6,11,5,11,12,12,13,8,12,8), and `k` scales linearly with depth, so a two-qubit depth of 18 cannot give `k_min = 20` either. Conclusion: the archived arrays were produced from a circuit that is no longer in the repository. The estimator is verified; the *provenance* of the archived samples is not.


**Sampling scheme.** `fast_sampler.VectorSampler` draws i.i.d. paths (multiplicity proportional to `qtilde`) and keeps every accepted path, i.e. the weights `w = 1/qtilde` again give a uniform-over-paths measure, as in the original de-duplicated sampler. `n_batches` is 0 because the batched sampler does not use per-call batches.

**Effective sample size.** `ESS = 1/sum(w^2)` shrinks quickly with depth: the weights `w = 2^k` have a dynamic range that grows with `m` (the spread of the branching count grows like sqrt(m) in absolute terms), so the weighted empirical distribution becomes controlled by a few paths. A useful scale for the KS column is `1/sqrt(ESS)`; values of `KS` at or below it are consistent with sampling noise. At the depths where `ESS` is still large (m = 6, 9) the normal fit is excellent (`r >= 0.995`); at larger `m` the statistic is noise-limited rather than showing a genuine departure from normality. Raising `--n-samples` from 2 000 to 20 000 left `ESS` unchanged at m = 15 (~25), so more sampling does not fix this with the present estimator.

## Results

| m | samples used | distinct paths | saturated | r | KS (weighted) | mu | sigma | ESS | sigma^2/m |
|---|---|---|---|---|---|---|---|---|---|
| 18 | 100000 | 99992 | no | 0.990886 | 0.07884 | -0.48153 | 0.02329 | 51.5 | 0.00003 |

## Figures

- `results\figures/qq_m18.png` - QQ plot of ln W at m = 18
- `results\figures/sweep_r_ks_vs_m.png` - r and the weighted KS statistic vs m, with the r = 0.9992 reference line
- `results\figures/sigma2_over_m_vs_m.png` - sigma^2/m vs m

## Raw data

One file per depth: `results/raw/raw_m<m>.npz`, written with `numpy.savez_compressed`, containing

- `W`       : float64 array, W_alpha = |Tr(rho_noisy O)| for every accepted path
- `qtilde`  : float64 array, the unnormalised path probability (= `pauli_weight`, the product of the branching probabilities, a power of 2)
- `k`       : float64 array, `-log2(qtilde)` (number of branching decisions)
- `path_id` : int64 array, index of the distinct Pauli path that produced the sample
- `meta_json` : JSON string with the full run metadata (J, dt, h, m, observable, seed, counts, timings)

`--analyse-only` recomputes every metric and figure from these files alone.

## Runtime

- m = 18: sampling 8.7 s, simulation 976.7 s

