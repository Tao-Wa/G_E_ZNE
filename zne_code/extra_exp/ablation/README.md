# Gaussian-term ablation for the hybrid Gaussian-exponential ZNE model

Referee-driven ablation study: **which term of the fitted ZNE ansatz (Eq. 94) is
responsible for the reported improvement?**

This folder only *adds* files. Nothing in
`zne_code/高斯-指数混合模型整合/` is modified, moved or deleted; `config.json` and
all random seeds are untouched, and **nothing is re-simulated** — the study reads
the cached raw data (`zne_separated_results_detailed*.txt`) and the cached exact
values (`results/expectation_values.txt`) that `zne.py` and `compare.py` already
produced, and re-runs only the cheap extrapolation step.

## What is compared

The shipped `poly_exp` entry of the model list is replaced by a reduced ansatz.
Data, noise coordinates, bounds convention, LOOCV, model selection and the error
aggregation are all unchanged.

### Eq. 94 family — `random`, `ising_trotter`

| Variant | Formula | Params | What it removes |
|---|---|---|---|
| **A** | `a*exp(c*x + b*x**2)` | 3 | the second branch |
| **B** | `a*exp(c*x + b*x**2) + d*x*exp(c*x)` | 4 | the constant part of the second branch (its linear factor is kept) |
| **C** | `a*exp(c*x + b*x**2) + (d + e*x)*exp(c*x)` — shipped `poly_exp_extrapolation2` (Eq. 94) | 5 | — |
| **D** | `(a + b*x)*exp(c*x)` | 3 | **the Gaussian `c₂x²`** |
| **E** | `a*exp(c*x + b*x**2) + d*exp(c*x)` | 4 | the linear factor `e*x` of the 2nd branch (its constant is kept) |

`b` is the Gaussian coefficient (c₂) and `c` the linear coefficient (c₁), matching
`extrapolation_model.py`. A is the referee's first suggestion ("retain only the
`a₀e^{c₂k²+c₁k}` term"). **D is the complementary ablation**: it is exactly
Eq. 94 with c₂ = 0, since `poly_exp_extrapolation2(x,a,b,c,d,e)` at `b = 0` is
`(a+d+e·x)e^{cx}`, whose two constant amplitudes merge. A alone cannot tell you
whether the Gaussian is necessary — A **and** D together can.

`p0`/`bounds` of A, B and E are the *prefix* of the shipped ones
(`p0 = [0.5, 0.01, -0.5, 0.3, 0.25]`,
`bounds = ([-inf, 0, -inf, -inf, -inf], [inf, inf, 0, inf, inf])`). D is the one
exception, because switching the Gaussian off changes what the parameters mean
(`a` becomes the merged amplitude, `b` the affine slope); it uses
`p0 = [0.8, 0.25, -0.5]`, `bounds = ([-inf, -inf, -inf], [inf, inf, 0])`.

### The `grover` family is re-centred on `poly_exp_extrapolation3`

`zne.py` ships `poly_exp_extrapolation3` = `(a + e*x)*exp(c*x + b*x**2) + d`
(paper Eq. 99) for Grover, so ablating Eq. 94 there is not apples-to-apples:

| Variant | Formula | Params |
|---|---|---|
| **A** — G-gaussNoC | `a*exp(c*x + b*x**2)` | 3 |
| **B** — G-gaussOnly | `a*exp(c*x + b*x**2) + d` | 4 |
| **D** — G-noGauss | `(a + b*x)*exp(c*x) + d` | 4 |
| **C** | Eq. 94 (`poly_exp_extrapolation2`) — *not* shipped for Grover | 5 |
| **SHIPPED** — G-full | `(a + e*x)*exp(c*x + b*x**2) + d` | 5 |

Here **SHIPPED is the baseline** (it is G-full, the published Grover model).
For `random`/`ising_trotter`, SHIPPED and C are the same model list and are
bit-identical; SHIPPED exists so the published baseline can be reproduced on
every family. Grover has n = 5 circuits (one per entry), so that family is
**indicative only**.

## Usage

```bash
PY=C:/Users/wangt/anaconda3/envs/zne_qiskit/python.exe

"$PY" ablate_gaussian_term.py --list
"$PY" ablate_gaussian_term.py --cir-type random
"$PY" ablate_gaussian_term.py --cir-type ising_trotter
"$PY" ablate_gaussian_term.py --cir-type grover
"$PY" ablate_gaussian_term.py --cir-type random --entries 1,2,3
```

| Flag | Meaning |
|---|---|
| `--cir-type {random,ising_trotter,grover}` | circuit family. Required, except when only `--list` is used; if omitted, the value from `config.json` is used (currently `grover`) and a notice is printed. |
| `--entries 1,2,3` | restrict to entry numbers (default: every entry that has cache) |
| `--outdir DIR` | output root (default `results/`) |
| `--list` | print available families, entries and which cached files exist, then exit |
| `--jobs N` | worker processes (default `min(cpu_count, 8)`; `--jobs 1` forces serial) |
| `--variant A,B,D` | evaluate a subset (default: every variant the family defines; Grover has no E) |
| `--max-circuits N` | debugging only — the modification factor then no longer matches `zne.py` |
| `--control-full-fit` | additionally run the robustness control described below |
| `--figures-only` | redraw the figures from an existing `per_circuit.csv` without recomputing |

If a family's cache is incomplete the script prints exactly which file is
missing for which entry and exits cleanly (exit code 1) instead of crashing.

## Outputs (per family, under `results/<cir_type>/`)

| File | Content |
|---|---|
| `per_circuit.csv` | one row per circuit: exact, noisy, and for each variant (A, B, C, D, E) the prediction, absolute error, `\|pred\| > 1.2` rejection flag and which model LOOCV selected; plus the `SHIPPED`, `exp`, `fidelity` and `purity_sqrt` reference columns |
| `summary.csv` | the same statistics aggregated per entry **and** overall (`entry = ALL`): mean / median / std / p90 / p99 / max of the absolute error, number and rate of `\|pred\| > 1.2` rejections, and how often LOOCV picked `poly_exp` |
| `C_vs_D_by_entry.csv` | the C-vs-D head-to-head, per entry and overall: mean/median/p90/p99/max, rejection counts, `mean(C)/mean(D)`, and the fraction of circuits where D beats C |
| `baseline_check.csv` | per entry: how far the `SHIPPED` prediction is from the archived `zne_separated_results.txt` (`Poly_exp ZNE` column) |
| `mean_abs_error_vs_step.png` | mean absolute error against the step index for every variant + the shipped `exp` member + unmitigated |
| `boxplot_step_<k>.png` | per-circuit absolute-error boxplot at the most-populated (median) step |
| `control_fullfit_*.csv` | only with `--control-full-fit` (see below) |

`results/RESULTS.md` holds the written-up numbers.

## Baseline reproduction check

`baseline_check.csv` compares the `SHIPPED` prediction with the archived
`zne_separated_results.txt` column 1 (`Poly_exp ZNE`), which is what the
published error analysis in `compare.py` / `error_data_compare.py` consumes.
Matching to `1e-6` is the strongest achievable agreement, because the archive
stores only 6 decimals.

## `--control-full-fit` (optional robustness control)

`zne_loocv` with `delate = 1` fits only the first `param` points, so the shipped
configuration fits A and D on 3 points, B and E on 4, and C on 5 — the reduced
variants are handicapped in the data budget as well as in the functional form.
`--control-full-fit` re-evaluates every variant with `param` left at 5 (which
changes only how many points the final fit sees, since `mod = 0` does not
otherwise use `param`), so all of them see the same 5 points. It isolates the
effect of the **functional form** from the effect of the **fit-point budget**.
Its results go to `control_fullfit_per_circuit.csv` /
`control_fullfit_summary.csv` (additional files per family, next to the primary
outputs). This matters most for the **C-vs-D verdict**, since D has fewer
parameters than C.

## Reused vs reimplemented

Reused verbatim from the simulation tree (read-only imports):

* `tools.zne_loocv` — the LOOCV selection and the `|pred| > 1.2` rejection rule.
* `extrapolation_model.poly_exp_extrapolation2` (variant C / `SHIPPED`),
  `poly_exp_extrapolation3` (grover `SHIPPED`), `exp_extrapolation`,
  `exp_extrapolation1`, `fid_extrapolation`, `purity_sqrt_extrapolation`.

Copied verbatim (identical code, no behavioural change) so that `zne.py` need
not be imported at run time:

* `natural_sort_key`, `load_previous_results`, `compute_modification_factor`
  (`compute_modification_factor` takes `len(qc.data)` directly instead of a
  `QuantumCircuit`, and the circuits are still loaded from the cached `.qasm`
  files exactly as `zne.py` does).
* the model-list construction of `get_extrapolation_models` and the error
  aggregation of `compare.py`.

Deliberate, documented improvements that cannot change any number:

* `--jobs` runs the per-circuit extrapolations in parallel; the arithmetic is
  bit-identical to a serial run (`--jobs 1` reproduces it exactly).
* for `random` / `ising_trotter` the `SHIPPED` list and variant C are the same
  list, so they are evaluated once instead of twice.
* the `print` chatter inside `tools.zne_loocv` is suppressed; scipy
  `OptimizeWarning`s are not suppressed (redirect them if you wish).
