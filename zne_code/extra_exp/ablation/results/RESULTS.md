# Gaussian-term ablation — results

Study run **entirely on cached data** (no re-simulation). Configuration taken
from `config.json`: `mod = num_train = 0`, `delate = fit_method = 1`,
`noise_scale_factors = [1, 3, 5, 7, 9]`, `num_qubits = 4`. The LOOCV, the
`|pred| > 1.2` rejection rule and the `y_data[0]` fallback are `tools.zne_loocv`
unchanged; the error aggregation is `compare.py`'s (`|prediction − exact|` against
`results/expectation_values.txt`).

## 0. The variants

**Eq. 94 family (`random`, `ising_trotter`)** — only the `poly_exp` entry of the
model list is swapped; data, coordinates, bounds convention, LOOCV and error
aggregation are untouched.

| | formula | params | what it removes |
|---|---|---|---|
| `exp` | `a*b^x + c` (shipped ensemble member) | 3 | reference |
| **A** | `a*exp(c*x + b*x**2)` | 3 | the whole second branch |
| **D** | `(a + b*x)*exp(c*x)` | 3 | **the Gaussian `c₂x²`** |
| **B** | `a*exp(c*x + b*x**2) + d` | 4 | the second branch (constant kept) |
| **E** | `a*exp(c*x + b*x**2) + d*exp(c*x)` | 4 | the linear factor `e*x` of the 2nd branch |
| **C** | `a*exp(c*x + b*x**2) + (d + e*x)*exp(c*x)` = `poly_exp_extrapolation2`, Eq. 94 | 5 | — (the shipped model) |

**D is exactly Eq. 94 with the Gaussian switched off**: `poly_exp_extrapolation2`
at `b = 0` is `(a+d+e·x)e^{cx}`, whose two constant amplitudes merge into one.
A alone (remove the second branch) cannot answer "is the Gaussian necessary?";
A **and** D together can, in both directions.

**Grover** is re-centred on its actual shipped model (§5).

Reports are the **shipped pipeline** numbers: for non-Grover families the
prediction is multiplied by the mean modification factor `zne.py` applies to the
`poly_exp` column (`mod_factor` in `per_circuit.csv`; raw output = `pred/mod_factor`).
Grover never applies that factor.

## 1. What was actually run

| family | entries | circuits | command | wall time |
|---|---|---|---|---|
| `random` | 19 (1–19) | 3800 | `--cir-type random --control-full-fit` | ~13 min |
| `ising_trotter` | 18 (1–18) | 3600 | `--cir-type ising_trotter --control-full-fit` | ~11 min |
| `grover` | 5 (1–5) | 5 | `--cir-type grover --control-full-fit` | ~7 s |

(`--jobs 8` on this 28-thread machine; the control costs roughly 1.8× the plain
run. `ising_trotter_circuits_19` and `grover_circuits_6` hold no `.qasm` and no
cache; they are reported and skipped.)

## 2. Overall numbers

### `random` — 3800 circuits

| variant | params | mean \|err\| | median | std | p90 | p99 | max | rejected (\|pred\|>1.2) |
|---|---|---|---|---|---|---|---|---|
| `exp` (ref) | 3 | 0.0222213 | 0.00743581 | 0.0647208 | 0.0434804 | 0.259767 | 1.06894 | — |
| A (Gauss only) | 3 | 0.0234624 | 0.00457346 | 0.0841708 | 0.0415567 | 0.390575 | 1.21161 | 107 (2.82 %) |
| **D (affine×exp, c₂=0)** | 3 | **0.0110288** | 0.00365979 | 0.0377046 | 0.0243701 | 0.0989389 | 1.1978 | 15 (0.395 %) |
| B (A + const) | 4 | 0.0210010 | 0.00431523 | 0.0771773 | 0.0362998 | 0.309299 | 1.1878 | 152 (4.00 %) |
| E (Gauss + const branch) | 4 | 0.0187091 | 0.00622701 | 0.0508372 | 0.0444397 | 0.189742 | 1.11324 | 52 (1.37 %) |
| **C = Eq. 94** | 5 | **0.00875834** | **0.00321434** | **0.0244394** | **0.0188175** | **0.0829368** | **0.751109** | **6 (0.158 %)** |
| SHIPPED (= C) | 5 | 0.00875834 | 0.00321434 | 0.0244394 | 0.0188175 | 0.0829368 | 0.751109 | — |
| unmitigated | — | 0.0849897 | 0.0538735 | 0.0918008 | 0.219498 | 0.377815 | 0.634983 | — |

### `ising_trotter` — 3600 circuits

| variant | params | mean \|err\| | median | std | p90 | p99 | max | rejected (\|pred\|>1.2) |
|---|---|---|---|---|---|---|---|---|
| `exp` (ref) | 3 | 0.0294523 | 0.00323796 | 0.0829124 | 0.0734525 | 0.411263 | 1.05243 | — |
| A (Gauss only) | 3 | 0.0322178 | 0.00344404 | 0.0897903 | 0.0839352 | 0.465016 | 1.15083 | 36 (1.00 %) |
| **D (affine×exp, c₂=0)** | 3 | **0.0168968** | 0.00307441 | 0.0476697 | **0.0437237** | 0.169248 | 1.19014 | 1 (0.028 %) |
| B (A + const) | 4 | 0.0397055 | 0.00284052 | 0.116645 | 0.0989504 | 0.695186 | 1.24656 | 112 (3.11 %) |
| E (Gauss + const branch) | 4 | 0.0209725 | 0.00395364 | 0.052109 | 0.0556945 | 0.195045 | 1.04181 | 4 (0.111 %) |
| **C = Eq. 94** | 5 | **0.0156133** | **0.00237208** | **0.0326505** | 0.0495135 | **0.16266** | **0.377823** | **0** |
| SHIPPED (= C) | 5 | 0.0156133 | 0.00237208 | 0.0326505 | 0.0495135 | 0.16266 | 0.377823 | — |
| unmitigated | — | 0.0962888 | 0.049392 | 0.122334 | 0.25435 | 0.575249 | 0.854605 | — |

### `grover` — 5 circuits (1 per entry; indicative only)

| variant | params | mean \|err\| | median | std | p90 | p99 | max |
|---|---|---|---|---|---|---|---|
| A (G-gaussNoC) | 3 | 0.0509506 | 0.056157 | 0.0428507 | 0.0991674 | 0.115668 | 0.117502 |
| B (G-gaussOnly) | 4 | **0.0343864** | 0.00863193 | 0.0513552 | 0.0909735 | 0.13137 | 0.135859 |
| C (Eq. 94 — *not* shipped here) | 5 | 0.0685401 | 0.0758199 | 0.0461015 | 0.118458 | 0.124558 | 0.125236 |
| D (G-noGauss) | 4 | 0.0744713 | 0.0678179 | 0.0453118 | 0.125215 | 0.129695 | 0.130193 |
| `exp` (double exp.) | — | 0.0779938 | 0.0705109 | 0.0456981 | 0.128312 | 0.130465 | 0.130705 |
| **SHIPPED = G-full (Eq. 99)** | 5 | **0.0353303** | 0.0138126 | 0.0484686 | 0.088978 | 0.126531 | 0.130704 |
| unmitigated | — | 0.315341 | 0.342672 | 0.205409 | 0.541812 | 0.586776 | 0.591772 |

`summary.csv` in each folder also carries the same statistics **per entry**
(19 / 18 / 5 blocks) plus the `ALL` row.

---

## 3. Does the Gaussian term matter?

**Short answer: it helps, but it is not what drives the result.** The `c₂x²`
Gaussian is a secondary, tail-level refinement; the *affine × exponential
structure* does the heavy lifting, and it does so **without any Gaussian at all**.

**The complementary ablation.** On `random`, deleting the Gaussian (C → D) raises
the mean absolute error from 0.00875834 to 0.0110288 — a factor of **1.26** — and
on `ising_trotter` from 0.0156133 to 0.0168968, a factor of only **1.08**. For
comparison, deleting the second branch instead (C → A) raises it to 0.0234624
(×2.68) and 0.0322178 (×2.06). Measured as the excess over C, the Gaussian is
worth **about one sixth** of what the second branch is worth on `random`
(0.0022705 vs 0.0147041) and **about one thirteenth** of it on `ising_trotter`
(0.0012835 vs 0.0166045).

**A plain affine-exponential already does most of the job.** D has *no* Gaussian
and only 3 parameters, yet on `random` it beats the shipped plain exponential by
2.0× (0.0110288 vs 0.0222213) and beats both Gaussian-based reduced forms A
(0.0234624) and B (0.0210010); on `ising_trotter` it beats the exponential by 1.7×
(0.0168968 vs 0.0294523) and again beats A and B. Whatever the hybrid model is
buying, most of it is available from `(a + b·x)e^{cx}` alone.

**The affine factor is worth more than the Gaussian.** E and D differ by exactly
one design choice: E keeps the Gaussian `e^{bx²}` but reduces the second branch to
a constant, D keeps the affine prefactor `(a + b·x)` but drops the Gaussian.
D wins on both families — 0.0110288 vs 0.0187091 (`random`, ×1.70) and 0.0168968
vs 0.0209725 (`ising_trotter`, ×1.24). The linear factor inside the exponential
prefactor is the more valuable of the two ingredients; the Gaussian is not.

### C vs D head-to-head

| | `random` (3800) | `ising_trotter` (3600) |
|---|---|---|
| C mean / D mean | 0.00875834 / 0.0110288 | 0.0156133 / 0.0168968 |
| mean ratio C/D | **0.794** | **0.924** |
| C median / D median | 0.00321434 / 0.00365979 | 0.00237208 / 0.00307441 |
| C p90 / D p90 | **0.0188175** / 0.0243701 | 0.0495135 / **0.0437237** |
| C p99 / D p99 | **0.0829368** / 0.0989389 | **0.16266** / 0.169248 |
| C max / D max | **0.751109** / 1.1978 | **0.377823** / 1.19014 |
| C rejected / D rejected | **6** / 15 | **0** / 1 |
| fraction of circuits where **D beats C** | 44.9 % (1705/3800) | **52.1 %** (1877/3600) |
| fraction of circuits where **C beats D** | 55.1 % (2095/3800) | 47.9 % (1723/3600) |
| entries where C has the lower mean | **16 / 19** | **12 / 18** |

The per-entry breakdown is in `C_vs_D_by_entry.csv` (every entry plus an `ALL`
row, with the same nine quantities).

**Reading of the table.** On `random` the Gaussian earns its place: C is better on
the mean, median, p90, p99 and max, on 55 % of individual circuits and on 16 of 19
depths. On `ising_trotter` the case is much weaker: C is better on the mean by only
8 % and on the median by 23 %, but **D is better on p90** (0.0437 vs 0.0495) and
**D wins on the majority of individual circuits (52.1 %)**. C's `ising_trotter`
advantage is concentrated almost entirely in the extreme tail — `max` 0.3778
versus 1.1901, i.e. the Gaussian is what stops a handful of extrapolations from
blowing up.

**Where the Gaussian pays off.** There is no clean "the Gaussian starts to pay off
beyond depth *k*" threshold. C's per-entry wins are not monotone in depth: on
`random` D wins at entries 7, 13 and 19 (steps 12, 24 and 36) but loses at every
other depth, including the deepest; on `ising_trotter` D wins at entries 2, 5, 6,
12, 17 and 18. The benefit is not depth-structured, it is *outlier*-structured —
it appears wherever a particular 5-point fit would otherwise run away.

**Robustness.** In neither family is C ever the *worst* variant at any entry
(0/19 and 0/18 entries), whereas A is worst at 10 and 3 entries, B at 4 and 9, and
E at 5 and 5. So while the Gaussian's average contribution is modest, C is the only
variant that is never the weakest choice.

**Not a fit-budget artefact.** `delate = 1` fits C on 5 points but D on only 3.
The control in §7 gives D all 5 points: D's mean becomes 0.012506 (`random`,
*worse*) and 0.0168995 (`ising_trotter`, unchanged), while C is 0.00875834 and
0.0156133. The C/D ratio moves from 0.794 → 0.700 on `random` and stays at 0.924
on `ising_trotter`. The verdict therefore survives, and on `random` it strengthens.

**Verdict.** The `c₂k²` Gaussian correction is **empirically helpful but not
necessary, and not the main driver of the reported gain**. A three-parameter affine
exponential leaves a mean error only 26 % larger than C's on `random` and 8 %
larger on `ising_trotter`, and it recovers 83 % / 91 % of the whole advantage that
the hybrid has over the shipped plain exponential — while containing no Gaussian at
all. One can drop the Gaussian and still beat `a·b^x + c` by 1.7–2.0×.
What the Gaussian genuinely buys is robustness: it lowers
the worst-case error by 1.6× (`random`) and 3.1× (`ising_trotter`) and essentially
eliminates `|pred| > 1.2` rejections (6 vs 15, and 0 vs 1). A referee asking whether
the Gaussian is *the* source of the improvement should be told plainly: no — it is
the affine-exponential structure of Eq. 94 that is load-bearing, and the Gaussian is
a useful stabiliser on top of it.

## 4. The other direction — is the second branch necessary?

Yes, emphatically. Removing it (C → A) is the single most destructive change in the
whole study: ×2.68 on `random`, ×2.06 on `ising_trotter`. And A is barely better
than the plain exponential — 0.0234624 vs 0.0222213 (`random`) and 0.0322178 vs
0.0294523 (`ising_trotter`) — so **the Gaussian-exponential term on its own carries
no advantage over `a·b^x + c`**. Adding a constant back (B) does not help: 0.0210010
and 0.0397055, the latter being worse than the plain exponential.

Ranking of the five reduced forms, by mean absolute error:

| | `random` | `ising_trotter` |
|---|---|---|
| best | C (0.00875834) | C (0.0156133) |
| 2nd | D (0.0110288) | D (0.0168968) |
| 3rd | E (0.0187091) | E (0.0209725) |
| 4th | B (0.0210010) | `exp` (0.0294523) |
| 5th | `exp` (0.0222213) | A (0.0322178) |
| worst | A (0.0234624) | B (0.0397055) |

`random`: C ≺ D ≺ E ≺ B ≺ exp ≺ A. `ising_trotter`: C ≺ D ≺ E ≺ exp ≺ A ≺ B.
The ordering of A, B and `exp` is unstable across families; only the top three
(C, D, E) are consistently ordered, and D is always second.

## 5. Grover — re-centred on the model Grover actually ships

`zne.py` uses `poly_exp_extrapolation3` = `(a + e·x)·exp(c·x + b·x²) + d`
(Eq. 99) for Grover, **not** Eq. 94, so the earlier C-based Grover comparison was
not apples-to-apples. Re-run against the shipped model as baseline:

| model | mean \|err\| | median | max |
|---|---|---|---|
| **G-full = SHIPPED (Eq. 99, 5p)** | **0.0353303** | 0.0138126 | 0.130704 |
| G-gaussOnly = B (4p) | 0.0343864 | 0.00863193 | 0.135859 |
| G-gaussNoC = A (3p) | 0.0509506 | 0.056157 | 0.117502 |
| G-noGauss = D (4p) | 0.0744713 | 0.0678179 | 0.130193 |
| Eq. 94 = C (5p, not shipped) | 0.0685401 | 0.0758199 | 0.125236 |
| `exp` (double exponential) | 0.0779938 | 0.0705109 | 0.130705 |
| unmitigated | 0.315341 | 0.342672 | 0.591772 |

Removing the Gaussian from the *shipped* Eq. 99 model (G-full → G-noGauss) more than
doubles the mean error, 0.0353303 → 0.0744713, and keeping only the Gaussian term
(G-gaussNoC) gives 0.0509506. So for Grover — on five circuits, **indicative only** —
both ingredients appear to matter, and the Gaussian looks more important here than it
does on `random`/`ising_trotter`. This does not contradict §3: Eq. 99 and Eq. 94 are
different functional families, and with n = 5 the spread across circuits is of the
same order as the differences (std 0.043–0.051). The published Grover number is
reproduced exactly by `SHIPPED`, not by C (C differs from the archive by up to 9.4e-02).

## 6. Baseline reproduction — variant C vs the published numbers

C is the shipped `poly_exp_extrapolation2`, so on the non-Grover families the
`SHIPPED` column and the C column are **bit-identical** (`max|SHIPPED − C| = 0.0`
over 3800 and 3600 circuits).

| family | circuits compared | agree with archived `Poly_exp ZNE` at 1e-6 | max \|Δ\| |
|---|---|---|---|
| `random` | 3800 / 3800 | **3800** | 5.0e-07 |
| `ising_trotter` | 3600 / 3600 | **3600** | 5.0e-07 |
| `grover` (`SHIPPED` = `poly_exp_extrapolation3`) | 5 / 5 | **5** | 4.3e-07 |

Independent cross-check against the archived per-entry averages in
`<circuit_dir>/error_analysis_results/detailed_error_analysis.txt`
(`Average Gaussian_exp ZNE Error:`): all 19 `random` and all 18 `ising_trotter`
entries agree, max \|Δ\| = 5e-07.

**Verdict: numerically identical to the archived baseline.** The 5e-07 residual is
exactly the 6-decimal rounding of the archive (`0.469736` vs our `0.46973643`), so
1e-6 is the strongest achievable agreement and byte-identity is not definable
against these files.

## 7. Robustness control — is it the term, or the number of fit points?

`delate = 1` fits A and D on 3 points, B and E on 4, C on 5. The control re-fits
**every** variant on all 5 points (`--control-full-fit`), then applies the same
modification-factor treatment. Files: `control_fullfit_summary.csv`,
`control_fullfit_per_circuit.csv`.

| variant | `random` (shipped budget) | `random` (5 fit pts) | `ising_trotter` (shipped) | `ising_trotter` (5 fit pts) |
|---|---|---|---|---|
| A | 0.0234624 | 0.0242635 | 0.0322178 | **0.0286100** |
| B | 0.0210010 | 0.0213331 | 0.0397055 | 0.0384538 |
| C | 0.00875834 | 0.00875834 | 0.0156133 | 0.0156133 |
| **D** | 0.0110288 | **0.0125060** | 0.0168968 | **0.0168995** |
| E | 0.0187091 | 0.0177034 | 0.0209725 | 0.0200335 |

Giving the reduced variants the full data budget does not rescue them. **The
C-vs-D verdict in particular is unaffected**: the C/D mean ratio goes from 0.794 to
0.700 on `random` (C's advantage *grows*) and stays at 0.924 on `ising_trotter`.
D's own mean gets worse under the control on `random` (0.0110288 → 0.0125060) and is
unchanged on `ising_trotter`. The Gaussian's contribution cannot be dismissed as a
fit-budget artefact of the three extra parameters.

## 8. Caveats

* `grover_circuits_5/.../zne_separated_results_detailed_readout_multi.txt` is
  missing (entries 1–4 have it). That file belongs to `zne_grover_shots.py`
  (shot-based, readout-mitigated), not to the `zne.py` pipeline this study mirrors;
  `zne.py`'s own cache exists for entries 1–5, so Grover ran on all 5 circuits. The
  script falls back to the readout_multi file only if the primary one is absent, and
  prints a notice when it does.
* `ising_trotter_circuits_19` and `grover_circuits_6` are empty (no `.qasm`, no
  cache); reported and skipped, not worked around.
* Grover has n = 5; every Grover statement here is indicative only.
* On the two shallowest entries of each family (entries 1–2) the ranking is
  unsettled and C is not always the best; see `C_vs_D_by_entry.csv`.
* Numbers are displayed rounded; `summary.csv` carries six significant digits,
  `per_circuit.csv` and `C_vs_D_by_entry.csv` ten.
