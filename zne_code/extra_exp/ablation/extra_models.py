# -*- coding: utf-8 -*-
"""Two extra ansaetze, tested on the same cached data (no re-simulation).

    F:  b * x * exp(c*x)            2 params   ("D without its constant")
    G:  b * x * exp(c*x) + d        3 params

Everything else follows the ablation: the five cached points x = [1,3,5,7,9],
the same p0/bounds convention, the same ``|pred| > 1.2 -> y[0]`` rejection rule,
and the same three-reference-circuit modification factor (''Random(123)''),
each model using its own predictions.

Note: at the mitigation point x = 0 the pure form F returns exactly 0, so its
prediction is 0 irrespective of the fit; that is expected and is reported as is.

    python extra_models.py            # writes results_extra/summary.csv
"""
import csv
import os
import random
import sys

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir,
                                     "高斯-指数混合模型整合"))
sys.path.insert(0, HERE)
import ablate_gaussian_term as abl  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402

OUT = os.path.join(HERE, "results_extra")


def f_model(x, b, c):
    return b * x * np.exp(c * x)


def g_model(x, b, c, d):
    return b * x * np.exp(c * x) + d


MODELS = {
    "F": dict(func=f_model, n=2, p0=[0.5, -0.5],
              bounds=([-np.inf, -np.inf], [np.inf, 0]),
              label="b*x*exp(c*x)"),
    "G": dict(func=g_model, n=3, p0=[0.5, -0.5, 0.3],
              bounds=([-np.inf, -np.inf, -np.inf], [np.inf, 0, np.inf]),
              label="b*x*exp(c*x)+d"),
}


def fit_predict(spec, x, y, target=0.0):
    try:
        popt, _ = curve_fit(spec["func"], x, y, p0=spec["p0"],
                            bounds=spec["bounds"], maxfev=200000)
        pred = float(spec["func"](target, *popt))
    except Exception:
        return float(y[0])
    if abs(pred) > 1.2:                 # same rejection rule as zne_loocv
        pred = float(y[0])
    return pred


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(ROOT, "config.json")) as fh:
        cfg = abl.json.load(fh) if hasattr(abl, "json") else None
    import json
    cfg = json.load(open(os.path.join(ROOT, "config.json")))
    ns = np.asarray(cfg["noise_scale_factors"], float)

    rows = []
    for fam in ("random", "ising_trotter"):
        entries = [r for r in abl.discover_entries(ROOT, fam, None) if r["ready"]]
        for r in entries:
            data, L, _ = abl.load_detailed(r["cached"], ns, 200)
            prev_exact, prev_noisy = abl.load_previous_results(r["dir"], cfg["row"])
            if prev_exact is None:
                continue
            files = sorted([f for f in os.listdir(r["dir"]) if f.endswith(".qasm")],
                           key=abl.natural_sort_key)
            sampled = sorted(random.Random(123).sample(range(len(files)),
                                                      min(3, len(files))))
            ng = {}
            for i in sampled:
                qc = QuantumCircuit.from_qasm_file(os.path.join(r["dir"], files[i]))
                qc.remove_final_measurements()
                ng[i] = len(qc.data)
            n = min(len(prev_exact), data.shape[0])
            for key, spec in MODELS.items():
                preds = [fit_predict(spec, ns, data[i, :L]) for i in range(n)]
                facs = [abl.compute_modification_factor(ng[i], preds[i], prev_exact[i])
                        for i in sampled if i < n]
                c = float(np.mean(facs)) if facs else 1.0
                errs = [abs(preds[i] * c - prev_exact[i]) for i in range(n)]
                rows.append(dict(family=fam, entry=r["entry"], model=key,
                                 mean_abs_err=float(np.mean(errs)),
                                 max_abs_err=float(np.max(errs))))

    # aggregate
    import collections
    agg = collections.defaultdict(list)
    for x in rows:
        agg[(x["family"], x["model"])].append(x)
    print("%-14s %-3s %10s %10s" % ("family", "model", "mean", "max"))
    for (fam, key), xs in sorted(agg.items()):
        m = np.mean([x["mean_abs_err"] for x in xs])
        mx = np.max([x["max_abs_err"] for x in xs])
        print("%-14s %-3s %10.5f %10.3f" % (fam, key, m, mx))
        rows.append(dict(family=fam, entry="ALL", model=key,
                         mean_abs_err=float(m), max_abs_err=float(mx)))

    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["family", "entry", "model",
                                           "mean_abs_err", "max_abs_err"])
        w.writeheader()
        w.writerows(rows)
    print("\nsaved", os.path.join(OUT, "summary.csv"))


if __name__ == "__main__":
    main()
