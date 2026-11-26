# -*- coding: utf-8 -*-
"""Collect the fitted parameters of the full model (Eq. 94/96) over all circuits.

Used only to justify, in the paper, why the reduced variants are so close to
the full model: we need the typical size of the Gaussian coefficient c2
relative to the linear c1, and of the constant amplitude d relative to the
linear one e.  The fit is identical to the production one (five cached points,
same p0 / bounds, ``delate = 1``).

    python fit_parameters.py [--max-circuits N]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir,
                                     "高斯-指数混合模型整合"))
sys.path.insert(0, HERE)
import ablate_gaussian_term as abl  # noqa: E402

_P0 = [0.5, 0.01, -0.5, 0.3, 0.25]
_BOUNDS = ([-np.inf, 0, -np.inf, -np.inf, -np.inf],
           [np.inf, np.inf, 0, np.inf, np.inf])


def fit_one(x, y):
    def f(t, a, b, c, d, e):
        return a * np.exp(c * t + b * t ** 2) + (d + e * t) * np.exp(c * t)
    try:
        popt, _ = curve_fit(f, x, y, p0=_P0, bounds=_BOUNDS, maxfev=200000)
        return popt
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-circuits", type=int, default=None)
    a = ap.parse_args()

    with open(os.path.join(ROOT, "config.json")) as fh:
        cfg = json.load(fh)
    ns = cfg["noise_scale_factors"]

    out = {}
    for fam in ("random", "ising_trotter"):
        report = abl.discover_entries(ROOT, fam, None)
        ready = [r for r in report if r["ready"]]
        rows = []
        for r in ready:
            data, L, _ = abl.load_detailed(r["cached"], ns, 200)
            n = data.shape[0] if a.max_circuits is None else min(a.max_circuits,
                                                                 data.shape[0])
            for i in range(n):
                y = data[i, :L]
                p = fit_one(np.asarray(ns, float), y)
                if p is not None:
                    rows.append(p)
        arr = np.asarray(rows)
        out[fam] = arr
        a_, b_, c_, d_, e_ = arr.T
        print("=" * 60)
        print(fam, "fits:", len(arr))
        for name, v in (("c1 (linear, <=0)", c_), ("c2 (Gaussian, >=0)", b_),
                        ("d (constant amp)", d_), ("e (linear amp)", e_)):
            print("  %-20s median |.| = %.4f   IQR = [%.4f, %.4f]"
                  % (name, np.median(np.abs(v)),
                     np.percentile(np.abs(v), 25), np.percentile(np.abs(v), 75)))
        print("  |c2| < |c1| in %.1f%% of circuits"
              % (100 * np.mean(np.abs(b_) < np.abs(c_))))
        print("  |d|  < |e|  in %.1f%% of circuits"
              % (100 * np.mean(np.abs(d_) < np.abs(e_))))
        print("  c2 = 0 (on the bound) in %.1f%% of circuits"
              % (100 * np.mean(b_ <= 1e-12)))

    with open(os.path.join(HERE, "results", "fitted_parameters.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "a", "c2", "c1", "d", "e"])
        for fam, arr in out.items():
            for row in arr:
                w.writerow([fam] + ["%.10g" % v for v in row])
    print("\nsaved results/fitted_parameters.csv")


if __name__ == "__main__":
    main()
