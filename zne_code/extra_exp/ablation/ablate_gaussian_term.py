#!/usr/bin/env python
"""Ablation of the Gaussian-exponential term of the published ZNE model (Eq. 94).

A *standalone*, cache-only study.  It re-simulates nothing, writes nothing into
the simulation tree, and does not touch ``config.json``.  It re-reads exactly
the cached artefacts that ``zne.py`` (``record = False``) and ``compare.py``
read, re-runs only the cheap extrapolation step, and reports how the error
changes when the shipped ``poly_exp`` ansatz is replaced by a reduced one.

Variants -- only the ``poly_exp`` entry of the model list is swapped; data,
coordinates, bound convention, LOOCV and error aggregation are untouched.
For ``random`` / ``ising_trotter`` (Eq. 94 family):

    A   a * exp(c*x + b*x**2)                             3 params
    B   a * exp(c*x + b*x**2) + d * x * exp(c*x)         4 params
    C   a * exp(c*x + b*x**2) + (d + e*x) * exp(c*x)      5 params  (= Eq. 94)
    D   (a + b*x) * exp(c*x)                              3 params  (= Eq. 94, c2 = 0)
    E   a * exp(c*x + b*x**2) + d * exp(c*x)              4 params  (= Eq. 94, e = 0)

Parameter naming follows ``extrapolation_model.py``: ``b`` is the Gaussian
coefficient c2, ``c`` the linear coefficient c1.  A removes the second branch;
**D is the complementary ablation** -- it removes the Gaussian c2*x**2 and keeps
the second branch, collapsing Eq. 94 to an affine x exponential.  B and E isolate the two
pieces of the second branch (its linear factor and its constant), so their comparison tells
which of the two matters.  A/D answer "is the Gaussian term necessary?" in both directions.

For ``grover`` the shipped ``poly_exp`` is ``poly_exp_extrapolation3`` (Eq. 99),
so the ablation is re-centred on that model:

    A   a * exp(c*x + b*x**2)                             3 params  (G-gaussNoC)
    B   a * exp(c*x + b*x**2) + d                         4 params  (G-gaussOnly)
    D   (a + b*x) * exp(c*x) + d                          4 params  (G-noGauss)
    C   a * exp(c*x + b*x**2) + (d + e*x) * exp(c*x)      5 params  (Eq. 94; NOT
                                                                     shipped here)
    SHIPPED  (a + e*x) * exp(c*x + b*x**2) + d            5 params  (G-full, Eq. 99)

Baseline reproduction
---------------------
For ``random`` / ``ising_trotter`` the shipped ``poly_exp`` ansatz *is*
``poly_exp_extrapolation2``, i.e. variant C.  For ``grover`` the shipped ansatz
is ``poly_exp_extrapolation3`` (a different function), so the script additionally
evaluates the *unmodified* ``zne.py`` model list (reported as ``SHIPPED``) on
every family; that column is what must reproduce the archived
``zne_separated_results.txt``.  ``SHIPPED`` is a verification reference and is
*also* the Grover baseline (G-full); it is not one of the ablation variants.
On the non-Grover families ``SHIPPED`` and variant C are the same model list and
are therefore bit-identical.

Usage
-----
    python ablate_gaussian_term.py --list
    python ablate_gaussian_term.py --cir-type random
    python ablate_gaussian_term.py --cir-type ising_trotter --entries 1,2,3
    python ablate_gaussian_term.py --cir-type grover
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import random
import re
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROJECT_ROOT = os.path.normpath(
    os.path.join(HERE, os.pardir, os.pardir, "高斯-指数混合模型整合")
)
DEFAULT_OUTDIR = os.path.join(HERE, "results")

FAMILY_DIRS = {
    "random": "random_circuits",
    "ising_trotter": "ising_trotter_circuits",
    "grover": "grover_circuits",
}
# Cached raw-data files, in order of preference.  zne.py (record=False) reads
# ``zne_separated_results_detailed.txt`` for every family; the second entry is
# the shot-based, readout-mitigated Grover cache written by zne_grover_shots.py
# and is only used as a fallback.
CACHED_DETAILED = [
    ("zne_separated_results/zne_separated_results_detailed.txt", True),
    ("zne_separated_results/zne_separated_results_detailed_readout_multi.txt", False),
]

# module-level handles into the simulation code (filled by _load_project_modules)
zne_loocv = None
poly_exp_extrapolation2 = poly_exp_extrapolation3 = None
exp_extrapolation = exp_extrapolation1 = None
fid_extrapolation = purity_sqrt_extrapolation = None


# --------------------------------------------------------------------------- #
# Variant ansatze  (the ONLY thing that differs between the runs)
# --------------------------------------------------------------------------- #
def variant_a(x, a, b, c):
    """A: Gaussian-exponential term only (the referee's first suggestion)."""
    return a * np.exp(c * x + b * (x ** 2))


def variant_b(x, a, b, c, d):
    """B (Grover, Eq. 99 family): A plus a constant offset.

    Used only for Grover, where Eq. 99 carries its constant *outside* the
    exponential, so this is the meaningful way to drop the second branch.
    """
    return a * np.exp(c * x + b * (x ** 2)) + d


def variant_b_lin(x, a, b, c, d):
    """B (Eq. 94 family): A plus the *linear* part of the second branch.

    ``a*exp(c*x + b*x**2) + d*x*exp(c*x)``.  Paired with E
    (``a*exp(c*x+b*x**2) + d*exp(c*x)``), it splits the second branch into its
    constant and its linear factor, which is the comparison this ablation is
    meant to isolate; a bare additive constant (the previous B) is not part of
    Eq. 94 at all.
    """
    return a * np.exp(c * x + b * (x ** 2)) + d * x * np.exp(c * x)


def variant_c(x, a, b, c, d, e):
    """C: the shipped Eq. 94 (alias of ``poly_exp_extrapolation2``)."""
    return a * np.exp(c * x + b * (x ** 2)) + (d + e * x) * np.exp(c * x)


def variant_d(x, a, b, c):
    """D: Eq. 94 with the Gaussian coefficient c2 switched off.

    ``poly_exp_extrapolation2(x, a, b, c, d, e)`` at ``b = 0`` is
    ``(a + d + e*x) * exp(c*x)``, whose two constant amplitudes merge into one;
    D is therefore exactly "the full model without the Gaussian term".
    Here ``b`` is the affine slope (the role of Eq. 94's ``e``) and ``c`` the
    linear exponent (Eq. 94's ``c1``); the merged amplitude is ``a``.
    """
    return (a + b * x) * np.exp(c * x)


def variant_e(x, a, b, c, d):
    """E: Eq. 94 with the affine prefactor ``(d + e*x)`` reduced to a constant.

    Same parameter convention as A/B/C: ``b`` is the Gaussian coefficient,
    ``c`` the linear one.  Separates "a second exponential branch exists" from
    "that branch needs the linear factor".
    """
    return a * np.exp(c * x + b * (x ** 2)) + d * np.exp(c * x)


def variant_d_grover(x, a, b, c, d):
    """Grover's Eq. 99 with the Gaussian coefficient switched off.

    ``poly_exp_extrapolation3(x, a, b, c, d, e) = (a + e*x)*exp(c*x + b*x**2) + d``
    at ``b = 0`` is ``(a + e*x)*exp(c*x) + d``; here ``b`` is the affine slope
    (Eq. 99's ``e``), ``c`` the linear exponent and ``d`` stays a separate offset
    because in Eq. 99 it sits *outside* the exponential.
    """
    return (a + b * x) * np.exp(c * x) + d


# --------------------------------------------------------------------------- #
# Per-family variant specification:
#   name -> (func, n_params, p0, bounds, formula label)
# ``p0`` / ``bounds`` follow one rule: the reduced forms inherit the shipped
# convention (a, b, c, d, e) = (amplitude, Gaussian coeff >= 0, linear coeff <= 0,
# constant, affine slope).  D is the only exception, because switching the
# Gaussian off changes what the parameters mean (see ``variant_d`` docstring).
# --------------------------------------------------------------------------- #
_LO = [-np.inf, 0, -np.inf, -np.inf, -np.inf]
_HI = [np.inf, np.inf, 0, np.inf, np.inf]
_P0 = [0.5, 0.01, -0.5, 0.3, 0.25]

# Eq. 94 family (random / ising_trotter).  Also used for Grover's A and B, whose
# shipped entry uses the very same p0 / bounds convention.
_E94 = {
    "A": (variant_a, 3, _P0[:3], (_LO[:3], _HI[:3]), "a*exp(c*x+b*x^2)"),
    "B": (variant_b_lin, 4, _P0[:4], (_LO[:4], _HI[:4]),
          "a*exp(c*x+b*x^2)+d*x*exp(c*x)"),
    "C": (variant_c, 5, list(_P0), (list(_LO), list(_HI)),
          "a*exp(c*x+b*x^2)+(d+e*x)*exp(c*x)  [Eq.94]"),
    "D": (variant_d, 3, [0.8, 0.25, -0.5],
          ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, 0]),
          "(a+b*x)*exp(c*x)  [Eq.94 with c2=0]"),
    "E": (variant_e, 4, _P0[:4], (_LO[:4], _HI[:4]),
          "a*exp(c*x+b*x^2)+d*exp(c*x)  [Eq.94 with e=0]"),
}

# Eq. 99 family (grover): the shipped ansatz is poly_exp_extrapolation3, so the
# ablation is re-centred on it.  G-gaussOnly == B and G-gaussNoC == A.
_GROVER = {
    "A": (variant_a, 3, _P0[:3], (_LO[:3], _HI[:3]),
          "a*exp(c*x+b*x^2)  [Eq.99, G-gaussNoC]"),
    "B": (variant_b, 4, _P0[:4], (_LO[:4], _HI[:4]),
          "a*exp(c*x+b*x^2)+d  [Eq.99, G-gaussOnly]"),
    "C": (variant_c, 5, list(_P0), (list(_LO), list(_HI)),
          "a*exp(c*x+b*x^2)+(d+e*x)*exp(c*x)  [Eq.94, not shipped for grover]"),
    "D": (variant_d_grover, 4, [0.5, 0.25, -0.5, 0.3],
          ([-np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, 0, np.inf]),
          "(a+b*x)*exp(c*x)+d  [Eq.99 with c2=0, G-noGauss]"),
}

FAMILY_SPECS = {"random": _E94, "ising_trotter": _E94, "grover": _GROVER}


def variant_spec(cir_type, variant):
    """``(func, n_params, p0, bounds, label)`` for a family / variant pair."""
    return FAMILY_SPECS[cir_type][variant]


def variant_order_for(cir_type):
    """Default variant order (and the set that exists) for a family."""
    return ["A", "B", "C", "D", "E"] if cir_type != "grover" else ["A", "B", "C", "D"]


# --------------------------------------------------------------------------- #
# Small helpers copied verbatim from the shipped pipeline
# --------------------------------------------------------------------------- #
def natural_sort_key(filename):
    """Identical to ``zne.py``'s ``natural_sort_key``."""
    numbers = re.findall(r"\d+", filename)
    return int(numbers[0]) if numbers else filename


def compute_modification_factor(ngates, pred, exact_val):
    """Verbatim copy of ``zne.py``'s ``compute_modification_factor``.

    ``ngates`` replaces the ``len(qc.data)`` argument so that no qiskit
    ``QuantumCircuit`` object is needed; the value is identical.
    """
    if exact_val != 0 and pred != 0:
        c = np.abs(exact_val / pred)
        if c < np.exp(ngates * 0.001):
            return c if c > 1 else 1.0
        else:
            return np.exp(ngates * 0.001)
    else:
        return 1.0


def load_previous_results(circuit_dir, row):
    """Verbatim logic of ``zne.py``'s ``load_previous_results``."""
    exact_expectations = []
    noisy_expectations = []
    prev_results_file = os.path.join(circuit_dir, "results", "expectation_values.txt")
    if not os.path.exists(prev_results_file):
        return None, None
    with open(prev_results_file, "r") as f:
        lines = f.readlines()
    for line in lines[row:]:
        if line.strip():
            parts = line.split()
            exact_expectations.append(float(parts[1]))
            noisy_expectations.append(float(parts[2]))
    return exact_expectations, noisy_expectations


class _NullWriter(io.TextIOBase):
    """Swallow the ``print`` chatter that ``tools.zne_loocv`` emits."""

    def write(self, s):  # noqa: D102
        return len(s)

    def flush(self):  # noqa: D102
        return None


_NULL = _NullWriter()


def _load_project_modules(project_root):
    """Import the (read-only) simulation modules.  Safe to call repeatedly."""
    global zne_loocv, poly_exp_extrapolation2, poly_exp_extrapolation3
    global exp_extrapolation, exp_extrapolation1, fid_extrapolation
    global purity_sqrt_extrapolation
    if zne_loocv is not None:
        return
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from tools import zne_loocv as _zl
    from extrapolation_model import (
        poly_exp_extrapolation2 as _p2, poly_exp_extrapolation3 as _p3,
        exp_extrapolation as _e1, exp_extrapolation1 as _e2,
        fid_extrapolation as _f, purity_sqrt_extrapolation as _ps)
    zne_loocv = _zl
    poly_exp_extrapolation2, poly_exp_extrapolation3 = _p2, _p3
    exp_extrapolation, exp_extrapolation1 = _e1, _e2
    fid_extrapolation, purity_sqrt_extrapolation = _f, _ps


# --------------------------------------------------------------------------- #
# Model-list construction (mirrors zne.py's get_extrapolation_models exactly)
# --------------------------------------------------------------------------- #
def _poly_exp_entry(func, n_params, p0, bounds, noise_scale_factors, exp_mean,
                    fit_points=None):
    return {
        "name": "poly_exp",
        "x": noise_scale_factors,
        "y": exp_mean,
        "func": func,
        "x0": 0,
        "param": n_params if fit_points is None else fit_points,
        "p0": list(p0),
        "bounds": (list(bounds[0]), list(bounds[1])),
    }


def shipped_models(cir_type, noise_scale_factors, exp_mean, factor, purity_mean):
    """The untouched ``zne.py`` model list for this family."""
    if cir_type != "grover":
        return [
            _poly_exp_entry(poly_exp_extrapolation2, 5, _P0, (_LO, _HI),
                            noise_scale_factors, exp_mean),
            {"name": "exp", "x": noise_scale_factors, "y": exp_mean,
             "func": exp_extrapolation, "x0": 0, "param": 3,
             "p0": [0.5, 0.5, 0.5],
             "bounds": ([-np.inf, 0, -np.inf], [np.inf, 1, np.inf])},
            {"name": "fidelity", "x": factor, "y": exp_mean,
             "func": fid_extrapolation, "x0": 0, "param": 2},
            {"name": "purity_sqrt", "x": purity_mean, "y": exp_mean,
             "func": purity_sqrt_extrapolation, "x0": 1, "param": 3,
             "p0": [0.5, min(purity_mean) / 2, 0.5],
             "bounds": ([-np.inf, 0, -np.inf], [np.inf, min(purity_mean), np.inf])},
        ]
    return [
        _poly_exp_entry(poly_exp_extrapolation3, 5, _P0, (_LO, _HI),
                        noise_scale_factors, exp_mean),
        {"name": "exp", "x": noise_scale_factors, "y": exp_mean,
         "func": exp_extrapolation1, "x0": 0, "param": 5,
         "p0": [0.5, 0.5, 0.5, 0.5, 0.5],
         "bounds": ([-np.inf, 0, -np.inf, 0, -np.inf], [np.inf, 1, np.inf, 1, np.inf])},
        {"name": "fidelity", "x": factor, "y": exp_mean,
         "func": fid_extrapolation, "x0": 0, "param": 2},
        {"name": "purity_sqrt", "x": purity_mean, "y": exp_mean,
         "func": purity_sqrt_extrapolation, "x0": 1, "param": 3,
         "p0": [0.5, min(purity_mean) / 2, 0.5],
         "bounds": ([-np.inf, 0, -np.inf], [np.inf, min(purity_mean), np.inf])},
    ]


def variant_models(base_models, cir_type, variant, noise_scale_factors, exp_mean,
                   fit_points=None):
    """Shipped model list with only the ``poly_exp`` entry swapped.

    For variant C the swapped entry is literally ``poly_exp_extrapolation2``, so
    the list is bit-identical to the shipped one on the non-Grover families.

    ``fit_points`` (optional) overrides the number of points used by the final
    fit.  It exists solely for ``--control-full-fit``, which isolates the
    functional form from the fit-point budget that ``delate == 1`` imposes
    (``delate == 1`` slices ``x_data[:param]``, so A, D are fitted on 3 points,
    B, E on 4 and C on 5 by default).  ``param`` is otherwise unused by
    ``zne_loocv`` when ``mod == 0``.
    """
    func, n_params, p0, bounds, _label = variant_spec(cir_type, variant)
    if variant == "C":
        # use the shipped callable itself so that C is bit-identical to zne.py
        func = poly_exp_extrapolation2
    models = [dict(m) for m in base_models]
    models[0] = _poly_exp_entry(func, n_params, p0, bounds, noise_scale_factors,
                                exp_mean, fit_points)
    return models


# --------------------------------------------------------------------------- #
# Data loading / discovery
# --------------------------------------------------------------------------- #
def find_cached_file(circuit_dir):
    """Return ``(path, is_primary)`` for the first cached detailed file found."""
    for rel, primary in CACHED_DETAILED:
        p = os.path.join(circuit_dir, rel)
        if os.path.isfile(p):
            return p, primary
    return None, None


def load_detailed(path, noise_scale_factors, n_circuits):
    """Load a cached ``zne_separated_results_detailed*.txt`` file.

    Returns ``(data, L, note)``; every row is
    ``[exp(L) | purity(L) | fidelity(L) | factor(L)]``.
    """
    data = np.loadtxt(path, delimiter="\t", skiprows=1)
    if data.ndim == 1:
        data = np.array([data])
    n_rows, n_cols = data.shape
    L = n_cols // 4
    if L != len(noise_scale_factors):
        raise ValueError(
            f"noise_scale_factors mismatch in {path}: file has {L} scale "
            f"factors, config has {len(noise_scale_factors)}")
    note = None
    if n_rows != n_circuits:
        if n_rows % n_circuits == 0:
            note = (f"file holds {n_rows // n_circuits} cached runs per circuit; "
                    f"run #0 of each circuit is used")
            data = data[:n_circuits]
        else:
            raise ValueError(f"{path}: {n_rows} data rows for {n_circuits} circuits")
    return data, L, note


def discover_entries(project_root, cir_type, requested=None):
    """Entry numbers with their cache status, sorted ascending."""
    family = FAMILY_DIRS[cir_type]
    base = os.path.join(project_root, family)
    found = []
    if not os.path.isdir(base):
        return found
    for name in os.listdir(base):
        m = re.fullmatch(re.escape(family) + r"_(\d+)", name)
        if not m:
            continue
        entry = int(m.group(1))
        if requested and entry not in requested:
            continue
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        has_qasm = any(f.endswith(".qasm") for f in os.listdir(d))
        cached, primary = find_cached_file(d)
        has_exp = os.path.isfile(os.path.join(d, "results", "expectation_values.txt"))
        found.append({"entry": entry, "dir": d, "has_qasm": has_qasm,
                      "cached": cached, "cached_primary": primary,
                      "has_expectation_values": has_exp,
                      "ready": bool(has_qasm and cached and has_exp)})
    found.sort(key=lambda r: r["entry"])
    return found


def load_archived_zne(path):
    """Same parsing as ``compare.py``'s ``load_zne_results``."""
    if not os.path.exists(path):
        return None
    out = []
    start = False
    with open(path, "r") as f:
        for line in f:
            if "Circuit" in line and "Poly_exp" in line:
                start = True
                continue
            if start and line.strip():
                parts = line.split()
                try:
                    out.append({"poly_exp": float(parts[1]), "exp": float(parts[2]),
                                "fidelity": float(parts[3]),
                                "purity_sqrt": float(parts[4]),
                                "loocv": float(parts[5]), "noisy": float(parts[6]),
                                "exact": float(parts[7])})
                except Exception:
                    pass
    return out


# --------------------------------------------------------------------------- #
# Worker (module level so that it survives a spawn-based process pool)
# --------------------------------------------------------------------------- #
def _worker_init(project_root):
    _load_project_modules(project_root)


def _process_items(job):
    """Evaluate the shipped list and all requested variants for a batch of circuits."""
    cir_type = job["cir_type"]
    ns = job["ns"]
    mod = job["mod"]
    delate = job["delate"]
    n_qubits = job["n_qubits"]
    variants = job["variants"]
    fit_control = job["fit_control"]

    out = []
    for it in job["items"]:
        exp_mean = it["exp"]
        purity = it["purity"]
        fidelity = it["fidelity"]
        factor = it["factor"]

        base = shipped_models(cir_type, ns, exp_mean, factor, purity)

        # For every family except grover the shipped poly_exp ansatz is
        # ``poly_exp_extrapolation2`` -- i.e. the shipped model list is
        # *bit-identical* to variant C's list -- so the SHIPPED reference and
        # variant C are the same zne_loocv call and are evaluated only once.
        reuse = cir_type != "grover" and "C" in variants
        cached_c = None
        if reuse:
            with contextlib.redirect_stdout(_NULL):
                pred, name, det = zne_loocv(
                    variant_models(base, cir_type, "C", ns, exp_mean),
                    q_num=n_qubits, mod=mod, delate=delate)
            cached_c = (pred, name, det)
            s_pred, s_name, s_det = pred, name, det
        else:
            with contextlib.redirect_stdout(_NULL):
                s_pred, s_name, s_det = zne_loocv(base, q_num=n_qubits, mod=mod,
                                                  delate=delate)

        rec = {
            "entry": it["entry"],
            "i": it["i"],
            "shipped_poly": float(next(d["pred"] for d in s_det
                                       if d["name"] == "poly_exp")),
            "shipped_loocv": float(s_pred),
            "shipped_loocv_model": s_name,
            "exp_pred": float(next(d["pred"] for d in s_det if d["name"] == "exp")),
            "fidelity_pred": float(next(d["pred"] for d in s_det
                                        if d["name"] == "fidelity")),
            "purity_pred": float(next(d["pred"] for d in s_det
                                      if d["name"] == "purity_sqrt")),
            "variants": {},
            "control": {},
        }

        for v in variants:
            if v == "C" and cached_c is not None:
                pred, name, det = cached_c
            else:
                models = variant_models(base, cir_type, v, ns, exp_mean)
                with contextlib.redirect_stdout(_NULL):
                    pred, name, det = zne_loocv(models, q_num=n_qubits, mod=mod,
                                                delate=delate)
            pe = next(d for d in det if d["name"] == "poly_exp")
            rec["variants"][v] = {"poly": float(pe["pred"]),
                                  "valid": bool(pe["valid"]),
                                  "success": bool(pe["success"]),
                                  "cv_error": float(pe["cv_error"]),
                                  "loocv": float(pred),
                                  "loocv_name": name}

        if fit_control is not None:
            for v in variants:
                if fit_control == variant_spec(cir_type, v)[1]:
                    # the control fit coincides with the primary one
                    p = rec["variants"][v]
                    rec["control"][v] = {"poly": p["poly"], "loocv": p["loocv"],
                                         "loocv_name": p["loocv_name"],
                                         "valid": p["valid"]}
                    continue
                models = variant_models(base, cir_type, v, ns, exp_mean,
                                        fit_points=fit_control)
                with contextlib.redirect_stdout(_NULL):
                    pred, name, det = zne_loocv(models, q_num=n_qubits, mod=mod,
                                                 delate=delate)
                pe = next(d for d in det if d["name"] == "poly_exp")
                rec["control"][v] = {"poly": float(pe["pred"]),
                                     "loocv": float(pred), "loocv_name": name,
                                     "valid": bool(pe["valid"])}
        out.append(rec)
    return out


def _run_jobs(jobs, jobs_n, project_root):
    """Run ``_process_items`` over ``jobs``, in parallel when possible."""
    results = []
    if jobs_n <= 1:
        _worker_init(project_root)
        for j in jobs:
            results.extend(_process_items(j))
        return results

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    ctx = mp.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=jobs_n, mp_context=ctx,
                                 initializer=_worker_init,
                                 initargs=(project_root,)) as ex:
            for chunk in ex.map(_process_items, jobs, chunksize=1):
                results.extend(chunk)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"WARNING: parallel execution failed ({exc}); falling back to "
              f"serial.", file=sys.stderr)
        results = []
        _worker_init(project_root)
        for j in jobs:
            results.extend(_process_items(j))
    return results


# --------------------------------------------------------------------------- #
# Core driver
# --------------------------------------------------------------------------- #
def run_family(cir_type, project_root, config, entries, variants, outdir,
               fit_points_control=None, max_circuits=None, jobs_n=1,
               apply_factor=True):
    mod = config["num_train"]
    delate = config["fit_method"]
    n_qubits = config["num_qubits"]
    ns = config["noise_scale_factors"]
    row = config["row"]

    report = discover_entries(project_root, cir_type, set(entries) if entries else None)
    ready = [r for r in report if r["ready"]]
    if not ready:
        lines = [f"No usable cached data for circuit family '{cir_type}'."]
        base = os.path.join(project_root, FAMILY_DIRS[cir_type])
        for r in report:
            if not r["has_qasm"]:
                lines.append(f"  - entry {r['entry']}: missing .qasm files in {r['dir']}")
            elif not r["cached"]:
                lines.append(f"  - entry {r['entry']}: missing cached raw data; none of "
                             + ", ".join(os.path.join(r["dir"], rel)
                                         for rel, _ in CACHED_DETAILED))
            elif not r["has_expectation_values"]:
                lines.append(f"  - entry {r['entry']}: missing "
                             f"{os.path.join(r['dir'], 'results', 'expectation_values.txt')}")
        if entries:
            seen = {r["entry"] for r in report}
            for e in sorted(set(entries) - seen):
                lines.append(f"  - entry {e}: no directory "
                             f"{os.path.join(base, FAMILY_DIRS[cir_type] + '_' + str(e))} "
                             f"at all (run --list to see what exists)")
        elif not report:
            lines.append(f"  - no '{FAMILY_DIRS[cir_type]}*' directories under {base}")
        return None, lines

    if cir_type == "grover":
        row = 10

    fam_outdir = os.path.join(outdir, cir_type)
    os.makedirs(fam_outdir, exist_ok=True)

    variant_order = list(variants)
    per_circuit_rows = []
    control_rows = []
    baseline_rows = []
    per_entry_info = []

    t0 = time.time()
    for k, rec in enumerate(ready):
        entry = rec["entry"]
        circuit_dir = rec["dir"]
        circuit_files = sorted((f for f in os.listdir(circuit_dir)
                                if f.endswith(".qasm")), key=natural_sort_key)
        if not circuit_files:
            continue
        prev_exact, prev_noisy = load_previous_results(circuit_dir, row)
        if prev_exact is None:
            continue
        data, L, note = load_detailed(rec["cached"], ns, len(circuit_files))
        if note:
            print(f"  [entry {entry}] NOTE: {note}")
        if not rec["cached_primary"]:
            print(f"  [entry {entry}] NOTE: using fallback cache "
                  f"{os.path.relpath(rec['cached'], project_root)}")

        step = entry - 1
        if cir_type == "random":
            step = 2 * (entry - 1)
        elif cir_type == "grover":
            step = entry

        n_circ = min(len(circuit_files), max_circuits or len(circuit_files))

        # circuits whose ngates zne.py needs for the modification factor
        sampled = []
        if cir_type != "grover":
            rand = random.Random(123)
            samples = rand.sample(range(len(circuit_files)),
                                  min(3, len(circuit_files)))
            sampled = sorted(s for s in samples if s < n_circ)

        per_entry_info.append({"entry": entry, "step": step, "n_circ": n_circ,
                               "prev_exact": prev_exact, "prev_noisy": prev_noisy,
                               "sampled": sampled, "circuit_dir": circuit_dir,
                               "circuit_files": circuit_files, "data": data, "L": L,
                               "arch": os.path.join(circuit_dir,
                                                    "zne_separated_results",
                                                    "zne_separated_results.txt")})

    if not per_entry_info:
        return None, [f"No circuits processed for '{cir_type}'."]

    # ---- build all jobs -------------------------------------------------- #
    # small chunks so that the pool actually spreads across cores
    chunk_size = max(1, min(16, 4 * jobs_n))
    jobs = []
    for info in per_entry_info:
        entry = info["entry"]
        data, L = info["data"], info["L"]
        items = []
        for i in range(info["n_circ"]):
            r = data[i, :]
            items.append({"entry": entry, "i": i,
                          "exp": np.array(r[:L]), "purity": np.array(r[L:2 * L]),
                          "fidelity": np.array(r[2 * L:3 * L]),
                          "factor": np.array(r[3 * L:])})
        for c in range(0, max(len(items), 1), chunk_size):
            chunk = items[c:c + chunk_size]
            if chunk:
                jobs.append({"cir_type": cir_type, "ns": list(ns), "mod": mod,
                             "delate": delate, "n_qubits": n_qubits,
                             "variants": variant_order,
                             "fit_control": fit_points_control, "items": chunk})

    results = _run_jobs(jobs, jobs_n, project_root)
    results.sort(key=lambda r: (r["entry"], r["i"]))

    # ---- assembly -------------------------------------------------------- #
    by_entry = {}
    for r in results:
        by_entry.setdefault(r["entry"], []).append(r)

    for info in per_entry_info:
        entry = info["entry"]
        res = sorted(by_entry.get(entry, []), key=lambda x: x["i"])
        if not res:
            continue
        prev_exact, prev_noisy = info["prev_exact"], info["prev_noisy"]
        n_circ = len(res)

        # modification factor, exactly as zne.py lines 378-382 (non-grover only)
        ngates = {}
        if info["sampled"]:
            from qiskit import QuantumCircuit
            for i in info["sampled"]:
                qc = QuantumCircuit.from_qasm_file(
                    os.path.join(info["circuit_dir"], info["circuit_files"][i]))
                qc.remove_final_measurements()
                ngates[i] = len(qc.data)

        def mean_factor(get_pred):
            mfs = [compute_modification_factor(ngates[i], get_pred(i), prev_exact[i])
                   for i in info["sampled"] if i < len(prev_exact)]
            return float(np.mean(mfs)) if mfs else 1.0

        scale = {v: 1.0 for v in variant_order}
        control_scale = {v: 1.0 for v in variant_order}
        shipped_scale = 1.0
        if cir_type != "grover" and info["sampled"] and apply_factor:
            for v in variant_order:
                scale[v] = mean_factor(lambda i, v=v: res[i]["variants"][v]["poly"])
                if fit_points_control is not None:
                    control_scale[v] = mean_factor(
                        lambda i, v=v: res[i]["control"][v]["poly"])
            shipped_scale = mean_factor(lambda i: res[i]["shipped_poly"])

        for i, r in enumerate(res):
            exact = float(prev_exact[i]) if i < len(prev_exact) else float("nan")
            noisy = (float(prev_noisy[i]) if prev_noisy and i < len(prev_noisy)
                     else float("nan"))
            sp = r["shipped_poly"] * shipped_scale
            out = {"entry": entry, "step": info["step"], "circuit": i,
                   "exact": exact, "noisy": noisy,
                   "noisy_abs_err": abs(noisy - exact),
                   "shipped_pred": sp, "shipped_abs_err": abs(sp - exact),
                   "shipped_loocv_model": r["shipped_loocv_model"]}
            for v in variant_order:
                d = r["variants"][v]
                p = d["poly"] * scale[v]
                out[f"{v}_pred"] = p
                out[f"{v}_abs_err"] = abs(p - exact)
                out[f"{v}_rejected"] = int(not d["valid"])
                out[f"{v}_loocv_pred"] = d["loocv"]
                out[f"{v}_loocv_model"] = d["loocv_name"]
            for key in ("exp", "fidelity", "purity"):
                p = r[f"{key}_pred"]
                out[f"{key}_pred"] = p
                out[f"{key}_abs_err"] = abs(p - exact)
            out["mod_factor"] = shipped_scale
            per_circuit_rows.append(out)

            if fit_points_control is not None:
                for v in variant_order:
                    c = r["control"][v]
                    control_rows.append({
                        "entry": entry, "step": info["step"], "circuit": i,
                        "variant": v, "pred": c["poly"] * control_scale[v],
                        "raw_pred": c["poly"], "mod_factor": control_scale[v],
                        "loocv": c["loocv"], "loocv_name": c["loocv_name"],
                        "valid": int(c["valid"]), "exact": exact})

        # baseline reproduction check
        arch = load_archived_zne(info["arch"])
        if arch and apply_factor:
            n = min(len(arch), n_circ)
            diffs = np.array([res[i]["shipped_poly"] * shipped_scale
                              - arch[i]["poly_exp"] for i in range(n)])
            baseline_rows.append({
                "entry": entry, "n_compared": n,
                "max_abs_diff": float(np.max(np.abs(diffs))),
                "n_match_1e-6": int(np.sum(np.abs(diffs) <= 1e-6)),
                "n_match_1e-9": int(np.sum(np.abs(diffs) <= 1e-9)),
                "n_exact": int(np.sum(diffs == 0.0)),
                "source": os.path.relpath(info["arch"], project_root)})
        elif not apply_factor:
            pass                              # no factor: archive check not applicable
        else:
            baseline_rows.append({"entry": entry, "n_compared": 0,
                                  "max_abs_diff": float("nan"), "n_match_1e-6": 0,
                                  "n_match_1e-9": 0, "n_exact": 0,
                                  "source": "MISSING " +
                                            os.path.relpath(info["arch"], project_root)})

        print(f"  entry {entry:>2}: {n_circ:>3} circuits   mod_factor={shipped_scale:.6f}"
              f"   ({time.time() - t0:.1f}s)")

    if not per_circuit_rows:
        return None, [f"No circuits processed for '{cir_type}'."]

    # ------------------------------------------------------------------ #
    # CSVs
    # ------------------------------------------------------------------ #
    with open(os.path.join(fam_outdir, "per_circuit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_circuit_rows[0].keys()))
        w.writeheader()
        for r in per_circuit_rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    def agg(rows, key_err, key_rej=None):
        e = np.array([r[key_err] for r in rows], dtype=float)
        return {"n_circuits": len(rows), "mean_abs_err": float(np.mean(e)),
                "median_abs_err": float(np.median(e)),
                "std_abs_err": float(np.std(e)),
                "p90_abs_err": float(np.percentile(e, 90)),
                "p99_abs_err": float(np.percentile(e, 99)),
                "max_abs_err": float(np.max(e)),
                "n_rejected": (int(np.sum([r[key_rej] for r in rows]))
                               if key_rej else "")}

    def summary_row(entry, step, name, formula, n_params, key_err,
                    key_rej=None, n_loocv=""):
        a = agg(per_circuit_rows if entry == "ALL"
                else [r for r in per_circuit_rows if r["entry"] == entry], key_err,
                key_rej)
        return {
            "cir_type": cir_type, "entry": entry, "step": step, "variant": name,
            "formula": formula, "n_params": n_params,
            "n_circuits": a["n_circuits"], "mean_abs_err": a["mean_abs_err"],
            "median_abs_err": a["median_abs_err"],
            "std_abs_err": a["std_abs_err"], "p90_abs_err": a["p90_abs_err"],
            "p99_abs_err": a["p99_abs_err"], "max_abs_err": a["max_abs_err"],
            "n_rejected_gt_1.2": a["n_rejected"],
            "reject_rate": (a["n_rejected"] / a["n_circuits"]
                            if (key_rej and a["n_circuits"]) else ""),
            "n_loocv_selected_poly_exp": n_loocv,
        }

    summary_rows = []
    entries_present = sorted({r["entry"] for r in per_circuit_rows})
    for entry in entries_present + ["ALL"]:
        step = "" if entry == "ALL" else next(
            r["step"] for r in per_circuit_rows if r["entry"] == entry)
        for v in variant_order:
            _f, n_params, _p0, _b, label = variant_spec(cir_type, v)
            n_loocv = int(np.sum(
                [r[f"{v}_loocv_model"] == "poly_exp" for r in per_circuit_rows
                 if entry == "ALL" or r["entry"] == entry]))
            summary_rows.append(summary_row(
                entry, step, v, label, n_params, f"{v}_abs_err",
                f"{v}_rejected", n_loocv))
        for key, name, formula in (
                ("exp_abs_err", "EXP",
                 "shipped 'exp' member (exp_extrapolation; _1 for grover)"),
                ("shipped_abs_err", "SHIPPED",
                 "unmodified zne.py model list (published baseline)"),
                ("noisy_abs_err", "NOISY", "unmitigated")):
            summary_rows.append(summary_row(entry, step, name, formula, "", key))

    # ---- C vs D head-to-head ------------------------------------------- #
    cvd_rows = []
    if "C" in variant_order and "D" in variant_order:
        for entry in entries_present + ["ALL"]:
            sub = (per_circuit_rows if entry == "ALL"
                   else [r for r in per_circuit_rows if r["entry"] == entry])
            eC = np.array([r["C_abs_err"] for r in sub])
            eD = np.array([r["D_abs_err"] for r in sub])
            n = len(sub)
            cvd_rows.append({
                "cir_type": cir_type, "entry": entry,
                "step": "" if entry == "ALL" else sub[0]["step"],
                "n_circuits": n,
                "C_mean": float(np.mean(eC)), "C_median": float(np.median(eC)),
                "C_p90": float(np.percentile(eC, 90)),
                "C_p99": float(np.percentile(eC, 99)),
                "C_max": float(np.max(eC)),
                "C_rejected_gt_1.2": int(np.sum([r["C_rejected"] for r in sub])),
                "D_mean": float(np.mean(eD)), "D_median": float(np.median(eD)),
                "D_p90": float(np.percentile(eD, 90)),
                "D_p99": float(np.percentile(eD, 99)),
                "D_max": float(np.max(eD)),
                "D_rejected_gt_1.2": int(np.sum([r["D_rejected"] for r in sub])),
                "mean_ratio_C_over_D": float(np.mean(eC) / np.mean(eD)),
                "n_D_beats_C": int(np.sum(eD < eC)),
                "frac_D_beats_C": float(np.mean(eD < eC)),
                "n_C_beats_D": int(np.sum(eC < eD)),
                "frac_C_beats_D": float(np.mean(eC < eD)),
                "n_ties": int(np.sum(eC == eD)),
            })
        with open(os.path.join(fam_outdir, "C_vs_D_by_entry.csv"), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cvd_rows[0].keys()))
            w.writeheader()
            for r in cvd_rows:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                            for k, v in r.items()})

    with open(os.path.join(fam_outdir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    control_summary = []
    if control_rows:
        with open(os.path.join(fam_outdir, "control_fullfit_per_circuit.csv"),
                  "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(control_rows[0].keys()))
            w.writeheader()
            for r in control_rows:
                w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v)
                            for k, v in r.items()})
        for v in variant_order:
            sub = [r for r in control_rows if r["variant"] == v]
            e = np.array([abs(r["pred"] - r["exact"]) for r in sub])
            _f, n_params, _p0, _b, label = variant_spec(cir_type, v)
            control_summary.append({
                "cir_type": cir_type, "variant": v, "formula": label,
                "fit_points": fit_points_control, "params": n_params,
                "n_circuits": len(sub), "mean_abs_err": float(np.mean(e)),
                "median_abs_err": float(np.median(e)),
                "std_abs_err": float(np.std(e)),
                "p90_abs_err": float(np.percentile(e, 90)),
                "p99_abs_err": float(np.percentile(e, 99)),
                "max_abs_err": float(np.max(e)),
                "n_rejected_gt_1.2": int(np.sum([not r["valid"] for r in sub]))})
        with open(os.path.join(fam_outdir, "control_fullfit_summary.csv"),
                  "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(control_summary[0].keys()))
            w.writeheader()
            for r in control_summary:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                            for k, v in r.items()})

    if baseline_rows:                      # empty when the factor is switched off
        with open(os.path.join(fam_outdir, "baseline_check.csv"), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(baseline_rows[0].keys()))
            w.writeheader()
            for r in baseline_rows:
                w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v)
                            for k, v in r.items()})

    # ------------------------------------------------------------------ #
    # Figures
    # ------------------------------------------------------------------ #
    mk_figures(fam_outdir, cir_type, per_circuit_rows, variant_order)

    return {"per_circuit_rows": per_circuit_rows, "summary_rows": summary_rows,
            "baseline_rows": baseline_rows, "control_summary": control_summary,
            "variant_order": variant_order, "entries": entries_present,
            "outdir": fam_outdir, "n_circuits": len(per_circuit_rows)}, []


def mk_figures(fam_outdir, cir_type, rows, variant_order):
    """Mean abs error vs step (+ standard-exponential reference) and a boxplot."""
    colors = {"A": "#1f77b4", "B": "#2ca02c", "C": "#d62728",
              "D": "#9467bd", "E": "#8c564b"}
    steps = sorted({r["step"] for r in rows})

    fig, ax = plt.subplots(figsize=(9, 6))
    for v in variant_order:
        _f, n_params, _p0, _b, label = variant_spec(cir_type, v)
        ys = [float(np.mean([r[f"{v}_abs_err"] for r in rows if r["step"] == s]))
              for s in steps]
        ax.plot(steps, ys, marker="o", color=colors[v],
                label=f"{v} ({n_params}p): {label}")
    for key, style in (("exp_abs_err", dict(marker="s", ls="--", color="#7f7f7f")),
                       ("noisy_abs_err", dict(marker="^", ls=":", color="#ff7f0e"))):
        ys = [float(np.mean([r[key] for r in rows if r["step"] == s])) for s in steps]
        label = ("shipped 'exp' member (reference)" if key == "exp_abs_err"
                 else "unmitigated")
        ax.plot(steps, ys, label=label, **style)
    ax.set_yscale("log")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlabel("step (family-specific, as in error_data_compare.py)")
    ax.set_title(f"Gaussian-term ablation - {cir_type} "
                 f"({len(rows)} circuits, mean |pred - exact|)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fam_outdir, "mean_abs_error_vs_step.png"), dpi=200)
    plt.close(fig)

    counts = {}
    for s in steps:
        counts[s] = len([r for r in rows if r["step"] == s])
    # representative = median step among the depths that have the most circuits
    best = max(counts.values())
    candidates = sorted(s for s, c in counts.items() if c == best)
    rep_step = candidates[len(candidates) // 2]
    sub = [r for r in rows if r["step"] == rep_step]
    data = [[r[f"{v}_abs_err"] for r in sub] for v in variant_order]
    data.append([r["exp_abs_err"] for r in sub])
    tick = [f"{v}\n({variant_spec(cir_type, v)[1]}p)" for v in variant_order] \
        + ["exp\n(ref)"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    bp = ax.boxplot(data, showfliers=True, patch_artist=True)
    ax.set_xticklabels(tick)
    for patch, c in zip(bp["boxes"], [colors[v] for v in variant_order] + ["#7f7f7f"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_ylabel("absolute error")
    ax.set_yscale("log")
    ax.set_title(f"{cir_type}: per-circuit absolute error at step={rep_step} "
                 f"({len(sub)} circuits)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fam_outdir, f"boxplot_step_{rep_step:g}.png"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def do_list(project_root):
    print(f"project root : {project_root}")
    if not os.path.isdir(project_root):
        print("  !! project root not found")
        return 2
    with open(os.path.join(project_root, "config.json")) as f:
        cfg = json.load(f)
    print(f"config       : mod(num_train)={cfg['num_train']} "
          f"delate(fit_method)={cfg['fit_method']} "
          f"noise_scale_factors={cfg['noise_scale_factors']} "
          f"num_qubits={cfg['num_qubits']} row={cfg['row']} "
          f"config cir_type={cfg['cir_type']}")
    for ct in ("random", "ising_trotter", "grover"):
        print(f"\n[{ct}]")
        for r in discover_entries(project_root, ct):
            cached = (os.path.relpath(r["cached"], project_root) if r["cached"]
                      else "MISSING (no detailed*.txt)")
            print(f"  {'READY ' if r['ready'] else 'SKIP  '} entry {r['entry']:>2}  "
                  f"qasm={'Y' if r['has_qasm'] else 'N'} "
                  f"exact={'Y' if r['has_expectation_values'] else 'N'}  {cached}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Gaussian-term ablation for the hybrid G-E ZNE model (Eq. 94).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cir-type", choices=["random", "ising_trotter", "grover"],
                    default=None,
                    help="circuit family to ablate. Required unless --list; if "
                         "omitted, config.json's cir_type is used "
                         "(currently 'grover').")
    ap.add_argument("--entries", default=None,
                    help="comma separated entry numbers (default: all with cache)")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR,
                    help=f"output root (default: {DEFAULT_OUTDIR})")
    ap.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT,
                    help="path to the simulation code root (contains config.json)")
    ap.add_argument("--list", action="store_true",
                    help="list available cir_types / entries / cached files and exit")
    ap.add_argument("--variant", default="all",
                    help="comma separated subset of A,B,C,D,E (default: all that "
                         "the family defines; grover has no E)")
    ap.add_argument("--max-circuits", type=int, default=None,
                    help="limit circuits per entry (debugging only; the "
                         "modification factor then differs from zne.py)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel worker processes (0 = auto). Use 1 for a "
                         "purely serial run.")
    ap.add_argument("--control-full-fit", action="store_true",
                    help="ALSO run a robustness control in which every variant is "
                         "fitted on all cached points, isolating the functional "
                         "form from the fit-point budget imposed by delate=1")
    ap.add_argument("--no-mod-factor", action="store_true",
                    help="switch OFF the modification factor of zne.py lines 378-382 "
                         "(raises the effective sample size question: without it the "
                         "predictions are the raw extrapolations)")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the figures from an existing per_circuit.csv "
                         "without recomputing anything")
    args = ap.parse_args(argv)

    project_root = os.path.abspath(args.project_root)
    if not os.path.isdir(project_root):
        print(f"ERROR: project root not found: {project_root}", file=sys.stderr)
        return 2

    if args.list:
        return do_list(project_root)

    try:
        _load_project_modules(project_root)
    except Exception as exc:
        print(f"ERROR: could not import the simulation modules from "
              f"{project_root}: {exc}", file=sys.stderr)
        return 2

    with open(os.path.join(project_root, "config.json")) as f:
        config = json.load(f)

    cir_type = args.cir_type or config["cir_type"]
    if args.cir_type is None:
        print(f"NOTE: --cir-type not given; using config.json's cir_type="
              f"'{cir_type}'.")

    entries = ([int(x) for x in args.entries.split(",") if x.strip()]
               if args.entries else None)

    def variants_for(ct):
        avail = variant_order_for(ct)
        if args.variant == "all":
            return avail
        want = [v.strip().upper() for v in args.variant.split(",") if v.strip()]
        bad = [v for v in want if v not in avail]
        if bad:
            raise SystemExit(f"ERROR: variant(s) {bad} not defined for "
                             f"'{ct}' (available: {avail})")
        return want

    variants = variants_for(cir_type)

    if args.figures_only:
        written = []
        for ct in ([cir_type] if args.cir_type else
                   ["random", "ising_trotter", "grover"]):
            p = os.path.join(os.path.abspath(args.outdir), ct, "per_circuit.csv")
            if not os.path.isfile(p):
                print(f"skip {ct}: no {p}")
                continue
            rows = []
            for r in csv.DictReader(open(p)):
                d = {}
                for k, v in r.items():
                    try:
                        d[k] = float(v)
                    except (TypeError, ValueError):
                        d[k] = v
                rows.append(d)
            have = [v for v in variants_for(ct) if f"{v}_abs_err" in rows[0]]
            mk_figures(os.path.dirname(p), ct, rows, have)
            written.append(ct)
        print(f"redrew figures for: {', '.join(written) or 'nothing'}")
        return 0

    jobs_n = args.jobs if args.jobs else min(os.cpu_count() or 1, 8)

    print("=" * 74)
    print(f"Gaussian-term ablation  |  cir_type={cir_type}  |  variants={variants}")
    print(f"project root : {project_root}")
    print(f"outdir       : {os.path.abspath(args.outdir)}")
    print(f"config       : mod={config['num_train']} delate={config['fit_method']} "
          f"ns={config['noise_scale_factors']} n_qubits={config['num_qubits']}")
    print(f"processes    : {jobs_n}")
    print("=" * 74)

    res, msgs = run_family(
        cir_type, project_root, config, entries, variants,
        os.path.abspath(args.outdir),
        fit_points_control=(len(config["noise_scale_factors"])
                            if args.control_full_fit else None),
        max_circuits=args.max_circuits, jobs_n=jobs_n,
        apply_factor=not args.no_mod_factor)
    for m in msgs:
        print(m)
    if res is None:
        return 1
    print(f"\nWrote {res['n_circuits']} circuit rows to {res['outdir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
