#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single sequential entry point for the simulation code of

    "Hybrid Gaussian-exponential zero-noise extrapolation for periodic circuits"

The original project was a loose collection of scripts that had to be launched
by hand, from the right working directory, in the right order.  This module
turns that collection into one reproducible pipeline.

Design rules (do not break them):

* This file *only orchestrates*.  It never re-implements physics, never touches a
  random seed and never edits a result.  Every step shells out to the original
  script, unchanged, with the original working directory.  Running a step here is
  byte-for-byte the same as running ``python <script>.py`` by hand.
* Each original script is executed in its **own** interpreter process.  That
  preserves module-level random-seed initialisation, ``if __name__ == "__main__"``
  semantics and avoids leaking global state (matplotlib rcParams, imported
  backends, numpy global RNG) from one step into the next.
* ``MPLBACKEND=Agg`` is exported for every step.  The original scripts call
  ``plt.show()`` after ``plt.savefig()``; with an interactive backend that blocks
  the pipeline until a window is closed by hand.  Agg makes ``show()`` a no-op
  and does not change a single pixel of the saved PNGs.
* Steps fail loudly.  A missing input raises ``StepError`` and the pipeline stops.

Usage
-----
    python main.py --list
    python main.py --all
    python main.py --steps 10,11,12
    python main.py --from 8
    python main.py --all --skip 14
    python main.py --all --with-production
    python main.py                      # interactive menu
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #

BASE_DIR: Path = Path(__file__).resolve().parent
MODULE_DIR: Path = BASE_DIR / "高斯-指数混合模型整合"
CONFIG_PATH: Path = MODULE_DIR / "config.json"
BACKUP_DIR: Path = BASE_DIR / "original_backup"

# Circuit-family name -> sub-directory prefix used throughout the project.
FAMILY_PREFIX: Dict[str, str] = {
    "random": "random_circuits",
    "ising_trotter": "ising_trotter_circuits",
    "grover": "grover_circuits",
}

# Raw circuit source directory per family (where .pk files live).
FAMILY_GEN_DIR: Dict[str, str] = {
    "random": "random_gen",
    "ising_trotter": "trotter_gen",
    "grover": "grover_gen",
}


class StepError(RuntimeError):
    """Raised when a step cannot run or exits with a non-zero status."""


# --------------------------------------------------------------------------- #
# Configuration access
# --------------------------------------------------------------------------- #

def load_config() -> dict:
    """Read ``config.json`` (never writes it)."""
    if not CONFIG_PATH.is_file():
        raise StepError(
            f"config.json not found at {CONFIG_PATH}. "
            "It must sit next to the pipeline scripts."
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Step description
# --------------------------------------------------------------------------- #

@dataclass
class Step:
    """One pipeline step: a thin wrapper around one original script."""

    number: int
    key: str
    name: str
    script: str                      # path relative to MODULE_DIR
    consumes: str                    # human readable input description
    produces: str                    # human readable output description
    required_inputs: Tuple[str, ...] = ()    # globs, relative to the step's workdir
    workdir: str = ""                # cwd for the step, relative to MODULE_DIR
    extra_argv: Tuple[str, ...] = ()
    needs_parent_on_path: bool = False
    production: bool = False         # writes raw data / re-simulates
    group: str = "analysis"          # "production" | "analysis" | "utility"
    # Optional predicate on config; returns (applicable, reason).
    applies: Callable[[dict], Tuple[bool, str]] = field(
        default=lambda cfg: (True, "")
    )
    # Pre-existing defect worth printing in the banner (never auto-fixed).
    known_issue: str = ""


def _family(config: dict) -> str:
    return config.get("cir_type", "grover")


def _applies_family(*families: str) -> Callable[[dict], Tuple[bool, str]]:
    def predicate(config: dict) -> Tuple[bool, str]:
        cir_type = _family(config)
        if cir_type in families:
            return True, ""
        return False, (
            f"only runs for cir_type in {list(families)} "
            f"(config.json has cir_type='{cir_type}')"
        )
    return predicate


def _missing(patterns: Sequence[str], base: Path = MODULE_DIR) -> List[str]:
    """Return the list of glob patterns that match nothing."""
    missing: List[str] = []
    for pattern in patterns:
        if not list(base.glob(pattern)):
            missing.append(pattern)
    return missing


# --------------------------------------------------------------------------- #
# Step registry  (order == dependency order)
# --------------------------------------------------------------------------- #

STEPS: List[Step] = [
    # ---------------- raw data production (excluded from --all) ------------- #
    Step(
        number=1, key="gen-grover", name="Generate Grover circuits",
        script="grover.py",
        consumes="nothing (builds circuits analytically)",
        produces="grover_gen/step_1.pk ... step_5.pk",
        required_inputs=(),
        production=True, group="production",
        applies=_applies_family("grover"),
    ),
    Step(
        number=2, key="gen-random", name="Generate random circuits",
        script="random_generator.py",
        consumes="nothing (random circuit synthesis, seed fixed inside the script)",
        produces="random_gen/step_0.pk ... step_36.pk",
        production=True, group="production",
        applies=_applies_family("random"),
    ),
    Step(
        number=3, key="gen-trotter", name="Generate Ising-Trotter circuits",
        script="trotter_generate.py",
        consumes="nothing (Trotterised Ising synthesis, seed 42 inside the script)",
        produces="trotter_gen/step_0.pk ... step_17.pk",
        production=True, group="production",
        applies=_applies_family("ising_trotter"),
    ),
    Step(
        number=4, key="clifford-sample", name="Clifford-road path sampling",
        script="clifford_road_sample.py",
        consumes="random_circuits/circuit_0.qasm",
        produces="weigh_data/*.txt  (relative to example_random/)",
        required_inputs=("random_circuits/circuit_0.qasm",),
        workdir="example_random",
        needs_parent_on_path=True,
        production=True, group="production",
    ),
    Step(
        number=5, key="export-circuits", name="Export .pk -> .qasm/.qpy",
        script="pk_file_deployment.py",
        consumes="{random_gen|trotter_gen|grover_gen}/step_*.pk",
        produces="{family}_circuits/<family>_circuits_*/circuit_*.qasm|.qpy",
        required_inputs=(),           # checked dynamically against cir_type
        production=True, group="production",
    ),
    Step(
        number=6, key="draw-circuits", name="Draw circuit diagrams",
        script="run_circuit.py",
        consumes="<family>_circuits_<n>/circuit_*.qasm",
        produces="<family>_circuits_<n>/pictures/circuit_0.png",
        production=True, group="production",
    ),
    Step(
        number=7, key="noisy-expectation", name="Exact / noisy expectation values",
        script="noise_exp_compute.py",
        consumes="<family>_circuits_<n>/circuit_*.qasm",
        produces="<family>_circuits_<n>/results/expectation_values.txt",
        production=True, group="production",
    ),

    # ---------------- analysis pipeline (default --all) -------------------- #
    Step(
        number=8, key="zne-density", name="ZNE (density-matrix path)",
        script="zne.py",
        consumes="config.json  +  zne_separated_results_detailed.txt when record=false",
        produces="<family>_circuits_<n>/zne_separated_results/zne_separated_results.txt",
        required_inputs=(),
        group="analysis",
        applies=_applies_family("random", "ising_trotter"),
    ),
    Step(
        number=9, key="zne-grover-shots", name="ZNE (sampled / Grover path)",
        script="zne_grover_shots.py",
        consumes="config.json  +  zne_separated_results_detailed_readout_multi.txt when record=false",
        produces="<family>_circuits_<n>/zne_separated_results/zne_separated_results_readout.txt",
        required_inputs=(),
        group="analysis",
        applies=_applies_family("grover"),
    ),
    Step(
        number=10, key="error-analysis", name="Per-circuit error analysis",
        script="compare.py",
        consumes="results/expectation_values.txt + zne_separated_results/zne_separated_results.txt",
        produces="<family>_circuits_<n>/error_analysis_results/detailed_error_analysis.txt",
        required_inputs=(),
        group="analysis",
    ),
    Step(
        number=11, key="error-boxplot", name="Error-distribution boxplot",
        script="single_set_compute.py",
        consumes="results/expectation_values.txt + zne_separated_results/zne_separated_results.txt",
        produces="<family>_circuits_<n>/comparison_results/Comparison of Error Distributions ...png",
        required_inputs=(),
        group="analysis",
    ),
    Step(
        number=12, key="error-vs-step", name="Average error vs step figure",
        script="error_data_compare.py",
        consumes="<family>_circuits_*/error_analysis_results/*.txt",
        produces="<family>_circuits/compare_plot/average_error_vs_step_<cir_type>.png",
        required_inputs=(),
        group="analysis",
    ),
    Step(
        number=13, key="grover-plots", name="Grover comparison figures",
        script="grover_data_plot.py",
        consumes="grover_circuits_*/zne_separated_results/zne_separated_results*.txt",
        produces="grover_circuits/compare_plot/*.png  +  grover_circuits_3/compare_plot/*.png",
        required_inputs=("grover_circuits/grover_circuits_*/zne_separated_results/*.txt",),
        group="analysis",
        applies=_applies_family("grover"),
    ),
    Step(
        number=14, key="stability", name="Extrapolation stability histogram",
        script="stable.py",
        consumes="grover_circuits_3/zne_separated_results/zne_separated_results_detailed.txt",
        produces="grover_circuits_3/stable_results/stability_histogram_entry_3.png",
        required_inputs=(
            "grover_circuits/grover_circuits_3/zne_separated_results/"
            "zne_separated_results_detailed.txt",
        ),
        group="analysis",
        applies=_applies_family("grover"),
    ),
    Step(
        number=15, key="normality-test", name="Normality test of noise weights",
        script="Normality_Test.py",
        consumes="weigh_data/noise_random_10_1_0.txt + pauli_weight_random_10_1_0.txt",
        produces="log_norm_test/*.png  (relative to example_random/)",
        required_inputs=(
            "weigh_data/noise_random_10_1_0.txt",
            "weigh_data/pauli_weight_random_10_1_0.txt",
        ),
        workdir="example_random",
        group="analysis",
    ),
    Step(
        number=16, key="device-properties", name="Fake-backend device properties",
        script="Device_properties.py",
        consumes="FakeQuito / FakeLima built-in calibration snapshots",
        produces="Device_properties/*.csv + *.png",
        required_inputs=(),
        group="utility",
        known_issue=(
            "PRE-EXISTING DEFECT (not fixed on purpose). The script calls "
            "plot_error_map(backend, ax=...) but the ax= keyword only exists in "
            "Qiskit >= 1.x, while this project needs Qiskit < 1.0 for "
            "'from qiskit import Aer'. The call therefore raises TypeError on "
            "every valid environment -- which is why the archived tree contains "
            "fake_backends_layout.png but no fake_backends_error_map.png. The "
            "CSV tables and the other three figures are written before the crash."
        ),
    ),
]

STEPS_BY_NUMBER: Dict[int, Step] = {step.number: step for step in STEPS}
LAST_STEP: int = STEPS[-1].number


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def _banner(step: Step, config: dict) -> None:
    print()
    print("=" * 78)
    print(f"  STEP {step.number}/{LAST_STEP}  |  {step.name}   [{step.key}]")
    print("-" * 78)
    print(f"  script   : {step.script}")
    if step.workdir:
        print(f"  cwd      : {step.workdir}/")
    print(f"  inputs   : {step.consumes}")
    print(f"  outputs  : {step.produces}")
    print(f"  cir_type : {_family(config)}    record={config.get('record')}    "
          f"sample={config.get('sample')}    readout_error={config.get('readout_error')}")
    if config.get("record"):
        print("  NOTE: record=True -> this step will RE-SIMULATE, not load the cache.")
    if step.known_issue:
        print("-" * 78)
        print(f"  {step.known_issue}")
    print("=" * 78)
    sys.stdout.flush()


def _step_env(step: Step) -> dict:
    env = os.environ.copy()
    # Never block on plt.show(); the figures are already written by savefig().
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    if step.needs_parent_on_path:
        # example_random/*.py import rand_seed / tools / Noise_model from the
        # project root, which is not their own directory.
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(MODULE_DIR) + (os.pathsep + existing if existing else "")
    return env


def run_step(step: Step, config: dict) -> float:
    """Execute one step in its own process.  Returns the elapsed seconds."""
    _banner(step, config)

    workdir = MODULE_DIR / step.workdir if step.workdir else MODULE_DIR

    # 1. Explicitly declared input files must exist.
    missing = _missing(step.required_inputs, workdir)
    # 2. Family directories must contain circuits for the steps that need them.
    family_prefix = FAMILY_PREFIX.get(_family(config))
    circuit_globs: Dict[int, str] = {
        5: f"{FAMILY_GEN_DIR.get(_family(config), '')}/step_*.pk",
        6: f"{family_prefix}/**/circuit_*.qasm",
        7: f"{family_prefix}/**/circuit_*.qasm",
        10: f"{family_prefix}/**/zne_separated_results/zne_separated_results.txt",
        11: f"{family_prefix}/**/zne_separated_results/zne_separated_results.txt",
    }
    if step.number in circuit_globs and family_prefix:
        missing += _missing((circuit_globs[step.number],))
    # With record=False the ZNE steps read their raw data from disk, so the
    # cached file is a hard requirement -- fail loudly instead of re-simulating.
    if not config.get("record"):
        cache_file = {
            8: ("zne_separated_results_detailed.txt", "zne-density"),
            9: ("zne_separated_results_detailed_readout_multi.txt", "zne-grover-shots"),
        }.get(step.number)
        if cache_file is not None and family_prefix:
            missing += _missing(
                (f"{family_prefix}/**/zne_separated_results/{cache_file[0]}",)
            )
    if missing:
        raise StepError(
            f"step {step.number} ({step.key}) is missing required input(s) "
            f"under {workdir}: " + ", ".join(missing)
        )

    # `script` is resolved relative to the step's workdir.
    script_path = workdir / step.script
    if not script_path.is_file():
        raise StepError(f"step {step.number}: script not found -> {script_path}")

    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(script_path), *step.extra_argv],
        cwd=str(workdir),
        env=_step_env(step),
    )
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        raise StepError(
            f"step {step.number} ({step.key}) failed with exit code "
            f"{completed.returncode} after {elapsed:.1f}s"
        )
    print(f"\n[step {step.number} done in {elapsed:.1f}s]")
    sys.stdout.flush()
    return elapsed


# --------------------------------------------------------------------------- #
# Selection logic
# --------------------------------------------------------------------------- #

def default_pipeline(with_production: bool = False,
                     with_utility: bool = False) -> List[int]:
    """Steps that ``--all`` runs: the cached-data analysis pipeline.

    ``production`` steps are excluded because their outputs are already in the
    tree and re-running them re-simulates.  ``utility`` steps are one-off device
    characterisation helpers that are not part of the ZNE experiment (see the
    ``known_issue`` note on step 16).
    """
    allowed = {"analysis"}
    if with_production:
        allowed.add("production")
    if with_utility:
        allowed.add("utility")
    return [step.number for step in STEPS if step.group in allowed]


def _parse_int_list(raw: str) -> List[int]:
    numbers: List[int] = []
    for chunk in raw.replace(" ", "").split(","):
        if not chunk:
            continue
        try:
            numbers.append(int(chunk))
        except ValueError:
            raise SystemExit(f"--steps: '{chunk}' is not a step number")
    return numbers


def resolve_selection(args: argparse.Namespace) -> Tuple[List[int], bool]:
    """Return (ordered step numbers, explicit_selection_flag)."""
    selected: List[int]
    explicit = False

    if args.steps:
        selected = _parse_int_list(args.steps)
        explicit = True
    elif args.from_step is not None:
        selected = [n for n in range(args.from_step, LAST_STEP + 1)]
        explicit = True
    elif args.all or args.with_production or args.with_utility:
        selected = default_pipeline(with_production=args.with_production,
                                    with_utility=args.with_utility)
    else:
        return [], False

    if args.skip:
        skipped = set(_parse_int_list(args.skip))
        selected = [n for n in selected if n not in skipped]

    unknown = [n for n in selected if n not in STEPS_BY_NUMBER]
    if unknown:
        raise SystemExit(
            f"unknown step number(s): {unknown}.  Valid range is 1..{LAST_STEP} "
            f"(run `python main.py --list`)."
        )
    # de-duplicate, keep registry order
    ordered = [n for n in range(1, LAST_STEP + 1) if n in set(selected)]
    return ordered, explicit


def _wrap(text: str, width: int) -> List[str]:
    """Cheap word wrapper (avoids importing textwrap for one call site)."""
    import textwrap
    return textwrap.wrap(text, width=width)


def print_steps() -> None:
    config = load_config()
    print()
    print("=" * 78)
    print("  PIPELINE STEPS")
    print("=" * 78)
    print(f"  {'#':>2}  {'group':<10}  {'key':<18}  name")
    print("-" * 78)
    for step in STEPS:
        print(f"  {step.number:>2}  {step.group:<10}  {step.key:<18}  {step.name}")
    print("-" * 78)
    print("  'production' steps write raw data / re-simulate and are NOT part of")
    print("  `--all` (their outputs are already committed in the tree).")
    print("  Run them explicitly, e.g.  python main.py --steps 5,7")
    print("  or: python main.py --all --with-production --with-utility")
    print("-" * 78)
    print("  'utility' = one-off device-characterisation helpers, also excluded")
    print("  from `--all`.  Step 16 has a pre-existing defect, see --list note.")
    print("=" * 78)
    for step in STEPS:
        if step.known_issue:
            print()
            print(f"  !! step {step.number} ({step.key}):")
            for line in _wrap(step.known_issue, 72):
                print(f"     {line}")
    print()
    print()
    print("  Pipeline mode read from config.json:")
    print(f"    cir_type      = {config.get('cir_type')!r}")
    print(f"    record        = {config.get('record')!r}"
          "   <- False = load raw data from disk, True = re-simulate")
    print(f"    sample        = {config.get('sample')!r}")
    print(f"    readout_error = {config.get('readout_error')!r}")
    applicable = [s.number for s in STEPS
                  if s.group == "analysis" and s.applies(config)[0]]
    print(f"    -> steps run by `--all` for this config: {applicable}")
    print(f"    -> production steps (need --with-production): "
          f"{[s.number for s in STEPS if s.group == 'production']}")
    print(f"    -> utility steps (need --with-utility): "
          f"{[s.number for s in STEPS if s.group == 'utility']}")
    print()


def interactive_menu() -> Tuple[List[int], bool]:
    print_steps()
    print("Enter a comma separated list of steps to run, 'all' for the default")
    print("pipeline, or 'q' to quit.")
    while True:
        try:
            raw = input("steps> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return [], False
        if raw.lower() in ("q", "quit", "exit"):
            return [], False
        if raw.lower() == "all":
            return default_pipeline(), False
        if raw:
            numbers = _parse_int_list(raw)
            unknown = [n for n in numbers if n not in STEPS_BY_NUMBER]
            if unknown:
                print(f"  unknown step(s) {unknown}; valid range 1..{LAST_STEP}")
                continue
            return [n for n in range(1, LAST_STEP + 1) if n in set(numbers)], True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Sequential pipeline for the Hybrid G-E ZNE paper code.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py --list\n"
            "  python main.py --all\n"
            "  python main.py --steps 10,11,12\n"
            "  python main.py --from 8\n"
            "  python main.py --all --skip 14\n"
            "  python main.py --all --with-production\n"
        ),
    )
    parser.add_argument("--list", action="store_true",
                        help="list all steps and the pipeline mode, then exit")
    parser.add_argument("--all", action="store_true",
                        help="run the default (cached-data) pipeline, steps in order")
    parser.add_argument("--steps", metavar="N,M,...",
                        help="run exactly these step numbers")
    parser.add_argument("--from", dest="from_step", type=int, metavar="N",
                        help="run every step from N to the last one")
    parser.add_argument("--skip", metavar="N,M,...",
                        help="remove these step numbers from the selection")
    parser.add_argument("--with-production", action="store_true",
                        help="with --all: also run the raw-data production steps")
    parser.add_argument("--with-utility", action="store_true",
                        help="with --all: also run the one-off utility steps")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue with later steps after a failure")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print_steps()
        return 0

    if not any([args.all, args.steps, args.from_step is not None,
                args.with_production, args.with_utility]):
        if not sys.stdin.isatty():
            print_steps()
            print("No steps selected and stdin is not a terminal; "
                  "use one of --all / --steps / --from.")
            return 0
        selected, explicit = interactive_menu()
    else:
        selected, explicit = resolve_selection(args)

    if not selected:
        print("Nothing to do.")
        return 0

    try:
        config = load_config()
    except StepError as error:
        print(f"ERROR: {error}")
        return 2

    print()
    print("#" * 78)
    print("#  Hybrid Gaussian-exponential ZNE -- sequential pipeline")
    print(f"#  project dir : {MODULE_DIR}")
    print(f"#  cir_type    : {config.get('cir_type')}")
    print(f"#  record      : {config.get('record')}"
          "  (False -> results are loaded from disk)")
    print(f"#  steps       : {selected}")
    print("#" * 78)

    timings: List[Tuple[int, str, float, str]] = []
    overall_start = time.perf_counter()

    for number in selected:
        step = STEPS_BY_NUMBER[number]
        applicable, reason = step.applies(config)
        if not applicable:
            if explicit:
                print(f"\nERROR: step {number} ({step.key}) is not applicable: {reason}")
                return 2
            print(f"\nSKIPPED step {number} ({step.key}): {reason}")
            timings.append((number, step.key, 0.0, "skipped"))
            continue
        try:
            elapsed = run_step(step, config)
            timings.append((number, step.key, elapsed, "ok"))
        except StepError as error:
            print(f"\nERROR: {error}")
            timings.append((number, step.key, 0.0, "FAILED"))
            if not args.keep_going:
                _print_summary(timings, time.perf_counter() - overall_start,
                               failed=True)
                return 1

    failed = any(status == "FAILED" for _, _, _, status in timings)
    _print_summary(timings, time.perf_counter() - overall_start, failed=failed)
    return 1 if failed else 0


def _print_summary(timings: List[Tuple[int, str, float, str]],
                   total: float, failed: bool) -> None:
    print()
    print("=" * 78)
    print("  SUMMARY")
    print("-" * 78)
    for number, key, elapsed, status in timings:
        marker = {"ok": "OK", "skipped": "--", "FAILED": "XX"}[status]
        print(f"  [{marker}] step {number:>2}  {key:<18}  {elapsed:>8.1f}s")
    print("-" * 78)
    print(f"  total: {total:.1f}s   {'(completed with failures)' if failed else ''}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
