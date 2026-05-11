#!/usr/bin/env python3
"""
run_hetero.py

Runs all 8 experimental conditions (4 baseline rule-based + 4 FVDM-derived)
in a single pass.  Each condition is written to its own subdirectory under
--outdir, giving a clean, analysis-ready dataset for results and discussion.

Conditions 1–4 (homogeneous baseline rule-based):
  1  rawSugarscape   — <outdir>/rawSugarscape/
  2  egoist          — <outdir>/egoist/
  3  altruist        — <outdir>/altruist/
  4  bentham         — <outdir>/bentham/

Conditions 5–8 (homogeneous FVDM-derived):
  5  fvdmRawSugarscape  — <outdir>/fvdmRawSugarscape/
  6  fvdmEgoist         — <outdir>/fvdmEgoist/
  7  fvdmAltruist       — <outdir>/fvdmAltruist/
  8  fvdmBentham        — <outdir>/fvdmBentham/

Requires prioritization_vectors.json and seeds.json to exist in --baseline-dir.
Run 'make baseline && make derive' first.

Usage:
    python run_hetero.py [--seeds N] [--timesteps N] [--agents N]
                         [--parallel N] [--outdir PATH]
                         [--baseline-dir PATH] [--force]
"""

import argparse
import json
import multiprocessing
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from run_baseline_conditions import (
    BASE_CONFIG,
    DEFAULT_N_SEEDS,
    run_single_simulation,
    _write_summary,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUT      = os.path.join(_SCRIPT_DIR, "data")
DEFAULT_BASELINE = os.path.join(_SCRIPT_DIR, "data", "baseline")

BASELINE_CONDITIONS = {
    "rawSugarscape": {"agentDecisionModels": ["none"],     "experimentalGroup": None},
    "egoist":        {"agentDecisionModels": ["egoist"],   "experimentalGroup": None},
    "altruist":      {"agentDecisionModels": ["altruist"], "experimentalGroup": None},
    "bentham":       {"agentDecisionModels": ["bentham"],  "experimentalGroup": None},
}

FVDM_MAP = {
    "fvdmRawSugarscape": "rawSugarscape",
    "fvdmEgoist":        "egoist",
    "fvdmAltruist":      "altruist",
    "fvdmBentham":       "bentham",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_vectors(baseline_dir):
    path = os.path.join(baseline_dir, "prioritization_vectors.json")
    if not os.path.exists(path):
        print(f"ERROR: prioritization vectors not found at {path}")
        print("Run 'make derive' first.")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    return {k: v["vector"] for k, v in data.items() if "vector" in v}


def _load_seeds(baseline_dir, n_seeds):
    path = os.path.join(baseline_dir, "seeds.json")
    if not os.path.exists(path):
        print(f"ERROR: seeds file not found at {path}")
        print("Run 'make baseline' first to generate matched seeds.")
        sys.exit(1)
    with open(path) as f:
        seeds = json.load(f)
    if len(seeds) < n_seeds:
        print(f"WARNING: seeds file has {len(seeds)} seeds, requested {n_seeds} — using all.")
        return seeds
    return seeds[:n_seeds]


def _build_all_conditions(vectors):
    conditions = dict(BASELINE_CONDITIONS)
    for fvdm_name, base_name in FVDM_MAP.items():
        if base_name not in vectors:
            print(f"WARNING: no vector for '{base_name}' — skipping {fvdm_name}")
            continue
        conditions[fvdm_name] = {
            "agentDecisionModels":       ["fvdm"],
            "experimentalGroup":         None,
            "agentPrioritizationVector": vectors[base_name],
        }
    return conditions

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds",    type=int, default=DEFAULT_N_SEEDS,
                   help="Number of matched seeds (default: %(default)s)")
    p.add_argument("--timesteps", type=int, default=None,
                   help="Simulation timesteps per run (default: BASE_CONFIG = 1000)")
    p.add_argument("--agents",   type=int, default=None,
                   help="Starting agents per run (default: BASE_CONFIG = 250)")
    p.add_argument("--parallel", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="Parallel simulation processes (default: CPU count - 1)")
    p.add_argument("--outdir",   default=DEFAULT_OUT,
                   help="Root output directory; each condition gets a subdirectory (default: %(default)s)")
    p.add_argument("--baseline-dir", default=DEFAULT_BASELINE, dest="baseline_dir",
                   help="Directory containing seeds.json and prioritization_vectors.json (default: %(default)s)")
    p.add_argument("--force",    action="store_true",
                   help="Re-run already-completed simulations")
    return p.parse_args()


def main():
    args = parse_args()

    vectors    = _load_vectors(args.baseline_dir)
    seeds      = _load_seeds(args.baseline_dir, args.seeds)
    conditions = _build_all_conditions(vectors)

    if args.timesteps is not None:
        BASE_CONFIG["timesteps"] = args.timesteps
    if args.agents is not None:
        BASE_CONFIG["startingAgents"] = args.agents

    # Create per-condition output directories
    for cond_name in conditions:
        os.makedirs(os.path.join(args.outdir, cond_name), exist_ok=True)

    # Build task list
    tasks = [
        (cond_name, cond_cfg, seed, os.path.join(args.outdir, cond_name), args.force)
        for cond_name, cond_cfg in conditions.items()
        for seed in seeds
    ]
    total = len(tasks)

    print(f"\nAll conditions ({len(conditions)} total):")
    for i, name in enumerate(conditions, 1):
        print(f"  {i:>2}.  {name:<24}  →  {os.path.join(args.outdir, name)}/")
    print(f"\nSeeds: {len(seeds)}  |  Timesteps: {BASE_CONFIG['timesteps']}"
          f"  |  Agents: {BASE_CONFIG['startingAgents']}")
    print(f"Running {total} simulations  ({args.parallel} parallel)\n")

    n_parallel = min(args.parallel, total)
    if n_parallel <= 1:
        completed = 0
        for task in tasks:
            cond, seed, status = run_single_simulation(task)
            completed += 1
            print(f"  [{completed:>{len(str(total))}}/{total}] {cond} seed={seed}: {status}")
    else:
        with multiprocessing.Pool(processes=n_parallel) as pool:
            completed = 0
            for cond, seed, status in pool.imap_unordered(run_single_simulation, tasks):
                completed += 1
                print(f"  [{completed:>{len(str(total))}}/{total}] {cond} seed={seed}: {status}")

    # Per-condition outcome summaries
    for cond_name in conditions:
        cond_dir = os.path.join(args.outdir, cond_name)
        _write_summary(cond_dir, [cond_name], seeds)

    print("\nDone.")


if __name__ == "__main__":
    main()
