#!/usr/bin/env python3
"""
derive_vectors.py

Derives FVDM prioritization vectors for conditions 5–8 (Raw Sugarscape Derived,
Egoist Derived, Altruist Derived, Bentham Derived) via Behavioral Feature
Expectation (BFE) applied to observed decision trajectories from the four
homogeneous baseline conditions.

Algorithm (Abbeel & Ng, 2004):
  1. Run homogeneous derivation simulations for each baseline model.
  2. At every agent decision step, record the combined feature vector of the
     chosen cell:
         f(c*) = E^imm(c*) + γ · E^fut(c*)
  3. After all derivation runs, compute the mean feature vector across all
     recorded decisions:
         μ = (1/N) Σ f(c*_t)
  4. Normalize μ to unit length → P_i.

Justification:
  Under feature expectation matching (Abbeel & Ng, 2004), an agent's policy is
  characterized by the expected feature vector of its observed decisions. The
  mean feature vector μ(π) = E_π[f(c*)] is the simplest estimator of this
  quantity. Normalization to unit length places P_i on the same hypersphere as
  the felicific effect vectors, making Euclidean distance a consistent metric.

  This approach does not control for feature availability across the candidate
  set (unlike MaxEnt IRL). However, given the stochastic resource distribution
  and dynamic population of the Digital Terrarium, feature availability varies
  substantially across timesteps and seeds, reducing this confound in practice.

Output:
  data/baseline/prioritization_vectors.json   — 4 derived vectors + metadata

Usage:
    python derive_vectors.py [--seeds N] [--timesteps N] [--outdir PATH]
                             [--parallel N] [--log-interval N] [--force]
"""

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import agent as agent_module
import sugarscape as sugarscape_module
from run_baseline_conditions import BASE_CONFIG, compute_felicific_vectors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUT          = os.path.join(_SCRIPT_DIR, "data", "baseline")
DEFAULT_N_SEEDS      = 15
DEFAULT_TIMESTEPS    = 2000
DEFAULT_LOG_INTERVAL = 500

# Baseline conditions to derive vectors from (same as conditions 1–4)
DERIVE_CONDITIONS = {
    "rawSugarscape": ["none"],
    "egoist":        ["egoist"],
    "altruist":      ["altruist"],
    "bentham":       ["bentham"],
}

# ---------------------------------------------------------------------------
# Feature vector helper
# ---------------------------------------------------------------------------

def _feature_vector(ag, cell):
    """f(c*) = E^imm(c*) + γ · E^fut(c*),  shape (5,)"""
    gamma = ag.decisionModelLookaheadDiscount if ag.decisionModelLookaheadFactor != 0 else 0.0
    E_imm, E_fut = compute_felicific_vectors(ag, cell)
    return np.array([
        E_imm["I"] + gamma * E_fut["If"],
        E_imm["D"] + gamma * E_fut["Df"],
        E_imm["C"] + gamma * E_fut["Cf"],
        E_imm["P"] + gamma * E_fut["Pf"],
        E_imm["X"] + gamma * E_fut["Xf"],
    ])

# ---------------------------------------------------------------------------
# Per-process BFE state
# ---------------------------------------------------------------------------

_feature_accumulator = []   # list of f(c*) arrays, one per agent decision

# Progress tracking (set before each seed run)
_condition_name  = ""
_seed_idx        = 0
_n_seeds         = 0
_timesteps_total = 0
_log_interval    = DEFAULT_LOG_INTERVAL
_t0_seed         = 0.0
_t0_condition    = 0.0

_orig_findBestCell = None
_orig_doTimestep   = None


def _patched_findBestCell(self):
    """Intercepts every agent cell selection and records f(c*)."""
    global _feature_accumulator

    chosen_cell = _orig_findBestCell(self)

    if chosen_cell is not None:
        try:
            _feature_accumulator.append(_feature_vector(self, chosen_cell))
        except Exception:
            pass

    return chosen_cell


def _patched_doTimestep(self):
    """Wraps Sugarscape.doTimestep to print a progress line every _log_interval timesteps."""
    _orig_doTimestep(self)

    t = self.timestep
    if t == 0 or (t % _log_interval != 0 and t != _timesteps_total):
        return

    # --- timing ---
    now             = time.time()
    elapsed         = now - _t0_seed
    frac_seed       = t / _timesteps_total if _timesteps_total > 0 else 1.0
    eta_seed        = (elapsed / frac_seed - elapsed) if frac_seed > 0 else 0.0
    seeds_done_frac = (_seed_idx + frac_seed) / _n_seeds if _n_seeds > 0 else 1.0
    total_elapsed   = now - _t0_condition
    eta_condition   = (total_elapsed / seeds_done_frac - total_elapsed) if seeds_done_frac > 0 else 0.0

    # --- running mean feature vector ---
    pop = self.runtimeStats.get("population", "?")
    n   = len(_feature_accumulator)
    if n > 0:
        mu = np.mean(_feature_accumulator, axis=0)
        mu_str = " ".join(f"{v:+.3f}" for v in mu)
    else:
        mu_str = "no data"

    print(
        f"  [{_condition_name}]"
        f"  seed {_seed_idx + 1}/{_n_seeds}"
        f"  t={t:>{len(str(_timesteps_total))}}/{_timesteps_total}"
        f"  pop={pop:>4}"
        f"  n={n:>9,}"
        f"  μ=[{mu_str}]"
        f"  seed_eta={_fmt_time(eta_seed)}"
        f"  cond_eta={_fmt_time(eta_condition)}"
    )


def _fmt_time(seconds):
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def _apply_patches():
    global _orig_findBestCell, _orig_doTimestep
    _orig_findBestCell = agent_module.Agent.findBestCell
    _orig_doTimestep   = sugarscape_module.Sugarscape.doTimestep
    agent_module.Agent.findBestCell          = _patched_findBestCell
    sugarscape_module.Sugarscape.doTimestep  = _patched_doTimestep


def _remove_patches():
    if _orig_findBestCell is not None:
        agent_module.Agent.findBestCell = _orig_findBestCell
    if _orig_doTimestep is not None:
        sugarscape_module.Sugarscape.doTimestep = _orig_doTimestep

# ---------------------------------------------------------------------------
# Derivation runner
# ---------------------------------------------------------------------------

def derive_condition(condition_name, models, seeds, timesteps, log_interval,
                     out_dir, force):
    """
    Runs all derivation seeds for one baseline condition, collecting f(c*)
    for every agent decision. Returns (Pi, n_decisions).
    """
    global _feature_accumulator
    global _condition_name, _seed_idx, _n_seeds, _timesteps_total
    global _log_interval, _t0_seed, _t0_condition

    vectors_path = os.path.join(out_dir, "prioritization_vectors.json")

    # Skip if already derived (unless forced)
    if not force and os.path.exists(vectors_path):
        try:
            with open(vectors_path) as f:
                saved = json.load(f)
            if condition_name in saved and "vector" in saved[condition_name]:
                vec = saved[condition_name]["vector"]
                print(f"  [{condition_name}] already derived — skipping")
                print(f"    Pi = {[round(v, 4) for v in vec]}")
                return np.array(vec), 0
        except Exception:
            pass

    # Reset accumulator and progress state
    _feature_accumulator = []
    _condition_name      = condition_name
    _n_seeds             = len(seeds)
    _timesteps_total     = timesteps
    _log_interval        = log_interval
    _t0_condition        = time.time()

    config = dict(BASE_CONFIG)
    config["agentDecisionModels"] = models
    config["experimentalGroup"]   = None
    config["timesteps"]           = timesteps
    config["headlessMode"]        = True
    config["logfile"]             = None
    config["agentLogfile"]        = None
    config["logfileFormat"]       = "csv"

    _apply_patches()
    try:
        for seed_idx, seed in enumerate(seeds):
            _seed_idx = seed_idx
            _t0_seed  = time.time()

            config["seed"] = seed
            random.seed(seed)
            sim = sugarscape_module.Sugarscape(config)
            try:
                sim.runSimulation(timesteps)
            except SystemExit:
                pass

            elapsed = time.time() - _t0_seed
            print(
                f"  [{condition_name}] seed {seed_idx + 1}/{len(seeds)} complete"
                f"  n={len(_feature_accumulator):,}"
                f"  ({_fmt_time(elapsed)})"
            )
    finally:
        _remove_patches()

    n = len(_feature_accumulator)
    if n == 0:
        print(f"  [{condition_name}] WARNING: no decisions recorded — defaulting to uniform")
        pi = np.full(5, 1.0 / math.sqrt(5))
    else:
        mu   = np.mean(_feature_accumulator, axis=0)
        norm = np.linalg.norm(mu)
        pi   = mu / norm if norm > 1e-10 else np.full(5, 1.0 / math.sqrt(5))

    total_time = time.time() - _t0_condition
    print(f"\n  [{condition_name}] DONE  Pi = {[round(v, 4) for v in pi.tolist()]}"
          f"  n={n:,}  ({_fmt_time(total_time)} total)\n")
    return pi, n

# ---------------------------------------------------------------------------
# Multiprocessing worker (must be top-level to be picklable)
# ---------------------------------------------------------------------------

def _derive_worker(args):
    condition_name, models, seeds, timesteps, log_interval, out_dir, force = args
    pi, n = derive_condition(
        condition_name, models, seeds, timesteps, log_interval, out_dir, force,
    )
    return condition_name, pi.tolist(), n

# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def _load_or_generate_seeds(seeds_path, n_seeds):
    if os.path.exists(seeds_path):
        with open(seeds_path) as f:
            seeds = json.load(f)
        if len(seeds) >= n_seeds:
            print(f"Reusing {n_seeds} seeds from {seeds_path}")
            return seeds[:n_seeds]
    seeds = []
    while len(seeds) < n_seeds:
        s = random.randint(0, sys.maxsize)
        if s not in seeds:
            seeds.append(s)
    with open(seeds_path, "w") as f:
        json.dump(seeds, f)
    print(f"Generated {n_seeds} derivation seeds → {seeds_path}")
    return seeds

# ---------------------------------------------------------------------------
# Save / merge results
# ---------------------------------------------------------------------------

def _save_vectors(vectors_path, results, decision_counts, seeds, timesteps):
    existing = {}
    if os.path.exists(vectors_path):
        try:
            with open(vectors_path) as f:
                existing = json.load(f)
        except Exception:
            pass

    for condition_name, pi in results.items():
        existing[condition_name] = {
            "vector":          [round(float(v), 6) for v in pi],
            "method":          "behavioral_feature_expectation",
            "seeds_used":      len(seeds),
            "timesteps":       timesteps,
            "total_decisions": decision_counts.get(condition_name, 0),
        }

    with open(vectors_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Prioritization vectors saved → {vectors_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds",        type=int, default=DEFAULT_N_SEEDS,
                   help="Derivation seeds per condition (default: %(default)s)")
    p.add_argument("--timesteps",    type=int, default=DEFAULT_TIMESTEPS,
                   help="Timesteps per derivation run (default: %(default)s)")
    p.add_argument("--outdir",       default=DEFAULT_OUT,
                   help="Output directory (default: %(default)s)")
    p.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL,
                   dest="log_interval",
                   help="Print progress every N timesteps (default: %(default)s)")
    p.add_argument("--parallel",     type=int, default=1,
                   help="Conditions to derive in parallel (default: %(default)s)")
    p.add_argument("--force",        action="store_true",
                   help="Re-derive even if vectors already exist")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    seeds_path   = os.path.join(args.outdir, "seeds.json")
    vectors_path = os.path.join(args.outdir, "prioritization_vectors.json")
    seeds        = _load_or_generate_seeds(seeds_path, args.seeds)
    n_workers    = min(args.parallel, len(DERIVE_CONDITIONS))

    print(
        f"\nDerivation settings (Behavioral Feature Expectation):"
        f"\n  method      : mean f(c*) normalized to unit length (Abbeel & Ng, 2004)"
        f"\n  conditions  : {list(DERIVE_CONDITIONS.keys())}"
        f"\n  seeds       : {args.seeds}"
        f"\n  timesteps   : {args.timesteps} (per seed)"
        f"\n  log_interval: every {args.log_interval} timesteps"
        f"\n  parallel    : {n_workers} condition(s) at a time"
        f"\n  output      : {vectors_path}\n"
    )

    worker_args = [
        (name, models, seeds, args.timesteps, args.log_interval, args.outdir, args.force)
        for name, models in DERIVE_CONDITIONS.items()
    ]

    results         = {}
    decision_counts = {}
    t0_total        = time.time()

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_derive_worker, a): a[0] for a in worker_args}
            for future in as_completed(futures):
                condition_name, pi_list, n = future.result()
                results[condition_name]         = np.array(pi_list)
                decision_counts[condition_name] = n
    else:
        for worker_arg in worker_args:
            condition_name, pi_list, n = _derive_worker(worker_arg)
            results[condition_name]         = np.array(pi_list)
            decision_counts[condition_name] = n

    _save_vectors(vectors_path, results, decision_counts, seeds, args.timesteps)

    total_time = time.time() - t0_total
    print(f"\n{'═'*68}")
    print(f"  {'Condition':<24}  {'p_I':>7}  {'p_D':>7}  {'p_C':>7}  {'p_P':>7}  {'p_X':>7}")
    print(f"{'─'*68}")
    for name, pi in results.items():
        vals = "  ".join(f"{v:+.4f}" for v in pi)
        print(f"  {name:<24}  {vals}")
    print(f"{'═'*68}")
    print(f"  Total derivation time: {_fmt_time(total_time)}")
    print(f"  Vectors written to: {vectors_path}\n")


if __name__ == "__main__":
    main()
