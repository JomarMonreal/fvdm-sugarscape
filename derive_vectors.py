#!/usr/bin/env python3
"""
derive_vectors.py

Derives FVDM prioritization vectors for conditions 5–8 (Raw Sugarscape Derived,
Egoist Derived, Altruist Derived, Bentham Derived) via Maximum Entropy Inverse
Reinforcement Learning (MaxEnt IRL) applied to behavioral trajectories from the
four homogeneous baseline conditions.

Algorithm (thesis §3.5.3):
  1. Run homogeneous derivation simulations for each baseline model.
  2. At every agent decision step, compute the combined feature vector
         f(c) = E^imm(c) + γ · E^fut(c)
     for the chosen cell AND every other candidate cell in the agent's vision.
  3. Apply online stochastic gradient ascent (Adagrad) on the MaxEnt IRL objective:
         ∇L(θ) = f(c_t) − E_θ[f(c')]   where c' ~ softmax(θ·f)
  4. After all derivation runs, normalize θ to unit length → P_i.

Minimum sample size:
  The reward function has d=5 parameters. MLE needs N >> d; practically, reliable
  estimates require ~1000+ gradient steps (one per agent per timestep). With 250
  agents, even a single 10-timestep run (~2500 steps) is statistically sufficient.
  Adagrad adapts its per-coordinate learning rate after the first few hundred steps.
  The default 30 seeds × 5000 timesteps (≈37.5M steps) ensures ecological diversity
  across population states, not statistical necessity. For a quick test, 5 seeds ×
  200 timesteps gives stable vectors in practice.

Output:
  data/baseline/prioritization_vectors.json   — 4 derived vectors + metadata

Usage:
    python derive_vectors.py [--seeds N] [--timesteps N] [--outdir PATH]
                             [--lr LR] [--log-interval N] [--force]
"""

import argparse
import json
import math
import os
import random
import sys
import time

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

DEFAULT_OUT                = os.path.join(_SCRIPT_DIR, "data", "baseline")
DEFAULT_N_SEEDS            = 30
DEFAULT_TIMESTEPS          = 5000
DEFAULT_LR                 = 0.1
DEFAULT_LOG_INTERVAL       = 500
DEFAULT_CONVERGE_THRESHOLD = 1e-3
DEFAULT_CONVERGE_PATIENCE  = 3      # consecutive intervals below threshold → stop seed early
ADAGRAD_EPS                = 1e-8
GRAD_WINDOW                = 500    # rolling window size for mean gradient norm display

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
    """
    f(c) = E^imm(c) + γ · E^fut(c),  shape (5,)

    Combines both temporal layers into the single 5-D feature vector used by
    the reward function R(c) = θᵀ f(c).
    """
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
# Per-process IRL + progress state
# ---------------------------------------------------------------------------

_theta             = None   # np.ndarray (5,): current parameter vector
_grad_sq           = None   # np.ndarray (5,): Adagrad accumulated squared gradients
_base_lr           = DEFAULT_LR
_step              = 0      # total gradient steps so far
_recent_grad_norms = []     # rolling window of per-step |∇| for display

# Progress tracking (set before each seed run)
_condition_name     = ""
_seed_idx           = 0
_n_seeds            = 0
_timesteps_total    = 0
_log_interval       = DEFAULT_LOG_INTERVAL
_converge_threshold = DEFAULT_CONVERGE_THRESHOLD
_converge_patience  = DEFAULT_CONVERGE_PATIENCE
_converge_stale     = 0   # consecutive log intervals with |∇| below threshold
_t0_seed            = 0.0   # wall-clock start of current seed
_t0_condition       = 0.0   # wall-clock start of current condition

_orig_findBestCell = None
_orig_doTimestep   = None


def _patched_findBestCell(self):
    """
    Intercepts every agent cell selection.
    Computes feature vectors for all candidate cells and performs one
    Adagrad gradient-ascent step on the MaxEnt IRL log-likelihood.
    """
    global _theta, _grad_sq, _step, _recent_grad_norms

    chosen_cell = _orig_findBestCell(self)

    if chosen_cell is None or not self.cellsInRange:
        return chosen_cell

    try:
        # Candidate set: all cells in vision range, plus current cell
        candidate_cells = list(self.cellsInRange.keys())
        if self.cell not in candidate_cells:
            candidate_cells.insert(0, self.cell)

        # Feature matrix F: shape (n_candidates, 5)
        F = np.array([_feature_vector(self, c) for c in candidate_cells])

        # Locate chosen cell
        try:
            chosen_idx = candidate_cells.index(chosen_cell)
        except ValueError:
            return chosen_cell

        # Softmax probabilities under current θ
        r = F @ _theta
        r -= r.max()          # log-sum-exp numerical stability
        exp_r = np.exp(r)
        probs = exp_r / exp_r.sum()

        # MaxEnt IRL gradient: ∇L = f(c_t) − E_θ[f(c')]
        g = F[chosen_idx] - probs @ F

        # Track gradient norm for progress display (rolling window)
        _recent_grad_norms.append(float(np.linalg.norm(g)))
        if len(_recent_grad_norms) > GRAD_WINDOW:
            _recent_grad_norms.pop(0)

        # Adagrad update
        _grad_sq += g * g
        alpha = _base_lr / np.sqrt(_grad_sq + ADAGRAD_EPS)
        _theta += alpha * g
        _step += 1

    except Exception:
        pass  # never let gradient errors crash the simulation

    return chosen_cell


def _patched_doTimestep(self):
    """
    Wraps Sugarscape.doTimestep to print a progress line every
    _log_interval timesteps and stop early on gradient convergence.
    """
    global _converge_stale

    _orig_doTimestep(self)

    t = self.timestep
    if t == 0 or (t % _log_interval != 0 and t != _timesteps_total):
        return

    # --- timing ---
    now        = time.time()
    elapsed    = now - _t0_seed
    frac_seed  = t / _timesteps_total if _timesteps_total > 0 else 1.0
    eta_seed   = (elapsed / frac_seed - elapsed) if frac_seed > 0 else 0.0

    seeds_done_frac = (_seed_idx + frac_seed) / _n_seeds if _n_seeds > 0 else 1.0
    total_elapsed   = now - _t0_condition
    eta_condition   = (total_elapsed / seeds_done_frac - total_elapsed) if seeds_done_frac > 0 else 0.0

    # --- stats ---
    pop       = self.runtimeStats.get("population", "?")
    mean_grad = (sum(_recent_grad_norms) / len(_recent_grad_norms)
                 if _recent_grad_norms else 0.0)
    theta_str = " ".join(f"{v:+.3f}" for v in _theta)

    # --- formatted line ---
    print(
        f"  [{_condition_name}]"
        f"  seed {_seed_idx + 1}/{_n_seeds}"
        f"  t={t:>{len(str(_timesteps_total))}}/{_timesteps_total}"
        f"  pop={pop:>4}"
        f"  steps={_step:>9,}"
        f"  |∇|={mean_grad:.4f}"
        f"  θ=[{theta_str}]"
        f"  seed_eta={_fmt_time(eta_seed)}"
        f"  cond_eta={_fmt_time(eta_condition)}"
    )

    # --- convergence check ---
    if _recent_grad_norms and mean_grad < _converge_threshold:
        _converge_stale += 1
        if _converge_stale >= _converge_patience:
            print(
                f"  [{_condition_name}]  seed {_seed_idx + 1}/{_n_seeds}"
                f"  CONVERGED at t={t}"
                f"  (|∇|={mean_grad:.4f} < {_converge_threshold}"
                f" for {_converge_patience} intervals) — stopping seed early"
            )
            self.endSimulation()
    else:
        _converge_stale = 0


def _fmt_time(seconds):
    """Format seconds into a human-readable string."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def _apply_irl_patches():
    global _orig_findBestCell, _orig_doTimestep
    _orig_findBestCell = agent_module.Agent.findBestCell
    _orig_doTimestep   = sugarscape_module.Sugarscape.doTimestep
    agent_module.Agent.findBestCell    = _patched_findBestCell
    sugarscape_module.Sugarscape.doTimestep = _patched_doTimestep


def _remove_irl_patches():
    if _orig_findBestCell is not None:
        agent_module.Agent.findBestCell = _orig_findBestCell
    if _orig_doTimestep is not None:
        sugarscape_module.Sugarscape.doTimestep = _orig_doTimestep

# ---------------------------------------------------------------------------
# Derivation runner
# ---------------------------------------------------------------------------

def derive_condition(condition_name, models, seeds, timesteps, lr, log_interval,
                     converge_threshold, converge_patience, out_dir, force):
    """
    Runs all derivation seeds for one baseline condition sequentially,
    accumulating Adagrad gradient updates into a single θ across seeds.
    Returns the final normalized prioritization vector Pi.
    """
    global _theta, _grad_sq, _base_lr, _step, _recent_grad_norms
    global _condition_name, _seed_idx, _n_seeds, _timesteps_total
    global _log_interval, _converge_threshold, _converge_patience, _converge_stale
    global _t0_seed, _t0_condition

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
                return np.array(vec)
        except Exception:
            pass

    # Initialize IRL state
    _theta             = np.zeros(5)
    _grad_sq           = np.zeros(5)
    _base_lr           = lr
    _step              = 0
    _recent_grad_norms = []

    # Progress state
    _condition_name     = condition_name
    _n_seeds            = len(seeds)
    _timesteps_total    = timesteps
    _log_interval       = log_interval
    _converge_threshold = converge_threshold
    _converge_patience  = converge_patience
    _t0_condition       = time.time()

    config = dict(BASE_CONFIG)
    config["agentDecisionModels"] = models
    config["experimentalGroup"]   = None
    config["timesteps"]           = timesteps
    config["headlessMode"]        = True
    config["logfile"]             = None
    config["agentLogfile"]        = None
    config["logfileFormat"]       = "csv"

    _apply_irl_patches()
    try:
        for seed_idx, seed in enumerate(seeds):
            _seed_idx       = seed_idx
            _converge_stale = 0
            _t0_seed        = time.time()

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
                f"  total_steps={_step:,}"
                f"  θ_norm={np.linalg.norm(_theta):.4f}"
                f"  ({_fmt_time(elapsed)})"
            )
    finally:
        _remove_irl_patches()

    # Normalize θ → Pi
    norm = np.linalg.norm(_theta)
    if norm < 1e-10:
        print(f"  [{condition_name}] WARNING: θ near zero — defaulting to uniform")
        pi = np.full(5, 1.0 / math.sqrt(5))
    else:
        pi = _theta / norm

    total_time = time.time() - _t0_condition
    print(f"\n  [{condition_name}] DONE  Pi = {[round(v, 4) for v in pi.tolist()]}"
          f"  ({_fmt_time(total_time)} total)\n")
    return pi

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

def _save_vectors(vectors_path, results, step_counts, seeds, timesteps, lr):
    existing = {}
    if os.path.exists(vectors_path):
        try:
            with open(vectors_path) as f:
                existing = json.load(f)
        except Exception:
            pass

    for condition_name, pi in results.items():
        existing[condition_name] = {
            "vector": [round(float(v), 6) for v in pi],
            "seeds_used": len(seeds),
            "timesteps": timesteps,
            "learning_rate": lr,
            "total_gradient_steps": step_counts.get(condition_name, 0),
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
    p.add_argument("--seeds",        type=int,   default=DEFAULT_N_SEEDS,
                   help="Derivation seeds per condition (default: %(default)s)")
    p.add_argument("--timesteps",    type=int,   default=DEFAULT_TIMESTEPS,
                   help="Timesteps per derivation run (default: %(default)s)")
    p.add_argument("--outdir",       default=DEFAULT_OUT,
                   help="Output directory (default: %(default)s)")
    p.add_argument("--lr",           type=float, default=DEFAULT_LR,
                   help="Adagrad base learning rate (default: %(default)s)")
    p.add_argument("--log-interval", type=int,   default=DEFAULT_LOG_INTERVAL,
                   dest="log_interval",
                   help="Print progress every N timesteps (default: %(default)s)")
    p.add_argument("--converge-threshold", type=float, default=DEFAULT_CONVERGE_THRESHOLD,
                   dest="converge_threshold",
                   help="Early-stop when rolling |∇| drops below this (default: %(default)s)")
    p.add_argument("--converge-patience", type=int, default=DEFAULT_CONVERGE_PATIENCE,
                   dest="converge_patience",
                   help="Consecutive intervals below threshold before stopping seed (default: %(default)s)")
    p.add_argument("--force",        action="store_true",
                   help="Re-derive even if vectors already exist")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    seeds_path   = os.path.join(args.outdir, "seeds.json")
    vectors_path = os.path.join(args.outdir, "prioritization_vectors.json")
    seeds = _load_or_generate_seeds(seeds_path, args.seeds)

    print(
        f"\nDerivation settings:"
        f"\n  conditions        : {list(DERIVE_CONDITIONS.keys())}"
        f"\n  seeds             : {args.seeds}"
        f"\n  timesteps         : {args.timesteps} (max per seed)"
        f"\n  lr (Adagrad)      : {args.lr}"
        f"\n  log_interval      : every {args.log_interval} timesteps"
        f"\n  converge_threshold: {args.converge_threshold}  (early-stop |∇| threshold)"
        f"\n  converge_patience : {args.converge_patience}   (consecutive intervals before stop)"
        f"\n  output            : {vectors_path}"
        f"\n"
        f"\nMinimum viable: 5 seeds × 200 timesteps (~250k gradient steps per condition)."
        f"\nDefault (30 × 5000) captures ecological diversity across population states.\n"
    )

    results    = {}
    step_counts = {}
    t0_total   = time.time()

    for condition_name, models in DERIVE_CONDITIONS.items():
        print(f"{'─'*60}")
        print(f"  Condition: {condition_name}")
        print(f"{'─'*60}")
        pi = derive_condition(
            condition_name, models, seeds,
            args.timesteps, args.lr, args.log_interval,
            args.converge_threshold, args.converge_patience,
            args.outdir, args.force
        )
        results[condition_name]     = pi
        step_counts[condition_name] = _step

    _save_vectors(vectors_path, results, step_counts, seeds, args.timesteps, args.lr)

    # Summary table
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
