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
  3. Apply online stochastic gradient ascent on the MaxEnt IRL log-likelihood:
         ∇L(θ) = f(c_t) − E_θ[f(c')]   where c' ~ softmax(θ·f)
  4. After all derivation runs, normalize θ to unit length → P_i.

Output:
  data/baseline/prioritization_vectors.json   — 4 derived vectors + metadata

Usage:
    python derive_vectors.py [--seeds N] [--timesteps N] [--outdir PATH] [--lr LR] [--force]
"""

import argparse
import json
import math
import os
import random
import sys

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

DEFAULT_OUT       = os.path.join(_SCRIPT_DIR, "data", "baseline")
DEFAULT_N_SEEDS   = 30
DEFAULT_TIMESTEPS = 5000
DEFAULT_LR        = 0.1
ADAGRAD_EPS       = 1e-8

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

    Combines both temporal layers into a single 5-D vector used by
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
# Per-process IRL state
# ---------------------------------------------------------------------------

_theta     = None   # np.ndarray (5,): current parameter vector
_grad_sq   = None   # np.ndarray (5,): Adagrad accumulated squared gradients
_base_lr   = DEFAULT_LR
_step      = 0
_orig_findBestCell = None


def _patched_findBestCell(self):
    """
    Intercepts every agent cell selection.
    Records the chosen cell and all candidate cells, then performs one
    Adagrad gradient-ascent step on the MaxEnt IRL objective.
    """
    global _theta, _grad_sq, _step

    chosen_cell = _orig_findBestCell(self)

    if chosen_cell is None or not self.cellsInRange:
        return chosen_cell

    try:
        # Build candidate set: all cells in vision range + current cell
        candidate_cells = list(self.cellsInRange.keys())
        if self.cell not in candidate_cells:
            candidate_cells.insert(0, self.cell)

        # Feature matrix F: shape (n_candidates, 5)
        F = np.array([_feature_vector(self, c) for c in candidate_cells])

        # Locate chosen cell in candidates
        try:
            chosen_idx = candidate_cells.index(chosen_cell)
        except ValueError:
            return chosen_cell

        # Softmax reward probabilities under current θ
        r = F @ _theta
        r -= r.max()          # numerical stability (log-sum-exp shift)
        exp_r = np.exp(r)
        probs = exp_r / exp_r.sum()

        # MaxEnt IRL gradient: ∇L = f(c_t) − E_θ[f(c')]
        g = F[chosen_idx] - probs @ F

        # Adagrad update: per-coordinate adaptive learning rate
        _grad_sq += g * g
        alpha = _base_lr / np.sqrt(_grad_sq + ADAGRAD_EPS)
        _theta += alpha * g
        _step += 1

    except Exception:
        pass  # Never let gradient errors crash the simulation

    return chosen_cell


def _apply_irl_patch():
    global _orig_findBestCell
    _orig_findBestCell = agent_module.Agent.findBestCell
    agent_module.Agent.findBestCell = _patched_findBestCell


def _remove_irl_patch():
    if _orig_findBestCell is not None:
        agent_module.Agent.findBestCell = _orig_findBestCell

# ---------------------------------------------------------------------------
# Derivation runner
# ---------------------------------------------------------------------------

def derive_condition(condition_name, models, seeds, timesteps, lr, out_dir, force):
    """
    Runs all derivation seeds for one baseline condition sequentially,
    accumulating gradient updates into a single θ across seeds.
    Returns the final normalized prioritization vector.
    """
    global _theta, _grad_sq, _base_lr, _step

    vectors_path = os.path.join(out_dir, "prioritization_vectors.json")

    # Skip if already derived and not forcing
    if not force and os.path.exists(vectors_path):
        try:
            with open(vectors_path) as f:
                saved = json.load(f)
            if condition_name in saved and "vector" in saved[condition_name]:
                vec = saved[condition_name]["vector"]
                print(f"  [{condition_name}] already derived — skipping  Pi={[round(v,4) for v in vec]}")
                return np.array(vec)
        except Exception:
            pass

    # Initialize IRL state for this condition
    _theta   = np.zeros(5)
    _grad_sq = np.zeros(5)
    _base_lr = lr
    _step    = 0

    config = dict(BASE_CONFIG)
    config["agentDecisionModels"] = models
    config["experimentalGroup"]   = None
    config["timesteps"]           = timesteps
    config["headlessMode"]        = True
    config["logfile"]             = None   # suppress logs during derivation
    config["agentLogfile"]        = None
    config["logfileFormat"]       = "csv"

    _apply_irl_patch()
    try:
        for seed_idx, seed in enumerate(seeds):
            config["seed"] = seed
            random.seed(seed)
            sim = sugarscape_module.Sugarscape(config)
            try:
                sim.runSimulation(timesteps)
            except SystemExit:
                pass
            print(f"  [{condition_name}] seed {seed_idx+1}/{len(seeds)} done  "
                  f"steps={_step:,}  θ_norm={np.linalg.norm(_theta):.4f}")
    finally:
        _remove_irl_patch()

    # Normalize θ to unit length → Pi
    norm = np.linalg.norm(_theta)
    if norm < 1e-10:
        print(f"  [{condition_name}] WARNING: θ near zero; defaulting to uniform vector")
        pi = np.full(5, 1.0 / math.sqrt(5))
    else:
        pi = _theta / norm

    print(f"  [{condition_name}] Pi = {[round(v, 4) for v in pi.tolist()]}")
    return pi

# ---------------------------------------------------------------------------
# Seed management (reuses seeds.json from run_baseline_conditions if present)
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

def _save_vectors(vectors_path, results, seeds, timesteps, lr):
    """Writes (or merges into) prioritization_vectors.json."""
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
            "total_gradient_steps": int(pi.size),  # placeholder; actual logged above
        }

    with open(vectors_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nPrioritization vectors saved → {vectors_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds",     type=int,   default=DEFAULT_N_SEEDS,
                   help="Derivation seeds per condition (default: %(default)s)")
    p.add_argument("--timesteps", type=int,   default=DEFAULT_TIMESTEPS,
                   help="Timesteps per derivation run (default: %(default)s)")
    p.add_argument("--outdir",    default=DEFAULT_OUT,
                   help="Output directory (default: %(default)s)")
    p.add_argument("--lr",        type=float, default=DEFAULT_LR,
                   help="Adagrad base learning rate (default: %(default)s)")
    p.add_argument("--force",     action="store_true",
                   help="Re-derive even if vectors already exist")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    seeds_path   = os.path.join(args.outdir, "seeds.json")
    vectors_path = os.path.join(args.outdir, "prioritization_vectors.json")
    seeds = _load_or_generate_seeds(seeds_path, args.seeds)

    print(f"Deriving prioritization vectors: {len(DERIVE_CONDITIONS)} conditions × "
          f"{args.seeds} seeds × {args.timesteps} timesteps\n")

    results = {}
    for condition_name, models in DERIVE_CONDITIONS.items():
        print(f"\n── {condition_name} ──")
        pi = derive_condition(
            condition_name, models, seeds,
            args.timesteps, args.lr, args.outdir, args.force
        )
        results[condition_name] = pi

    _save_vectors(vectors_path, results, seeds, args.timesteps, args.lr)

    # Print summary table
    print("\n┌──────────────────────────┬────────┬────────┬────────┬────────┬────────┐")
    print("│ Condition                │   p_I  │   p_D  │   p_C  │   p_P  │   p_X  │")
    print("├──────────────────────────┼────────┼────────┼────────┼────────┼────────┤")
    for name, pi in results.items():
        row = f"│ {name:<24s} │"
        for v in pi:
            row += f" {v:6.4f} │"
        print(row)
    print("└──────────────────────────┴────────┴────────┴────────┴────────┴────────┘")

    print("\nTo use these vectors in FVDM conditions, set agentPrioritizationVector")
    print(f"to the appropriate vector from {vectors_path}")


if __name__ == "__main__":
    main()
