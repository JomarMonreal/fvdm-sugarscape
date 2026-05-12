#!/usr/bin/env python3
"""
pilot_behavior.py

Diagnostic script: runs a short heterogeneous simulation and records per-type
cell-selection statistics to verify that egoist, altruist, rawSugarscape, and
bentham agents make meaningfully different choices.

Three signals are measured at every findBestCell call:
  1. contested_rate  — fraction of moves that go to an occupied cell (O=1)
  2. avg_occ_wealth  — when O=1, mean normalized wealth of the occupant
  3. avg_nb_wealth   — mean normalized wealth of destination-cell neighbors

If behavioral differences exist in the simulation they will appear here before
any feature-space redesign is attempted.

Usage:
    python pilot_behavior.py [--seeds N] [--timesteps N] [--agents N]
"""

import argparse
import os
import random
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import agent as agent_module
import sugarscape as sugarscape_module
from run_baseline_conditions import BASE_CONFIG

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_TO_CONDITION = {
    "none":     "rawSugarscape",
    "egoist":   "egoist",
    "altruist": "altruist",
    "bentham":  "bentham",
}
HETERO_MODELS = list(MODEL_TO_CONDITION.keys())

# ---------------------------------------------------------------------------
# Per-process accumulator
# ---------------------------------------------------------------------------

# model -> {"moves": int, "contested": int,
#           "occ_wealth_sum": float, "nb_wealth_sum": float, "nb_wealth_n": int}
_stats = {}

_orig_findBestCell = None
_W_max = 1.0   # updated each seed from environment


def _reset_stats():
    global _stats
    _stats = {
        m: {"moves": 0, "contested": 0,
            "occ_wealth_sum": 0.0, "occ_wealth_n": 0,
            "nb_wealth_sum": 0.0, "nb_wealth_n": 0}
        for m in MODEL_TO_CONDITION
    }


def _patched_findBestCell(self):
    chosen = _orig_findBestCell(self)
    if chosen is None:
        return chosen

    model = self.decisionModel
    if model not in _stats:
        return chosen

    s = _stats[model]
    s["moves"] += 1

    W_max = max(_W_max, 1.0)
    occupant = chosen.agent

    # Contested: occupied by a different-tribe agent
    if occupant is not None and occupant is not self and occupant.tribe != self.tribe:
        s["contested"] += 1
        occ_w = (occupant.sugar + occupant.spice) / W_max
        s["occ_wealth_sum"] += occ_w
        s["occ_wealth_n"]   += 1

    # Neighbor wealth at destination: agents adjacent to chosen cell
    nb_agents = [c.agent for c in chosen.neighbors.values() if c.agent is not None]
    if nb_agents:
        nb_w = sum(a.sugar + a.spice for a in nb_agents) / (len(nb_agents) * W_max)
        s["nb_wealth_sum"] += nb_w
        s["nb_wealth_n"]   += 1

    return chosen


def _apply_patch():
    global _orig_findBestCell
    _orig_findBestCell = agent_module.Agent.findBestCell
    agent_module.Agent.findBestCell = _patched_findBestCell


def _remove_patch():
    if _orig_findBestCell is not None:
        agent_module.Agent.findBestCell = _orig_findBestCell

# ---------------------------------------------------------------------------
# Run one seed
# ---------------------------------------------------------------------------

def _run_seed(seed, timesteps, agents, seed_idx, n_seeds):
    global _W_max
    _reset_stats()

    config = dict(BASE_CONFIG)
    config["agentDecisionModels"] = HETERO_MODELS
    config["experimentalGroup"]   = None
    config["timesteps"]           = timesteps
    config["startingAgents"]      = agents
    config["headlessMode"]        = True
    config["logfile"]             = None
    config["agentLogfile"]        = None
    config["logfileFormat"]       = "csv"
    config["seed"]                = seed

    _apply_patch()
    try:
        random.seed(seed)
        sim = sugarscape_module.Sugarscape(config)
        # Grab W_max from environment after init
        try:
            env = sim.environment
            _W_max = max(
                max(c.maxSugar for row in env.grid for c in row),
                1.0,
            )
        except Exception:
            _W_max = 400.0
        try:
            sim.runSimulation(timesteps)
        except SystemExit:
            pass
    finally:
        _remove_patch()

    print(f"  seed {seed_idx + 1}/{n_seeds} done")
    return {m: dict(v) for m, v in _stats.items()}

# ---------------------------------------------------------------------------
# Aggregate and print
# ---------------------------------------------------------------------------

def _merge(totals, result):
    for model, s in result.items():
        t = totals[model]
        t["moves"]          += s["moves"]
        t["contested"]      += s["contested"]
        t["occ_wealth_sum"] += s["occ_wealth_sum"]
        t["occ_wealth_n"]   += s["occ_wealth_n"]
        t["nb_wealth_sum"]  += s["nb_wealth_sum"]
        t["nb_wealth_n"]    += s["nb_wealth_n"]


def _print_table(totals, seeds, timesteps):
    W = 88
    print(f"\n{'═' * W}")
    print(f"  Behavioral pilot  ({seeds} seed(s) × {timesteps} timesteps, heterogeneous population)")
    print(f"{'─' * W}")
    print(f"  {'Type':<18}  {'moves':>8}  {'contested%':>11}  "
          f"{'avg_occ_wealth':>15}  {'avg_nb_wealth':>14}")
    print(f"{'─' * W}")

    for model, cond in MODEL_TO_CONDITION.items():
        s = totals[model]
        moves      = s["moves"]
        contested  = s["contested"]
        c_rate     = 100.0 * contested / moves if moves > 0 else 0.0
        occ_w      = s["occ_wealth_sum"] / s["occ_wealth_n"] if s["occ_wealth_n"] > 0 else float("nan")
        nb_w       = s["nb_wealth_sum"]  / s["nb_wealth_n"]  if s["nb_wealth_n"]  > 0 else float("nan")

        occ_str = f"{occ_w:.4f}" if s["occ_wealth_n"] > 0 else "  —"
        nb_str  = f"{nb_w:.4f}"  if s["nb_wealth_n"]  > 0 else "  —"

        print(f"  {cond:<18}  {moves:>8,}  {c_rate:>10.2f}%  {occ_str:>15}  {nb_str:>14}")

    print(f"{'═' * W}")
    print()
    print("  Signals to interpret:")
    print("  contested%     — egoist > rawSugarscape > altruist expected")
    print("  avg_occ_wealth — egoist should pick richer targets than altruist")
    print("  avg_nb_wealth  — altruist should move to cells with wealthier neighbors")
    print(f"{'═' * W}\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds",     type=int, default=5,
                   help="Number of seeds to run (default: %(default)s)")
    p.add_argument("--timesteps", type=int, default=1000,
                   help="Timesteps per seed (default: %(default)s)")
    p.add_argument("--agents",    type=int, default=250,
                   help="Starting agents (default: %(default)s)")
    return p.parse_args()


def main():
    args  = parse_args()
    seeds = [random.randint(0, 2**31) for _ in range(args.seeds)]

    print(f"\nBehavioral pilot: {args.seeds} seed(s) × {args.timesteps} timesteps "
          f"| {args.agents} agents | heterogeneous population\n")

    totals = {
        m: {"moves": 0, "contested": 0,
            "occ_wealth_sum": 0.0, "occ_wealth_n": 0,
            "nb_wealth_sum": 0.0, "nb_wealth_n": 0}
        for m in MODEL_TO_CONDITION
    }

    for idx, seed in enumerate(seeds):
        result = _run_seed(seed, args.timesteps, args.agents, idx, args.seeds)
        _merge(totals, result)

    _print_table(totals, args.seeds, args.timesteps)


if __name__ == "__main__":
    main()
