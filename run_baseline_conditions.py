#!/usr/bin/env python3
"""
run_baseline_conditions.py

Runs the 4 baseline experimental conditions (Raw Sugarscape, Egoist, Altruist, Bentham)
across N_SEEDS matched seeds and records:
  - Per-timestep runtimeStats CSV (population, Gini, TTL, wealth, interaction counts, tribe stats)
  - Per-agent cell-choice felicific profiles CSV: E^imm(c*) and E^fut(c*) each timestep
  - Per-tribe population and wealth metrics CSV each timestep
  - Final outcome measures JSON per run
  - Cross-run outcomes summary CSV

Also supports heterogeneous conditions via optional experimentalGroup parameter.

Usage:
    python run_baseline_conditions.py [--seeds N] [--parallel N] [--outdir PATH] [--force]
"""

import argparse
import csv
import json
import math
import multiprocessing
import os
import random
import sys

# ---------------------------------------------------------------------------
# Project import path
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import agent as agent_module
import sugarscape as sugarscape_module

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

DEFAULT_OUT = os.path.join(_SCRIPT_DIR, "data", "baseline")
DEFAULT_N_SEEDS = 30

_VECTORS_FILE = os.path.join(DEFAULT_OUT, "prioritization_vectors.json")


def _load_derived_vectors():
    """Load IRL-derived prioritization vectors if available, else return None."""
    if not os.path.exists(_VECTORS_FILE):
        return None
    try:
        with open(_VECTORS_FILE) as f:
            data = json.load(f)
        return {k: v["vector"] for k, v in data.items() if "vector" in v}
    except Exception:
        return None


def _build_conditions():
    """
    Builds the full 8-condition dict.
    Conditions 1–4: baseline rule-based agents.
    Conditions 5–8: FVDM agents using IRL-derived prioritization vectors.
    FVDM conditions are included only when prioritization_vectors.json exists.
    """
    conds = {
        "rawSugarscape": {
            "agentDecisionModels": ["none"],
            "experimentalGroup": None,
        },
        "egoist": {
            "agentDecisionModels": ["egoist"],
            "experimentalGroup": None,
        },
        "altruist": {
            "agentDecisionModels": ["altruist"],
            "experimentalGroup": None,
        },
        "bentham": {
            "agentDecisionModels": ["bentham"],
            "experimentalGroup": None,
        },
    }

    vectors = _load_derived_vectors()
    if vectors is not None:
        fvdm_map = {
            "fvdmRawSugarscape": "rawSugarscape",
            "fvdmEgoist":        "egoist",
            "fvdmAltruist":      "altruist",
            "fvdmBentham":       "bentham",
        }
        for fvdm_name, base_name in fvdm_map.items():
            if base_name in vectors:
                conds[fvdm_name] = {
                    "agentDecisionModels": ["fvdm"],
                    "experimentalGroup": None,
                    "agentPrioritizationVector": vectors[base_name],
                }
        print(f"Loaded {len(vectors)} prioritization vectors → FVDM conditions added.")
    else:
        print(f"No prioritization vectors found at {_VECTORS_FILE}.")
        print("Run 'make derive' first to add FVDM conditions 5–8.")

    return conds


CONDITIONS = _build_conditions()

# Base simulation config matching the project defaults in config.json
BASE_CONFIG = {
    "agentAggressionFactor": [1, 1],
    "agentBaseInterestRate": [0.05, 0.10],
    "agentConditions": ["none"],
    "agentDecisionModels": ["none"],
    "agentDecisionModelFactor": [1, 1],
    "agentDecisionModelLookaheadDiscount": [0.5, 0.5],
    "agentDecisionModelLookaheadFactor": 0.5,
    "agentDecisionModelTribalFactor": [-1, -1],
    "agentDepressionPercentage": 0,
    "agentDiseaseProtectionChance": [0.0, 0.0],
    "agentDynamicSelfishnessFactor": [0.0, 0.0],
    "agentDynamicTemperanceFactor": [0, 0],
    "agentFemaleInfertilityAge": [40, 50],
    "agentFemaleFertilityAge": [12, 15],
    "agentFertilityFactor": [1, 1],
    "agentImmuneSystemLength": 35,
    "agentInheritancePolicy": "children",
    "agentLeader": False,
    "agentLendingFactor": [1, 1],
    "agentLoanDuration": [5, 5],
    "agentLogfile": None,
    "agentLookaheadFactor": [1, 10],
    "agentMaleInfertilityAge": [50, 60],
    "agentMaleFertilityAge": [12, 15],
    "agentMaleToFemaleRatio": 1,
    "agentMaxAge": [60, 100],
    "agentMaxFriends": [5, 10],
    "agentMovement": [1, 6],
    "agentMovementMode": "cardinal",
    "agentReplacements": 0,
    "agentSelfishnessFactor": [-1, -1],
    "agentSpiceMetabolism": [1, 4],
    "agentStartingSpice": [10, 40],
    "agentStartingSugar": [10, 40],
    "agentSugarMetabolism": [1, 4],
    "agentTagging": True,
    "agentTagPreferences": False,
    "agentTagStringLength": 11,
    "agentTradeFactor": [1, 1],
    "agentTemperanceFactor": [0, 0],
    "agentUniversalSpice": [0, 0],
    "agentUniversalSugar": [0, 0],
    "agentVision": [1, 6],
    "agentVisionMode": "cardinal",
    "debugMode": ["none"],
    "diseaseAggressionPenalty": [-1, 1],
    "diseaseFertilityPenalty": [-1, 1],
    "diseaseFriendlinessPenalty": [0, 0],
    "diseaseHappinessPenalty": [0, 0],
    "diseaseIncubationPeriod": [0, 0],
    "diseaseList": [],
    "diseaseMovementPenalty": [0, 0],
    "diseaseSpiceMetabolismPenalty": [1, 3],
    "diseaseSugarMetabolismPenalty": [1, 3],
    "diseaseTagStringLength": [11, 21],
    "diseaseTimeframe": [0, 0],
    "diseaseTransmissionChance": [1.0, 1.0],
    "diseaseVisionPenalty": [-1, 1],
    "environmentEquator": -1,
    "environmentFile": None,
    "environmentHeight": 50,
    "environmentMaxCombatLoot": 2,
    "environmentMaxSpice": 4,
    "environmentMaxSugar": 4,
    "environmentMaxTribes": 3,
    "environmentPollutionDiffusionDelay": 0,
    "environmentPollutionDiffusionTimeframe": [0, 0],
    "environmentPollutionTimeframe": [0, 0],
    "environmentQuadrantSizeFactor": 1,
    "environmentSeasonalGrowbackDelay": 0,
    "environmentSeasonInterval": 0,
    "environmentSpiceConsumptionPollutionFactor": 0,
    "environmentSpicePeaks": [[15, 15, 4], [35, 35, 4]],
    "environmentSpiceProductionPollutionFactor": 0,
    "environmentSpiceRegrowRate": 1,
    "environmentStartingQuadrants": [1, 2, 3, 4],
    "environmentSugarConsumptionPollutionFactor": 0,
    "environmentSugarPeaks": [[15, 35, 4], [35, 15, 4]],
    "environmentSugarProductionPollutionFactor": 0,
    "environmentSugarRegrowRate": 1,
    "environmentTribePerQuadrant": False,
    "environmentUniversalSpiceIncomeInterval": 0,
    "environmentUniversalSugarIncomeInterval": 0,
    "environmentWidth": 50,
    "environmentWraparound": True,
    "experimentalGroup": None,
    "headlessMode": True,
    "interfaceHeight": 1400,
    "interfaceWidth": 1200,
    "keepAlivePostExtinction": False,
    "keepAliveAtEnd": False,
    "logfile": None,
    "logfileFormat": "csv",
    "neighborhoodMode": "vonNeumann",
    "profileMode": False,
    "screenshots": False,
    "seed": -1,
    "startingAgents": 250,
    "startingDiseases": 50,
    "startingDiseasesPerAgent": [0, 0],
    "timesteps": 1000,
}

# ---------------------------------------------------------------------------
# Felicific coordinate computation
# ---------------------------------------------------------------------------

def compute_felicific_vectors(ag, chosen_cell):
    """
    Compute E^imm(c*) and E^fut(c*) for agent ag choosing chosen_cell.

    Called at decision time (before the agent moves), so chosen_cell still
    holds the pre-collection resource values.  Derives coordinates analytically
    from Bentham.findEthicalValueOfCell in ethics.py.

    Returns (E_imm, E_fut) as dicts:
      E^imm: {I, D, C, P, X}   — immediate felicific coordinates
      E^fut: {If, Df, Cf, Pf, Xf} — future felicific coordinates
    """
    env = ag.cell.environment

    # Cell wealth at decision time, including potential combat loot
    combat_cap = env.maxCombatLoot * 2
    W_c = chosen_cell.sugar + chosen_cell.spice
    W_c_max = chosen_cell.maxSugar + chosen_cell.maxSpice
    if chosen_cell.agent is not None and chosen_cell.agent is not ag:
        loot = min(chosen_cell.agent.sugar + chosen_cell.agent.spice, combat_cap)
        W_c += loot
        W_c_max += loot

    cell_neighbor_wealth = chosen_cell.findNeighborWealth()
    W_max = env.globalMaxSugar + env.globalMaxSpice
    m_i = ag.sugarMetabolism + ag.spiceMetabolism
    H_i = ag.findTimeToLive()

    # Certainty: chosen cell is always reachable (it was selected from cellsInRange)
    C = 1.0

    # Distance from agent's current cell to chosen cell (for Propinquity)
    dx = abs(chosen_cell.x - ag.cell.x)
    dy = abs(chosen_cell.y - ag.cell.y)
    if env.wraparound:
        dx = min(dx, env.width - dx)
        dy = min(dy, env.height - dy)
    d = math.sqrt(dx * dx + dy * dy)

    # Propinquity: spatial nearness as proxy for temporal delay
    P = 1.0 / (1.0 + d)

    # Intensity I = 1 / ((1+H_i)(1+pollution_c))
    denom_I = (1.0 + H_i) * (1.0 + chosen_cell.pollution)
    I = 1.0 / denom_I if denom_I > 0.0 else 0.0

    # Duration D = W_c / (m_i * W_c_max)
    D = W_c / (m_i * W_c_max) if m_i > 0.0 and W_c_max > 0.0 else 0.0

    # Extent X = |N_i| / |V_i|
    cells_in_range = len(ag.cellsInRange) if ag.cellsInRange else 1
    neighborhood_size = len(ag.neighborhood) if ag.neighborhood else 1
    X = neighborhood_size / cells_in_range if cells_in_range > 0 else 1.0

    E_imm = {
        "I": round(I, 6),
        "D": round(D, 6),
        "C": C,
        "P": P,
        "X": round(X, 6),
    }

    # Future intensity I_f = cellNeighborWealth / (W_max * |adj(ag.cell)|)
    adj_count = len(ag.cell.neighbors)
    I_f = cell_neighbor_wealth / (W_max * adj_count) if W_max > 0.0 and adj_count > 0 else 0.0

    # Future duration D_f = (W_c - m_i) / (m_i * W_c_max)
    D_f = (W_c - m_i) / (m_i * W_c_max) if m_i > 0.0 and W_c_max > 0.0 else 0.0

    # Lookahead discount γ — only when lookahead is enabled
    gamma = ag.decisionModelLookaheadDiscount if ag.decisionModelLookaheadFactor != 0 else 0.0
    P_f = gamma

    # Future extent X_f = |N_i(c*)| / |V_i| from chosen cell's vantage
    if ag.decisionModelLookaheadFactor != 0:
        future_n = len(ag.findNeighborhood(chosen_cell))
        X_f = future_n / cells_in_range if cells_in_range > 0 else 1.0
    else:
        X_f = 1.0

    E_fut = {
        "If": round(I_f, 6),
        "Df": round(D_f, 6),
        "Cf": C,
        "Pf": P_f,
        "Xf": round(X_f, 6),
    }

    return E_imm, E_fut


# ---------------------------------------------------------------------------
# Monkey-patch infrastructure
# ---------------------------------------------------------------------------

# Module-level buffers; each worker process has its own copy.
_felicific_buffer = []
_tribe_buffer = []
_original_findBestCell = None
_original_doTimestep = None


def _patched_findBestCell(self):
    """Wraps Agent.findBestCell to record E^imm/E^fut for the chosen cell."""
    global _felicific_buffer
    best_cell = _original_findBestCell(self)
    if best_cell is None:
        return best_cell
    try:
        E_imm, E_fut = compute_felicific_vectors(self, best_cell)
        _felicific_buffer.append({
            "timestep": self.timestep,
            "agentID": self.ID,
            "tribe": self.tribe,
            "decisionModel": self.decisionModel,
            "cellX": best_cell.x,
            "cellY": best_cell.y,
            "E_imm_I": E_imm["I"],
            "E_imm_D": E_imm["D"],
            "E_imm_C": E_imm["C"],
            "E_imm_P": E_imm["P"],
            "E_imm_X": E_imm["X"],
            "E_fut_If": E_fut["If"],
            "E_fut_Df": E_fut["Df"],
            "E_fut_Cf": E_fut["Cf"],
            "E_fut_Pf": E_fut["Pf"],
            "E_fut_Xf": E_fut["Xf"],
        })
    except Exception:
        pass  # Never let recording errors crash the simulation
    return best_cell


def _patched_doTimestep(self):
    """Wraps Sugarscape.doTimestep to capture per-tribe stats after each step."""
    global _tribe_buffer
    _original_doTimestep(self)
    # Collect tribe breakdown from live agents after the step finishes
    tribes = {}
    for ag in self.agents:
        t = ag.tribe
        if t not in tribes:
            tribes[t] = {"population": 0, "totalWealth": 0.0}
        tribes[t]["population"] += 1
        tribes[t]["totalWealth"] += ag.sugar + ag.spice
    for tribe_id, stats in sorted(tribes.items()):
        _tribe_buffer.append({
            "timestep": self.timestep,
            "tribe": tribe_id,
            "population": stats["population"],
            "totalWealth": round(stats["totalWealth"], 2),
            "meanWealth": round(stats["totalWealth"] / stats["population"], 2)
                          if stats["population"] > 0 else 0.0,
        })


def _apply_patches():
    global _original_findBestCell, _original_doTimestep
    _original_findBestCell = agent_module.Agent.findBestCell
    _original_doTimestep = sugarscape_module.Sugarscape.doTimestep
    agent_module.Agent.findBestCell = _patched_findBestCell
    sugarscape_module.Sugarscape.doTimestep = _patched_doTimestep


def _remove_patches():
    if _original_findBestCell is not None:
        agent_module.Agent.findBestCell = _original_findBestCell
    if _original_doTimestep is not None:
        sugarscape_module.Sugarscape.doTimestep = _original_doTimestep


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

FELICIFIC_FIELDS = [
    "timestep", "agentID", "tribe", "decisionModel", "cellX", "cellY",
    "E_imm_I", "E_imm_D", "E_imm_C", "E_imm_P", "E_imm_X",
    "E_fut_If", "E_fut_Df", "E_fut_Cf", "E_fut_Pf", "E_fut_Xf",
]

TRIBE_FIELDS = ["timestep", "tribe", "population", "totalWealth", "meanWealth"]

OUTCOME_FIELDS = [
    "condition", "seed",
    "finalPopulation", "extinct",
    "finalGini", "finalMeanTimeToLive",
    "finalAgentWealth", "finalEnvWealth",
    "totalCombatDeaths", "totalStarvationDeaths", "totalAgingDeaths",
    "totalTradeVolume", "remainingTribes", "timestepsRun",
]


def _write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _compute_outcome_from_csv(condition_name, seed, timestep_csv_path):
    """Read the timestep log CSV to compute point-in-time and cumulative outcome measures."""
    try:
        rows = _read_csv(timestep_csv_path)
    except Exception:
        return None
    if not rows:
        return None

    last = rows[-1]
    total_combat    = sum(int(r.get("agentCombatDeaths", 0))     for r in rows)
    total_starve    = sum(int(r.get("agentStarvationDeaths", 0)) for r in rows)
    total_aging     = sum(int(r.get("agentAgingDeaths", 0))      for r in rows)
    total_trade_vol = sum(float(r.get("tradeVolume", 0))         for r in rows)

    return {
        "condition":           condition_name,
        "seed":                seed,
        "finalPopulation":     int(last.get("population", 0)),
        "extinct":             int(last.get("population", 0)) == 0,
        "finalGini":           round(float(last.get("giniCoefficient", 0)), 4),
        "finalMeanTimeToLive": round(float(last.get("agentMeanTimeToLive", 0)), 2),
        "finalAgentWealth":    round(float(last.get("agentWealthTotal", 0)), 2),
        "finalEnvWealth":      round(float(last.get("environmentWealthTotal", 0)), 2),
        "totalCombatDeaths":   total_combat,
        "totalStarvationDeaths": total_starve,
        "totalAgingDeaths":    total_aging,
        "totalTradeVolume":    round(total_trade_vol, 2),
        "remainingTribes":     int(last.get("remainingTribes", 0)),
        "timestepsRun":        int(last.get("timestep", 0)),
        "complete":            True,
    }


# ---------------------------------------------------------------------------
# Simulation worker
# ---------------------------------------------------------------------------

def run_single_simulation(args):
    """
    Worker: runs one (condition, seed) pair and writes all output files.
    Safe to run in a multiprocessing pool; each process patches its own
    copy of the agent and sugarscape modules.
    """
    condition_name, condition_cfg, seed, out_dir, force = args

    prefix = os.path.join(out_dir, f"{condition_name}_{seed}")
    timestep_log  = prefix + "_timestep.csv"
    agent_log     = prefix + "_agents.csv"
    felicific_log = prefix + "_felicific.csv"
    tribe_log     = prefix + "_tribes.csv"
    outcome_file  = prefix + "_outcome.json"

    # Skip already-completed runs unless --force
    if not force and os.path.exists(outcome_file):
        try:
            with open(outcome_file) as f:
                saved = json.load(f)
            if saved.get("complete"):
                return condition_name, seed, "skipped"
        except Exception:
            pass

    # Reset per-process buffers
    global _felicific_buffer, _tribe_buffer
    _felicific_buffer = []
    _tribe_buffer = []

    # Build per-run simulation config
    config = dict(BASE_CONFIG)
    config["agentDecisionModels"] = condition_cfg["agentDecisionModels"]
    config["experimentalGroup"]   = condition_cfg.get("experimentalGroup")
    config["seed"]         = seed
    config["logfile"]      = timestep_log
    config["agentLogfile"] = agent_log
    config["logfileFormat"] = "csv"
    config["headlessMode"] = True
    # Pass FVDM prioritization vector when present (conditions 5–8)
    if "agentPrioritizationVector" in condition_cfg:
        config["agentPrioritizationVector"] = condition_cfg["agentPrioritizationVector"]

    _apply_patches()
    try:
        random.seed(seed)
        sim = sugarscape_module.Sugarscape(config)
        try:
            sim.runSimulation(config["timesteps"])
        except SystemExit:
            pass  # endSimulation() calls exit(0); catch it so we continue

        # Write felicific profiles
        _write_csv(felicific_log, FELICIFIC_FIELDS, _felicific_buffer)

        # Write per-tribe metrics
        _write_csv(tribe_log, TRIBE_FIELDS, _tribe_buffer)

        # Derive outcome measures from the written timestep CSV
        outcome = _compute_outcome_from_csv(condition_name, seed, timestep_log)
        if outcome is None:
            outcome = {
                "condition": condition_name,
                "seed": seed,
                "complete": False,
                "error": "timestep log missing or empty",
            }
        with open(outcome_file, "w") as f:
            json.dump(outcome, f, indent=2)

        return condition_name, seed, "completed"

    except Exception as exc:
        return condition_name, seed, f"error: {exc}"
    finally:
        _remove_patches()


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def _load_or_generate_seeds(seeds_path, n_seeds):
    if os.path.exists(seeds_path):
        with open(seeds_path) as f:
            seeds = json.load(f)
        if len(seeds) >= n_seeds:
            print(f"Loaded {len(seeds)} existing seeds from {seeds_path}")
            return seeds[:n_seeds]
    # Generate fresh matched seeds
    seeds = []
    while len(seeds) < n_seeds:
        s = random.randint(0, sys.maxsize)
        if s not in seeds:
            seeds.append(s)
    with open(seeds_path, "w") as f:
        json.dump(seeds, f)
    print(f"Generated {n_seeds} new seeds → {seeds_path}")
    return seeds


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _write_summary(out_dir, conditions, seeds):
    rows = []
    for cond in conditions:
        for seed in seeds:
            outcome_file = os.path.join(out_dir, f"{cond}_{seed}_outcome.json")
            if not os.path.exists(outcome_file):
                continue
            try:
                with open(outcome_file) as f:
                    rows.append(json.load(f))
            except Exception:
                continue
    summary_path = os.path.join(out_dir, "outcomes_summary.csv")
    _write_csv(summary_path, OUTCOME_FIELDS, rows)
    print(f"Summary written → {summary_path}  ({len(rows)} runs)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds",    type=int, default=DEFAULT_N_SEEDS,
                   help="Number of random seeds (default: %(default)s)")
    p.add_argument("--parallel", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="Parallel simulation processes (default: CPU count - 1)")
    p.add_argument("--outdir",   default=DEFAULT_OUT,
                   help="Output directory (default: %(default)s)")
    p.add_argument("--timesteps", type=int, default=None,
                   help="Simulation timesteps per run (default: from BASE_CONFIG = 1000)")
    p.add_argument("--agents",   type=int, default=None,
                   help="Starting agents per run (default: from BASE_CONFIG = 250)")
    p.add_argument("--force",    action="store_true",
                   help="Re-run already-completed simulations")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    seeds_path = os.path.join(out_dir, "seeds.json")
    seeds = _load_or_generate_seeds(seeds_path, args.seeds)

    # Apply any CLI overrides to the base config
    if args.timesteps is not None:
        BASE_CONFIG["timesteps"] = args.timesteps
    if args.agents is not None:
        BASE_CONFIG["startingAgents"] = args.agents

    # Build task list: (condition_name, condition_cfg, seed, out_dir, force)
    tasks = [
        (name, cfg, seed, out_dir, args.force)
        for name, cfg in CONDITIONS.items()
        for seed in seeds
    ]
    total = len(tasks)
    print(f"Running {len(CONDITIONS)} conditions × {len(seeds)} seeds = {total} simulations"
          f" ({args.parallel} parallel)")

    n_parallel = min(args.parallel, total)

    if n_parallel <= 1:
        completed = 0
        for task in tasks:
            cond, seed, status = run_single_simulation(task)
            completed += 1
            print(f"  [{completed}/{total}] {cond} seed={seed}: {status}")
    else:
        with multiprocessing.Pool(processes=n_parallel) as pool:
            completed = 0
            for cond, seed, status in pool.imap_unordered(run_single_simulation, tasks):
                completed += 1
                print(f"  [{completed}/{total}] {cond} seed={seed}: {status}")

    _write_summary(out_dir, CONDITIONS.keys(), seeds)
    print("Done.")


if __name__ == "__main__":
    main()
