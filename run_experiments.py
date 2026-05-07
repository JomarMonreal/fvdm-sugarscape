#!/usr/bin/env python3
"""
run_experiments.py
------------------
Baseline replication experiment runner for the Digital Terrarium / Sugarscape simulation.

Runs homogeneous conditions (egoist, altruist, rawSugarscape, bentham) and one
heterogeneous (mixed) condition across N seeds, collecting per-timestep and
aggregated metrics for replication of the original Digital Terrarium study.

Recorded metrics per timestep:
  - population
  - agent mean wealth, min wealth, max wealth, societal wealth (agentWealthTotal)
  - Gini coefficient (wealth inequality)
  - agent deaths (total, aging, starvation, combat)
  - agents born
  - mean time-to-live (TTL)
  - trade volume, mean trade price
  - action-selection frequencies:
      movementActions, combatActions, tradeActions, reproductionActions, lendingActions

Aggregated (final) metrics:
  - extinction (bool), final population, total deaths, total born
  - execution time (seconds)
  - mean/std of per-timestep metrics across seeds

Usage:
  python run_experiments.py [options]

See --help for full flag list.
"""

import argparse
import csv
import json
import math
import multiprocessing
import os
import random
import subprocess
import sys
import time

# ─────────────────────────────────────────────────────────────────
# Condition definitions
# ─────────────────────────────────────────────────────────────────

CONDITIONS = {
    "rawSugarscape":  ["rawSugarscape"],
    "egoist":         ["egoist"],
    "altruist":       ["altruist"],
    "bentham":        ["bentham"],
    "heterogeneous":  ["egoist", "altruist", "bentham"],
}

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def load_base_config(config_path: str) -> dict:
    """Load and return the sugarscapeOptions block from the master config."""
    with open(config_path, "r") as f:
        full = json.load(f)
    opts = full.get("sugarscapeOptions", full)
    return opts


def make_run_config(base: dict, seed: int, decision_models: list,
                    timesteps: int, num_agents: int, output_dir: str,
                    condition_name: str) -> dict:
    """Build a per-run configuration dict from the base options."""
    cfg = dict(base)   # shallow copy is fine for primitives / lists

    cfg["seed"] = seed
    cfg["agentDecisionModels"] = decision_models
    cfg["timesteps"] = timesteps
    cfg["startingAgents"] = num_agents
    cfg["startingDiseases"] = 0

    # Force headless, no GUI, no interactive output
    cfg["headlessMode"] = True
    cfg["debugMode"] = ["none"]
    cfg["keepAlivePostExtinction"] = False
    cfg["keepAliveAtEnd"] = False
    cfg["screenshots"] = False
    cfg["profileMode"] = False

    # Per-run log paths — agent log disabled for performance
    cfg["logfile"] = os.path.join(output_dir, f"{condition_name}_{seed}.json")
    cfg["agentLogfile"] = None   # not needed: action counts now in main log
    cfg["logfileFormat"] = "json"

    return cfg


def write_run_config(cfg: dict, path: str):
    """Serialise a run config dict to a .config file."""
    with open(path, "w") as f:
        json.dump(cfg, f)


def run_one_simulation(args):
    """
    Worker function executed in each subprocess.
    args: (config_path, python_alias, job_idx, total_jobs, counter, lock)
    Returns (config_path, duration) on completion.
    """
    config_path, python_alias, job_idx, total_jobs, counter, lock = args
    cmd = [python_alias, "sugarscape.py", "--conf", config_path]
    
    start = time.time()
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    duration = time.time() - start
    
    with lock:
        counter.value += 1
        pct = (counter.value / total_jobs) * 100
        print(f"\r  [{counter.value:>4}/{total_jobs}]  {pct:5.1f}%  last: {os.path.basename(config_path):<50}",
              end="", flush=True)
    return config_path, duration


# ─────────────────────────────────────────────────────────────────
# Log parsing
# ─────────────────────────────────────────────────────────────────

def safe_json_load(path: str):
    """Return parsed JSON or None if file is missing / malformed."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def parse_sugarscape_log(log_path: str, condition: str,
                          seed: int, timesteps: int, duration: float = 0.0):
    """
    Parse a completed simulation log and return:
      - per_timestep: list of dicts (one per recorded timestep)
      - summary: dict of aggregated/final stats
    Returns (None, None) if logs are invalid.

    Action counts (movementActions, combatActions, tradeActions,
    reproductionActions, lendingActions) are now written directly to
    the main simulation log by sugarscape.py — no agent log needed.
    """
    sim_log = safe_json_load(log_path)

    if sim_log is None or len(sim_log) == 0:
        return None, None

    # ── Build per-timestep records from main sim log ──
    per_timestep = []
    total_deaths = 0
    total_born = 0
    extinct = False
    final_pop = 0

    for entry in sim_log:
        ts = int(entry.get("timestep", 0))
        pop = int(entry.get("population", 0))
        deaths_this_ts = int(entry.get("agentDeaths", 0))
        born_this_ts = int(entry.get("agentsBorn", 0))
        total_deaths += deaths_this_ts
        total_born += born_this_ts
        if pop == 0:
            extinct = True

        # Action counts from main log (written by sugarscape.updateRuntimeStats)
        mv = int(entry.get("actionMovements", 0))
        cb = int(entry.get("actionCombats", 0))
        tr = int(entry.get("actionTrades", 0))
        rp = int(entry.get("actionReproductions", 0))
        ln = int(entry.get("actionLendings", 0))
        total_actions = mv + cb + tr + rp + ln

        rec = {
            "condition": condition,
            "seed": seed,
            "timestep": ts,
            # Population
            "population": pop,
            # Wealth
            "meanWealth": float(entry.get("meanWealth", 0)),
            "minWealth": float(entry.get("minWealth", 0)),
            "maxWealth": float(entry.get("maxWealth", 0)),
            "societalWealth": float(entry.get("agentWealthTotal", 0)),
            "giniCoefficient": float(entry.get("giniCoefficient", 0)),
            # Deaths
            "agentDeaths": deaths_this_ts,
            "agentAgingDeaths": int(entry.get("agentAgingDeaths", 0)),
            "agentStarvationDeaths": int(entry.get("agentStarvationDeaths", 0)),
            "agentCombatDeaths": int(entry.get("agentCombatDeaths", 0)),
            "agentsBorn": born_this_ts,
            # Survival
            "meanTimeToLive": float(entry.get("agentMeanTimeToLive", 0)),
            "meanAge": float(entry.get("meanAge", 0)),
            "meanAgeAtDeath": float(entry.get("meanAgeAtDeath", 0)),
            # Trade
            "tradeVolume": float(entry.get("tradeVolume", 0)),
            "meanTradePrice": float(entry.get("meanTradePrice", 0)),
            # Happiness
            "meanHappiness": float(entry.get("meanHappiness", 0)),
            # Sickness
            "sickAgents": int(entry.get("sickAgents", 0)),
            # Action frequencies
            "movementActions": mv,
            "combatActions": cb,
            "tradeActions": tr,
            "reproductionActions": rp,
            "lendingActions": ln,
            "totalActions": total_actions,
        }
        per_timestep.append(rec)
        final_pop = pop

    # ── Build summary/aggregated record ──
    summary = {
        "condition": condition,
        "seed": seed,
        "executionTime": round(duration, 2),
        "extinct": extinct,
        "finalPopulation": final_pop,
        "totalDeaths": total_deaths,
        "totalBorn": total_born,
        "finalTimestep": per_timestep[-1]["timestep"] if per_timestep else 0,
        "finalMeanWealth": per_timestep[-1]["meanWealth"] if per_timestep else 0,
        "finalSocietalWealth": per_timestep[-1]["societalWealth"] if per_timestep else 0,
        "finalGini": per_timestep[-1]["giniCoefficient"] if per_timestep else 0,
        "finalMeanTimeToLive": per_timestep[-1]["meanTimeToLive"] if per_timestep else 0,
        "totalTradeVolume": sum(r["tradeVolume"] for r in per_timestep),
        "totalCombatActions": sum(r["combatActions"] for r in per_timestep),
        "totalTradeActions": sum(r["tradeActions"] for r in per_timestep),
        "totalReproductionActions": sum(r["reproductionActions"] for r in per_timestep),
        "totalLendingActions": sum(r["lendingActions"] for r in per_timestep),
    }
    return per_timestep, summary


# ─────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────

def write_csv(rows: list, path: str):
    """Write a list of dicts to CSV, creating parent dirs as needed."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_seed_stats(summaries: list) -> dict:
    """
    Given a list of per-seed summary dicts (same condition),
    compute mean and std for numeric fields.
    Returns a flat dict with keys like meanFinalPopulation, stdFinalPopulation, etc.
    """
    if not summaries:
        return {}
    numeric_keys = [k for k, v in summaries[0].items()
                    if isinstance(v, (int, float)) and k not in ("seed",)]
    agg = {"condition": summaries[0]["condition"], "numSeeds": len(summaries)}
    agg["extinctionRate"] = sum(1 for s in summaries if s["extinct"]) / len(summaries)
    for k in numeric_keys:
        vals = [s[k] for s in summaries]
        mean_v = sum(vals) / len(vals)
        variance = sum((v - mean_v) ** 2 for v in vals) / len(vals)
        std_v = math.sqrt(variance)
        agg[f"mean_{k}"] = round(mean_v, 4)
        agg[f"std_{k}"] = round(std_v, 4)
    return agg


# ─────────────────────────────────────────────────────────────────
# Main experiment pipeline
# ─────────────────────────────────────────────────────────────────

def run_experiments(args):
    python_alias = args.python
    config_path = args.config
    output_root = args.output
    num_seeds = args.seeds
    num_agents = args.agents
    timesteps = args.timesteps
    num_cores = args.cores
    gui_mode = args.gui

    # Cap cores to available CPUs
    max_cores = os.cpu_count() or 1
    if num_cores < 1:
        num_cores = 1
    if num_cores > max_cores:
        print(f"  [warn] Requested {num_cores} cores but only {max_cores} available. Using {max_cores}.")
        num_cores = max_cores

    # GUI mode: single seed, single run, interactive
    if gui_mode:
        num_seeds = 1
        print("  [gui] GUI mode enabled. Running 1 seed with GUI.")

    base_cfg = load_base_config(config_path)

    # Override key study parameters to match the paper's setup (5000 ts, 250 agents)
    # These can also be overridden by CLI args
    base_cfg["timesteps"] = timesteps
    base_cfg["startingAgents"] = num_agents

    os.makedirs(output_root, exist_ok=True)

    sim_dir = os.path.join(output_root, "sim_logs")
    results_dir = os.path.join(output_root, "results")
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Generate seeds once so all conditions share the same seeds
    random.seed(42)
    seeds = []
    while len(seeds) < num_seeds:
        s = random.randint(0, sys.maxsize)
        if s not in seeds:
            seeds.append(s)

    print(f"\n{'='*60}")
    print(f"  Digital Terrarium Baseline Replication")
    print(f"{'='*60}")
    print(f"  Seeds:      {num_seeds}")
    print(f"  Timesteps:  {timesteps}")
    print(f"  Agents:     {num_agents}")
    print(f"  Cores:      {num_cores}")
    print(f"  Output:     {output_root}")
    print(f"{'='*60}\n")

    # ── Phase 1: Build and run all simulation configs ──────────────

    all_run_configs = []   # list of (condition_name, seed, cfg_path, log_path, agent_log_path)

    for condition_name, decision_models in CONDITIONS.items():
        condition_sim_dir = os.path.join(sim_dir, condition_name)
        os.makedirs(condition_sim_dir, exist_ok=True)
        for seed in seeds:
            cfg = make_run_config(
                base=base_cfg,
                seed=seed,
                decision_models=decision_models,
                timesteps=timesteps,
                num_agents=num_agents,
                output_dir=condition_sim_dir,
                condition_name=condition_name,
            )
            cfg_path = os.path.join(condition_sim_dir, f"{condition_name}_{seed}.config")
            write_run_config(cfg, cfg_path)
            all_run_configs.append((
                condition_name, seed, cfg_path, cfg["logfile"]
            ))

    # Filter out already-completed runs
    pending = []
    for (cname, seed, cfg_path, log_path) in all_run_configs:
        if os.path.exists(log_path):
            log_data = safe_json_load(log_path)
            if log_data and len(log_data) > 0:
                last_ts = log_data[-1].get("timestep", 0)
                last_pop = log_data[-1].get("population", -1)
                if int(last_ts) >= timesteps or int(last_pop) == 0:
                    continue   # completed
        pending.append((cname, seed, cfg_path, log_path))

    total_jobs = len(all_run_configs)
    skip_count = total_jobs - len(pending)
    print(f"  Total simulation runs:    {total_jobs}")
    print(f"  Already completed:        {skip_count}")
    print(f"  Queued to run:            {len(pending)}\n")

    if pending:
        if gui_mode:
            # In GUI mode, launch sugarscape directly (interactive)
            print("  [gui] Launching GUI simulation …")
            _, _, cfg_path, _ = pending[0]
            # Remove headlessMode override for GUI run
            with open(cfg_path) as f:
                gui_cfg = json.load(f)
            gui_cfg["headlessMode"] = False
            with open(cfg_path, "w") as f:
                json.dump(gui_cfg, f)
            subprocess.run([python_alias, "sugarscape.py", "--conf", cfg_path])
            print("\n  [gui] Simulation finished. No batch results will be saved.\n")
            return
        
        # Headless batch mode
        start_time = time.time()
        manager = multiprocessing.Manager()
        counter = manager.Value('i', 0)
        lock = manager.Lock()

        worker_args = [
            (cfg_path, python_alias, i, len(pending), counter, lock)
            for i, (_, _, cfg_path, _) in enumerate(pending)
        ]

        print(f"  Running simulations using {num_cores} core(s) …")
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(run_one_simulation, worker_args)
        
        # Build duration map for the current session
        session_durations = {path: dur for path, dur in results}

        elapsed = time.time() - start_time
        print(f"\n\n  All simulations completed in {elapsed:.1f}s\n")
    else:
        session_durations = {}

    # ── Phase 2: Parse logs and collect results ────────────────────

    print("  Parsing simulation logs …")
    all_per_timestep = []
    all_summaries = []
    condition_summaries = {cname: [] for cname in CONDITIONS}

    for (cname, seed, cfg_path, log_path) in all_run_configs:
        duration = session_durations.get(cfg_path, 0.0)
        pts, summary = parse_sugarscape_log(
            log_path, cname, seed, timesteps, duration=duration
        )
        if pts is None:
            print(f"  [warn] Could not parse log for {cname} seed={seed}")
            continue
        all_per_timestep.extend(pts)
        all_summaries.append(summary)
        condition_summaries[cname].append(summary)

    # ── Phase 3: Write per-timestep CSV ──────────────────────────

    pts_path = os.path.join(results_dir, "per_timestep.csv")
    write_csv(all_per_timestep, pts_path)
    print(f"  Wrote per-timestep data  → {pts_path}")

    # ── Phase 4: Write per-run summary CSV ───────────────────────

    summary_path = os.path.join(results_dir, "per_seed_summary.csv")
    write_csv(all_summaries, summary_path)
    print(f"  Wrote per-seed summaries → {summary_path}")

    # ── Phase 5: Write condition-level aggregated CSV ─────────────

    agg_rows = []
    for cname, summs in condition_summaries.items():
        if summs:
            agg = aggregate_seed_stats(summs)
            agg_rows.append(agg)

    agg_path = os.path.join(results_dir, "condition_aggregates.csv")
    write_csv(agg_rows, agg_path)
    print(f"  Wrote condition aggregates → {agg_path}")

    # ── Phase 6: Print condensed table to console ─────────────────

    print(f"\n{'='*70}")
    print(f"  {'Condition':<18} {'Seeds':>6} {'Extinct%':>9} {'MeanFinalPop':>13} {'MeanGini':>9} {'MeanTTL':>8}")
    print(f"  {'-'*66}")
    for row in agg_rows:
        cname = row.get("condition", "?")
        nseeds = row.get("numSeeds", 0)
        ext_pct = row.get("extinctionRate", 0) * 100
        mean_fp = row.get("mean_finalPopulation", 0)
        mean_gini = row.get("mean_finalGini", 0)
        mean_ttl = row.get("mean_finalMeanTimeToLive", 0)
        print(f"  {cname:<18} {nseeds:>6} {ext_pct:>8.1f}% {mean_fp:>13.1f} {mean_gini:>9.3f} {mean_ttl:>8.2f}")
    print(f"{'='*70}\n")

    print("  Done.\n")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline Digital Terrarium replication experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config",
        default="config.json",
        help="Path to the master config.json file.",
    )
    parser.add_argument(
        "-o", "--output",
        default="experiment_results",
        help="Root directory for all output (logs + results CSVs).",
    )
    parser.add_argument(
        "-s", "--seeds",
        type=int,
        default=500,
        help="Number of random seeds (simulation replications) per condition.",
    )
    parser.add_argument(
        "-a", "--agents",
        type=int,
        default=250,
        help="Starting number of agents per simulation.",
    )
    parser.add_argument(
        "-t", "--timesteps",
        type=int,
        default=5000,
        help="Number of timesteps per simulation.",
    )
    parser.add_argument(
        "-j", "--cores",
        type=int,
        default=1,
        help="Number of parallel CPU cores to use.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help=(
            "Run one seed with the GUI enabled (interactive mode). "
            "Forces --seeds=1 and disables multiprocessing."
        ),
    )
    parser.add_argument(
        "--python",
        default="python",
        help="Python interpreter alias (e.g. python3).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(args)
