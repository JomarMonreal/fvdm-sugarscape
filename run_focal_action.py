#!/usr/bin/env python3
"""
run_focal_action.py
-------------------
Targeted Focal-Action Data Generation for FVDM derivation.

Runs biased-agent simulations to collect dense state-action-outcome
observations for each of the four discretionary action classes:
  Combat, Trade, Reproduction, and Lending.

Each bias configuration overrides specific agent parameters to maximise
execution of the targeted action (see BiasedFocalAction in ethics.py).
The resulting agent-level log data is then post-processed into a single
derivation dataset suitable for training the felicific coordinate
prediction functions (NGBoost, multinomial gradient boosting) and for
deriving prioritization vectors via MaxEnt IRL.

Recorded per observation (one row per agent per timestep):
  - Local state s_i  (wealth, sugar, spice, metabolism, age, TTL,
                       neighbours, vision, position, tribe, etc.)
  - Selected action   (combat / trade / reproduction / lending / none)
  - Outcome over H_i  (wealth change, sugar gained, spice gained,
                        prey killed, trade partners, mates, loans)

Reference: Thesis Section 3.6.2 – Targeted Focal-Action Data Generation.

Usage:
  python run_focal_action.py [options]

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
# Bias condition definitions
# ─────────────────────────────────────────────────────────────────

BIAS_CONDITIONS = {
    "biasedCombat":       ["biasedCombat"],
    "biasedTrade":        ["biasedTrade"],
    "biasedReproduction": ["biasedReproduction"],
    "biasedLending":      ["biasedLending"],
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


def make_bias_run_config(base: dict, seed: int, decision_models: list,
                         timesteps: int, num_agents: int, output_dir: str,
                         condition_name: str) -> dict:
    """Build a per-run configuration dict with biased-agent settings."""
    cfg = dict(base)

    cfg["seed"] = seed
    cfg["agentDecisionModels"] = decision_models
    cfg["timesteps"] = timesteps
    cfg["startingAgents"] = num_agents
    cfg["startingDiseases"] = 0

    # Headless, non-interactive
    cfg["headlessMode"] = True
    cfg["debugMode"] = ["none"]
    cfg["keepAlivePostExtinction"] = False
    cfg["keepAliveAtEnd"] = False
    cfg["screenshots"] = False
    cfg["profileMode"] = False

    # Log paths — ENABLE agent log for per-agent state-action-outcome data
    cfg["logfile"] = os.path.join(output_dir, f"{condition_name}_{seed}.json")
    cfg["agentLogfile"] = os.path.join(output_dir, f"{condition_name}_{seed}_agents.json")
    cfg["logfileFormat"] = "json"

    return cfg


def write_run_config(cfg: dict, path: str):
    """Serialise a run config dict to a .config file."""
    with open(path, "w") as f:
        json.dump(cfg, f)


def run_one_simulation(args):
    """Worker function executed in each subprocess."""
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
# Agent log parsing → derivation observations
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


def classify_action(record: dict) -> str:
    """Determine which discretionary action the agent performed this timestep.

    Priority order matches the simulation's action execution sequence:
    Combat > Trade > Reproduction > Lending > None.
    """
    if record.get("preyKilled", False):
        return "combat"
    if record.get("tradePartners", 0) > 0:
        return "trade"
    if record.get("mates", 0) > 0:
        return "reproduction"
    if record.get("lendingPartners", 0) > 0:
        return "lending"
    return "none"


def parse_agent_log(agent_log_path: str, condition: str, seed: int) -> list:
    """Parse the agent-level JSON log into derivation observation rows.

    Each row captures the local state s_i, the selected action, and the
    observed outcome — the three components needed for training felicific
    coordinate prediction functions and for MaxEnt IRL trajectory input.
    """
    data = safe_json_load(agent_log_path)
    if data is None or len(data) == 0:
        return []

    observations = []
    for record in data:
        action = classify_action(record)

        obs = {
            # ── Metadata ──
            "condition": condition,
            "seed": seed,
            "timestep": record.get("timestep", 0),
            "agentID": record.get("ID", -1),

            # ── Local state s_i ──
            "age": record.get("age", 0),
            "wealth": record.get("wealth", 0),
            "sugar": record.get("sugar", 0),
            "spice": record.get("spice", 0),
            "timeToLive": record.get("timeToLive", 0),
            "movement": record.get("movement", 0),
            "neighbors": record.get("neighbors", 0),
            "neighborsInTribe": record.get("neighborsInTribe", 0),
            "neighborsNotInTribe": record.get("neighborsNotInTribe", 0),
            "validMoves": record.get("validMoves", 0),
            "compositeHappiness": record.get("compositeHappiness", 0),
            "depression": int(record.get("depression", False)),

            # ── Selected action ──
            "action": action,

            # ── Observed outcome (over evaluation horizon) ──
            "wealthGained": record.get("wealthGained", 0),
            "sugarGained": record.get("sugarGained", 0),
            "spiceGained": record.get("spiceGained", 0),
            "preyKilled": int(record.get("preyKilled", False)),
            "preyWealth": record.get("preyWealth", 0),
            "tradePartners": record.get("tradePartners", 0),
            "mates": record.get("mates", 0),
            "lendingPartners": record.get("lendingPartners", 0),
            "diseasesSpread": record.get("diseasesSpread", 0),
            "didMove": int(record.get("didMove", False)),
            "moveRank": record.get("moveRank", 0),
            "timeToLiveDifference": record.get("timeToLiveDifference", 0),
            "pollutionDifference": record.get("pollutionDifference", 0),
        }
        observations.append(obs)

    return observations


# ─────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────

def write_csv(rows: list, path: str):
    """Write a list of dicts to CSV, creating parent dirs as needed."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    all_keys = list(rows[0].keys())
    seen = set(all_keys)
    for row in rows[1:]:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, restval="")
        writer.writeheader()
        writer.writerows(rows)


def compute_action_distribution(observations: list) -> dict:
    """Compute per-action counts from parsed observations."""
    counts = {"combat": 0, "trade": 0, "reproduction": 0, "lending": 0, "none": 0}
    for obs in observations:
        action = obs.get("action", "none")
        counts[action] = counts.get(action, 0) + 1
    return counts


# ─────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────

def run_focal_action(args):
    python_alias = args.python
    config_path = args.config
    output_root = args.output
    num_seeds = args.seeds
    num_agents = args.agents
    timesteps = args.timesteps
    num_cores = args.cores

    max_cores = os.cpu_count() or 1
    if num_cores < 1:
        num_cores = 1
    if num_cores > max_cores:
        print(f"  [warn] Requested {num_cores} cores but only {max_cores} available. Using {max_cores}.")
        num_cores = max_cores

    base_cfg = load_base_config(config_path)
    base_cfg["timesteps"] = timesteps
    base_cfg["startingAgents"] = num_agents

    os.makedirs(output_root, exist_ok=True)

    sim_dir = os.path.join(output_root, "sim_logs")
    results_dir = os.path.join(output_root, "results")
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Generate deterministic seeds
    random.seed(42)
    seeds = []
    while len(seeds) < num_seeds:
        s = random.randint(0, sys.maxsize)
        if s not in seeds:
            seeds.append(s)

    print(f"\n{'='*60}")
    print(f"  Targeted Focal-Action Data Generation")
    print(f"{'='*60}")
    print(f"  Bias conditions:  {list(BIAS_CONDITIONS.keys())}")
    print(f"  Seeds:            {num_seeds}")
    print(f"  Timesteps:        {timesteps}")
    print(f"  Agents:           {num_agents}")
    print(f"  Cores:            {num_cores}")
    print(f"  Output:           {output_root}")
    print(f"{'='*60}\n")

    # ── Phase 1: Build and run simulation configs ────────────────

    all_run_configs = []

    for condition_name, decision_models in BIAS_CONDITIONS.items():
        condition_sim_dir = os.path.join(sim_dir, condition_name)
        os.makedirs(condition_sim_dir, exist_ok=True)
        for seed in seeds:
            cfg = make_bias_run_config(
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
                condition_name, seed, cfg_path, cfg["logfile"], cfg["agentLogfile"]
            ))

    # Filter out already-completed runs
    pending = []
    for (cname, seed, cfg_path, log_path, agent_log_path) in all_run_configs:
        if os.path.exists(agent_log_path):
            agent_data = safe_json_load(agent_log_path)
            if agent_data and len(agent_data) > 0:
                continue   # already collected
        pending.append((cname, seed, cfg_path, log_path, agent_log_path))

    total_jobs = len(all_run_configs)
    skip_count = total_jobs - len(pending)
    print(f"  Total simulation runs:    {total_jobs}")
    print(f"  Already completed:        {skip_count}")
    print(f"  Queued to run:            {len(pending)}\n")

    if pending:
        start_time = time.time()
        manager = multiprocessing.Manager()
        counter = manager.Value('i', 0)
        lock = manager.Lock()

        worker_args = [
            (cfg_path, python_alias, i, len(pending), counter, lock)
            for i, (_, _, cfg_path, _, _) in enumerate(pending)
        ]

        print(f"  Running biased simulations using {num_cores} core(s) …")
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.map(run_one_simulation, worker_args)

        elapsed = time.time() - start_time
        print(f"\n\n  All simulations completed in {elapsed:.1f}s\n")

    # ── Phase 2: Parse agent logs → derivation dataset ───────────

    print("  Parsing agent logs into derivation observations …")
    all_observations = []
    condition_counts = {cname: 0 for cname in BIAS_CONDITIONS}

    for (cname, seed, cfg_path, log_path, agent_log_path) in all_run_configs:
        obs = parse_agent_log(agent_log_path, cname, seed)
        if not obs:
            print(f"  [warn] No agent data for {cname} seed={seed}")
            continue
        all_observations.extend(obs)
        condition_counts[cname] += len(obs)

    # ── Phase 3: Write combined derivation CSV ───────────────────

    combined_path = os.path.join(results_dir, "focal_action_derivation.csv")
    write_csv(all_observations, combined_path)
    print(f"  Wrote combined derivation data → {combined_path}")
    print(f"    Total observations: {len(all_observations)}")

    # ── Phase 4: Write per-condition CSVs ────────────────────────

    print("\n  Per-condition breakdown:")
    for cname in BIAS_CONDITIONS:
        cond_obs = [o for o in all_observations if o["condition"] == cname]
        if cond_obs:
            cond_dir = os.path.join(results_dir, cname)
            os.makedirs(cond_dir, exist_ok=True)
            write_csv(cond_obs, os.path.join(cond_dir, "derivation_observations.csv"))

            dist = compute_action_distribution(cond_obs)
            total = len(cond_obs)
            print(f"    {cname:<25} {total:>8} obs  |  "
                  f"combat={dist['combat']:>5}  trade={dist['trade']:>5}  "
                  f"repro={dist['reproduction']:>5}  lending={dist['lending']:>5}  "
                  f"none={dist['none']:>5}")

    # ── Phase 5: Action distribution summary CSV ─────────────────

    summary_rows = []
    for cname in BIAS_CONDITIONS:
        cond_obs = [o for o in all_observations if o["condition"] == cname]
        dist = compute_action_distribution(cond_obs)
        total = len(cond_obs)
        row = {
            "condition": cname,
            "totalObservations": total,
            "combatActions": dist["combat"],
            "tradeActions": dist["trade"],
            "reproductionActions": dist["reproduction"],
            "lendingActions": dist["lending"],
            "noAction": dist["none"],
        }
        if total > 0:
            row["combatPct"] = round(dist["combat"] / total * 100, 2)
            row["tradePct"] = round(dist["trade"] / total * 100, 2)
            row["reproductionPct"] = round(dist["reproduction"] / total * 100, 2)
            row["lendingPct"] = round(dist["lending"] / total * 100, 2)
            row["noActionPct"] = round(dist["none"] / total * 100, 2)
        summary_rows.append(row)

    summary_path = os.path.join(results_dir, "action_distribution_summary.csv")
    write_csv(summary_rows, summary_path)
    print(f"\n  Wrote action distribution summary → {summary_path}")

    # ── Phase 6: Sample-size enforcement ───────────────────────────
    # Ensure ≥ 2,640 discretionary observations (thesis §3.4.4).
    # If short, automatically run additional seeds and re-parse.

    target_min = 2640  # 22 vars × 10 EPV × 4 actions × 3 (boosting factor)
    max_retries = 5
    retry = 0

    while True:
        discretionary_total = sum(
            1 for o in all_observations if o["action"] != "none"
        )

        print(f"\n{'='*60}")
        print(f"  Sample Size Check (attempt {retry + 1})")
        print(f"  Target minimum (per thesis §3.4.4): {target_min}")
        print(f"  Discretionary action observations:  {discretionary_total}")

        if discretionary_total >= target_min:
            print(f"  ✓ Sufficient for coordinate model training")
            print(f"{'='*60}\n")
            break

        deficit = target_min - discretionary_total
        retry += 1

        if retry > max_retries:
            print(f"  ✗ Still short by {deficit} after {max_retries} retry rounds.")
            print(f"    Increase --timesteps or --agents and re-run manually.")
            print(f"{'='*60}\n")
            break

        # Estimate how many extra seeds to run (across all conditions)
        obs_per_seed = len(all_observations) / max(num_seeds * len(BIAS_CONDITIONS), 1)
        disc_rate = discretionary_total / max(len(all_observations), 1)
        disc_per_seed = obs_per_seed * disc_rate
        extra_seeds_needed = max(1, int(math.ceil(deficit / max(disc_per_seed, 1))))

        print(f"  ✗ Short by {deficit} — auto-generating {extra_seeds_needed} extra seed(s) per condition …")
        print(f"{'='*60}\n")

        # Generate new unique seeds
        new_seeds = []
        existing_seeds = set(seeds)
        while len(new_seeds) < extra_seeds_needed:
            s = random.randint(0, sys.maxsize)
            if s not in existing_seeds:
                new_seeds.append(s)
                existing_seeds.add(s)
        seeds.extend(new_seeds)
        num_seeds += extra_seeds_needed

        # Build and run configs for new seeds only
        new_run_configs = []
        for condition_name, decision_models in BIAS_CONDITIONS.items():
            condition_sim_dir = os.path.join(sim_dir, condition_name)
            for seed in new_seeds:
                cfg = make_bias_run_config(
                    base=base_cfg, seed=seed,
                    decision_models=decision_models,
                    timesteps=timesteps, num_agents=num_agents,
                    output_dir=condition_sim_dir,
                    condition_name=condition_name,
                )
                cfg_path = os.path.join(
                    condition_sim_dir, f"{condition_name}_{seed}.config"
                )
                write_run_config(cfg, cfg_path)
                new_run_configs.append((
                    condition_name, seed, cfg_path,
                    cfg["logfile"], cfg["agentLogfile"]
                ))

        # Run the new simulations
        manager = multiprocessing.Manager()
        counter = manager.Value('i', 0)
        lock = manager.Lock()
        worker_args = [
            (cfg_path, python_alias, i, len(new_run_configs), counter, lock)
            for i, (_, _, cfg_path, _, _) in enumerate(new_run_configs)
        ]
        print(f"  Running {len(new_run_configs)} extra simulations …")
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.map(run_one_simulation, worker_args)
        print()

        # Parse new agent logs and append
        for (cname, seed, cfg_path, log_path, agent_log_path) in new_run_configs:
            obs = parse_agent_log(agent_log_path, cname, seed)
            if obs:
                all_observations.extend(obs)
                condition_counts[cname] += len(obs)
        all_run_configs.extend(new_run_configs)

        # Re-write the combined CSV with new data included
        write_csv(all_observations, combined_path)

    print("  Done.\n")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Targeted Focal-Action Data Generation for FVDM derivation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config",
        default="config.json",
        help="Path to the master config.json file.",
    )
    parser.add_argument(
        "-o", "--output",
        default="focal_action_results",
        help="Root directory for all output (logs + derivation CSVs).",
    )
    parser.add_argument(
        "-s", "--seeds",
        type=int,
        default=10,
        help="Number of random seeds (replications) per bias condition.",
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
        default=2500,
        help="Number of timesteps per simulation.",
    )
    parser.add_argument(
        "-j", "--cores",
        type=int,
        default=1,
        help="Number of parallel CPU cores to use.",
    )
    parser.add_argument(
        "--python",
        default="python",
        help="Python interpreter alias (e.g. python3).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_focal_action(args)
