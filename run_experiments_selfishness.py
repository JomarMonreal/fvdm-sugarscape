#!/usr/bin/env python3
"""
run_experiments_selfishness.py
-------------------------------
Replicates the selfishness-factor sweep from Herman & Kremer (2024)
Section VII-C.  All agents use the "bentham" hedonic-calculus model with a
fixed selfishnessFactor φ that is swept from 0.0 (pure altruist) to 1.0
(pure egoist) in configurable increments.

Parameters (paper defaults):
  250 agents, 5 000 timesteps, 30 seeds, φ step 0.05 → 21 conditions.

Outputs (under --output/):
  sim_logs/          — raw JSON simulation logs (one per seed per condition)
  results/
    per_timestep.csv       — timestep-level population metrics per seed
    per_seed_summary.csv   — one row per (condition, seed) with final metrics
    condition_aggregates.csv
    selfishness_spearman.csv  — Spearman r between φ and each outcome metric

Usage:
  python run_experiments_selfishness.py
  python run_experiments_selfishness.py --seeds 30 --cores 8
  python run_experiments_selfishness.py --phi-step 0.01   # full KH replication
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

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Spearman correlation
# ─────────────────────────────────────────────────────────────────────────────

try:
    from scipy.stats import spearmanr as _scipy_spearmanr
    def spearman_r(xs, ys):
        r, p = _scipy_spearmanr(xs, ys)
        return round(float(r), 4), round(float(p), 4)
except ImportError:
    def spearman_r(xs, ys):
        n = len(xs)
        if n < 3:
            return 0.0, None
        rx = [sorted(xs).index(x) + 1 for x in xs]
        ry = [sorted(ys).index(y) + 1 for y in ys]
        d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
        r = 1 - 6 * d2 / (n * (n ** 2 - 1))
        return round(r, 4), None

# ─────────────────────────────────────────────────────────────────────────────
# Seed generation (deterministic, shared with other runners)
# ─────────────────────────────────────────────────────────────────────────────

def generate_seeds(n: int, master_seed: int = 42) -> list:
    random.seed(master_seed)
    seeds = []
    while len(seeds) < n:
        s = random.randint(0, sys.maxsize)
        if s not in seeds:
            seeds.append(s)
    return seeds


# ─────────────────────────────────────────────────────────────────────────────
# Condition helpers
# ─────────────────────────────────────────────────────────────────────────────

def phi_to_condition_name(phi: float) -> str:
    """'phi_050' style name for phi=0.50."""
    return f"phi_{round(phi * 100):03d}"


def make_phi_levels(step: float) -> list:
    """Return sorted list of phi values from 0.0 to 1.0 inclusive."""
    n_steps = round(1.0 / step)
    return [round(i * step, 10) for i in range(n_steps + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_base_config(path: str) -> dict:
    with open(path) as f:
        full = json.load(f)
    return full.get("sugarscapeOptions", full)


def make_run_config(base: dict, seed: int, phi: float,
                    timesteps: int, num_agents: int,
                    log_path: str) -> dict:
    cfg = dict(base)
    cfg["seed"]                    = seed
    cfg["agentDecisionModels"]     = ["bentham"]
    cfg["agentSelfishnessFactor"]  = [phi, phi]
    cfg["timesteps"]               = timesteps
    cfg["startingAgents"]          = num_agents
    cfg["startingDiseases"]        = 0
    cfg["headlessMode"]            = True
    cfg["debugMode"]               = ["none"]
    cfg["keepAlivePostExtinction"] = False
    cfg["keepAliveAtEnd"]          = False
    cfg["screenshots"]             = False
    cfg["profileMode"]             = False
    cfg["logfile"]                 = log_path
    cfg["agentLogfile"]            = ""   # no agent log needed for this sweep
    cfg["logfileFormat"]           = "json"
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one_simulation(args):
    config_path, python_alias, counter, lock, total = args
    t = time.time()
    subprocess.run([python_alias, "sugarscape.py", "--conf", config_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dur = time.time() - t
    with lock:
        counter.value += 1
        pct = counter.value / total * 100
        print(f"\r  [{counter.value:>4}/{total}]  {pct:5.1f}%  "
              f"last={os.path.basename(config_path):<45}", end="", flush=True)
    return config_path, dur


# ─────────────────────────────────────────────────────────────────────────────
# Log parsing
# ─────────────────────────────────────────────────────────────────────────────

def safe_json_load(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def parse_sim_log(log_path: str, condition: str, phi: float, seed: int,
                  duration: float = 0.0):
    """
    Returns (per_timestep, summary) or (None, None) on failure.
    """
    sim_log = safe_json_load(log_path)
    if not sim_log or len(sim_log) == 0:
        return None, None

    per_timestep = []
    total_deaths = 0
    total_born   = 0
    final_pop    = 0

    for entry in sim_log:
        ts  = int(entry.get("timestep", 0))
        pop = int(entry.get("population", 0))
        d   = int(entry.get("agentDeaths", 0))
        b   = int(entry.get("agentsBorn", 0))
        total_deaths += d
        total_born   += b

        rec = {
            "condition":             condition,
            "phi":                   phi,
            "seed":                  seed,
            "timestep":              ts,
            "population":            pop,
            "meanWealth":            float(entry.get("meanWealth", 0)),
            "societalWealth":        float(entry.get("agentWealthTotal", 0)),
            "giniCoefficient":       float(entry.get("giniCoefficient", 0)),
            "agentDeaths":           d,
            "agentsBorn":            b,
            "meanTimeToLive":        float(entry.get("agentMeanTimeToLive", 0)),
            "meanAge":               float(entry.get("meanAge", 0)),
            "meanAgeAtDeath":        float(entry.get("meanAgeAtDeath", 0)),
            "combatActions":         int(entry.get("actionCombats", 0)),
            "tradeActions":          int(entry.get("actionTrades", 0)),
            "reproductionActions":   int(entry.get("actionReproductions", 0)),
            "meanSelfishness":       float(entry.get("meanSelfishness", phi)),
        }
        per_timestep.append(rec)
        final_pop = pop

    last = per_timestep[-1] if per_timestep else {}
    n_ts = len(per_timestep)

    mean_deaths_per_ts = total_deaths / n_ts if n_ts > 0 else 0
    # mean age at death: average of non-zero entries (only meaningful when deaths occur)
    age_at_death_vals = [r["meanAgeAtDeath"] for r in per_timestep if r["meanAgeAtDeath"] > 0]
    mean_age_at_death = sum(age_at_death_vals) / len(age_at_death_vals) if age_at_death_vals else 0

    summary = {
        "condition":            condition,
        "phi":                  phi,
        "seed":                 seed,
        "executionTime":        round(duration, 2),
        "extinct":              (final_pop == 0),
        "finalPopulation":      final_pop,
        "totalDeaths":          total_deaths,
        "totalBorn":            total_born,
        "finalTimestep":        last.get("timestep", 0),
        "finalMeanWealth":      last.get("meanWealth", 0),
        "finalSocietalWealth":  last.get("societalWealth", 0),
        "finalGini":            last.get("giniCoefficient", 0),
        "finalMeanTimeToLive":  last.get("meanTimeToLive", 0),
        "finalMeanAge":         last.get("meanAge", 0),
        "meanDeathsPerTimestep": round(mean_deaths_per_ts, 4),
        "meanAgeAtDeath":       round(mean_age_at_death, 4),
        "totalCombatActions":   sum(r["combatActions"]       for r in per_timestep),
        "totalTradeActions":    sum(r["tradeActions"]        for r in per_timestep),
        "totalReproductionActions": sum(r["reproductionActions"] for r in per_timestep),
    }

    return per_timestep, summary


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(rows: list, path: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    seen = set(keys)
    for r in rows[1:]:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, restval="")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_seeds(summaries: list) -> dict:
    if not summaries:
        return {}
    num_keys = [k for k, v in summaries[0].items()
                if isinstance(v, (int, float)) and k not in ("seed", "phi")]
    phi = summaries[0]["phi"]
    cname = summaries[0]["condition"]
    agg = {
        "condition":     cname,
        "phi":           phi,
        "numSeeds":      len(summaries),
        "extinctionRate": sum(1 for s in summaries if s["extinct"]) / len(summaries),
    }
    for k in num_keys:
        vals = [s[k] for s in summaries]
        mu   = sum(vals) / len(vals)
        std  = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
        med  = float(np.median(vals))
        q1   = float(np.percentile(vals, 25))
        q3   = float(np.percentile(vals, 75))
        agg[f"mean_{k}"]   = round(mu,  4)
        agg[f"std_{k}"]    = round(std, 4)
        agg[f"median_{k}"] = round(med, 4)
        agg[f"q1_{k}"]     = round(q1,  4)
        agg[f"q3_{k}"]     = round(q3,  4)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_selfishness_sweep(args):
    config_path  = args.config
    output_root  = args.output
    num_seeds    = args.seeds
    num_agents   = args.agents
    timesteps    = args.timesteps
    num_cores    = args.cores
    python_alias = args.python
    force        = args.force
    phi_step     = args.phi_step

    num_cores = max(1, min(num_cores, os.cpu_count() or 1))

    base_cfg  = load_base_config(config_path)
    seeds     = generate_seeds(num_seeds)
    phi_vals  = make_phi_levels(phi_step)

    sim_dir     = os.path.join(output_root, "sim_logs")
    results_dir = os.path.join(output_root, "results")
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Selfishness Factor Sweep  (Herman & Kremer 2024, §VII-C)")
    print(f"{'='*60}")
    print(f"  φ levels:  {len(phi_vals)}  ({phi_vals[0]:.2f} → {phi_vals[-1]:.2f}, step {phi_step})")
    print(f"  Seeds:     {num_seeds}   Timesteps: {timesteps}   Agents: {num_agents}")
    print(f"  Cores:     {num_cores}   Output:    {output_root}")
    print(f"{'='*60}\n")

    # ── Build run configs ──────────────────────────────────────────────────
    all_runs = []
    for phi in phi_vals:
        cname = phi_to_condition_name(phi)
        cdir  = os.path.join(sim_dir, cname)
        os.makedirs(cdir, exist_ok=True)
        for seed in seeds:
            log_path = os.path.join(cdir, f"{cname}_{seed}.json")
            cfg      = make_run_config(base_cfg, seed, phi, timesteps, num_agents, log_path)
            cfg_path = os.path.join(cdir, f"{cname}_{seed}.config")
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
            all_runs.append((cname, phi, seed, cfg_path, log_path))

    pending = []
    for (cname, phi, seed, cfg_path, log_path) in all_runs:
        if not force and os.path.exists(log_path):
            d = safe_json_load(log_path)
            if d and len(d) > 0:
                last_ts  = d[-1].get("timestep", 0)
                last_pop = d[-1].get("population", -1)
                if int(last_ts) >= timesteps or int(last_pop) == 0:
                    continue
        pending.append((cname, phi, seed, cfg_path, log_path))

    total = len(all_runs)
    print(f"  Total runs: {total}  |  Queued: {len(pending)}  |  Skipped: {total - len(pending)}\n")

    # ── Run simulations ────────────────────────────────────────────────────
    session_durations = {}
    if pending:
        t_start = time.time()
        manager = multiprocessing.Manager()
        counter = manager.Value("i", 0)
        lock    = manager.Lock()
        worker_args = [(cfg_path, python_alias, counter, lock, len(pending))
                       for (_, _, _, cfg_path, _) in pending]
        print(f"  Running {len(pending)} simulations on {num_cores} core(s) …")
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(run_one_simulation, worker_args)
        session_durations = {path: dur for path, dur in results}
        print(f"\n\n  Completed in {time.time() - t_start:.1f}s\n")

    # ── Parse results ──────────────────────────────────────────────────────
    print("  Parsing logs …")
    all_pts       = []
    all_summaries = []
    cond_summaries = {phi_to_condition_name(phi): [] for phi in phi_vals}

    for (cname, phi, seed, cfg_path, log_path) in all_runs:
        dur = session_durations.get(cfg_path, 0.0)
        pts, summary = parse_sim_log(log_path, cname, phi, seed, dur)
        if pts is None:
            print(f"  [warn] Could not parse {cname} seed={seed}")
            continue
        all_pts.extend(pts)
        all_summaries.append(summary)
        cond_summaries[cname].append(summary)

    # ── Write CSVs ─────────────────────────────────────────────────────────
    write_csv(all_pts,       os.path.join(results_dir, "per_timestep.csv"))
    write_csv(all_summaries, os.path.join(results_dir, "per_seed_summary.csv"))

    agg_rows = []
    for phi in phi_vals:
        cname = phi_to_condition_name(phi)
        summs = cond_summaries[cname]
        if summs:
            agg_rows.append(aggregate_seeds(summs))
    write_csv(agg_rows, os.path.join(results_dir, "condition_aggregates.csv"))

    # ── Spearman correlations ──────────────────────────────────────────────
    outcome_metrics = [
        "finalPopulation", "finalSocietalWealth", "finalMeanWealth",
        "finalMeanTimeToLive", "meanDeathsPerTimestep", "meanAgeAtDeath",
        "finalGini", "totalCombatActions", "totalTradeActions",
        "totalReproductionActions",
    ]
    agg_lookup = {r["condition"]: r for r in agg_rows}
    spearman_rows = []
    for metric in outcome_metrics:
        xs, ys = [], []
        for phi in phi_vals:
            cname = phi_to_condition_name(phi)
            row   = agg_lookup.get(cname)
            if row and f"mean_{metric}" in row:
                xs.append(phi)
                ys.append(row[f"mean_{metric}"])
        if len(xs) >= 3:
            r_val, p_val = spearman_r(xs, ys)
            spearman_rows.append({
                "metric":     metric,
                "spearman_r": r_val,
                "p_value":    p_val if p_val is not None else "",
                "n_levels":   len(xs),
            })
    write_csv(spearman_rows,
              os.path.join(results_dir, "selfishness_spearman.csv"))

    # ── Console summary ────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {'φ':>5} {'Extinct%':>9} {'FinalPop':>10} {'MeanWealth':>11} "
          f"{'TTL':>7} {'Deaths/ts':>10}")
    print(f"  {'-'*60}")
    for phi in phi_vals:
        cname = phi_to_condition_name(phi)
        row   = agg_lookup.get(cname, {})
        ext   = row.get("extinctionRate", 0) * 100
        fp    = row.get("mean_finalPopulation", 0)
        mw    = row.get("mean_finalMeanWealth", 0)
        ttl   = row.get("mean_finalMeanTimeToLive", 0)
        dtts  = row.get("mean_meanDeathsPerTimestep", 0)
        print(f"  {phi:>5.2f} {ext:>8.1f}% {fp:>10.1f} {mw:>11.2f} "
              f"{ttl:>7.2f} {dtts:>10.4f}")
    print(f"{'='*72}\n")

    if spearman_rows:
        print(f"  Spearman r (φ vs outcome):")
        for row in spearman_rows:
            p_str = f"p={row['p_value']:.4f}" if row["p_value"] != "" else "p=N/A "
            print(f"    {row['metric']:<34} r={row['spearman_r']:>7.4f}  {p_str}")
        print()

    print(f"  Outputs written to {results_dir}/\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Selfishness-factor sweep (KH2024 §VII-C replication)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c", "--config",    default="config.json")
    p.add_argument("-o", "--output",    default="selfishness_results")
    p.add_argument("-s", "--seeds",     type=int,   default=30)
    p.add_argument("-a", "--agents",    type=int,   default=250)
    p.add_argument("-t", "--timesteps", type=int,   default=5000)
    p.add_argument("-j", "--cores",     type=int,   default=1)
    p.add_argument("--phi-step",        type=float, default=0.05,
                   help="Increment for φ sweep (0.05 → 21 levels; 0.01 → 101 levels)")
    p.add_argument("--python",          default="python3")
    p.add_argument("--force",           action="store_true", default=False,
                   help="Re-run even if log files already exist")
    return p.parse_args()


if __name__ == "__main__":
    run_selfishness_sweep(parse_args())
