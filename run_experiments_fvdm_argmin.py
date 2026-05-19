#!/usr/bin/env python3
"""
run_experiments_fvdm_argmin.py
-------------------------------
Runs the four consistent-argmin FVDM conditions.

Decision rule: c* = argmin_c { ||μ_imm − v_net_imm(c)||₂ + ||μ_fut − v_net_fut(c)||₂ }

where v_net(c) = φ·v_self(c) − (1−φ)·mean_k[v_k(c)]

Both the profile μ (from bfe_profiles_phi.json) and the candidates v_net(c) live
in the same opportunity-cost space — this is the first consistent test of FVDM argmin.

Conditions:
  argminRaw       (fvdmArgminRaw,      φ=1.0)
  argminEgoist    (fvdmArgminEgoist,   φ=1.0)
  argminAltruist  (fvdmArgminAltruist, φ=0.0)
  argminBentham   (fvdmArgminBentham,  φ=0.5)

Outputs (under results/argmin/):
  per_seed_summary.csv    — extinction, final pop, mean wealth, Gini, TTL per seed
  condition_aggregates.csv — mean ± sd across seeds
  bfs_vs_derived.csv      — BFS(μ_obs, μ_profile) per condition

Usage:
  python run_experiments_fvdm_argmin.py
  python run_experiments_fvdm_argmin.py -s 10 -t 500 -a 250 -j 8
  python run_experiments_fvdm_argmin.py -s 30 -t 5000 -a 250 -j 30
  python run_experiments_fvdm_argmin.py --force
"""

import argparse
import csv as csv_mod
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
# Conditions
# ─────────────────────────────────────────────────────────────────────────────

CONDITIONS = {
    "argminRaw":      ["fvdmArgminRaw"],
    "argminEgoist":   ["fvdmArgminEgoist"],
    "argminAltruist": ["fvdmArgminAltruist"],
    "argminBentham":  ["fvdmArgminBentham"],
}

PROFILE_PAIRS = {
    "argminRaw":      "rawSugarscape",
    "argminEgoist":   "egoist",
    "argminAltruist": "altruist",
    "argminBentham":  "bentham",
}

IMM_LABELS = ["I", "D", "C", "P", "E"]
FUT_LABELS = ["I", "D", "C", "P", "E"]

PROFILE_PATH = "fvdm_vectors/bfe_profiles_phi.json"
OUTPUT_DIR   = os.path.join("results", "argmin")

# ─────────────────────────────────────────────────────────────────────────────
# Seed generation — matches other experiment runners (seed=42)
# ─────────────────────────────────────────────────────────────────────────────

def generate_seeds(n: int, seed: int = 42) -> list:
    random.seed(seed)
    seeds = []
    while len(seeds) < n:
        s = random.randint(0, sys.maxsize)
        if s not in seeds:
            seeds.append(s)
    return seeds


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_base_config(path: str) -> dict:
    with open(path) as f:
        full = json.load(f)
    return full.get("sugarscapeOptions", full)


def make_run_config(base: dict, seed: int, decision_models: list,
                    timesteps: int, num_agents: int,
                    log_path: str, agent_log_path: str) -> dict:
    cfg = dict(base)
    cfg["seed"]                     = seed
    cfg["agentDecisionModels"]      = decision_models
    cfg["timesteps"]                = timesteps
    cfg["startingAgents"]           = num_agents
    cfg["startingDiseases"]         = 0
    cfg["headlessMode"]             = True
    cfg["debugMode"]                = ["none"]
    cfg["keepAlivePostExtinction"]  = False
    cfg["keepAliveAtEnd"]           = False
    cfg["screenshots"]              = False
    cfg["profileMode"]              = False
    cfg["logfile"]                  = log_path
    cfg["agentLogfile"]             = agent_log_path
    cfg["logfileFormat"]            = "json"
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one_simulation(args):
    cfg_path, python_alias, counter, lock, total = args
    t = time.time()
    subprocess.run([python_alias, "sugarscape.py", "--conf", cfg_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dur = time.time() - t
    with lock:
        counter.value += 1
        n = counter.value
        bar = "█" * int(30 * n / total) + "░" * (30 - int(30 * n / total))
        print(f"\r  [{bar}] {n:>4}/{total}  {n/total*100:5.1f}%  "
              f"{os.path.basename(cfg_path):<48}", end="", flush=True)
    return cfg_path, dur


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


def parse_sim_log(log_path: str, condition: str, seed: int, timesteps: int):
    sim_log = safe_json_load(log_path)
    if not sim_log:
        return None

    last = sim_log[-1]
    final_ts  = int(last.get("timestep", 0))
    final_pop = int(last.get("population", 0))
    extinct   = int(final_ts < timesteps - 1 or final_pop == 0)

    wealth_vals = [float(e.get("meanWealth", 0)) for e in sim_log]
    gini_vals   = [float(e.get("giniCoefficient", 0)) for e in sim_log]
    ttl_vals    = [float(e.get("meanTimeToLive", 0)) for e in sim_log if e.get("meanTimeToLive") is not None]

    return {
        "condition":    condition,
        "seed":         seed,
        "extinct":      extinct,
        "final_pop":    final_pop,
        "mean_wealth":  float(np.mean(wealth_vals)) if wealth_vals else 0.0,
        "mean_gini":    float(np.mean(gini_vals))   if gini_vals  else 0.0,
        "mean_ttl":     float(np.mean(ttl_vals))    if ttl_vals   else 0.0,
    }


def parse_agent_log_bfe(agent_log_path: str, condition: str, seed: int):
    """Read argmin_v_imm_* / argmin_v_fut_* columns; return per-seed mean vectors."""
    data = safe_json_load(agent_log_path)
    if not data:
        return None

    imm_acc = np.zeros(5)
    fut_acc = np.zeros(5)
    n = 0

    rows = data if isinstance(data, list) else []
    for row in rows:
        # Only include rows with social context (neighbours > 0)
        if int(row.get("neighbors", row.get("neighbourhood", 0))) <= 0:
            continue
        try:
            imm = np.array([float(row.get(f"argmin_v_imm_{l}", 0.0)) for l in IMM_LABELS])
            fut = np.array([float(row.get(f"argmin_v_fut_{l}", 0.0)) for l in FUT_LABELS])
            imm_acc += imm
            fut_acc += fut
            n += 1
        except Exception:
            continue

    if n == 0:
        return None
    return {
        "condition": condition,
        "seed":      seed,
        "n_obs":     n,
        "mu_imm":    (imm_acc / n).tolist(),
        "mu_fut":    (fut_acc / n).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BFS computation
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def compute_bfs_vs_derived(felicific_rows: list, profile_path: str):
    profiles = {}
    if os.path.exists(profile_path):
        with open(profile_path) as f:
            profiles = json.load(f).get("profiles", {})

    results = []
    by_cond = {}
    for row in felicific_rows:
        by_cond.setdefault(row["condition"], []).append(row)

    for cond, rows in by_cond.items():
        profile_key = PROFILE_PAIRS.get(cond, "bentham")
        profile = profiles.get(profile_key, {})
        if not profile:
            continue
        mu_prof_imm = np.array(profile["mu_imm"])
        mu_prof_fut = np.array(profile["mu_fut"])

        for row in rows:
            mu_obs_imm = np.array(row["mu_imm"])
            mu_obs_fut = np.array(row["mu_fut"])
            obs  = np.concatenate([mu_obs_imm, mu_obs_fut])
            prof = np.concatenate([mu_prof_imm, mu_prof_fut])
            results.append({
                "condition":   cond,
                "seed":        row["seed"],
                "n_obs":       row["n_obs"],
                "bfs":         cosine_sim(obs, prof),
                "profile_key": profile_key,
            })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CSV writing
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(path: str, rows: list):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path}")


def aggregate_summaries(summary_rows: list):
    by_cond = {}
    for r in summary_rows:
        by_cond.setdefault(r["condition"], []).append(r)

    agg = []
    for cond, rows in sorted(by_cond.items()):
        ext   = [r["extinct"]    for r in rows]
        pop   = [r["final_pop"]  for r in rows]
        wlth  = [r["mean_wealth"] for r in rows]
        gini  = [r["mean_gini"]  for r in rows]
        ttl   = [r["mean_ttl"]   for r in rows]
        agg.append({
            "condition":        cond,
            "n_seeds":          len(rows),
            "extinction_rate":  round(float(np.mean(ext)),  4),
            "mean_final_pop":   round(float(np.mean(pop)),  2),
            "sd_final_pop":     round(float(np.std(pop)),   2),
            "mean_wealth":      round(float(np.mean(wlth)), 4),
            "mean_gini":        round(float(np.mean(gini)), 4),
            "mean_ttl":         round(float(np.mean(ttl)),  4),
        })
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_dir = os.path.join(OUTPUT_DIR, "runs")
    os.makedirs(run_dir, exist_ok=True)

    base   = load_base_config("config.json")
    seeds  = generate_seeds(args.seeds)
    python = sys.executable

    # Build run list
    run_meta  = []   # (cfg_path, condition, seed, log_path, agent_log_path)
    run_queue = []   # args for pool

    for cond, dm_list in CONDITIONS.items():
        for seed in seeds:
            tag        = f"{cond}_{seed}"
            log_path   = os.path.join(run_dir, f"{tag}.json")
            alog_path  = os.path.join(run_dir, f"{tag}_agents.json")
            cfg        = make_run_config(base, seed, dm_list, args.timesteps,
                                         args.agents, log_path, alog_path)
            cfg_path   = os.path.join(run_dir, f"{tag}_cfg.json")

            if not args.force and os.path.exists(log_path):
                run_meta.append((cfg_path, cond, seed, log_path, alog_path))
                continue

            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
            run_meta.append((cfg_path, cond, seed, log_path, alog_path))
            run_queue.append((cfg_path, python))

    total = len(run_queue)
    if total == 0:
        print("  All runs already complete. Use --force to re-run.")
    else:
        n_cond = len(CONDITIONS)
        print(f"\n{'='*60}")
        print(f"  Consistent-Argmin FVDM Experiment")
        print(f"  Conditions: {n_cond}  Seeds: {args.seeds}  "
              f"Timesteps: {args.timesteps}  Total runs: {total}")
        print(f"  Profile: {PROFILE_PATH}")
        print(f"{'='*60}\n")

        mgr     = multiprocessing.Manager()
        counter = mgr.Value("i", 0)
        lock    = mgr.Lock()
        pool_args = [(cfg, py, counter, lock, total) for cfg, py in run_queue]

        t0 = time.time()
        with multiprocessing.Pool(processes=min(args.jobs, total)) as pool:
            for _ in pool.imap_unordered(run_one_simulation, pool_args):
                pass
        print(f"\n\n  Done in {time.time() - t0:.1f}s\n")

    # ── Parse results ─────────────────────────────────────────────────────────
    print("  Parsing logs ...")
    summary_rows   = []
    felicific_rows = []

    for cfg_path, cond, seed, log_path, alog_path in run_meta:
        row = parse_sim_log(log_path, cond, seed, args.timesteps)
        if row:
            summary_rows.append(row)

        bfe_row = parse_agent_log_bfe(alog_path, cond, seed)
        if bfe_row:
            felicific_rows.append(bfe_row)

    # ── BFS vs derived ────────────────────────────────────────────────────────
    bfs_rows = compute_bfs_vs_derived(felicific_rows, PROFILE_PATH)

    # ── Write CSVs ────────────────────────────────────────────────────────────
    write_csv(os.path.join(OUTPUT_DIR, "per_seed_summary.csv"),    summary_rows)
    write_csv(os.path.join(OUTPUT_DIR, "condition_aggregates.csv"), aggregate_summaries(summary_rows))
    write_csv(os.path.join(OUTPUT_DIR, "bfs_vs_derived.csv"),       bfs_rows)

    # ── Print summary table ───────────────────────────────────────────────────
    agg = aggregate_summaries(summary_rows)
    print(f"\n  {'Condition':<20}  {'Ext%':>6}  {'Avg pop':>8}  {'Avg TTL':>8}  {'Avg wealth':>10}")
    print(f"  {'-'*60}")
    for r in agg:
        print(f"  {r['condition']:<20}  {r['extinction_rate']*100:>5.1f}%"
              f"  {r['mean_final_pop']:>8.1f}  {r['mean_ttl']:>8.2f}"
              f"  {r['mean_wealth']:>10.2f}")

    if bfs_rows:
        by_cond_bfs = {}
        for r in bfs_rows:
            by_cond_bfs.setdefault(r["condition"], []).append(r["bfs"])
        print(f"\n  {'Condition':<20}  {'BFS vs derived (mean)':>22}  {'n_seeds':>8}")
        print(f"  {'-'*55}")
        for cond in sorted(by_cond_bfs):
            vals = [v for v in by_cond_bfs[cond] if not (isinstance(v, float) and math.isnan(v))]
            mean_bfs = float(np.mean(vals)) if vals else float("nan")
            print(f"  {cond:<20}  {mean_bfs:>22.4f}  {len(vals):>8}")

    print(f"\n  Output: {OUTPUT_DIR}/\n")


def parse_args():
    p = argparse.ArgumentParser(description="Consistent-argmin FVDM experiment")
    p.add_argument("-s", "--seeds",     type=int, default=5)
    p.add_argument("-t", "--timesteps", type=int, default=500)
    p.add_argument("-a", "--agents",    type=int, default=250)
    p.add_argument("-j", "--jobs",      type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--force",           action="store_true",
                   help="Re-run even if output files already exist")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
