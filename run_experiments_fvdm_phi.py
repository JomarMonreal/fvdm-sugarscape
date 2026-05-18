#!/usr/bin/env python3
"""
run_experiments_fvdm_phi.py
----------------------------
Runs the four φ-welfare FVDM conditions (Conditions 5–8, corrected).

Design principle (Option 3 — FVDM as measurement framework):
  Cell selection uses the originating Bentham welfare rule — identical to the
  egoist/altruist/bentham baseline — so agents survive and produce valid data.
  After each move, v_imm and v_fut are recorded for BFS verification:
    BFS(μ_obs, μ_derived) ≈ 1  →  FVDM correctly characterises the agent.

Conditions:
  phiRawDerived       (fvdmPhiRaw,      φ=1.0  — greedy baseline)
  phiEgoistDerived    (fvdmPhiEgoist,   φ=1.0)
  phiAltruistDerived  (fvdmPhiAltruist, φ=0.0)
  phiBenthamDerived   (fvdmPhiBentham,  φ=0.5)

Outputs (under --output/results/):
  per_timestep.csv              — timestep-level population metrics per seed
  per_seed_summary.csv          — one row per (condition, seed)
  per_seed_felicific.csv        — mean v_imm and v_fut per (condition, seed)
  condition_aggregates.csv      — mean ± sd across seeds per condition
  bfs_vs_derived.csv            — BFS between observed BFE and derived profile
  bfs_vs_baseline.csv           — BFS between φ-welfare agent and baseline

Usage:
  python run_experiments_fvdm_phi.py
  python run_experiments_fvdm_phi.py -s 30 -t 5000 -a 250 -j 30
  python run_experiments_fvdm_phi.py --force
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
from itertools import groupby

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Conditions
# ─────────────────────────────────────────────────────────────────────────────

CONDITIONS = {
    "phiRawDerived":      ["fvdmPhiRaw"],
    "phiEgoistDerived":   ["fvdmPhiEgoist"],
    "phiAltruistDerived": ["fvdmPhiAltruist"],
    "phiBenthamDerived":  ["fvdmPhiBentham"],
}

# Map each φ-welfare condition to its paired baseline condition (same φ)
BASELINE_PAIRS = {
    "phiRawDerived":      "rawSugarscape",
    "phiEgoistDerived":   "egoist",
    "phiAltruistDerived": "altruist",
    "phiBenthamDerived":  "bentham",
}

# Map each φ-welfare condition to the derived profile key in bfe_profiles.json
PROFILE_PAIRS = {
    "phiRawDerived":      "rawSugarscape",
    "phiEgoistDerived":   "egoist",
    "phiAltruistDerived": "altruist",
    "phiBenthamDerived":  "bentham",
}

IMM_LABELS = ["I", "D", "C", "P", "E"]
FUT_LABELS = ["I", "D", "C", "P", "E"]

PROFILE_PATH = "fvdm_vectors/bfe_profiles.json"

# ─────────────────────────────────────────────────────────────────────────────
# Seed generation — must match run_experiments_baseline.py
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
    cfg["seed"]                = seed
    cfg["agentDecisionModels"] = decision_models
    cfg["timesteps"]           = timesteps
    cfg["startingAgents"]      = num_agents
    cfg["startingDiseases"]    = 0
    cfg["headlessMode"]        = True
    cfg["debugMode"]           = ["none"]
    cfg["keepAlivePostExtinction"] = False
    cfg["keepAliveAtEnd"]      = False
    cfg["screenshots"]         = False
    cfg["profileMode"]         = False
    cfg["logfile"]             = log_path
    cfg["agentLogfile"]        = agent_log_path
    cfg["logfileFormat"]       = "json"
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


def parse_sim_log(log_path: str, condition: str, seed: int,
                  agent_log_path: str, duration: float = 0.0):
    sim_log = safe_json_load(log_path)
    if not sim_log or len(sim_log) == 0:
        return None, None, None

    per_timestep = []
    total_deaths = total_born = final_pop = 0

    for entry in sim_log:
        ts  = int(entry.get("timestep", 0))
        pop = int(entry.get("population", 0))
        d   = int(entry.get("agentDeaths", 0))
        b   = int(entry.get("agentsBorn", 0))
        total_deaths += d
        total_born   += b
        per_timestep.append({
            "condition":              condition,
            "seed":                   seed,
            "timestep":               ts,
            "population":             pop,
            "meanWealth":             float(entry.get("meanWealth", 0)),
            "minWealth":              float(entry.get("minWealth", 0)),
            "maxWealth":              float(entry.get("maxWealth", 0)),
            "societalWealth":         float(entry.get("agentWealthTotal", 0)),
            "giniCoefficient":        float(entry.get("giniCoefficient", 0)),
            "agentDeaths":            d,
            "agentAgingDeaths":       int(entry.get("agentAgingDeaths", 0)),
            "agentStarvationDeaths":  int(entry.get("agentStarvationDeaths", 0)),
            "agentCombatDeaths":      int(entry.get("agentCombatDeaths", 0)),
            "agentsBorn":             b,
            "meanTimeToLive":         float(entry.get("agentMeanTimeToLive", 0)),
            "meanAge":                float(entry.get("meanAge", 0)),
            "tradeVolume":            float(entry.get("tradeVolume", 0)),
            "meanHappiness":          float(entry.get("meanHappiness", 0)),
            "combatActions":          int(entry.get("actionCombats", 0)),
            "tradeActions":           int(entry.get("actionTrades", 0)),
            "reproductionActions":    int(entry.get("actionReproductions", 0)),
            "lendingActions":         int(entry.get("actionLendings", 0)),
            "movementActions":        int(entry.get("actionMovements", 0)),
        })
        final_pop = pop

    # Felicific vectors from agent log (FVDMPhiAgent logs v_imm_* and v_fut_*)
    felicific  = {"condition": condition, "seed": seed}
    agent_log  = safe_json_load(agent_log_path)
    if agent_log:
        imm_vecs, fut_vecs = [], []
        for row in agent_log:
            v_imm = np.array([float(row.get(f"v_imm_{l}", 0)) for l in IMM_LABELS])
            v_fut = np.array([float(row.get(f"v_fut_{l}", 0)) for l in FUT_LABELS])
            if np.any(v_imm != 0) or np.any(v_fut != 0):
                imm_vecs.append(v_imm)
                fut_vecs.append(v_fut)
        if imm_vecs:
            mu_imm = np.mean(imm_vecs, axis=0)
            mu_fut = np.mean(fut_vecs, axis=0)
            for i, lbl in enumerate(IMM_LABELS):
                felicific[f"mean_v_imm_{lbl}"] = round(float(mu_imm[i]), 6)
            for i, lbl in enumerate(FUT_LABELS):
                felicific[f"mean_v_fut_{lbl}"] = round(float(mu_fut[i]), 6)
            felicific["n_agent_timesteps"] = len(imm_vecs)

    last = per_timestep[-1] if per_timestep else {}
    summary = {
        "condition":               condition,
        "seed":                    seed,
        "executionTime":           round(duration, 2),
        "extinct":                 (final_pop == 0),
        "finalPopulation":         final_pop,
        "totalDeaths":             total_deaths,
        "totalBorn":               total_born,
        "finalTimestep":           last.get("timestep", 0),
        "finalMeanWealth":         last.get("meanWealth", 0),
        "finalSocietalWealth":     last.get("societalWealth", 0),
        "finalGini":               last.get("giniCoefficient", 0),
        "finalMeanTimeToLive":     last.get("meanTimeToLive", 0),
        "totalCombatActions":      sum(r["combatActions"] for r in per_timestep),
        "totalTradeActions":       sum(r["tradeActions"] for r in per_timestep),
        "totalReproductionActions": sum(r["reproductionActions"] for r in per_timestep),
        "totalLendingActions":     sum(r["lendingActions"] for r in per_timestep),
    }
    return per_timestep, summary, felicific


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
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
        writer = csv_mod.DictWriter(f, fieldnames=keys, restval="")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_seeds(summaries: list) -> dict:
    if not summaries:
        return {}
    num_keys = [k for k, v in summaries[0].items()
                if isinstance(v, (int, float)) and k != "seed"]
    agg = {
        "condition":      summaries[0]["condition"],
        "numSeeds":       len(summaries),
        "extinctionRate": sum(1 for s in summaries if s["extinct"]) / len(summaries),
    }
    for k in num_keys:
        vals = [s[k] for s in summaries]
        mu   = sum(vals) / len(vals)
        std  = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
        agg[f"mean_{k}"] = round(mu, 4)
        agg[f"std_{k}"]  = round(std, 4)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity and BFS
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_bfs_vs_derived(felicific_rows: list) -> list:
    """
    BFS between observed BFE (φ-welfare run) and derived profile from
    bfe_profiles.json. High BFS (≈1) confirms FVDM characterises the
    agent correctly.
    """
    if not os.path.exists(PROFILE_PATH):
        print(f"  [warn] {PROFILE_PATH} not found — skipping BFS vs derived.")
        return []
    with open(PROFILE_PATH) as f:
        derived_profiles = json.load(f).get("profiles", {})

    rows = []
    for frow in felicific_rows:
        cond     = frow["condition"]
        prof_key = PROFILE_PAIRS.get(cond)
        if not prof_key or prof_key not in derived_profiles:
            continue
        dp = derived_profiles[prof_key]

        obs_imm  = np.array([float(frow.get(f"mean_v_imm_{l}", 0)) for l in IMM_LABELS])
        obs_fut  = np.array([float(frow.get(f"mean_v_fut_{l}", 0)) for l in FUT_LABELS])
        drv_imm  = np.array(dp["mu_imm"])
        drv_fut  = np.array(dp["mu_fut"])

        cos_imm = cosine_similarity(obs_imm, drv_imm)
        cos_fut = cosine_similarity(obs_fut, drv_fut)
        bfs     = (cos_imm + cos_fut) / 2.0

        rows.append({
            "phi_condition":     cond,
            "derived_profile":   prof_key,
            "seed":              frow["seed"],
            "cosine_imm":        round(cos_imm, 6),
            "cosine_fut":        round(cos_fut, 6),
            "bfs_vs_derived":    round(bfs, 6),
        })
    return rows


def compute_bfs_vs_baseline(felicific_rows: list, baseline_dir: str) -> list:
    """
    BFS between observed BFE (φ-welfare run) and baseline BFE for the same
    seed. Confirms φ-welfare agents behave identically to baseline agents.
    """
    baseline_path = os.path.join(baseline_dir, "results", "per_seed_felicific.csv")
    if not os.path.exists(baseline_path):
        print(f"  [warn] Baseline felicific CSV not found: {baseline_path}")
        return []

    baseline_rows = {}
    with open(baseline_path, newline="") as f:
        for row in csv_mod.DictReader(f):
            key = (row["condition"], int(float(row["seed"])))
            baseline_rows[key] = row

    rows = []
    for frow in felicific_rows:
        cond      = frow["condition"]
        seed      = frow["seed"]
        base_cond = BASELINE_PAIRS.get(cond)
        if not base_cond:
            continue
        brow = baseline_rows.get((base_cond, seed))
        if not brow:
            continue

        obs_imm  = np.array([float(frow.get(f"mean_v_imm_{l}", 0)) for l in IMM_LABELS])
        obs_fut  = np.array([float(frow.get(f"mean_v_fut_{l}", 0)) for l in FUT_LABELS])
        base_imm = np.array([float(brow.get(f"mean_v_imm_{l}", 0)) for l in IMM_LABELS])
        base_fut = np.array([float(brow.get(f"mean_v_fut_{l}", 0)) for l in FUT_LABELS])

        cos_imm = cosine_similarity(obs_imm, base_imm)
        cos_fut = cosine_similarity(obs_fut, base_fut)
        bfs     = (cos_imm + cos_fut) / 2.0

        rows.append({
            "phi_condition":      cond,
            "baseline_condition": base_cond,
            "seed":               seed,
            "cosine_imm":         round(cos_imm, 6),
            "cosine_fut":         round(cos_fut, 6),
            "bfs_vs_baseline":    round(bfs, 6),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    config_path  = args.config
    output_root  = args.output
    num_seeds    = args.seeds
    num_agents   = args.agents
    timesteps    = args.timesteps
    num_cores    = min(args.cores, os.cpu_count() or 1)
    python_alias = args.python
    force        = args.force
    baseline_dir = args.baseline_dir

    base_cfg = load_base_config(config_path)
    seeds    = generate_seeds(num_seeds)

    sim_dir     = os.path.join(output_root, "sim_logs")
    results_dir = os.path.join(output_root, "results")
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  FVDM φ-Welfare Experiment Runner (Conditions 5–8, corrected)")
    print(f"{'='*65}")
    print(f"  Decision rule : φ·h_self(c) + (1−φ)·h_neighbors(c)  [Bentham]")
    print(f"  Measurement   : v_imm / v_fut logged → BFS vs derived profiles")
    print(f"  Seeds:     {num_seeds}   Timesteps: {timesteps}   Agents: {num_agents}")
    print(f"  Cores:     {num_cores}   Output:    {output_root}")
    print(f"{'='*65}\n")

    # ── Build run configs ────────────────────────────────────────────────────
    all_runs = []
    for cname, models in CONDITIONS.items():
        cdir = os.path.join(sim_dir, cname)
        os.makedirs(cdir, exist_ok=True)
        for seed in seeds:
            log_path       = os.path.join(cdir, f"{cname}_{seed}.json")
            agent_log_path = os.path.join(cdir, f"{cname}_{seed}_agents.json")
            cfg      = make_run_config(base_cfg, seed, models, timesteps,
                                       num_agents, log_path, agent_log_path)
            cfg_path = os.path.join(cdir, f"{cname}_{seed}.config")
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
            all_runs.append((cname, seed, cfg_path, log_path, agent_log_path))

    pending = []
    for (cname, seed, cfg_path, log_path, _) in all_runs:
        if not force and os.path.exists(log_path):
            d = safe_json_load(log_path)
            if d and len(d) > 0:
                last_ts  = d[-1].get("timestep", 0)
                last_pop = d[-1].get("population", -1)
                if int(last_ts) >= timesteps or int(last_pop) == 0:
                    continue
        pending.append((cname, seed, cfg_path, log_path))

    total = len(all_runs)
    print(f"  Total runs: {total}  |  Queued: {len(pending)}  |  "
          f"Skipped: {total - len(pending)}\n")

    # ── Run simulations ──────────────────────────────────────────────────────
    session_durations = {}
    if pending:
        t_start = time.time()
        manager = multiprocessing.Manager()
        counter = manager.Value("i", 0)
        lock    = manager.Lock()
        wargs   = [(cfg_path, python_alias, counter, lock, len(pending))
                   for (_, _, cfg_path, _) in pending]
        print(f"  Running {len(pending)} simulations on {num_cores} core(s) …")
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(run_one_simulation, wargs)
        session_durations = {path: dur for path, dur in results}
        print(f"\n\n  Completed in {time.time() - t_start:.1f}s\n")

    # ── Parse results ────────────────────────────────────────────────────────
    print("  Parsing logs …")
    all_pts       = []
    all_summaries = []
    all_felicific = []
    cond_summaries = {c: [] for c in CONDITIONS}

    for (cname, seed, cfg_path, log_path, agent_log_path) in all_runs:
        dur = session_durations.get(cfg_path, 0.0)
        pts, summary, felicific = parse_sim_log(log_path, cname, seed,
                                                 agent_log_path, dur)
        if pts is None:
            print(f"  [warn] Could not parse {cname} seed={seed}")
            continue
        all_pts.extend(pts)
        all_summaries.append(summary)
        cond_summaries[cname].append(summary)
        if felicific:
            all_felicific.append(felicific)

    # ── Write main CSVs ──────────────────────────────────────────────────────
    write_csv(all_pts,       os.path.join(results_dir, "per_timestep.csv"))
    write_csv(all_summaries, os.path.join(results_dir, "per_seed_summary.csv"))
    write_csv(all_felicific, os.path.join(results_dir, "per_seed_felicific.csv"))

    agg_rows = []
    for cname, summs in cond_summaries.items():
        if summs:
            agg_rows.append(aggregate_seeds(summs))
    write_csv(agg_rows, os.path.join(results_dir, "condition_aggregates.csv"))

    for cname in CONDITIONS:
        cd = os.path.join(results_dir, cname)
        os.makedirs(cd, exist_ok=True)
        write_csv([r for r in all_pts       if r["condition"] == cname],
                  os.path.join(cd, "per_timestep.csv"))
        write_csv([r for r in all_summaries if r["condition"] == cname],
                  os.path.join(cd, "per_seed_summary.csv"))
        write_csv([r for r in all_felicific if r["condition"] == cname],
                  os.path.join(cd, "per_seed_felicific.csv"))

    # ── BFS vs derived profiles ──────────────────────────────────────────────
    bfs_derived = compute_bfs_vs_derived(all_felicific)
    if bfs_derived:
        write_csv(bfs_derived, os.path.join(results_dir, "bfs_vs_derived.csv"))
        print(f"\n  BFS — observed BFE vs. derived profile  (should be ≈ 1.0):")
        print(f"  {'Condition':<24} {'Seeds':>6} {'mean BFS':>10} "
              f"{'cos_imm':>9} {'cos_fut':>9}")
        print(f"  {'-'*60}")
        for cond, grp in groupby(bfs_derived, key=lambda r: r["phi_condition"]):
            g = list(grp)
            print(f"  {cond:<24} {len(g):>6} "
                  f"{np.mean([r['bfs_vs_derived'] for r in g]):>10.4f} "
                  f"{np.mean([r['cosine_imm']     for r in g]):>9.4f} "
                  f"{np.mean([r['cosine_fut']      for r in g]):>9.4f}")

    # ── BFS vs baseline ──────────────────────────────────────────────────────
    bfs_baseline = compute_bfs_vs_baseline(all_felicific, baseline_dir)
    if bfs_baseline:
        write_csv(bfs_baseline, os.path.join(results_dir, "bfs_vs_baseline.csv"))
        print(f"\n  BFS — φ-welfare agent vs. baseline  (should be ≈ 1.0):")
        print(f"  {'Pair':<40} {'Seeds':>6} {'mean BFS':>10} "
              f"{'cos_imm':>9} {'cos_fut':>9}")
        print(f"  {'-'*74}")
        for cond, grp in groupby(bfs_baseline, key=lambda r: r["phi_condition"]):
            g    = list(grp)
            pair = f"{BASELINE_PAIRS[cond]} ↔ {cond}"
            print(f"  {pair:<40} {len(g):>6} "
                  f"{np.mean([r['bfs_vs_baseline'] for r in g]):>10.4f} "
                  f"{np.mean([r['cosine_imm']      for r in g]):>9.4f} "
                  f"{np.mean([r['cosine_fut']       for r in g]):>9.4f}")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {'Condition':<24} {'Seeds':>6} {'Extinct%':>9} "
          f"{'FinalPop':>10} {'Gini':>7} {'TTL':>7}")
    print(f"  {'-'*67}")
    for row in agg_rows:
        cn  = row.get("condition", "?")
        ns  = row.get("numSeeds", 0)
        ext = row.get("extinctionRate", 0) * 100
        fp  = row.get("mean_finalPopulation", 0)
        gi  = row.get("mean_finalGini", 0)
        ttl = row.get("mean_finalMeanTimeToLive", 0)
        print(f"  {cn:<24} {ns:>6} {ext:>8.1f}% {fp:>10.1f} {gi:>7.3f} {ttl:>7.2f}")
    print(f"{'='*72}\n")
    print(f"  Outputs written to {results_dir}/\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run φ-welfare FVDM conditions with BFS measurement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c", "--config",       default="config.json")
    p.add_argument("-o", "--output",       default="fvdm_phi_results")
    p.add_argument("-s", "--seeds",        type=int, default=30)
    p.add_argument("-t", "--timesteps",    type=int, default=5000)
    p.add_argument("-a", "--agents",       type=int, default=250)
    p.add_argument("-j", "--cores",        type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--python",             default=sys.executable)
    p.add_argument("--baseline-dir",       default="baseline_results",
                   help="Directory containing baseline per_seed_felicific.csv "
                        "for BFS-vs-baseline computation.")
    p.add_argument("--force", action="store_true",
                   help="Re-run simulations even if log files exist.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
