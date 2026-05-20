#!/usr/bin/env python3
"""
run_experiments_factored_argmin.py
------------------------------------
Experiment 5 (factored argmin variant): runs four FVDM conditions using the
factored four-vector profiles derived from derive_vectors_factored.py.

Decision rule (FVDMArgminFactoredAgent):
  score(c) = ‖ φ·v_self_imm(c) − μ_self_imm ‖₂ + ‖ φ·v_self_fut(c) − μ_self_fut ‖₂
           + ‖ (1−φ)·v_nbr_imm(c) − μ_nbr_imm ‖₂ + ‖ (1−φ)·v_nbr_fut(c) − μ_nbr_fut ‖₂
  c* = argmin score(c)

Profiles are loaded from fvdm_vectors/bfe_profiles_factored.json.
For egoist (φ=1): neighbor terms vanish → agents seek high-resource cells → no extinction.

Conditions:
  factoredRaw      (fvdmFactoredRaw,      φ=1.0)
  factoredEgoist   (fvdmFactoredEgoist,   φ=1.0)
  factoredAltruist (fvdmFactoredAltruist, φ=0.0)
  factoredBentham  (fvdmFactoredBentham,  φ=0.5)

Outputs (under --output/results/):
  per_timestep.csv              — timestep-level population metrics per seed
  per_seed_summary.csv          — one row per (condition, seed)
  per_seed_felicific.csv        — mean φ-scaled self/nbr vectors per (condition, seed)
  condition_aggregates.csv      — mean ± sd across seeds per condition
  bfs_vs_derived.csv            — BFS between observed and derived factored profiles

Usage:
  python run_experiments_factored_argmin.py
  python run_experiments_factored_argmin.py -s 30 -t 5000 -a 250 -j 30
  python run_experiments_factored_argmin.py --force
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
    "factoredRaw":      ["fvdmFactoredRaw"],
    "factoredEgoist":   ["fvdmFactoredEgoist"],
    "factoredAltruist": ["fvdmFactoredAltruist"],
    "factoredBentham":  ["fvdmFactoredBentham"],
}

PROFILE_PAIRS = {
    "factoredRaw":      "rawSugarscape",
    "factoredEgoist":   "egoist",
    "factoredAltruist": "altruist",
    "factoredBentham":  "bentham",
}

LABELS = ["I", "D", "C", "P", "E"]

PROFILE_PATH = "fvdm_vectors/bfe_profiles_factored.json"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def generate_seeds(n: int, seed: int = 42) -> list:
    random.seed(seed)
    seeds = []
    while len(seeds) < n:
        s = random.randint(0, sys.maxsize)
        if s not in seeds:
            seeds.append(s)
    return seeds


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


def safe_json_load(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


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

    # Felicific vectors: fct_self_imm_*, fct_self_fut_*, fct_nbr_imm_*, fct_nbr_fut_*
    felicific = {"condition": condition, "seed": seed}
    agent_log = safe_json_load(agent_log_path)
    if agent_log:
        si_vecs, sf_vecs, ni_vecs, nf_vecs = [], [], [], []
        for row in agent_log:
            si = np.array([float(row.get(f"fct_self_imm_{l}", 0)) for l in LABELS])
            sf = np.array([float(row.get(f"fct_self_fut_{l}",  0)) for l in LABELS])
            ni = np.array([float(row.get(f"fct_nbr_imm_{l}",  0)) for l in LABELS])
            nf = np.array([float(row.get(f"fct_nbr_fut_{l}",  0)) for l in LABELS])
            if np.any(si != 0) or np.any(ni != 0):
                si_vecs.append(si); sf_vecs.append(sf)
                ni_vecs.append(ni); nf_vecs.append(nf)
        if si_vecs:
            mu_si = np.mean(si_vecs, axis=0)
            mu_sf = np.mean(sf_vecs, axis=0)
            mu_ni = np.mean(ni_vecs, axis=0)
            mu_nf = np.mean(nf_vecs, axis=0)
            for i, lbl in enumerate(LABELS):
                felicific[f"mu_self_imm_{lbl}"] = round(float(mu_si[i]), 6)
                felicific[f"mu_self_fut_{lbl}"]  = round(float(mu_sf[i]), 6)
                felicific[f"mu_nbr_imm_{lbl}"]  = round(float(mu_ni[i]), 6)
                felicific[f"mu_nbr_fut_{lbl}"]   = round(float(mu_nf[i]), 6)
            felicific["n_agent_timesteps"] = len(si_vecs)

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
                keys.append(k); seen.add(k)
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
# BFS — four-vector cosine similarity
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def compute_bfs_vs_derived(felicific_rows: list) -> list:
    """
    BFS = mean cosine similarity across the four profile vectors.
    BFS ≈ 1.0 means the factored argmin agent chose cells whose scaled vectors
    match the derived profile — confirming correct behavioral fidelity.
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

        obs_si = np.array([float(frow.get(f"mu_self_imm_{l}", 0)) for l in LABELS])
        obs_sf = np.array([float(frow.get(f"mu_self_fut_{l}",  0)) for l in LABELS])
        obs_ni = np.array([float(frow.get(f"mu_nbr_imm_{l}",  0)) for l in LABELS])
        obs_nf = np.array([float(frow.get(f"mu_nbr_fut_{l}",  0)) for l in LABELS])

        drv_si = np.array(dp.get("mu_self_imm", [0.0] * 5))
        drv_sf = np.array(dp.get("mu_self_fut",  [0.0] * 5))
        drv_ni = np.array(dp.get("mu_nbr_imm",  [0.0] * 5))
        drv_nf = np.array(dp.get("mu_nbr_fut",   [0.0] * 5))

        cos_si = cosine_similarity(obs_si, drv_si)
        cos_sf = cosine_similarity(obs_sf, drv_sf)
        cos_ni = cosine_similarity(obs_ni, drv_ni)
        cos_nf = cosine_similarity(obs_nf, drv_nf)

        valid  = [c for c in [cos_si, cos_sf, cos_ni, cos_nf] if not math.isnan(c)]
        bfs    = sum(valid) / len(valid) if valid else float("nan")

        rows.append({
            "condition":      cond,
            "derived_profile": prof_key,
            "seed":            frow["seed"],
            "cos_self_imm":    round(cos_si, 6) if not math.isnan(cos_si) else "",
            "cos_self_fut":    round(cos_sf, 6) if not math.isnan(cos_sf) else "",
            "cos_nbr_imm":    round(cos_ni, 6) if not math.isnan(cos_ni) else "",
            "cos_nbr_fut":     round(cos_nf, 6) if not math.isnan(cos_nf) else "",
            "bfs":             round(bfs, 6) if not math.isnan(bfs) else "",
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

    base_cfg = load_base_config(config_path)
    seeds    = generate_seeds(num_seeds)

    sim_dir     = os.path.join(output_root, "sim_logs")
    results_dir = os.path.join(output_root, "results")
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  FVDM Factored Argmin Experiment Runner")
    print(f"{'='*65}")
    print(f"  Decision  : argmin φ·dist_self(c) + (1−φ)·dist_nbr(c)")
    print(f"  Profiles  : {PROFILE_PATH}")
    print(f"  Seeds:    {num_seeds}   Timesteps: {timesteps}   Agents: {num_agents}")
    print(f"  Cores:    {num_cores}   Output:    {output_root}")
    print(f"{'='*65}\n")

    if not os.path.exists(PROFILE_PATH):
        print(f"  [error] Profile file not found: {PROFILE_PATH}")
        print(f"          Run derive_vectors_factored.py --homogeneous first.")
        sys.exit(1)

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
    all_pts        = []
    all_summaries  = []
    all_felicific  = []
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
    bfs_rows = compute_bfs_vs_derived(all_felicific)
    if bfs_rows:
        write_csv(bfs_rows, os.path.join(results_dir, "bfs_vs_derived.csv"))
        print(f"\n  BFS — observed vs. factored derived profile  (≈1.0 = good fidelity):")
        print(f"  {'Condition':<22}  {'Seeds':>6}  {'BFS':>8}  "
              f"{'cos_si':>8}  {'cos_sf':>8}  {'cos_ni':>8}  {'cos_nf':>8}")
        print(f"  {'-'*76}")
        for cond, grp in groupby(bfs_rows, key=lambda r: r["condition"]):
            g = list(grp)
            def mean_col(col):
                vals = [r[col] for r in g if isinstance(r[col], float)]
                return sum(vals) / len(vals) if vals else float("nan")
            print(f"  {cond:<22}  {len(g):>6}  "
                  f"{mean_col('bfs'):>8.4f}  "
                  f"{mean_col('cos_self_imm'):>8.4f}  "
                  f"{mean_col('cos_self_fut'):>8.4f}  "
                  f"{mean_col('cos_nbr_imm'):>8.4f}  "
                  f"{mean_col('cos_nbr_fut'):>8.4f}")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {'Condition':<22}  {'Seeds':>6}  {'Extinct%':>9}  "
          f"{'FinalPop':>10}  {'Gini':>7}  {'TTL':>7}")
    print(f"  {'-'*67}")
    for row in agg_rows:
        cn  = row.get("condition", "?")
        ns  = row.get("numSeeds", 0)
        ext = row.get("extinctionRate", 0) * 100
        fp  = row.get("mean_finalPopulation", 0)
        gi  = row.get("mean_finalGini", 0)
        ttl = row.get("mean_finalMeanTimeToLive", 0)
        print(f"  {cn:<22}  {ns:>6}  {ext:>8.1f}%  {fp:>10.1f}  {gi:>7.3f}  {ttl:>7.2f}")
    print(f"{'='*72}\n")
    print(f"  Outputs written to {results_dir}/\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run factored argmin FVDM conditions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c", "--config",    default="config.json")
    p.add_argument("-o", "--output",    default="fvdm_factored_results")
    p.add_argument("-s", "--seeds",     type=int, default=30)
    p.add_argument("-t", "--timesteps", type=int, default=5000)
    p.add_argument("-a", "--agents",    type=int, default=250)
    p.add_argument("-j", "--cores",     type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--python",          default=sys.executable)
    p.add_argument("--force", action="store_true",
                   help="Re-run simulations even if log files exist.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
