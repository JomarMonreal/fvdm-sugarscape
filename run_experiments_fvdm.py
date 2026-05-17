#!/usr/bin/env python3
"""
run_experiments_fvdm.py
------------------------
Runs the four homogeneous FVDM-derived conditions (conditions 5-8):
  5. fvdmRawDerived      (profile: rawSugarscape)
  6. fvdmEgoistDerived   (profile: egoist)
  7. fvdmAltruistDerived (profile: altruist)
  8. fvdmBenthamDerived  (profile: bentham)

Uses the SAME 30 matched seeds as run_experiments_baseline.py (generated
deterministically with random.seed(42)) so that paired comparisons across
baseline and FVDM conditions are seed-matched.

FVDMAgent logs v_imm_* and v_fut_* (the analytically computed effect vectors
of the chosen cell) directly in the agent log via updateRuntimeStats().
These are read and averaged per run to produce per_seed_felicific.csv,
which is the primary data source for:
  • Cosine similarity between baseline and FVDM conditions (BFS)
  • Kruskal-Wallis across FVDM-derived conditions

Requires:
  fvdm_vectors/bfe_profiles.json  (produced by derive_vectors.py)

Outputs (under --output/results/):
  per_timestep.csv       — timestep-level population metrics per seed
  per_seed_summary.csv   — one row per (condition, seed) with all final metrics
  per_seed_felicific.csv — mean v_imm and v_fut per (condition, seed)
  condition_aggregates.csv

Usage:
  python run_experiments_fvdm.py [options]
  python run_experiments_fvdm.py --seeds 30 --cores 8
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
# Condition definitions (FVDM-derived only)
# ─────────────────────────────────────────────────────────────────────────────

CONDITIONS = {
    "fvdmRawDerived":      ["fvdmRawDerived"],
    "fvdmEgoistDerived":   ["fvdmEgoistDerived"],
    "fvdmAltruistDerived": ["fvdmAltruistDerived"],
    "fvdmBenthamDerived":  ["fvdmBenthamDerived"],
}

# Paired baseline condition for each FVDM condition (for BFS computation)
BASELINE_PAIRS = {
    "fvdmRawDerived":      "rawSugarscape",
    "fvdmEgoistDerived":   "egoist",
    "fvdmAltruistDerived": "altruist",
    "fvdmBenthamDerived":  "bentham",
}

IMM_LABELS = ["I", "D", "C", "P", "E"]
FUT_LABELS = ["I", "D", "C", "P", "E"]

# Step size (percentage points) for the heterogeneous Egoist-Bentham sweep
HETERO_STEP = 10

# ─────────────────────────────────────────────────────────────────────────────
# Shared seed generation (must match run_experiments_baseline.py)
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
# Heterogeneous Egoist-Bentham sweep helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_models_list_fvdm(pct_bentham: int, total_agents: int = 250) -> list:
    """Return the agentDecisionModels list for a given Bentham proportion (0–100)."""
    if pct_bentham == 0:
        return ["fvdmEgoistDerived"]
    if pct_bentham == 100:
        return ["fvdmBenthamDerived"]
    from math import gcd
    n_bentham = round(pct_bentham / 100 * total_agents)
    n_egoist  = total_agents - n_bentham
    g = gcd(n_bentham, n_egoist)
    return ["fvdmEgoistDerived"] * (n_egoist // g) + ["fvdmBenthamDerived"] * (n_bentham // g)


def make_fvdm_hetero_conditions(step: int = HETERO_STEP):
    """
    Build the heterogeneous FVDM conditions (one per proportion level).
    Returns (fvdm_conds, base_pairs) where:
      fvdm_conds : dict  condition_name -> models list
      base_pairs : dict  condition_name -> matching baseline condition name
    """
    fvdm_conds = {}
    base_pairs = {}
    for pct in range(0, 101, step):
        fname = f"fvdmHeteroMix_p{pct:03d}"
        bname = f"heteroMix_p{pct:03d}"
        fvdm_conds[fname] = make_models_list_fvdm(pct)
        base_pairs[fname]  = bname
    return fvdm_conds, base_pairs


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


def parse_sim_log(log_path: str, condition: str, seed: int,
                  agent_log_path: str, duration: float = 0.0):
    """
    Parse one FVDM simulation run.
    FVDMAgent writes v_imm_* and v_fut_* directly to each agent log entry,
    so no re-computation is needed.
    """
    sim_log = safe_json_load(log_path)
    if not sim_log or len(sim_log) == 0:
        return None, None, None

    per_timestep  = []
    total_deaths  = 0
    total_born    = 0
    final_pop     = 0

    for entry in sim_log:
        ts  = int(entry.get("timestep", 0))
        pop = int(entry.get("population", 0))
        d   = int(entry.get("agentDeaths", 0))
        b   = int(entry.get("agentsBorn", 0))
        total_deaths += d
        total_born   += b

        rec = {
            "condition":           condition,
            "seed":                seed,
            "timestep":            ts,
            "population":          pop,
            "meanWealth":          float(entry.get("meanWealth", 0)),
            "minWealth":           float(entry.get("minWealth", 0)),
            "maxWealth":           float(entry.get("maxWealth", 0)),
            "societalWealth":      float(entry.get("agentWealthTotal", 0)),
            "giniCoefficient":     float(entry.get("giniCoefficient", 0)),
            "agentDeaths":         d,
            "agentAgingDeaths":    int(entry.get("agentAgingDeaths", 0)),
            "agentStarvationDeaths": int(entry.get("agentStarvationDeaths", 0)),
            "agentCombatDeaths":   int(entry.get("agentCombatDeaths", 0)),
            "agentsBorn":          b,
            "meanTimeToLive":      float(entry.get("agentMeanTimeToLive", 0)),
            "meanAge":             float(entry.get("meanAge", 0)),
            "tradeVolume":         float(entry.get("tradeVolume", 0)),
            "meanHappiness":       float(entry.get("meanHappiness", 0)),
            "combatActions":       int(entry.get("actionCombats", 0)),
            "tradeActions":        int(entry.get("actionTrades", 0)),
            "reproductionActions": int(entry.get("actionReproductions", 0)),
            "lendingActions":      int(entry.get("actionLendings", 0)),
            "movementActions":     int(entry.get("actionMovements", 0)),
        }
        per_timestep.append(rec)
        final_pop = pop

    # ── Felicific profiles from agent log (v_imm_*, v_fut_* logged by FVDMAgent) ──
    felicific = {"condition": condition, "seed": seed}
    agent_log = safe_json_load(agent_log_path)

    if agent_log:
        imm_vecs = []
        fut_vecs = []
        for row in agent_log:
            v_imm = np.array([
                float(row.get("v_imm_I", 0)),
                float(row.get("v_imm_D", 0)),
                float(row.get("v_imm_C", 0)),
                float(row.get("v_imm_P", 0)),
                float(row.get("v_imm_E", 0)),
            ])
            v_fut = np.array([
                float(row.get("v_fut_I", 0)),
                float(row.get("v_fut_D", 0)),
                float(row.get("v_fut_C", 0)),
                float(row.get("v_fut_P", 0)),
                float(row.get("v_fut_E", 0)),
            ])
            # Skip zero vectors (agent did not yet record a valid choice)
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
        "condition":          condition,
        "seed":               seed,
        "executionTime":      round(duration, 2),
        "extinct":            (final_pop == 0),
        "finalPopulation":    final_pop,
        "totalDeaths":        total_deaths,
        "totalBorn":          total_born,
        "finalTimestep":      last.get("timestep", 0),
        "finalMeanWealth":    last.get("meanWealth", 0),
        "finalSocietalWealth": last.get("societalWealth", 0),
        "finalGini":          last.get("giniCoefficient", 0),
        "finalMeanTimeToLive": last.get("meanTimeToLive", 0),
        "totalCombatActions": sum(r["combatActions"] for r in per_timestep),
        "totalTradeActions":  sum(r["tradeActions"] for r in per_timestep),
        "totalReproductionActions": sum(r["reproductionActions"] for r in per_timestep),
        "totalLendingActions": sum(r["lendingActions"] for r in per_timestep),
    }

    return per_timestep, summary, felicific


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
# Behavioral Fidelity Score (BFS) computation
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_bfs(fvdm_felicific: list, baseline_dir: str) -> list:
    """
    Compute per-seed Behavioral Fidelity Score for each FVDM condition.
    BFS = mean cosine similarity across both temporal layers.

    Reads the baseline per_seed_felicific.csv from baseline_dir if available.
    Returns list of dicts with BFS per (fvdm_condition, seed).
    """
    baseline_path = os.path.join(baseline_dir, "results", "per_seed_felicific.csv")
    if not os.path.exists(baseline_path):
        print(f"  [warn] Baseline felicific CSV not found: {baseline_path}")
        print(f"         Run run_experiments_baseline.py first for BFS computation.")
        return []

    import csv as csv_mod
    baseline_rows = {}
    with open(baseline_path, newline="") as f:
        for row in csv_mod.DictReader(f):
            key = (row["condition"], int(float(row["seed"])))
            baseline_rows[key] = row

    bfs_rows = []
    for frow in fvdm_felicific:
        fvdm_cond = frow["condition"]
        seed      = frow["seed"]
        base_cond = BASELINE_PAIRS.get(fvdm_cond)
        if not base_cond:
            continue

        brow = baseline_rows.get((base_cond, seed))
        if not brow:
            continue

        # Build vectors from both rows
        fvdm_imm  = np.array([float(frow.get(f"mean_v_imm_{l}", 0)) for l in IMM_LABELS])
        fvdm_fut  = np.array([float(frow.get(f"mean_v_fut_{l}", 0)) for l in FUT_LABELS])
        base_imm  = np.array([float(brow.get(f"mean_v_imm_{l}", 0)) for l in IMM_LABELS])
        base_fut  = np.array([float(brow.get(f"mean_v_fut_{l}", 0)) for l in FUT_LABELS])

        cos_imm = cosine_similarity(fvdm_imm, base_imm)
        cos_fut = cosine_similarity(fvdm_fut, base_fut)
        bfs     = (cos_imm + cos_fut) / 2.0

        bfs_rows.append({
            "fvdm_condition":  fvdm_cond,
            "baseline_condition": base_cond,
            "seed":            seed,
            "cosine_imm":      round(cos_imm, 6),
            "cosine_fut":      round(cos_fut, 6),
            "bfs":             round(bfs, 6),
        })

    return bfs_rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_fvdm(args):
    config_path   = args.config
    output_root   = args.output
    num_seeds     = args.seeds
    num_agents    = args.agents
    timesteps     = args.timesteps
    num_cores     = args.cores
    python_alias  = args.python
    force         = args.force
    baseline_dir  = args.baseline_dir

    num_cores = max(1, min(num_cores, os.cpu_count() or 1))

    # Verify bfe_profiles.json exists
    profile_path = "fvdm_vectors/bfe_profiles.json"
    if not os.path.exists(profile_path):
        print(f"[error] {profile_path} not found. Run derive_vectors.py first.")
        sys.exit(1)

    base_cfg = load_base_config(config_path)
    seeds    = generate_seeds(num_seeds)

    sim_dir     = os.path.join(output_root, "sim_logs")
    results_dir = os.path.join(output_root, "results")
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  FVDM Experiment Runner (Conditions 5–8)")
    print(f"{'='*60}")
    print(f"  Seeds:     {num_seeds}   Timesteps: {timesteps}   Agents: {num_agents}")
    print(f"  Cores:     {num_cores}   Profiles:  {profile_path}")
    print(f"  Output:    {output_root}")
    print(f"{'='*60}\n")

    # ── Build run configs ───────────────────────────────────────────────────
    all_runs = []
    for cname, models in CONDITIONS.items():
        cdir = os.path.join(sim_dir, cname)
        os.makedirs(cdir, exist_ok=True)
        for seed in seeds:
            log_path       = os.path.join(cdir, f"{cname}_{seed}.json")
            agent_log_path = os.path.join(cdir, f"{cname}_{seed}_agents.json")
            cfg = make_run_config(base_cfg, seed, models, timesteps, num_agents,
                                  log_path, agent_log_path)
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
    print(f"  Total runs: {total}  |  Queued: {len(pending)}  |  Skipped: {total - len(pending)}\n")

    # ── Run simulations ─────────────────────────────────────────────────────
    session_durations = {}
    if pending:
        t_start = time.time()
        manager = multiprocessing.Manager()
        counter = manager.Value("i", 0)
        lock    = manager.Lock()
        worker_args = [(cfg_path, python_alias, counter, lock, len(pending))
                       for (_, _, cfg_path, _) in pending]
        print(f"  Running {len(pending)} simulations on {num_cores} core(s) …")
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(run_one_simulation, worker_args)
        session_durations = {path: dur for path, dur in results}
        print(f"\n\n  Completed in {time.time() - t_start:.1f}s\n")

    # ── Parse results ───────────────────────────────────────────────────────
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

    # ── Write main CSVs ─────────────────────────────────────────────────────
    write_csv(all_pts,       os.path.join(results_dir, "per_timestep.csv"))
    write_csv(all_summaries, os.path.join(results_dir, "per_seed_summary.csv"))
    write_csv(all_felicific, os.path.join(results_dir, "per_seed_felicific.csv"))

    agg_rows = []
    for cname, summs in cond_summaries.items():
        if summs:
            agg_rows.append(aggregate_seeds(summs))
    write_csv(agg_rows, os.path.join(results_dir, "condition_aggregates.csv"))

    # Per-condition subdirectories
    for cname in CONDITIONS:
        cd = os.path.join(results_dir, cname)
        os.makedirs(cd, exist_ok=True)
        write_csv([r for r in all_pts       if r["condition"] == cname],
                  os.path.join(cd, "per_timestep.csv"))
        write_csv([r for r in all_summaries if r["condition"] == cname],
                  os.path.join(cd, "per_seed_summary.csv"))
        write_csv([r for r in all_felicific if r["condition"] == cname],
                  os.path.join(cd, "per_seed_felicific.csv"))
        write_csv([r for r in agg_rows      if r.get("condition") == cname],
                  os.path.join(cd, "condition_aggregates.csv"))

    # ── Behavioral Fidelity Score ────────────────────────────────────────────
    bfs_rows = compute_bfs(all_felicific, baseline_dir)
    if bfs_rows:
        write_csv(bfs_rows, os.path.join(results_dir, "behavioral_fidelity_scores.csv"))
        print(f"\n  Behavioral Fidelity Scores (BFS):")
        print(f"  {'Pair':<40} {'Seeds':>6} {'mean BFS':>10} {'mean cos_imm':>13} {'mean cos_fut':>13}")
        print(f"  {'-'*82}")
        from itertools import groupby
        for fvdm_cond, group in groupby(bfs_rows, key=lambda r: r["fvdm_condition"]):
            g = list(group)
            mean_bfs  = np.mean([r["bfs"]        for r in g])
            mean_ci   = np.mean([r["cosine_imm"] for r in g])
            mean_cf   = np.mean([r["cosine_fut"] for r in g])
            pair      = f"{BASELINE_PAIRS[fvdm_cond]} ↔ {fvdm_cond}"
            print(f"  {pair:<40} {len(g):>6} {mean_bfs:>10.4f} {mean_ci:>13.4f} {mean_cf:>13.4f}")

    # ── Console summary ─────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {'Condition':<22} {'Seeds':>6} {'Extinct%':>9} {'FinalPop':>10} "
          f"{'Gini':>7} {'TTL':>7}")
    print(f"  {'-'*65}")
    for row in agg_rows:
        cn  = row.get("condition", "?")
        ns  = row.get("numSeeds", 0)
        ext = row.get("extinctionRate", 0) * 100
        fp  = row.get("mean_finalPopulation", 0)
        gi  = row.get("mean_finalGini", 0)
        ttl = row.get("mean_finalMeanTimeToLive", 0)
        print(f"  {cn:<22} {ns:>6} {ext:>8.1f}% {fp:>10.1f} {gi:>7.3f} {ttl:>7.2f}")
    print(f"{'='*72}\n")
    print(f"  Outputs written to {results_dir}/\n")

    # ── Heterogeneous FVDM Egoist-Bentham sweep ──────────────────────────────
    if not args.no_hetero:
        hetero_step = args.hetero_step
        hetero_conds, hetero_base_pairs = make_fvdm_hetero_conditions(hetero_step)

        hetero_sim_dir     = os.path.join(output_root, "hetero_sim_logs")
        hetero_results_dir = os.path.join(output_root, "hetero_results")
        os.makedirs(hetero_sim_dir, exist_ok=True)
        os.makedirs(hetero_results_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  FVDM Heterogeneous Egoist-Bentham Sweep")
        print(f"{'='*60}")
        print(f"  Conditions: {len(hetero_conds)}  (step={hetero_step}%)")
        print(f"{'='*60}\n")

        # Build run configs for hetero conditions
        hetero_all_runs = []
        for cname, models in hetero_conds.items():
            cdir = os.path.join(hetero_sim_dir, cname)
            os.makedirs(cdir, exist_ok=True)
            for seed in seeds:
                log_path       = os.path.join(cdir, f"{cname}_{seed}.json")
                agent_log_path = os.path.join(cdir, f"{cname}_{seed}_agents.json")
                cfg = make_run_config(base_cfg, seed, models, timesteps, num_agents,
                                      log_path, agent_log_path)
                cfg_path = os.path.join(cdir, f"{cname}_{seed}.config")
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f)
                hetero_all_runs.append((cname, seed, cfg_path, log_path, agent_log_path))

        hetero_pending = []
        for (cname, seed, cfg_path, log_path, _) in hetero_all_runs:
            if not force and os.path.exists(log_path):
                d = safe_json_load(log_path)
                if d and len(d) > 0:
                    last_ts  = d[-1].get("timestep", 0)
                    last_pop = d[-1].get("population", -1)
                    if int(last_ts) >= timesteps or int(last_pop) == 0:
                        continue
            hetero_pending.append((cname, seed, cfg_path, log_path))

        h_total = len(hetero_all_runs)
        print(f"  Total runs: {h_total}  |  Queued: {len(hetero_pending)}"
              f"  |  Skipped: {h_total - len(hetero_pending)}\n")

        # Run simulations
        hetero_durations = {}
        if hetero_pending:
            t_start = time.time()
            manager = multiprocessing.Manager()
            counter = manager.Value("i", 0)
            lock    = manager.Lock()
            worker_args = [(cfg_path, python_alias, counter, lock, len(hetero_pending))
                           for (_, _, cfg_path, _) in hetero_pending]
            print(f"  Running {len(hetero_pending)} simulations on {num_cores} core(s) …")
            with multiprocessing.Pool(processes=num_cores) as pool:
                h_results = pool.map(run_one_simulation, worker_args)
            hetero_durations = {path: dur for path, dur in h_results}
            print(f"\n\n  Completed in {time.time() - t_start:.1f}s\n")

        # Parse results
        print("  Parsing hetero logs …")
        h_pts        = []
        h_summaries  = []
        h_felicific  = []
        h_cond_summs = {c: [] for c in hetero_conds}

        for (cname, seed, cfg_path, log_path, agent_log_path) in hetero_all_runs:
            dur = hetero_durations.get(cfg_path, 0.0)
            pts, summary, felicific = parse_sim_log(log_path, cname, seed,
                                                     agent_log_path, dur)
            if pts is None:
                print(f"  [warn] Could not parse {cname} seed={seed}")
                continue
            h_pts.extend(pts)
            h_summaries.append(summary)
            h_cond_summs[cname].append(summary)
            if felicific:
                h_felicific.append(felicific)

        # Write hetero CSVs
        write_csv(h_pts,       os.path.join(hetero_results_dir, "per_timestep.csv"))
        write_csv(h_summaries, os.path.join(hetero_results_dir, "per_seed_summary.csv"))
        write_csv(h_felicific, os.path.join(hetero_results_dir, "per_seed_felicific.csv"))

        h_agg_rows = []
        for cname, summs in h_cond_summs.items():
            if summs:
                h_agg_rows.append(aggregate_seeds(summs))
        write_csv(h_agg_rows, os.path.join(hetero_results_dir, "condition_aggregates.csv"))

        # ── Hetero BFS (against baseline hetero felicific) ───────────────────
        baseline_hetero_path = os.path.join(
            baseline_dir, "hetero_results", "per_seed_felicific.csv"
        )
        hetero_bfs_rows = []
        if not os.path.exists(baseline_hetero_path):
            print(f"  [warn] Baseline hetero felicific CSV not found: {baseline_hetero_path}")
            print(f"         Run run_experiments_baseline.py hetero sweep first for BFS computation.")
        else:
            import csv as csv_mod
            base_hetero_rows = {}
            with open(baseline_hetero_path, newline="") as f:
                for row in csv_mod.DictReader(f):
                    key = (row["condition"], int(float(row["seed"])))
                    base_hetero_rows[key] = row

            for frow in h_felicific:
                fvdm_cond = frow["condition"]
                seed      = frow["seed"]
                base_cond = hetero_base_pairs.get(fvdm_cond)
                if not base_cond:
                    continue
                brow = base_hetero_rows.get((base_cond, seed))
                if not brow:
                    continue

                fvdm_imm = np.array([float(frow.get(f"mean_v_imm_{l}", 0)) for l in IMM_LABELS])
                fvdm_fut = np.array([float(frow.get(f"mean_v_fut_{l}", 0)) for l in FUT_LABELS])
                base_imm = np.array([float(brow.get(f"mean_v_imm_{l}", 0)) for l in IMM_LABELS])
                base_fut = np.array([float(brow.get(f"mean_v_fut_{l}", 0)) for l in FUT_LABELS])

                cos_imm = cosine_similarity(fvdm_imm, base_imm)
                cos_fut = cosine_similarity(fvdm_fut, base_fut)
                bfs     = (cos_imm + cos_fut) / 2.0

                # Extract pct_bentham from condition name (fvdmHeteroMix_p###)
                pct_bentham = int(fvdm_cond.split("_p")[-1])

                hetero_bfs_rows.append({
                    "fvdm_condition":      fvdm_cond,
                    "baseline_condition":  base_cond,
                    "pct_bentham":         pct_bentham,
                    "seed":                seed,
                    "cosine_imm":          round(cos_imm, 6),
                    "cosine_fut":          round(cos_fut, 6),
                    "bfs":                 round(bfs, 6),
                })

            write_csv(hetero_bfs_rows, os.path.join(hetero_results_dir, "hetero_bfs.csv"))

        # ── Hetero Spearman correlations ─────────────────────────────────────
        # Metrics to correlate against pct_bentham
        spearman_metrics = [
            ("finalPopulation",    "mean_finalPopulation"),
            ("finalGini",          "mean_finalGini"),
            ("finalMeanTimeToLive","mean_finalMeanTimeToLive"),
            ("finalSocietalWealth","mean_finalSocietalWealth"),
        ]

        def spearman_r(x, y):
            """Compute Spearman rank correlation between two lists."""
            n = len(x)
            if n < 2:
                return float("nan")
            def ranks(v):
                sv = sorted(range(n), key=lambda i: v[i])
                r  = [0.0] * n
                i  = 0
                while i < n:
                    j = i
                    while j < n - 1 and v[sv[j + 1]] == v[sv[j]]:
                        j += 1
                    avg_rank = (i + j) / 2.0 + 1
                    for k in range(i, j + 1):
                        r[sv[k]] = avg_rank
                    i = j + 1
                return r
            rx, ry = ranks(x), ranks(y)
            mu_rx = sum(rx) / n
            mu_ry = sum(ry) / n
            num  = sum((rx[i] - mu_rx) * (ry[i] - mu_ry) for i in range(n))
            den  = math.sqrt(
                sum((rx[i] - mu_rx) ** 2 for i in range(n)) *
                sum((ry[i] - mu_ry) ** 2 for i in range(n))
            )
            return num / den if den > 1e-10 else float("nan")

        # Aggregate per (pct_bentham) for FVDM outcomes
        pct_levels = sorted({int(c.split("_p")[-1]) for c in hetero_conds})

        fvdm_spearman_rows = []
        for label, agg_key in spearman_metrics:
            pct_vals    = []
            metric_vals = []
            for row in h_agg_rows:
                cname = row.get("condition", "")
                if not cname.startswith("fvdmHeteroMix_p"):
                    continue
                pct = int(cname.split("_p")[-1])
                pct_vals.append(pct)
                metric_vals.append(row.get(agg_key, 0))
            rho = spearman_r(pct_vals, metric_vals)
            fvdm_spearman_rows.append({
                "metric":  label,
                "context": "fvdm_hetero",
                "rho":     round(rho, 6) if not math.isnan(rho) else "",
                "n":       len(pct_vals),
            })

        write_csv(fvdm_spearman_rows,
                  os.path.join(hetero_results_dir, "hetero_spearman.csv"))

        # ── Per-proportion BFS aggregation ───────────────────────────────────
        pct_bfs_summary = {}
        for row in hetero_bfs_rows:
            p = row["pct_bentham"]
            pct_bfs_summary.setdefault(p, []).append(row["bfs"])

        # ── Console summary table ─────────────────────────────────────────────
        # Build a lookup for fvdm spearman rho by metric
        fvdm_rho_by_metric = {r["metric"]: r["rho"] for r in fvdm_spearman_rows}

        print(f"\n  Heterogeneous FVDM sweep — per-proportion summary:")
        print(f"  {'Pct':>5}  {'FVDM Condition':<26}  {'Seeds':>6}  "
              f"{'FinalPop':>10}  {'Gini':>7}  {'TTL':>7}  {'mean BFS':>10}")
        print(f"  {'-'*80}")

        for pct in pct_levels:
            fname = f"fvdmHeteroMix_p{pct:03d}"
            arow  = next((r for r in h_agg_rows if r.get("condition") == fname), {})
            ns    = arow.get("numSeeds", 0)
            fp    = arow.get("mean_finalPopulation", 0)
            gi    = arow.get("mean_finalGini", 0)
            ttl   = arow.get("mean_finalMeanTimeToLive", 0)
            bfs_vals = pct_bfs_summary.get(pct, [])
            mean_bfs = np.mean(bfs_vals) if bfs_vals else float("nan")
            bfs_str  = f"{mean_bfs:.4f}" if not math.isnan(mean_bfs) else "   n/a"
            print(f"  {pct:>5}  {fname:<26}  {ns:>6}  "
                  f"{fp:>10.1f}  {gi:>7.3f}  {ttl:>7.2f}  {bfs_str:>10}")

        print(f"  {'-'*80}")
        print(f"\n  Spearman rho (pct_bentham vs metric):")
        for row in fvdm_spearman_rows:
            rho_str = f"{row['rho']}" if row["rho"] != "" else "n/a"
            print(f"    {row['metric']:<26}  fvdm rho={rho_str}")

        print(f"\n  Hetero outputs written to {hetero_results_dir}/\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FVDM-derived condition runner (conditions 5–8)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c", "--config",       default="config.json")
    p.add_argument("-o", "--output",       default="fvdm_results")
    p.add_argument("-b", "--baseline-dir", default="baseline_results",
                   help="Root directory of run_experiments_baseline.py output "
                        "(used for BFS computation).")
    p.add_argument("-s", "--seeds",     type=int, default=30)
    p.add_argument("-a", "--agents",    type=int, default=250)
    p.add_argument("-t", "--timesteps", type=int, default=5000)
    p.add_argument("-j", "--cores",     type=int, default=1)
    p.add_argument("--python",          default="python3")
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--hetero-step", type=int, default=HETERO_STEP,
                   help="Percentage-point step for the heterogeneous sweep (default: 10).")
    p.add_argument("--no-hetero", action="store_true", default=False,
                   help="Skip the heterogeneous Egoist-Bentham sweep.")
    return p.parse_args()


if __name__ == "__main__":
    run_fvdm(parse_args())
