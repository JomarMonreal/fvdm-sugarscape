#!/usr/bin/env python3
"""
run_experiments_baseline.py
----------------------------
Runs the four homogeneous baseline conditions (conditions 1-4):
  1. rawSugarscape
  2. egoist
  3. altruist
  4. bentham

Then (unless --no-hetero is given) runs the heterogeneous Egoist–Bentham sweep
following Herman & Kremer (2024): Bentham proportion swept from 0% to 100% in
HETERO_STEP (default 10) percentage-point increments, giving 11 conditions.

Parameters (paper defaults):
  250 starting agents, 5 000 timesteps, 30 matched seeds.

Agent logging is enabled so that per-agent chosen-cell data (bfe_* fields)
can be post-processed to compute mean felicific effect vectors per run:
  v_bar_imm_g  and  v_bar_fut_g  per condition per seed.

Outputs (under --output/results/):
  per_timestep.csv       — timestep-level population metrics per seed
  per_seed_summary.csv   — one row per (condition, seed) with all final metrics
  per_seed_felicific.csv — mean v_imm and v_fut per (condition, seed) for BFS
  condition_aggregates.csv

Heterogeneous outputs (under --output/hetero_results/):
  per_timestep.csv       — same schema, all 11 hetero conditions
  per_seed_summary.csv
  per_seed_felicific.csv
  condition_aggregates.csv
  hetero_spearman.csv    — Spearman r between % Bentham and each outcome metric

Usage:
  python run_experiments_baseline.py [options]
  python run_experiments_baseline.py --seeds 30 --cores 8
  python run_experiments_baseline.py --no-hetero          # skip hetero sweep
  python run_experiments_baseline.py --hetero-step 5      # finer sweep
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
# Condition definitions (baseline only)
# ─────────────────────────────────────────────────────────────────────────────

CONDITIONS = {
    "rawSugarscape": ["rawSugarscape"],
    "egoist":        ["egoist"],
    "altruist":      ["altruist"],
    "bentham":       ["bentham"],
}

HETERO_STEP = 10  # percentage-point increment for heterogeneous sweep

# Coordinate labels
IMM_LABELS = ["I", "D", "C", "P", "E"]
FUT_LABELS = ["I", "D", "C", "P", "E"]

GAMMA_DEFAULT = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Spearman correlation (scipy if available, otherwise rank-based fallback)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from scipy.stats import spearmanr as _scipy_spearmanr
    def spearman_r(xs, ys):
        r, p = _scipy_spearmanr(xs, ys)
        return round(float(r), 4), round(float(p), 4)
except ImportError:
    def spearman_r(xs, ys):
        n = len(xs)
        rx = [sorted(xs).index(x) + 1 for x in xs]
        ry = [sorted(ys).index(y) + 1 for y in ys]
        d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
        r = 1 - 6 * d2 / (n * (n ** 2 - 1))
        return round(r, 4), None

# ─────────────────────────────────────────────────────────────────────────────
# Heterogeneous Egoist–Bentham sweep helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_models_list(pct_bentham: int, total_agents: int = 250) -> list:
    """Return the minimal repeating round-robin agent-type list for a given
    Bentham percentage (0–100), mixing Egoist and Bentham agents."""
    if pct_bentham == 0:
        return ["egoist"]
    if pct_bentham == 100:
        return ["bentham"]
    from math import gcd
    n_bentham = round(pct_bentham / 100 * total_agents)
    n_egoist  = total_agents - n_bentham
    g = gcd(n_bentham, n_egoist)
    return ["egoist"] * (n_egoist // g) + ["bentham"] * (n_bentham // g)


def make_hetero_conditions(step: int = HETERO_STEP) -> dict:
    """Build a dict mapping heterogeneous condition name to models list."""
    conds = {}
    for pct in range(0, 101, step):
        name = f"heteroMix_p{pct:03d}"
        conds[name] = make_models_list(pct)
    return conds

# ─────────────────────────────────────────────────────────────────────────────
# Shared seed generation (must match run_experiments_fvdm.py)
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
                    log_path: str, agent_log_path: str = "") -> dict:
    cfg = dict(base)
    cfg["seed"]               = seed
    cfg["agentDecisionModels"] = decision_models
    cfg["timesteps"]          = timesteps
    cfg["startingAgents"]     = num_agents
    cfg["startingDiseases"]   = 0
    cfg["headlessMode"]       = True
    cfg["debugMode"]          = ["none"]
    cfg["keepAlivePostExtinction"] = False
    cfg["keepAliveAtEnd"]     = False
    cfg["screenshots"]        = False
    cfg["profileMode"]        = False
    cfg["logfile"]            = log_path
    cfg["agentLogfile"]       = agent_log_path
    cfg["logfileFormat"]      = "json"
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
# Analytical felicific coordinate computation (mirrors FVDMAgent formulas)
# ─────────────────────────────────────────────────────────────────────────────

def compute_v_imm(row: dict) -> np.ndarray:
    ttl  = max(0.0, float(row.get("timeToLive", 0)))
    poll = max(0.0, float(row.get("bfe_pollution", 0)))
    m    = max(1.0, float(row.get("sugarMetabolism", 1)) + float(row.get("spiceMetabolism", 1)))
    w_c     = max(0.0, float(row.get("bfe_w_c", 0)))
    w_c_max = max(1.0, float(row.get("bfe_w_c_max", 1)))
    v       = max(1,   int(row.get("bfe_cells_in_range", 1)))

    I = 1.0 / ((1.0 + ttl) * (1.0 + poll))
    D = min(1.0, w_c / (m * w_c_max))
    return np.array([I, D, 1.0, 1.0, 1.0 / v])


def compute_v_fut(row: dict, gamma: float = GAMMA_DEFAULT) -> np.ndarray:
    m       = max(1.0, float(row.get("sugarMetabolism", 1)) + float(row.get("spiceMetabolism", 1)))
    w_c     = max(0.0, float(row.get("bfe_w_c", 0)))
    w_c_max = max(1.0, float(row.get("bfe_w_c_max", 1)))
    v       = max(1,   int(row.get("bfe_cells_in_range", 1)))
    w_adj      = max(0.0, float(row.get("bfe_adj_wealth", 0)))
    n_adj      = max(1,   int(row.get("bfe_num_adj", 1)))
    w_glob_max = max(1.0, float(row.get("globalMaxWealth", 1)))

    J  = w_adj / (w_glob_max * n_adj)
    Df = max(0.0, w_c - m) / (m * w_c_max)
    return np.array([J, Df, 1.0, gamma, 1.0 / v])


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
                  agent_log_path: str, gamma: float, duration: float = 0.0):
    """
    Parse one simulation run.  Returns:
      per_timestep : list[dict]  — population metrics per timestep
      summary      : dict        — final/aggregated stats
      felicific    : dict        — mean v_imm and v_fut across all agent-timesteps
    """
    sim_log = safe_json_load(log_path)
    if not sim_log or len(sim_log) == 0:
        return None, None, None

    # ── Per-timestep population metrics ──
    per_timestep = []
    total_deaths = 0
    total_born = 0
    final_pop = 0

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

    # ── Felicific profiles from agent log ──
    felicific = {"condition": condition, "seed": seed}
    agent_log = safe_json_load(agent_log_path)

    if agent_log:
        imm_vecs = []
        fut_vecs = []
        for row in agent_log:
            if not row.get("bfe_w_c_max"):
                continue   # skip rows without BFE data (e.g. first timestep)
            v_imm = compute_v_imm(row)
            v_fut = compute_v_fut(row, gamma=gamma)
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

    # ── Summary ──
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
        "condition":     summaries[0]["condition"],
        "numSeeds":      len(summaries),
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(args):
    config_path   = args.config
    output_root   = args.output
    num_seeds     = args.seeds
    num_agents    = args.agents
    timesteps     = args.timesteps
    num_cores     = args.cores
    python_alias  = args.python
    force         = args.force
    gamma         = args.gamma

    num_cores = max(1, min(num_cores, os.cpu_count() or 1))

    base_cfg = load_base_config(config_path)
    seeds    = generate_seeds(num_seeds)

    sim_dir     = os.path.join(output_root, "sim_logs")
    results_dir = os.path.join(output_root, "results")
    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Baseline Experiment Runner (Conditions 1–4)")
    print(f"{'='*60}")
    print(f"  Seeds:     {num_seeds}   Timesteps: {timesteps}   Agents: {num_agents}")
    print(f"  Cores:     {num_cores}   Gamma:     {gamma}")
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
    all_pts       = []
    all_summaries = []
    all_felicific = []
    cond_summaries = {c: [] for c in CONDITIONS}

    for (cname, seed, cfg_path, log_path, agent_log_path) in all_runs:
        dur = session_durations.get(cfg_path, 0.0)
        pts, summary, felicific = parse_sim_log(log_path, cname, seed,
                                                 agent_log_path, gamma, dur)
        if pts is None:
            print(f"  [warn] Could not parse {cname} seed={seed}")
            continue
        all_pts.extend(pts)
        all_summaries.append(summary)
        cond_summaries[cname].append(summary)
        if felicific:
            all_felicific.append(felicific)

    # ── Write CSVs ──────────────────────────────────────────────────────────
    write_csv(all_pts,       os.path.join(results_dir, "per_timestep.csv"))
    write_csv(all_summaries, os.path.join(results_dir, "per_seed_summary.csv"))
    write_csv(all_felicific, os.path.join(results_dir, "per_seed_felicific.csv"))

    if args.delete_logs:
        for (_, _, cfg_path, log_path, agent_log_path) in all_runs:
            for p in (log_path, agent_log_path, cfg_path):
                if p and os.path.exists(p):
                    os.remove(p)

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

    # ── Heterogeneous Egoist–Bentham sweep ──────────────────────────────────
    hetero_spearman_rows = []
    if not args.no_hetero:
        HETERO_CONDITIONS = make_hetero_conditions(step=args.hetero_step)
        hetero_results_dir = os.path.join(output_root, "hetero_results")
        hetero_sim_dir     = os.path.join(output_root, "hetero_sim_logs")
        os.makedirs(hetero_results_dir, exist_ok=True)
        os.makedirs(hetero_sim_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Heterogeneous Egoist–Bentham Sweep")
        print(f"  Step: {args.hetero_step}%  Conditions: {len(HETERO_CONDITIONS)}")
        print(f"{'='*60}\n")

        # Build all hetero run configs (no agent logs — not used for hetero sweep)
        all_runs_hetero = []
        for cname, models in HETERO_CONDITIONS.items():
            cdir = os.path.join(hetero_sim_dir, cname)
            os.makedirs(cdir, exist_ok=True)
            for seed in seeds:
                log_path = os.path.join(cdir, f"{cname}_{seed}.json")
                cfg = make_run_config(base_cfg, seed, models, timesteps, num_agents,
                                      log_path)
                cfg_path = os.path.join(cdir, f"{cname}_{seed}.config")
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f)
                all_runs_hetero.append((cname, seed, cfg_path, log_path, ""))

        pending_hetero = []
        for (cname, seed, cfg_path, log_path, _) in all_runs_hetero:
            if not force and os.path.exists(log_path):
                d = safe_json_load(log_path)
                if d and len(d) > 0:
                    last_ts  = d[-1].get("timestep", 0)
                    last_pop = d[-1].get("population", -1)
                    if int(last_ts) >= timesteps or int(last_pop) == 0:
                        continue
            pending_hetero.append((cname, seed, cfg_path, log_path))

        total_hetero = len(all_runs_hetero)
        print(f"  Total hetero runs: {total_hetero}  |  Queued: {len(pending_hetero)}"
              f"  |  Skipped: {total_hetero - len(pending_hetero)}\n")

        hetero_session_durations = {}
        if pending_hetero:
            t_start = time.time()
            manager = multiprocessing.Manager()
            counter = manager.Value("i", 0)
            lock    = manager.Lock()
            worker_args = [(cfg_path, python_alias, counter, lock, len(pending_hetero))
                           for (_, _, cfg_path, _) in pending_hetero]
            print(f"  Running {len(pending_hetero)} hetero simulations on {num_cores} core(s) …")
            with multiprocessing.Pool(processes=num_cores) as pool:
                hetero_results_raw = pool.map(run_one_simulation, worker_args)
            hetero_session_durations = {path: dur for path, dur in hetero_results_raw}
            print(f"\n\n  Completed in {time.time() - t_start:.1f}s\n")

        # Parse hetero results
        print("  Parsing hetero logs …")
        h_pts       = []
        h_summaries = []
        h_felicific = []
        h_cond_summaries = {c: [] for c in HETERO_CONDITIONS}

        for (cname, seed, cfg_path, log_path, agent_log_path) in all_runs_hetero:
            dur = hetero_session_durations.get(cfg_path, 0.0)
            pts, summary, felicific = parse_sim_log(log_path, cname, seed,
                                                     agent_log_path, gamma, dur)
            if pts is None:
                print(f"  [warn] Could not parse {cname} seed={seed}")
                continue
            h_pts.extend(pts)
            h_summaries.append(summary)
            h_cond_summaries[cname].append(summary)
            if felicific:
                h_felicific.append(felicific)

        # Write hetero CSVs
        write_csv(h_pts,       os.path.join(hetero_results_dir, "per_timestep.csv"))
        write_csv(h_summaries, os.path.join(hetero_results_dir, "per_seed_summary.csv"))
        write_csv(h_felicific, os.path.join(hetero_results_dir, "per_seed_felicific.csv"))

        h_agg_rows = []
        for cname, summs in h_cond_summaries.items():
            if summs:
                h_agg_rows.append(aggregate_seeds(summs))
        write_csv(h_agg_rows, os.path.join(hetero_results_dir, "condition_aggregates.csv"))

        # Compute Spearman correlations across proportion levels
        # x-axis: pct_bentham (0, 10, 20, ..., 100)
        pct_levels = list(range(0, 101, args.hetero_step))
        outcome_metrics = [
            "mean_finalPopulation",
            "mean_finalGini",
            "mean_finalMeanTimeToLive",
            "mean_finalMeanWealth",
            "mean_finalSocietalWealth",
            "mean_totalCombatActions",
            "mean_totalTradeActions",
            "mean_totalReproductionActions",
            "extinctionRate",
        ]
        # Build lookup: condition_name -> agg_row
        agg_lookup = {r["condition"]: r for r in h_agg_rows}

        for metric in outcome_metrics:
            xs = []
            ys = []
            for pct in pct_levels:
                cname = f"heteroMix_p{pct:03d}"
                row   = agg_lookup.get(cname)
                if row and metric in row:
                    xs.append(pct)
                    ys.append(row[metric])
            if len(xs) >= 3:
                r_val, p_val = spearman_r(xs, ys)
                hetero_spearman_rows.append({
                    "metric":         metric,
                    "spearman_r":     r_val,
                    "p_value":        p_val if p_val is not None else "",
                    "n_proportions":  len(xs),
                })

        write_csv(hetero_spearman_rows,
                  os.path.join(hetero_results_dir, "hetero_spearman.csv"))

        if args.delete_logs:
            for (_, _, cfg_path, log_path, _agent) in all_runs_hetero:
                for p in (log_path, cfg_path):
                    if p and os.path.exists(p):
                        os.remove(p)

        print(f"  Hetero outputs written to {hetero_results_dir}/\n")

    # ── Console summary ─────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {'Condition':<16} {'Seeds':>6} {'Extinct%':>9} {'FinalPop':>10} "
          f"{'Gini':>7} {'TTL':>7} {'mu_imm_I':>10} {'mu_imm_D':>10}")
    print(f"  {'-'*70}")
    for row in agg_rows:
        cn  = row.get("condition", "?")
        ns  = row.get("numSeeds", 0)
        ext = row.get("extinctionRate", 0) * 100
        fp  = row.get("mean_finalPopulation", 0)
        gi  = row.get("mean_finalGini", 0)
        ttl = row.get("mean_finalMeanTimeToLive", 0)
        # Get mu_imm from felicific rows
        fc  = [r for r in all_felicific if r["condition"] == cn]
        mi_I = np.mean([r.get("mean_v_imm_I", 0) for r in fc]) if fc else 0
        mi_D = np.mean([r.get("mean_v_imm_D", 0) for r in fc]) if fc else 0
        print(f"  {cn:<16} {ns:>6} {ext:>8.1f}% {fp:>10.1f} "
              f"{gi:>7.3f} {ttl:>7.2f} {mi_I:>10.4f} {mi_D:>10.4f}")
    print(f"{'='*72}\n")
    print(f"  Outputs written to {results_dir}/\n")

    # ── Heterogeneous Spearman summary ──────────────────────────────────────
    if not args.no_hetero and hetero_spearman_rows:
        print(f"{'='*72}")
        print(f"  Spearman Correlations: % Bentham vs Outcome Metrics")
        print(f"  {'Metric':<36} {'r':>8} {'p':>10} {'n':>4}")
        print(f"  {'-'*60}")
        for row in hetero_spearman_rows:
            p_str = f"{row['p_value']:.4f}" if row["p_value"] != "" else "  N/A "
            print(f"  {row['metric']:<36} {row['spearman_r']:>8.4f} {p_str:>10} "
                  f"{row['n_proportions']:>4}")
        print(f"{'='*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Baseline condition runner (conditions 1–4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c", "--config",    default="config.json")
    p.add_argument("-o", "--output",    default="baseline_results")
    p.add_argument("-s", "--seeds",     type=int, default=30)
    p.add_argument("-a", "--agents",    type=int, default=250)
    p.add_argument("-t", "--timesteps", type=int, default=5000)
    p.add_argument("-j", "--cores",     type=int, default=1)
    p.add_argument("-g", "--gamma",     type=float, default=0.5)
    p.add_argument("--python",          default="python3")
    p.add_argument("--force",       action="store_true", default=False)
    p.add_argument("--delete-logs", action="store_true", default=False,
                   help="Delete raw JSON sim logs after parsing (saves disk space)")
    p.add_argument("--hetero-step", type=int, default=10,
                   help="Percentage-point increment for Egoist--Bentham sweep")
    p.add_argument("--no-hetero",  action="store_true", default=False,
                   help="Skip the heterogeneous Egoist--Bentham sweep")
    return p.parse_args()


if __name__ == "__main__":
    run_baseline(parse_args())
