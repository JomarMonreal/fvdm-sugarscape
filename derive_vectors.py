#!/usr/bin/env python3
"""
derive_vectors.py
-----------------
Prioritization Profile Derivation via Behavioral Feature Expectation (BFE).

Runs a mixed-population derivation simulation containing all four baseline
agent types simultaneously.  At each decision step where at least one
neighbor agent is present (neighbors > 0), the chosen cell's raw properties
are read from the agent log and the two felicific effect vectors are computed
analytically:

  v_imm(c) = ( 1/((1+TTL)(1+pollution)),   W_c/(m*W_cmax),      1, 1,     1/|V| )
  v_fut(c)  = ( W_adj/(Wgmax*n_adj),         max(0,W_c-m)/(m*W_cmax), 1, gamma, 1/|V| )

Both vectors are averaged separately per agent type across all qualifying
observations, yielding one prioritization profile (mu_imm, mu_fut) per type.

NOTE: sugarscape.py maps the "rawSugarscape" decision model to "none" internally
(sugarscape.py line ~668).  This script remaps it back when reading logs.

Variance analysis:
  Per-seed mu vectors are computed first.  Their variance across seeds drives a
  sample-size estimate: n_seeds >= ceil(sigma^2 / epsilon^2) per component.
  The script reports the required number of seeds and ensures they are run before
  computing the final pooled profiles.

Output: fvdm_vectors/bfe_profiles.json

Usage:
  python derive_vectors.py [options]
  python derive_vectors.py --seeds 10 --timesteps 5000 --agents 250 --cores 8
  python derive_vectors.py --target-se 0.005   # tighter convergence
"""

import argparse
import json
import multiprocessing
import os
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

AGENT_TYPES = ["rawSugarscape", "egoist", "altruist", "bentham"]

# sugarscape.py translates "rawSugarscape" -> "none" before storing in the agent.
# This map lets us look up the correct log value for each agent type.
DECISIONMODEL_IN_LOG = {
    "rawSugarscape": "none",
    "egoist":        "egoist",
    "altruist":      "altruist",
    "bentham":       "bentham",
}

# gamma fallback when decisionModelLookaheadDiscount is not in the agent log
GAMMA_DEFAULT = 0.5

# Default target standard error for sample-size estimation (per vector component).
# Components are in [0, 1], so 0.005 ≈ 0.5% precision.
DEFAULT_TARGET_SE = 0.02

COORD_LABELS = ["I", "D", "C", "P", "E"]

# Only the columns needed to compute v_imm / v_fut.  Everything else is
# discarded immediately on load so raw JSON logs can be deleted right after
# per-seed stats are extracted, keeping peak disk usage to one batch at a time.
BFE_COLUMNS = frozenset([
    "decisionModel", "neighbors",
    "timeToLive", "bfe_pollution",
    "sugarMetabolism", "spiceMetabolism",
    "bfe_w_c", "bfe_w_c_max", "bfe_cells_in_range",
    "bfe_adj_wealth", "bfe_num_adj", "globalMaxWealth",
])


# ─────────────────────────────────────────────────────────────────────────────
# Analytical felicific coordinate computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_v_imm(row: dict) -> np.ndarray:
    """Compute v_imm(c) = (I, D, C=1, P=1, E=1/|V|) from a logged agent row."""
    ttl  = max(0.0, float(row.get("timeToLive", 0)))
    poll = max(0.0, float(row.get("bfe_pollution", 0)))
    m    = max(1.0, float(row.get("sugarMetabolism", 1)) + float(row.get("spiceMetabolism", 1)))
    w_c      = max(0.0, float(row.get("bfe_w_c", 0)))
    w_c_max  = max(1.0, float(row.get("bfe_w_c_max", 1)))
    v        = max(1,   int(row.get("bfe_cells_in_range", 1)))

    I = 1.0 / ((1.0 + ttl) * (1.0 + poll))
    D = min(1.0, w_c / (m * w_c_max))
    C = 1.0
    P = 1.0
    E = 1.0 / v
    return np.array([I, D, C, P, E])


def compute_v_fut(row: dict, gamma: float = GAMMA_DEFAULT) -> np.ndarray:
    """Compute v_fut(c) = (J, Df, C=1, P=gamma, E=1/|V|) from a logged agent row."""
    m       = max(1.0, float(row.get("sugarMetabolism", 1)) + float(row.get("spiceMetabolism", 1)))
    w_c     = max(0.0, float(row.get("bfe_w_c", 0)))
    w_c_max = max(1.0, float(row.get("bfe_w_c_max", 1)))
    v       = max(1,   int(row.get("bfe_cells_in_range", 1)))

    # W_adj / (W_globalmax * n_adj) — normalised adjacent cell wealth
    w_adj      = max(0.0, float(row.get("bfe_adj_wealth", 0)))
    n_adj      = max(1,   int(row.get("bfe_num_adj", 1)))
    w_glob_max = max(1.0, float(row.get("globalMaxWealth", 1)))  # written by sim if available

    J  = w_adj / (w_glob_max * n_adj)
    Df = max(0.0, w_c - m) / (m * w_c_max)
    C  = 1.0
    P  = gamma
    E  = 1.0 / v
    return np.array([J, Df, C, P, E])


# ─────────────────────────────────────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_base_config(path: str) -> dict:
    with open(path) as f:
        full = json.load(f)
    return full.get("sugarscapeOptions", full)


def run_one_sim(args):
    cfg_path, python_alias, counter, lock, total = args
    subprocess.run([python_alias, "sugarscape.py", "--conf", cfg_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with lock:
        counter.value += 1
        pct = counter.value / total * 100
        print(f"\r  [{counter.value:>3}/{total}] {pct:5.1f}%", end="", flush=True)


def safe_json_load(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# BFE derivation from agent log DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def agent_log_to_df(path: str, seed_tag=None, slim: bool = True) -> pd.DataFrame:
    """Parse a raw agent JSON log into a flat DataFrame.

    slim:     if True (default) keep only BFE_COLUMNS, discarding everything
              else immediately to minimise memory.  The raw JSON file can then
              be deleted right after this call.
    seed_tag: if provided, adds a 'seed' column for per-seed variance.
    """
    data = safe_json_load(path)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if slim:
        keep = [c for c in df.columns if c in BFE_COLUMNS]
        df = df[keep]
    if seed_tag is not None:
        df["seed"] = seed_tag
    return df


def _filter_has_neighbor(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where at least one neighbor was present (neighbors > 0)."""
    if "neighbors" not in df.columns:
        return df   # can't filter; return everything
    return df[df["neighbors"] > 0]


def _rows_for_type(df: pd.DataFrame, atype: str) -> pd.DataFrame:
    """Return rows belonging to atype, accounting for rawSugarscape -> 'none'."""
    log_model = DECISIONMODEL_IN_LOG.get(atype, atype)
    return df[df["decisionModel"] == log_model].copy()


def _vecs_for_rows(rows: pd.DataFrame, gamma: float):
    """Compute (imm_vecs, fut_vecs) arrays from a filtered row DataFrame."""
    imm, fut = [], []
    for row_dict in rows.to_dict(orient="records"):
        imm.append(compute_v_imm(row_dict))
        fut.append(compute_v_fut(row_dict, gamma=gamma))
    return np.array(imm), np.array(fut)


# ── Variance analysis ─────────────────────────────────────────────────────────

def compute_per_seed_stats(df: pd.DataFrame, gamma: float = GAMMA_DEFAULT
                           ) -> dict:
    """
    Compute per-seed mu_imm and mu_fut for each agent type.

    Returns:
        dict atype -> {"seed_mus_imm": ndarray(n_seeds, 5),
                       "seed_mus_fut": ndarray(n_seeds, 5),
                       "seed_n_obs":   list[int]}
    """
    if "seed" not in df.columns:
        print("  [warn] No 'seed' column — cannot compute per-seed variance.")
        return {}

    seeds = sorted(df["seed"].unique())
    result = {atype: {"seed_mus_imm": [], "seed_mus_fut": [], "seed_n_obs": []}
              for atype in AGENT_TYPES}

    for seed in seeds:
        seed_df = df[df["seed"] == seed]
        for atype in AGENT_TYPES:
            rows = _filter_has_neighbor(_rows_for_type(seed_df, atype))
            if len(rows) == 0:
                continue
            imm_vecs, fut_vecs = _vecs_for_rows(rows, gamma)
            result[atype]["seed_mus_imm"].append(imm_vecs.mean(axis=0))
            result[atype]["seed_mus_fut"].append(fut_vecs.mean(axis=0))
            result[atype]["seed_n_obs"].append(len(rows))

    for atype in AGENT_TYPES:
        r = result[atype]
        if r["seed_mus_imm"]:
            r["seed_mus_imm"] = np.array(r["seed_mus_imm"])
            r["seed_mus_fut"] = np.array(r["seed_mus_fut"])
        else:
            r["seed_mus_imm"] = np.empty((0, 5))
            r["seed_mus_fut"] = np.empty((0, 5))

    return result


def estimate_required_seeds(per_seed_stats: dict,
                             target_se: float = DEFAULT_TARGET_SE,
                             z: float = 1.96) -> dict:
    """
    For each agent type, estimate how many seeds are needed so that the
    standard error of every vector component is <= target_se.

    Uses: n_required = ceil((sigma / target_se)^2)

    Returns:
        dict atype -> {"n_current": int,
                       "n_required": int,
                       "sigma_imm": ndarray(5),
                       "sigma_fut": ndarray(5),
                       "se_imm": ndarray(5),
                       "se_fut": ndarray(5)}
    """
    report = {}
    for atype, r in per_seed_stats.items():
        mus_imm = r["seed_mus_imm"]
        mus_fut = r["seed_mus_fut"]
        n = len(mus_imm)
        if n < 2:
            report[atype] = {"n_current": n, "n_required": None,
                             "note": "need >= 2 seeds for variance estimate"}
            continue

        sigma_imm = mus_imm.std(axis=0, ddof=1)
        sigma_fut = mus_fut.std(axis=0, ddof=1)
        se_imm    = sigma_imm / np.sqrt(n)
        se_fut    = sigma_fut / np.sqrt(n)

        # worst-case component (largest sigma)
        n_req_imm = np.ceil((sigma_imm / target_se) ** 2).astype(int)
        n_req_fut = np.ceil((sigma_fut / target_se) ** 2).astype(int)
        n_required = int(max(n_req_imm.max(), n_req_fut.max()))

        mean_obs_per_seed = (np.mean(r["seed_n_obs"])
                             if r["seed_n_obs"] else 0)
        obs_required = int(np.ceil(n_required * mean_obs_per_seed))

        report[atype] = {
            "n_current":          n,
            "n_required":         n_required,
            "obs_required":       obs_required,
            "mean_obs_per_seed":  round(float(mean_obs_per_seed), 1),
            "sigma_imm":          sigma_imm,
            "sigma_fut":          sigma_fut,
            "se_imm":             se_imm,
            "se_fut":             se_fut,
        }
    return report


def print_variance_report(sample_size_report: dict, target_se: float):
    print(f"\n  Variance Analysis (target SE ≤ {target_se})")
    print(f"  {'Type':<14} {'n_seeds':>8} {'n_req':>7} "
          f"{'obs/seed':>9} {'obs_req':>8}  "
          f"{'max_σ_imm':>10} {'max_σ_fut':>10}  "
          f"{'max_SE_imm':>11} {'max_SE_fut':>11}")
    print(f"  {'-'*100}")
    for atype in AGENT_TYPES:
        r = sample_size_report.get(atype, {})
        if not r or r.get("n_required") is None:
            n_cur = r.get("n_current", 0)
            print(f"  {atype:<14} {n_cur:>8}  (insufficient seeds for estimate)")
            continue
        max_si = float(r["sigma_imm"].max())
        max_sf = float(r["sigma_fut"].max())
        max_sei = float(r["se_imm"].max())
        max_sef = float(r["se_fut"].max())
        print(f"  {atype:<14} {r['n_current']:>8} {r['n_required']:>7} "
              f"{r['mean_obs_per_seed']:>9.1f} {r['obs_required']:>8}  "
              f"{max_si:>10.5f} {max_sf:>10.5f}  "
              f"{max_sei:>11.5f} {max_sef:>11.5f}")

    global_n_req = max(
        (r["n_required"] for r in sample_size_report.values()
         if r.get("n_required") is not None),
        default=None,
    )
    global_obs_req = max(
        (r["obs_required"] for r in sample_size_report.values()
         if r.get("obs_required") is not None),
        default=None,
    )
    print(f"\n  → Seeds needed (most demanding type): {global_n_req}")
    print(f"  → Total observations (max across types): {global_obs_req}")
    return global_n_req, global_obs_req


# ── Final pooled profiles ─────────────────────────────────────────────────────

def derive_profiles_bfe(df: pd.DataFrame, gamma: float = GAMMA_DEFAULT) -> dict:
    """
    Derive (mu_imm, mu_fut) per agent type from a combined agent log DataFrame.
    Includes all moves where neighbors > 0 (not just contested moves).
    rawSugarscape agents are identified by decisionModel == 'none' in the log.

    Returns:
        dict atype -> {"mu_imm": list[5], "mu_fut": list[5], "n_obs": int}
    """
    if df.empty:
        return {}

    if "decisionModel" not in df.columns:
        print("  [warn] 'decisionModel' column missing from agent log.")
        return {}

    profiles = {}

    for atype in AGENT_TYPES:
        type_df = _rows_for_type(df, atype)
        if type_df.empty:
            log_model = DECISIONMODEL_IN_LOG.get(atype, atype)
            print(f"  [warn] No log rows for agent type: {atype} "
                  f"(looked for decisionModel='{log_model}')")
            continue

        fdf = _filter_has_neighbor(type_df)
        n_total = len(type_df)
        n_filtered = len(fdf)

        if n_filtered == 0:
            print(f"  [{atype}] {n_total:,} total rows but 0 with neighbors>0; "
                  f"falling back to all rows.")
            fdf = type_df

        print(f"  [{atype}] {n_filtered:,} / {n_total:,} observations "
              f"(neighbors>0 filter)")

        imm_vecs, fut_vecs = _vecs_for_rows(fdf, gamma)
        mu_imm = imm_vecs.mean(axis=0).tolist()
        mu_fut  = fut_vecs.mean(axis=0).tolist()

        profiles[atype] = {
            "mu_imm": [round(x, 6) for x in mu_imm],
            "mu_fut":  [round(x, 6) for x in mu_fut],
            "n_obs":   len(fdf),
        }

        print(f"         mu_imm: {dict(zip(COORD_LABELS, [round(x, 4) for x in mu_imm]))}")
        print(f"         mu_fut: {dict(zip(COORD_LABELS, [round(x, 4) for x in mu_fut]))}")

    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# Streaming accumulator — avoids keeping the full combined DataFrame in memory
# and allows raw logs to be deleted immediately after per-seed stats are taken.
# ─────────────────────────────────────────────────────────────────────────────

def make_accum() -> dict:
    return {atype: {
        "seed_mus_imm": [], "seed_mus_fut": [], "seed_n_obs": [],
        "sum_imm": np.zeros(5), "sum_fut": np.zeros(5), "total_obs": 0,
    } for atype in AGENT_TYPES}


def _accum_one_seed(seed_df: pd.DataFrame, gamma: float, accum: dict) -> None:
    """Fold one seed's slim DataFrame into the accumulator (in-place)."""
    for atype in AGENT_TYPES:
        rows = _filter_has_neighbor(_rows_for_type(seed_df, atype))
        if len(rows) == 0:
            continue
        imm_vecs, fut_vecs = _vecs_for_rows(rows, gamma)
        n = len(rows)
        accum[atype]["seed_mus_imm"].append(imm_vecs.mean(axis=0))
        accum[atype]["seed_mus_fut"].append(fut_vecs.mean(axis=0))
        accum[atype]["seed_n_obs"].append(n)
        accum[atype]["sum_imm"]   += imm_vecs.sum(axis=0)
        accum[atype]["sum_fut"]   += fut_vecs.sum(axis=0)
        accum[atype]["total_obs"] += n


def compute_per_seed_stats_from_accum(accum: dict) -> dict:
    """Extract per-seed variance data from a completed accumulator."""
    result = {}
    for atype in AGENT_TYPES:
        a = accum[atype]
        result[atype] = {
            "seed_mus_imm": np.array(a["seed_mus_imm"]) if a["seed_mus_imm"] else np.empty((0, 5)),
            "seed_mus_fut": np.array(a["seed_mus_fut"]) if a["seed_mus_fut"] else np.empty((0, 5)),
            "seed_n_obs":   a["seed_n_obs"],
        }
    return result


def derive_profiles_from_accum(accum: dict) -> dict:
    """Compute final observation-weighted BFE profiles from a completed accumulator."""
    profiles = {}
    for atype in AGENT_TYPES:
        a = accum[atype]
        if a["total_obs"] == 0:
            log_model = DECISIONMODEL_IN_LOG.get(atype, atype)
            print(f"  [warn] No qualifying observations for {atype} "
                  f"(log value '{log_model}')")
            continue
        mu_imm = a["sum_imm"] / a["total_obs"]
        mu_fut = a["sum_fut"] / a["total_obs"]
        profiles[atype] = {
            "mu_imm": [round(x, 6) for x in mu_imm],
            "mu_fut":  [round(x, 6) for x in mu_fut],
            "n_obs":   a["total_obs"],
        }
        print(f"  [{atype}] {a['total_obs']:,} qualifying observations")
        print(f"         mu_imm: {dict(zip(COORD_LABELS, [round(x, 4) for x in mu_imm]))}")
        print(f"         mu_fut:  {dict(zip(COORD_LABELS, [round(x, 4) for x in mu_fut]))}")
    return profiles


def _run_batch_and_accumulate(batch: list, cfg_to_log: dict, python_alias: str,
                               accum: dict, gamma: float, num_cores: int,
                               counter_offset: int, total_pending: int,
                               keep_logs: bool) -> int:
    """Run one batch of simulations, accumulate stats, optionally delete logs.

    Returns the number of missing/empty logs in the batch.
    """
    manager = multiprocessing.Manager()
    counter = manager.Value("i", counter_offset)
    lock    = manager.Lock()
    worker_args = [(p, python_alias, counter, lock, total_pending) for p in batch]
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(run_one_sim, worker_args)

    n_missing = 0
    for cfg_path in batch:
        seed, agent_log_path = cfg_to_log[cfg_path]
        df = agent_log_to_df(agent_log_path, seed_tag=seed, slim=True)
        if df.empty:
            n_missing += 1
        else:
            _accum_one_seed(df, gamma, accum)
            if not keep_logs:
                try:
                    os.remove(agent_log_path)
                except OSError:
                    pass
    return n_missing


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    t0 = time.time()

    output_dir   = args.output
    config_path  = args.config
    num_seeds    = args.seeds
    timesteps    = args.timesteps
    num_agents   = args.agents
    num_cores    = args.cores
    python_alias = args.python
    gamma        = args.gamma
    force        = args.force

    os.makedirs(output_dir, exist_ok=True)
    sim_dir = os.path.join(output_dir, "derivation_logs")
    os.makedirs(sim_dir, exist_ok=True)

    max_cores = os.cpu_count() or 1
    num_cores = min(num_cores, max_cores)

    print(f"\n{'='*60}")
    print(f"  FVDM Prioritization Profile Derivation (BFE)")
    print(f"{'='*60}")
    print(f"  Seeds:      {num_seeds}")
    print(f"  Timesteps:  {timesteps}")
    print(f"  Agents:     {num_agents}  (all 4 types, round-robin)")
    print(f"  Gamma:      {gamma}")
    print(f"  Cores:      {num_cores}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    base_cfg = load_base_config(config_path)

    # ── Generate matched seeds ──────────────────────────────────────────────
    random.seed(42)
    seeds = []
    while len(seeds) < num_seeds:
        s = random.randint(0, sys.maxsize)
        if s not in seeds:
            seeds.append(s)

    keep_logs = args.keep_logs

    # ── Build per-seed simulation configs (mixed population) ────────────────
    pending    = []   # cfg paths that still need running
    cfg_to_log = {}   # cfg_path -> (seed, agent_log_path)
    cached     = {}   # seed    -> agent_log_path  (already-done seeds)

    def _make_seed_cfg(seed):
        cfg = dict(base_cfg)
        cfg["seed"]               = seed
        cfg["timesteps"]          = timesteps
        cfg["startingAgents"]     = num_agents
        cfg["startingDiseases"]   = 0
        cfg["headlessMode"]       = True
        cfg["debugMode"]          = ["none"]
        cfg["keepAlivePostExtinction"] = False
        cfg["keepAliveAtEnd"]     = False
        cfg["screenshots"]        = False
        cfg["profileMode"]        = False
        cfg["agentDecisionModels"] = AGENT_TYPES
        log_path       = os.path.join(sim_dir, f"deriv_{seed}.json")
        agent_log_path = os.path.join(sim_dir, f"deriv_{seed}_agents.json")
        cfg["logfile"]       = log_path
        cfg["agentLogfile"]  = agent_log_path
        cfg["logfileFormat"] = "json"
        cfg_path_out = os.path.join(sim_dir, f"deriv_{seed}.config")
        with open(cfg_path_out, "w") as f:
            json.dump(cfg, f)
        return cfg_path_out, agent_log_path

    for seed in seeds:
        cfg_path_out, agent_log_path = _make_seed_cfg(seed)
        cfg_to_log[cfg_path_out] = (seed, agent_log_path)
        already_done = (not force and os.path.exists(agent_log_path)
                        and safe_json_load(agent_log_path))
        if already_done:
            cached[seed] = agent_log_path
        else:
            pending.append(cfg_path_out)

    skip = num_seeds - len(pending)
    print(f"  {num_seeds} derivation sims: {skip} already done, {len(pending)} queued\n")

    # ── Streaming accumulator — never builds a combined DataFrame ───────────
    accum    = make_accum()
    n_missing = 0

    # Load already-done (cached) seeds first
    if cached:
        print(f"  Loading {len(cached)} cached seed(s) …")
        for seed, path in cached.items():
            df = agent_log_to_df(path, seed_tag=seed, slim=True)
            if df.empty:
                n_missing += 1
            else:
                _accum_one_seed(df, gamma, accum)
                if not keep_logs:
                    try: os.remove(path)
                    except OSError: pass
        print()

    # Run pending seeds in batches of num_cores, process+delete after each batch
    if pending:
        batches = [pending[i:i + num_cores] for i in range(0, len(pending), num_cores)]
        print(f"  Running {len(pending)} simulations ({num_cores} cores, "
              f"{len(batches)} batch(es)) …")
        offset = 0
        for batch in batches:
            n_miss = _run_batch_and_accumulate(
                batch, cfg_to_log, python_alias, accum, gamma,
                num_cores, offset, len(pending), keep_logs)
            n_missing += n_miss
            offset += len(batch)
        print()

    n_seeds_done = sum(len(a["seed_n_obs"]) for a in accum.values()
                       if a["seed_n_obs"]) // max(1, len(AGENT_TYPES))
    total_rows = sum(a["total_obs"] for a in accum.values())
    if n_missing:
        print(f"  [warn] {n_missing} log file(s) missing or empty.")
    if total_rows == 0:
        print("  [error] No agent log data found.  Aborting.")
        sys.exit(1)
    print(f"  Accumulated: ~{total_rows:,} qualifying observations "
          f"across {n_seeds_done} seed(s)\n")

    # ── Variance analysis & sample-size estimation ──────────────────────────
    print("  Variance analysis …\n")
    per_seed_stats  = compute_per_seed_stats_from_accum(accum)
    sample_size_rpt = estimate_required_seeds(per_seed_stats,
                                              target_se=args.target_se)
    global_n_req, global_obs_req = print_variance_report(sample_size_rpt,
                                                         args.target_se)

    # ── Expand seed pool if required ────────────────────────────────────────
    if global_n_req is not None and global_n_req > num_seeds:
        extra_needed = global_n_req - num_seeds
        print(f"\n  Variance requires {global_n_req} seeds "
              f"({extra_needed} more than the {num_seeds} run so far).")
        print(f"  Running {extra_needed} additional seed(s) …\n")

        all_seen = set(seeds)
        extra_seeds = []
        candidate = num_seeds + 1
        while len(extra_seeds) < extra_needed:
            random.seed(candidate)
            s = random.randint(0, sys.maxsize)
            if s not in all_seen:
                extra_seeds.append(s)
                all_seen.add(s)
            candidate += 1

        extra_pending = []
        for seed in extra_seeds:
            cfg_path_out, agent_log_path = _make_seed_cfg(seed)
            cfg_to_log[cfg_path_out] = (seed, agent_log_path)
            if os.path.exists(agent_log_path) and safe_json_load(agent_log_path):
                df = agent_log_to_df(agent_log_path, seed_tag=seed, slim=True)
                if not df.empty:
                    _accum_one_seed(df, gamma, accum)
                    if not keep_logs:
                        try: os.remove(agent_log_path)
                        except OSError: pass
            else:
                extra_pending.append(cfg_path_out)

        if extra_pending:
            batches = [extra_pending[i:i + num_cores]
                       for i in range(0, len(extra_pending), num_cores)]
            print(f"  Running {len(extra_pending)} extra simulations "
                  f"({num_cores} cores, {len(batches)} batch(es)) …")
            offset = 0
            for batch in batches:
                n_miss = _run_batch_and_accumulate(
                    batch, cfg_to_log, python_alias, accum, gamma,
                    num_cores, offset, len(extra_pending), keep_logs)
                n_missing += n_miss
                offset += len(batch)
            print()

        total_rows = sum(a["total_obs"] for a in accum.values())
        print(f"  Expanded accumulator: ~{total_rows:,} qualifying observations\n")
    else:
        if global_n_req is not None:
            print(f"\n  Current {num_seeds} seeds already satisfy target SE "
                  f"(required {global_n_req}).\n")

    # ── Derive BFE profiles from accumulator ────────────────────────────────
    print("  Computing BFE profiles …\n")
    profiles = derive_profiles_from_accum(accum)

    # ── φ-linear validation: predict Bentham as 0.5·Egoist + 0.5·Altruist ──
    phi_bfs = None
    phi_pred = None
    if "egoist" in profiles and "altruist" in profiles and "bentham" in profiles:
        mu_e_imm = np.array(profiles["egoist"]["mu_imm"])
        mu_a_imm = np.array(profiles["altruist"]["mu_imm"])
        mu_e_fut = np.array(profiles["egoist"]["mu_fut"])
        mu_a_fut = np.array(profiles["altruist"]["mu_fut"])
        pred_imm = (0.5 * mu_e_imm + 0.5 * mu_a_imm).tolist()
        pred_fut = (0.5 * mu_e_fut + 0.5 * mu_a_fut).tolist()

        def _cos(u, v):
            u, v = np.array(u), np.array(v)
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            return float(np.dot(u, v) / (nu * nv)) if nu > 1e-10 and nv > 1e-10 else 0.0

        cos_imm = _cos(pred_imm, profiles["bentham"]["mu_imm"])
        cos_fut = _cos(pred_fut, profiles["bentham"]["mu_fut"])
        phi_bfs = round((cos_imm + cos_fut) / 2.0, 6)
        phi_pred = {
            "mu_imm": [round(x, 6) for x in pred_imm],
            "mu_fut": [round(x, 6) for x in pred_fut],
            "cos_imm": round(cos_imm, 6),
            "cos_fut": round(cos_fut, 6),
            "phi_bfs": phi_bfs,
        }
        print(f"\n  φ-Linear Validation (Bentham ≈ 0.5·Egoist + 0.5·Altruist)")
        print(f"  cos_imm = {cos_imm:.4f}   cos_fut = {cos_fut:.4f}   φ-BFS = {phi_bfs:.4f}")
        if phi_bfs >= 0.99:
            print("  → φ linearly parameterises felicific space (Bentham BFE is redundant).")
        else:
            print("  → argmin nonlinearity is material; Bentham BFE is not redundant.")

    # ── Save output ─────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, "bfe_profiles.json")
    output = {
        "description": "BFE prioritization profiles for FVDM agent types",
        "coordinate_labels": {
            "imm": ["I (intensity)", "D (duration)", "C (certainty)", "P=1 (propinquity)", "E=1/|V| (extent)"],
            "fut": ["J (future intensity)", "Df (future duration)", "C (certainty)", "P=gamma (propinquity)", "E=1/|V| (extent)"],
        },
        "derivation_config": {
            "seeds": num_seeds,
            "timesteps": timesteps,
            "agents": num_agents,
            "gamma": gamma,
            "filter": "neighbors_gt_0",
            "agent_types": AGENT_TYPES,
        },
        "profiles": {k: {kk: vv for kk, vv in v.items()} for k, v in profiles.items()},
        "phi_linear_validation": phi_pred,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = round(time.time() - t0, 1)
    print(f"\n{'='*60}")
    print(f"  {'Type':<20} {'n_obs':>8}  {'mu_imm[:2]':>22}  {'mu_fut[:2]':>22}")
    print(f"  {'-'*56}")
    for atype, prof in profiles.items():
        mi = prof["mu_imm"]
        mf = prof["mu_fut"]
        mi_str = f"I={mi[0]:.3f} D={mi[1]:.3f}"
        mf_str = f"J={mf[0]:.3f} Df={mf[1]:.3f}"
        print(f"  {atype:<20} {prof['n_obs']:>8}  {mi_str:>22}  {mf_str:>22}")
    if phi_bfs is not None:
        print(f"  {'φ-BFS (Bentham)':<20} {'':>8}  {'':>22}  {phi_bfs:>22.4f}")
    print(f"{'='*60}")
    print(f"  Saved → {out_path}")
    print(f"  Done in {elapsed}s\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Derive FVDM prioritization profiles via BFE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c", "--config",    default="config.json")
    p.add_argument("-o", "--output",    default="fvdm_vectors")
    p.add_argument("-s", "--seeds",     type=int, default=128,
                   help="Number of mixed-population derivation seeds.")
    p.add_argument("-a", "--agents",    type=int, default=250,
                   help="Starting agents (round-robin across 4 types).")
    p.add_argument("-t", "--timesteps", type=int, default=5000)
    p.add_argument("-j", "--cores",     type=int, default=1)
    p.add_argument("-g", "--gamma",     type=float, default=0.5,
                   help="Lookahead discount factor (gamma) for future layer.")
    p.add_argument("--python",     default="python3")
    p.add_argument("--force",      action="store_true", default=False,
                   help="Re-run all derivation sims even if logs exist.")
    p.add_argument("--keep-logs",  action="store_true", default=False,
                   help="Keep raw agent JSON logs after processing (default: "
                        "delete immediately to save disk space).")
    p.add_argument("--target-se",  type=float, default=DEFAULT_TARGET_SE,
                   help="Target standard error per vector component for "
                        "sample-size estimation. Script will run additional "
                        "seeds automatically if the initial batch is insufficient.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
