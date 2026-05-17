#!/usr/bin/env python3
"""
derive_vectors.py
-----------------
Prioritization Profile Derivation via Behavioral Feature Expectation (BFE).

Runs a single mixed-population derivation simulation containing all four
baseline agent types simultaneously.  At each contested decision step
(chosen cell occupied by a different-tribe agent, bfe_is_contested=True),
the chosen cell's raw properties are read from the agent log and the two
felicific effect vectors are computed analytically:

  v_imm(c) = ( 1/((1+TTL)(1+pollution)),   W_c/(m*W_cmax),      1, 1,     1/|V| )
  v_fut(c)  = ( W_adj/(Wgmax*n_adj),         max(0,W_c-m)/(m*W_cmax), 1, gamma, 1/|V| )

Both vectors are averaged separately across all contested observations per
agent type, yielding one prioritization profile (mu_imm, mu_fut) per type.

Output: fvdm_vectors/bfe_profiles.json

Usage:
  python derive_vectors.py [options]
  python derive_vectors.py --seeds 128 --timesteps 5000 --agents 250 --cores 8
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

# gamma fallback when decisionModelLookaheadDiscount is not in the agent log
GAMMA_DEFAULT = 0.5


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

def agent_log_to_df(path: str) -> pd.DataFrame:
    """Parse a raw agent JSON log into a flat DataFrame."""
    data = safe_json_load(path)
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def derive_profiles_bfe(df: pd.DataFrame,
                        gamma: float = GAMMA_DEFAULT,
                        contested_only: bool = True) -> dict:
    """
    Derive (mu_imm, mu_fut) per agent type from a combined agent log DataFrame.

    Args:
        df: Combined agent log with columns produced by agent.updateRuntimeStats()
            plus the bfe_* columns added by this study.
        gamma: Discount factor for future layer propinquity.
        contested_only: If True, restrict BFE observations to contested moves
                        (bfe_is_contested == True).  Falls back to all moves
                        if a type has zero contested observations.

    Returns:
        dict mapping agent-type string -> {"mu_imm": list[5], "mu_fut": list[5],
                                            "n_obs": int, "contested_only": bool}
    """
    if df.empty:
        return {}

    # Identify agent type from decisionModel column (added by sugarscape)
    if "decisionModel" not in df.columns:
        print("  [warn] 'decisionModel' column missing from agent log.")
        return {}

    profiles = {}

    for atype in AGENT_TYPES:
        type_df = df[df["decisionModel"] == atype].copy()
        if type_df.empty:
            print(f"  [warn] No log rows for agent type: {atype}")
            continue

        # Filter to contested moves where possible
        use_contested = contested_only
        cdf = type_df[type_df["bfe_is_contested"] == True] if "bfe_is_contested" in type_df.columns else pd.DataFrame()
        if len(cdf) == 0:
            print(f"  [{atype}] No contested observations; falling back to all moves.")
            cdf = type_df
            use_contested = False

        print(f"  [{atype}] {len(cdf):,} observations (contested={use_contested})")

        imm_vecs = []
        fut_vecs = []
        for _, row in cdf.iterrows():
            row_dict = row.to_dict()
            v_imm = compute_v_imm(row_dict)
            v_fut = compute_v_fut(row_dict, gamma=gamma)
            imm_vecs.append(v_imm)
            fut_vecs.append(v_fut)

        mu_imm = np.mean(imm_vecs, axis=0).tolist()
        mu_fut = np.mean(fut_vecs, axis=0).tolist()

        profiles[atype] = {
            "mu_imm": [round(x, 6) for x in mu_imm],
            "mu_fut": [round(x, 6) for x in mu_fut],
            "n_obs": len(cdf),
            "contested_only": use_contested,
        }

        labels = ["I", "D", "C", "P", "E"]
        print(f"         mu_imm: {dict(zip(labels, [round(x, 4) for x in mu_imm]))}")
        print(f"         mu_fut: {dict(zip(labels, [round(x, 4) for x in mu_fut]))}")

    return profiles


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

    # ── Build per-seed simulation configs (mixed population) ────────────────
    pending = []
    agent_log_paths = []

    for seed in seeds:
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
        # Mixed population: all four types in round-robin
        cfg["agentDecisionModels"] = AGENT_TYPES

        log_path       = os.path.join(sim_dir, f"deriv_{seed}.json")
        agent_log_path = os.path.join(sim_dir, f"deriv_{seed}_agents.json")
        cfg["logfile"]       = log_path
        cfg["agentLogfile"]  = agent_log_path
        cfg["logfileFormat"] = "json"

        cfg_path_out = os.path.join(sim_dir, f"deriv_{seed}.config")
        with open(cfg_path_out, "w") as f:
            json.dump(cfg, f)

        agent_log_paths.append(agent_log_path)

        if not force and os.path.exists(agent_log_path):
            existing = safe_json_load(agent_log_path)
            if existing and len(existing) > 0:
                continue
        pending.append(cfg_path_out)

    skip = num_seeds - len(pending)
    print(f"  {num_seeds} derivation sims: {skip} already done, {len(pending)} queued\n")

    # ── Run simulations ─────────────────────────────────────────────────────
    if pending:
        manager = multiprocessing.Manager()
        counter = manager.Value("i", 0)
        lock    = manager.Lock()
        worker_args = [(p, python_alias, counter, lock, len(pending)) for p in pending]
        print(f"  Running {len(pending)} simulations ({num_cores} cores) …")
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.map(run_one_sim, worker_args)
        print()

    # ── Parse agent logs ────────────────────────────────────────────────────
    print("  Parsing agent logs …")
    frames = []
    missing = 0
    for path in agent_log_paths:
        df = agent_log_to_df(path)
        if df.empty:
            missing += 1
        else:
            frames.append(df)

    if not frames:
        print("  [error] No agent log data found.  Aborting.")
        sys.exit(1)

    if missing:
        print(f"  [warn] {missing} log file(s) missing or empty.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Combined log: {len(combined):,} agent-timestep records\n")

    # ── Derive BFE profiles ─────────────────────────────────────────────────
    print("  Computing BFE profiles …\n")
    profiles = derive_profiles_bfe(combined, gamma=gamma, contested_only=True)

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
            "contested_only": True,
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
    p.add_argument("--python",          default="python3")
    p.add_argument("--force", action="store_true", default=False,
                   help="Re-run all derivation sims even if logs exist.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
