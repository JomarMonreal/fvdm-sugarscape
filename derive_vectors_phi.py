#!/usr/bin/env python3
"""
derive_vectors_phi.py
----------------------
Derives corrected BFE prioritization profiles using the opportunity-cost
formulation of the welfare function:

  h(c) = φ × h_self(c)  −  (1−φ) × Σ_k h_k(c)

At each timestep the FVDMBFEAgent computes and logs the net feature vector:

  v_net = φ × v_self(c*)  −  (1−φ) × mean_k( v_k(c*) )

where v_k(c*) is what neighbour k would have gotten from c* — the value
denied to them.  Averaging v_net across all qualifying timesteps gives the
corrected profile μ_net, which correctly encodes both what the agent gains
and what it costs its neighbours.

This fixes the core methodological gap in derive_vectors.py, which only
recorded v_self (the agent's own gain) and ignored the opportunity-cost term.

Agent types and their φ values:
  phiBfeRaw      φ = 1.0  (greedy baseline)
  phiBfeEgoist   φ = 1.0
  phiBfeAltruist φ = 0.0
  phiBfeBentham  φ = 0.5

Usage:
  python derive_vectors_phi.py
  python derive_vectors_phi.py --homogeneous -s 10 -t 5000 -a 250 -j 30
  python derive_vectors_phi.py --homogeneous -s 10 -j 30 --force
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

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

AGENT_TYPES = ["phiBfeRaw", "phiBfeEgoist", "phiBfeAltruist", "phiBfeBentham"]

# Labels used in the log by FVDMBFEAgent.updateRuntimeStats()
IMM_LABELS = ["I", "D", "C", "P", "E"]
FUT_LABELS = ["I", "D", "C", "P", "E"]

# Profile key to map each type to in the output (matches bfe_profiles.json keys)
PROFILE_KEY = {
    "phiBfeRaw":      "rawSugarscape",
    "phiBfeEgoist":   "egoist",
    "phiBfeAltruist": "altruist",
    "phiBfeBentham":  "bentham",
}

DEFAULT_TARGET_SE = 0.02


# ─────────────────────────────────────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_base_config(path: str) -> dict:
    with open(path) as f:
        full = json.load(f)
    return full.get("sugarscapeOptions", full)


def safe_json_load(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def run_one_sim(args):
    cfg_path, python_alias, counter, lock, total = args
    subprocess.run([python_alias, "sugarscape.py", "--conf", cfg_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with lock:
        counter.value += 1
        n = counter.value
        bar = "█" * int(30 * n / total) + "░" * (30 - int(30 * n / total))
        print(f"\r  [{bar}] {n:>3}/{total}  {n/total*100:5.1f}%", end="", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Log reading — reads pre-computed v_net from FVDMBFEAgent log fields
# ─────────────────────────────────────────────────────────────────────────────

def read_net_vecs(agent_log_path: str, atype: str, seed_tag=None):
    """
    Read bfe_v_imm_* and bfe_v_fut_* from an agent log produced by FVDMBFEAgent.

    Returns (imm_vecs, fut_vecs) as np.ndarrays of shape (N, 5), filtered to:
      - rows belonging to atype (by decisionModel field)
      - rows where neighbors > 0

    Returns (None, None) if the log is missing or empty.
    """
    data = safe_json_load(agent_log_path)
    if not data:
        return None, None

    dm_key = atype.lower()   # e.g. "phibfeegoist"

    imm_vecs, fut_vecs = [], []
    for row in data:
        # Filter by agent type
        dm = row.get("decisionModel", "").lower()
        if dm_key not in dm and dm != dm_key:
            # Also accept exact match after stripping "phibfe" prefix for rawSugarscape
            # (sugarscape maps rawSugarscape -> "none" internally)
            if atype == "phiBfeRaw" and dm not in ("none", "phibferaw"):
                continue
            elif atype != "phiBfeRaw":
                continue

        # Filter: at least one neighbour present
        if int(row.get("neighbors", 0)) <= 0:
            continue

        # Read pre-computed net vectors
        v_imm = np.array([float(row.get(f"bfe_v_imm_{l}", 0.0)) for l in IMM_LABELS])
        v_fut = np.array([float(row.get(f"bfe_v_fut_{l}", 0.0)) for l in FUT_LABELS])

        # Skip zero vectors (agent had not yet moved)
        if not (np.any(v_imm != 0) or np.any(v_fut != 0)):
            continue

        imm_vecs.append(v_imm)
        fut_vecs.append(v_fut)

    if not imm_vecs:
        return None, None
    return np.array(imm_vecs), np.array(fut_vecs)


# ─────────────────────────────────────────────────────────────────────────────
# Accumulator
# ─────────────────────────────────────────────────────────────────────────────

def make_accum(agent_types=None):
    types = agent_types or AGENT_TYPES
    return {t: {
        "seed_mus_imm": [],
        "seed_mus_fut": [],
        "seed_n_obs":   [],
        "sum_imm":      np.zeros(5),
        "sum_fut":      np.zeros(5),
        "total_obs":    0,
    } for t in types}


def accum_one_seed(agent_log_path: str, seed: int, accum: dict):
    """Fold one seed's agent log into the accumulator."""
    for atype in accum:
        imm_vecs, fut_vecs = read_net_vecs(agent_log_path, atype, seed_tag=seed)
        if imm_vecs is None:
            continue
        n = len(imm_vecs)
        accum[atype]["seed_mus_imm"].append(imm_vecs.mean(axis=0))
        accum[atype]["seed_mus_fut"].append(fut_vecs.mean(axis=0))
        accum[atype]["seed_n_obs"].append(n)
        accum[atype]["sum_imm"]   += imm_vecs.sum(axis=0)
        accum[atype]["sum_fut"]   += fut_vecs.sum(axis=0)
        accum[atype]["total_obs"] += n


def derive_profiles_from_accum(accum: dict) -> dict:
    profiles = {}
    for atype, a in accum.items():
        if a["total_obs"] == 0:
            print(f"  [warn] No qualifying observations for {atype}")
            continue
        mu_imm = a["sum_imm"] / a["total_obs"]
        mu_fut = a["sum_fut"] / a["total_obs"]
        key = PROFILE_KEY[atype]
        profiles[key] = {
            "mu_imm": [round(x, 6) for x in mu_imm],
            "mu_fut":  [round(x, 6) for x in mu_fut],
            "n_obs":   a["total_obs"],
        }
        print(f"  [{atype}]  n={a['total_obs']:,}")
        print(f"    mu_imm: { {l: round(mu_imm[i], 4) for i, l in enumerate(IMM_LABELS)} }")
        print(f"    mu_fut:  { {l: round(mu_fut[i], 4) for i, l in enumerate(FUT_LABELS)} }")
    return profiles


def print_variance_report(accum: dict):
    print(f"\n  Variance across seeds (std dev per component):")
    print(f"  {'Type':<22}  {'n_seeds':>8}  {'max_σ_imm':>10}  {'max_σ_fut':>10}")
    print(f"  {'-'*58}")
    for atype, a in accum.items():
        mus_imm = np.array(a["seed_mus_imm"]) if a["seed_mus_imm"] else np.empty((0, 5))
        mus_fut = np.array(a["seed_mus_fut"]) if a["seed_mus_fut"] else np.empty((0, 5))
        n = len(mus_imm)
        if n < 2:
            print(f"  {atype:<22}  {n:>8}  (need ≥2 seeds)")
            continue
        max_si = float(mus_imm.std(axis=0, ddof=1).max())
        max_sf = float(mus_fut.std(axis=0, ddof=1).max())
        print(f"  {atype:<22}  {n:>8}  {max_si:>10.5f}  {max_sf:>10.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# Config builders
# ─────────────────────────────────────────────────────────────────────────────

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
# Derivation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_homogeneous(args, sim_dir: str) -> dict:
    """
    Run one isolated batch per agent type (homogeneous mode).
    All types' seeds are queued into a single shared pool for maximum
    core utilisation.
    Returns a completed accumulator.
    """
    base_cfg  = load_base_config(args.config)
    num_cores = min(args.cores, os.cpu_count() or 1)

    random.seed(42)
    seeds = []
    while len(seeds) < args.seeds:
        s = random.randint(0, sys.maxsize)
        if s not in seeds:
            seeds.append(s)

    accum      = make_accum(AGENT_TYPES)
    cfg_to_log = {}   # cfg_path -> (atype, seed, agent_log_path)
    all_cfgs   = []

    for atype in AGENT_TYPES:
        type_dir = os.path.join(sim_dir, atype)
        os.makedirs(type_dir, exist_ok=True)
        for seed in seeds:
            tag            = f"{atype}_{seed}"
            log_path       = os.path.join(type_dir, f"{tag}.json")
            agent_log_path = os.path.join(type_dir, f"{tag}_agents.json")
            cfg_path       = os.path.join(type_dir, f"{tag}.config")

            if not args.force and os.path.exists(agent_log_path):
                all_cfgs.append(cfg_path)
                cfg_to_log[cfg_path] = (atype, seed, agent_log_path)
                continue

            cfg = make_run_config(base_cfg, seed, [atype],
                                  args.timesteps, args.agents,
                                  log_path, agent_log_path)
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
            all_cfgs.append(cfg_path)
            cfg_to_log[cfg_path] = (atype, seed, agent_log_path)

    pending = [p for p in all_cfgs
               if not os.path.exists(cfg_to_log[p][2]) or args.force]

    print(f"  Total runs: {len(all_cfgs)}  |  "
          f"Queued: {len(pending)}  |  Skipped: {len(all_cfgs)-len(pending)}\n")

    if pending:
        manager = multiprocessing.Manager()
        counter = manager.Value("i", 0)
        lock    = manager.Lock()
        wargs   = [(p, args.python, counter, lock, len(pending)) for p in pending]
        print(f"  Running {len(pending)} simulations on {num_cores} core(s) …")
        t0 = time.time()
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.map(run_one_sim, wargs)
        print(f"\n\n  Done in {time.time()-t0:.1f}s\n")

    print("  Accumulating logs …")
    for cfg_path in all_cfgs:
        atype, seed, agent_log_path = cfg_to_log[cfg_path]
        accum_one_seed(agent_log_path, seed, {atype: accum[atype]})
        if not args.keep_logs:
            try:
                os.remove(agent_log_path)
            except OSError:
                pass

    return accum


def run_mixed(args, sim_dir: str) -> dict:
    """Run all types together in each simulation (mixed-population mode)."""
    base_cfg  = load_base_config(args.config)
    num_cores = min(args.cores, os.cpu_count() or 1)

    random.seed(42)
    seeds = []
    while len(seeds) < args.seeds:
        s = random.randint(0, sys.maxsize)
        if s not in seeds:
            seeds.append(s)

    accum      = make_accum(AGENT_TYPES)
    cfg_to_log = {}
    all_cfgs   = []

    for seed in seeds:
        tag            = f"mixed_{seed}"
        log_path       = os.path.join(sim_dir, f"{tag}.json")
        agent_log_path = os.path.join(sim_dir, f"{tag}_agents.json")
        cfg_path       = os.path.join(sim_dir, f"{tag}.config")

        cfg = make_run_config(base_cfg, seed, AGENT_TYPES,
                              args.timesteps, args.agents,
                              log_path, agent_log_path)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        all_cfgs.append(cfg_path)
        cfg_to_log[cfg_path] = (seed, agent_log_path)

    pending = [p for p in all_cfgs
               if not os.path.exists(cfg_to_log[p][1]) or args.force]

    if pending:
        manager = multiprocessing.Manager()
        counter = manager.Value("i", 0)
        lock    = manager.Lock()
        wargs   = [(p, args.python, counter, lock, len(pending)) for p in pending]
        print(f"  Running {len(pending)} simulations on {num_cores} core(s) …")
        t0 = time.time()
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.map(run_one_sim, wargs)
        print(f"\n\n  Done in {time.time()-t0:.1f}s\n")

    print("  Accumulating logs …")
    for cfg_path in all_cfgs:
        seed, agent_log_path = cfg_to_log[cfg_path]
        accum_one_seed(agent_log_path, seed, accum)
        if not args.keep_logs:
            try:
                os.remove(agent_log_path)
            except OSError:
                pass

    return accum


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    sim_dir = os.path.join(args.sim_dir,
                           "homogeneous" if args.homogeneous else "mixed")
    os.makedirs(sim_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Corrected BFE Derivation  (opportunity-cost formulation)")
    print(f"{'='*65}")
    print(f"  v_net = φ·v_self(c*) − (1−φ)·mean_k(v_k(c*))")
    print(f"  Mode   : {'homogeneous' if args.homogeneous else 'mixed'}")
    print(f"  Seeds  : {args.seeds}   Timesteps: {args.timesteps}   Agents: {args.agents}")
    print(f"  Cores  : {args.cores}")
    print(f"{'='*65}\n")

    if args.homogeneous:
        accum = run_homogeneous(args, sim_dir)
    else:
        accum = run_mixed(args, sim_dir)

    print_variance_report(accum)

    print(f"\n  Deriving profiles …\n")
    profiles = derive_profiles_from_accum(accum)

    # ── BFS comparison vs original bfe_profiles.json ─────────────────────────
    orig_path = "fvdm_vectors/bfe_profiles.json"
    if os.path.exists(orig_path):
        with open(orig_path) as f:
            orig_profiles = json.load(f).get("profiles", {})

        def cos(u, v):
            u, v = np.array(u, float), np.array(v, float)
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            return float(np.dot(u, v) / (nu * nv)) if nu > 1e-9 and nv > 1e-9 else 0.0

        print(f"\n  BFS vs original bfe_profiles.json (self-only derivation):")
        print(f"  {'Type':<14}  {'cos_imm':>9}  {'cos_fut':>9}  {'BFS':>9}")
        print(f"  {'-'*46}")
        for key, prof in profiles.items():
            if key in orig_profiles:
                ci = cos(prof["mu_imm"], orig_profiles[key]["mu_imm"])
                cf = cos(prof["mu_fut"],  orig_profiles[key]["mu_fut"])
                bfs = (ci + cf) / 2
                print(f"  {key:<14}  {ci:>9.4f}  {cf:>9.4f}  {bfs:>9.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "description": (
            "Corrected BFE profiles from h(c) = φ·h_self − (1−φ)·Σ h_k. "
            "v_net = φ·v_self(c*) − (1−φ)·mean_k(v_k(c*)) averaged over "
            "qualifying timesteps (neighbors > 0)."
        ),
        "coordinate_labels": {
            "imm": ["I_net", "D_net", "C_net", "P_net", "E_net"],
            "fut": ["J_net", "Df_net", "C_net", "Pgamma_net", "E_net"],
        },
        "derivation_config": {
            "method":     "phi_opportunity_cost",
            "mode":       "homogeneous" if args.homogeneous else "mixed",
            "seeds":      args.seeds,
            "timesteps":  args.timesteps,
            "agents":     args.agents,
        },
        "profiles": profiles,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {args.output}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Derive corrected BFE profiles via opportunity-cost formulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c", "--config",     default="config.json")
    p.add_argument("-o", "--output",     default="fvdm_vectors/bfe_profiles_phi.json")
    p.add_argument("--sim-dir",          default="derivation_phi_logs")
    p.add_argument("-s", "--seeds",      type=int, default=10)
    p.add_argument("-t", "--timesteps",  type=int, default=5000)
    p.add_argument("-a", "--agents",     type=int, default=250)
    p.add_argument("-j", "--cores",      type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--python",           default=sys.executable)
    p.add_argument("--homogeneous",      action="store_true",
                   help="Run each type in isolation (recommended).")
    p.add_argument("--keep-logs",        action="store_true",
                   help="Keep raw agent log files after accumulation.")
    p.add_argument("--force",            action="store_true",
                   help="Re-run simulations even if logs exist.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
