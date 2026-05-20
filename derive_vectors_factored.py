#!/usr/bin/env python3
"""
derive_vectors_factored.py
---------------------------
Derives four-vector BFE profiles by keeping the self and neighbor felicific
effects SEPARATE rather than collapsing them into a single v_net.

Derived from h(c) = φ × h_self(c) − (1−φ) × Σ h_other_k(c).

For each qualifying timestep of a rule-based agent, four φ-scaled vectors are logged:
  v1 = φ · v_self_imm(c*)        — immediate self welfare, φ-weighted
  v2 = φ · v_self_fut(c*)        — future self welfare, φ-weighted
  v3 = (1−φ) · v_nbr_imm(c*)    — mean immediate neighbour welfare, (1−φ)-weighted
  v4 = (1−φ) · v_nbr_fut(c*)    — mean future neighbour welfare, (1−φ)-weighted

Averaging over all qualifying timesteps (bfe_has_nbrs=1) gives:
  (μ1, μ2, μ3, μ4)

The decision rule for a new FVDM argmin agent is then simply:
  c* = argmin  dist(φ·v_self_imm(c), μ1) + dist(φ·v_self_fut(c), μ2)
             + dist((1−φ)·v_nbr_imm(c), μ3) + dist((1−φ)·v_nbr_fut(c), μ4)

Because the same (φ, 1−φ) scaling is applied at both derivation and decision time,
the four profile vectors form a consistent prioritization space. For egoist (φ=1):
μ3 = μ4 = 0, the neighbor terms vanish, and the agent purely matches μ1/μ2 which
reflect high-resource cells — no extinction.

Requires: FVDMBFEAgent logs bfe_self_v_imm_*, bfe_nbr_v_imm_*, bfe_has_nbrs
          (added alongside the existing bfe_v_imm_* net-vector fields in ethics.py)

Usage:
  python derive_vectors_factored.py --homogeneous -s 10 -t 5000 -a 250 -j 30
  python derive_vectors_factored.py --homogeneous -s 10 -j 30 --force
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

IMM_LABELS = ["I", "D", "C", "P", "E"]
FUT_LABELS = ["I", "D", "C", "P", "E"]

PROFILE_KEY = {
    "phiBfeRaw":      "rawSugarscape",
    "phiBfeEgoist":   "egoist",
    "phiBfeAltruist": "altruist",
    "phiBfeBentham":  "bentham",
}


# ─────────────────────────────────────────────────────────────────────────────
# Simulation helpers (shared with derive_vectors_phi.py)
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
# Log reading — reads separate self/neighbor vectors from FVDMBFEAgent logs
# ─────────────────────────────────────────────────────────────────────────────

def read_factored_vecs(agent_log_path: str, atype: str):
    """
    Read bfe_self_v_imm/fut and bfe_nbr_v_imm/fut from an FVDMBFEAgent log.

    Filters to rows where:
      - decisionModel matches atype
      - bfe_has_nbrs == 1  (neighbor vectors are valid)

    Returns (self_imm, self_fut, nbr_imm, nbr_fut) as (N,5) arrays,
    or (None, None, None, None) if the log is missing/empty.
    """
    data = safe_json_load(agent_log_path)
    if not data:
        return None, None, None, None

    dm_key = atype.lower()

    rows_si, rows_sf, rows_ni, rows_nf = [], [], [], []
    for row in data:
        dm = row.get("decisionModel", "").lower()
        if dm_key not in dm and dm != dm_key:
            if atype == "phiBfeRaw" and dm not in ("none", "phibferaw"):
                continue
            elif atype != "phiBfeRaw":
                continue

        if not int(row.get("bfe_has_nbrs", 0)):
            continue

        si = np.array([float(row.get(f"bfe_self_v_imm_{l}", 0.0)) for l in IMM_LABELS])
        sf = np.array([float(row.get(f"bfe_self_v_fut_{l}",  0.0)) for l in FUT_LABELS])
        ni = np.array([float(row.get(f"bfe_nbr_v_imm_{l}",   0.0)) for l in IMM_LABELS])
        nf = np.array([float(row.get(f"bfe_nbr_v_fut_{l}",   0.0)) for l in FUT_LABELS])

        if not (np.any(si != 0) or np.any(sf != 0)):
            continue

        rows_si.append(si); rows_sf.append(sf)
        rows_ni.append(ni); rows_nf.append(nf)

    if not rows_si:
        return None, None, None, None
    return (np.array(rows_si), np.array(rows_sf),
            np.array(rows_ni), np.array(rows_nf))


# ─────────────────────────────────────────────────────────────────────────────
# Accumulator
# ─────────────────────────────────────────────────────────────────────────────

def make_accum(agent_types=None):
    types = agent_types or AGENT_TYPES
    return {t: {
        "sum_self_imm":       np.zeros(5),
        "sum_self_fut":        np.zeros(5),
        "sum_nbr_imm":        np.zeros(5),
        "sum_nbr_fut":         np.zeros(5),
        "total_obs":           0,
        "seed_mus_self_imm":  [],
        "seed_mus_self_fut":   [],
        "seed_mus_nbr_imm":   [],
        "seed_mus_nbr_fut":    [],
        "seed_n_obs":          [],
    } for t in types}


def accum_one_seed(agent_log_path: str, accum: dict):
    for atype in accum:
        si, sf, ni, nf = read_factored_vecs(agent_log_path, atype)
        if si is None:
            continue
        n = len(si)
        accum[atype]["seed_mus_self_imm"].append(si.mean(axis=0))
        accum[atype]["seed_mus_self_fut"].append(sf.mean(axis=0))
        accum[atype]["seed_mus_nbr_imm"].append(ni.mean(axis=0))
        accum[atype]["seed_mus_nbr_fut"].append(nf.mean(axis=0))
        accum[atype]["seed_n_obs"].append(n)
        accum[atype]["sum_self_imm"] += si.sum(axis=0)
        accum[atype]["sum_self_fut"]  += sf.sum(axis=0)
        accum[atype]["sum_nbr_imm"]  += ni.sum(axis=0)
        accum[atype]["sum_nbr_fut"]   += nf.sum(axis=0)
        accum[atype]["total_obs"]    += n


def derive_profiles_from_accum(accum: dict) -> dict:
    profiles = {}
    for atype, a in accum.items():
        if a["total_obs"] == 0:
            print(f"  [warn] No qualifying observations for {atype}")
            continue
        n = a["total_obs"]
        mu_si = a["sum_self_imm"] / n
        mu_sf = a["sum_self_fut"]  / n
        mu_ni = a["sum_nbr_imm"]  / n
        mu_nf = a["sum_nbr_fut"]   / n

        key = PROFILE_KEY[atype]
        profiles[key] = {
            "mu_self_imm": [round(x, 6) for x in mu_si],
            "mu_self_fut":  [round(x, 6) for x in mu_sf],
            "mu_nbr_imm":  [round(x, 6) for x in mu_ni],
            "mu_nbr_fut":   [round(x, 6) for x in mu_nf],
            "n_obs": n,
        }
        print(f"  [{atype}]  n={n:,}")
        print(f"    mu_self_imm: { {l: round(mu_si[i], 4) for i, l in enumerate(IMM_LABELS)} }")
        print(f"    mu_self_fut:  { {l: round(mu_sf[i], 4) for i, l in enumerate(FUT_LABELS)} }")
        print(f"    mu_nbr_imm:  { {l: round(mu_ni[i], 4) for i, l in enumerate(IMM_LABELS)} }")
        print(f"    mu_nbr_fut:   { {l: round(mu_nf[i], 4) for i, l in enumerate(FUT_LABELS)} }")
    return profiles


def print_variance_report(accum: dict):
    print(f"\n  Variance across seeds (max std dev in self_imm / nbr_imm):")
    print(f"  {'Type':<22}  {'n_seeds':>8}  {'max_σ_self_imm':>14}  {'max_σ_nbr_imm':>13}")
    print(f"  {'-'*64}")
    for atype, a in accum.items():
        mus_si = np.array(a["seed_mus_self_imm"]) if a["seed_mus_self_imm"] else np.empty((0, 5))
        mus_ni = np.array(a["seed_mus_nbr_imm"])  if a["seed_mus_nbr_imm"]  else np.empty((0, 5))
        n = len(mus_si)
        if n < 2:
            print(f"  {atype:<22}  {n:>8}  (need ≥2 seeds)")
            continue
        max_si = float(mus_si.std(axis=0, ddof=1).max())
        max_ni = float(mus_ni.std(axis=0, ddof=1).max()) if len(mus_ni) >= 2 else float("nan")
        print(f"  {atype:<22}  {n:>8}  {max_si:>14.5f}  {max_ni:>13.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# Config builder
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
# Derivation runners
# ─────────────────────────────────────────────────────────────────────────────

def run_homogeneous(args, sim_dir: str) -> dict:
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
        accum_one_seed(agent_log_path, {atype: accum[atype]})
        if not args.keep_logs:
            try:
                os.remove(agent_log_path)
            except OSError:
                pass

    return accum


def run_mixed(args, sim_dir: str) -> dict:
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
        accum_one_seed(agent_log_path, accum)
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
    print(f"  Factored BFE Derivation  (separate self / neighbor vectors)")
    print(f"{'='*65}")
    print(f"  Profile: (μ_self_imm, μ_self_fut, μ_nbr_imm, μ_nbr_fut)")
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

    # ── Sanity check: self D should be positive for all types ─────────────────
    print(f"\n  Self-D sanity check (should be > 0 for all types — survival signal):")
    print(f"  {'Type':<14}  {'mu_self_imm_D':>14}  {'mu_nbr_imm_D':>13}")
    print(f"  {'-'*46}")
    d_idx = IMM_LABELS.index("D")
    for key, prof in profiles.items():
        sd = prof["mu_self_imm"][d_idx]
        nd = prof["mu_nbr_imm"][d_idx]
        flag = "  ✓" if sd > 0 else "  ✗ PROBLEM"
        print(f"  {key:<14}  {sd:>14.4f}  {nd:>13.4f}{flag}")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "description": (
            "Factored BFE profiles: v_self and v_nbr stored separately. "
            "Decision rule: c* = argmin φ·dist(v_self(c), μ_self) + (1−φ)·dist(v_nbr(c), μ_nbr). "
            "Derived from rule-based FVDMBFEAgent runs; filter: bfe_has_nbrs=1."
        ),
        "coordinate_labels": {
            "self_imm": ["I", "D", "C", "P", "E"],
            "self_fut":  ["J", "Df", "C", "Pgamma", "E"],
            "nbr_imm":  ["I", "D", "C", "P", "E"],
            "nbr_fut":   ["J", "Df", "C", "Pgamma", "E"],
        },
        "derivation_config": {
            "method":    "factored_bfe",
            "mode":      "homogeneous" if args.homogeneous else "mixed",
            "seeds":     args.seeds,
            "timesteps": args.timesteps,
            "agents":    args.agents,
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
        description="Derive factored BFE profiles (separate self / neighbor vectors)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c", "--config",    default="config.json")
    p.add_argument("-o", "--output",    default="fvdm_vectors/bfe_profiles_factored.json")
    p.add_argument("--sim-dir",         default="derivation_factored_logs")
    p.add_argument("-s", "--seeds",     type=int, default=10)
    p.add_argument("-t", "--timesteps", type=int, default=5000)
    p.add_argument("-a", "--agents",    type=int, default=250)
    p.add_argument("-j", "--cores",     type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--python",          default=sys.executable)
    p.add_argument("--homogeneous",     action="store_true",
                   help="Run each type in isolation (recommended).")
    p.add_argument("--keep-logs",       action="store_true",
                   help="Keep raw agent log files after accumulation.")
    p.add_argument("--force",           action="store_true",
                   help="Re-run simulations even if logs exist.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
