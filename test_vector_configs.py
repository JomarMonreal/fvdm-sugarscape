#!/usr/bin/env python3
"""
test_vector_configs.py
----------------------
Quick survival test for different BFE vector configurations.

Runs 3 seeds × each config in bfe_profiles_test_configs.json, reports
whether agents survive past timestep 200 and final population size.

Usage:
  python test_vector_configs.py
  python test_vector_configs.py --seeds 5 --timesteps 1000 --cores 8
  python test_vector_configs.py --configs my_profiles.json
"""

import argparse
import json
import math
import multiprocessing
import os
import random
import subprocess
import sys
import tempfile
import time

PYTHON = sys.executable
PROFILE_PATH = "fvdm_vectors/bfe_profiles_test_configs.json"
BASE_CONFIG   = "config.json"
AGENT_MODEL   = "fvdmEgoistDerived"   # reuse key mapping — egoist reads "egoist" profile
TIMESTEPS     = 500
N_SEEDS       = 3
NUM_AGENTS    = 250
SURVIVAL_MIN_TIMESTEP = 200   # agent must survive past this to count as "surviving"


def load_base_config(path):
    with open(path) as f:
        full = json.load(f)
    return full.get("sugarscapeOptions", full)


def make_cfg(base, seed, profile_key, timesteps, n_agents, log_path, agent_log_path):
    cfg = dict(base)
    cfg["seed"]                = seed
    cfg["agentDecisionModels"] = [AGENT_MODEL]
    cfg["timesteps"]           = timesteps
    cfg["startingAgents"]      = n_agents
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


def _build_profile_file(test_profiles, profile_key):
    """Write a temporary bfe_profiles.json containing only `profile_key` mapped to 'egoist'."""
    profile = test_profiles[profile_key]
    data = {"profiles": {"egoist": {"mu_imm": profile["mu_imm"], "mu_fut": profile["mu_fut"]}}}
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    return f.name


def run_one(args):
    cfg_path, profile_file, seed, profile_key, counter, lock, total = args
    env = os.environ.copy()
    env["FVDM_PROFILE_PATH"] = profile_file
    t0 = time.time()
    subprocess.run(
        [PYTHON, "sugarscape.py", "--conf", cfg_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env,
    )
    dur = time.time() - t0
    with lock:
        counter.value += 1
        n = counter.value
        bar_filled = int(30 * n / total)
        bar = "█" * bar_filled + "░" * (30 - bar_filled)
        pct = n / total * 100
        print(f"\r  [{bar}] {n:>3}/{total}  {pct:5.1f}%  {profile_key:<22} seed={seed}  ({dur:.1f}s)",
              end="", flush=True)
    return cfg_path, seed, profile_key


def check_survival(log_path, min_ts):
    if not os.path.exists(log_path):
        return 0, False
    try:
        with open(log_path) as f:
            data = json.load(f)
        ticks = data if isinstance(data, list) else data.get("ticks", [])
        if not ticks:
            return 0, False
        last = ticks[-1]
        ts = last.get("timestep", last.get("tick", 0))
        pop = last.get("livingAgents", last.get("agents", 0))
        survived = ts >= min_ts and pop > 0
        return pop, survived
    except Exception:
        return 0, False


def main(args):
    with open(PROFILE_PATH) as f:
        test_data = json.load(f)
    test_profiles = test_data["profiles"]

    base = load_base_config(BASE_CONFIG)
    random.seed(99)
    seeds = [random.randint(1, 10**9) for _ in range(args.seeds)]

    tmp_dir = tempfile.mkdtemp(prefix="fvdm_test_")
    log_dir = os.path.join(tmp_dir, "logs")
    os.makedirs(log_dir)

    # Build all (profile, seed) combinations
    worker_args = []
    cfg_meta = {}   # cfg_path -> (profile_key, seed)

    for pkey in test_profiles:
        if pkey.startswith("_"):
            continue
        profile_file = _build_profile_file(test_profiles, pkey)
        for seed in seeds:
            tag      = f"{pkey}_{seed}"
            log_path = os.path.join(log_dir, f"{tag}.json")
            alog     = os.path.join(log_dir, f"{tag}_agents.json")
            cfg      = make_cfg(base, seed, pkey, args.timesteps, args.agents, log_path, alog)
            cfg_path = os.path.join(log_dir, f"{tag}_cfg.json")
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
            worker_args.append((cfg_path, profile_file, seed, pkey))
            cfg_meta[cfg_path] = (pkey, seed, log_path)

    total = len(worker_args)
    n_configs = sum(1 for k in test_profiles if not k.startswith("_"))
    print(f"\n{'='*60}")
    print(f"  Vector Config Survival Test")
    print(f"  Configs: {n_configs}  Seeds: {args.seeds}  "
          f"Timesteps: {args.timesteps}  Total runs: {total}")
    print(f"  Survival threshold: past timestep {SURVIVAL_MIN_TIMESTEP}")
    print(f"{'='*60}\n")

    mgr     = multiprocessing.Manager()
    counter = mgr.Value("i", 0)
    lock    = mgr.Lock()
    worker_args = [(cfg, pf, s, pk, counter, lock, total)
                   for (cfg, pf, s, pk) in worker_args]

    cores = min(args.cores, total)
    t0 = time.time()
    with multiprocessing.Pool(processes=cores) as pool:
        for _ in pool.imap_unordered(run_one, worker_args):
            pass
    elapsed = time.time() - t0
    print(f"\n\n  Done in {elapsed:.1f}s\n")

    # Aggregate results
    results = {}   # profile_key -> list of (pop, survived)
    for (cfg_path, pf, seed, pkey, *_rest) in worker_args:
        _, _, log_path = cfg_meta[cfg_path]
        pop, survived = check_survival(log_path, SURVIVAL_MIN_TIMESTEP)
        results.setdefault(pkey, []).append((pop, survived))

    print(f"  {'Config':<22}  {'Survived':<10}  {'Avg final pop':<16}  Prediction")
    print(f"  {'-'*70}")
    for pkey, runs in results.items():
        n_survived  = sum(1 for _, s in runs if s)
        avg_pop     = sum(p for p, _ in runs) / max(1, len(runs))
        note        = test_profiles[pkey].get("_note", "")
        pred        = note.split("Prediction: ")[-1].split(".")[0] if "Prediction:" in note else "?"
        print(f"  {pkey:<22}  {n_survived}/{len(runs):<9}  {avg_pop:<16.1f}  {pred}")

    print(f"\n  Temp files in: {tmp_dir}")
    print(f"  (delete with: rm -rf {tmp_dir})\n")


def parse_args():
    p = argparse.ArgumentParser(description="Test BFE vector configurations for survival")
    p.add_argument("--configs",    default=PROFILE_PATH,
                   help="Path to test configs JSON")
    p.add_argument("--seeds",      type=int, default=N_SEEDS)
    p.add_argument("--timesteps",  type=int, default=TIMESTEPS)
    p.add_argument("--agents",     type=int, default=NUM_AGENTS)
    p.add_argument("--cores",      type=int, default=min(8, os.cpu_count() or 1))
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
