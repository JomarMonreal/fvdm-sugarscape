#!/usr/bin/env python3
"""
derive_vectors.py
-----------------
Prioritization Vector Derivation via Maximum Entropy IRL.

Derives 8 prioritization vectors P_i = (p_I, p_D, p_C, p_P, p_X):
  1-4. rawDerived, egoistDerived, altruistDerived, benthamDerived
       — from short baseline derivation sims (3 seeds × 1000 timesteps)
  5-8. combatDerived, tradeDerived, reproductionDerived, lendingDerived
       — REUSED from the existing focal-action derivation CSV (no new sims)

Speed optimizations vs. original:
  • Biased vectors: zero simulation time (reads existing CSV)
  • Baseline vectors: 3 seeds × 1000 ts instead of 10 × 5000
  • MaxEnt IRL: vectorized numpy, 100 iterations default
  • Total: ~5-10 minutes instead of hours

Usage:
  .venv/bin/python derive_vectors.py [options]
"""

import argparse
import json
import multiprocessing
import os
import random
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

STATE_FEATURES = [
    "age", "wealth", "sugar", "spice", "timeToLive", "movement",
    "neighbors", "neighborsInTribe", "neighborsNotInTribe",
    "validMoves", "compositeHappiness", "depression",
]

DISCRETIONARY_ACTIONS = ["combat", "trade", "reproduction", "lending"]

# Baseline conditions require new sims (no existing agent logs)
BASELINE_CONDITIONS = {
    "rawDerived":     ["rawSugarscape"],
    "egoistDerived":  ["egoist"],
    "altruistDerived": ["altruist"],
    "benthamDerived": ["bentham"],
}

# Biased conditions reuse the focal-action derivation CSV
BIASED_CONDITIONS = {
    "combatDerived":       "biasedCombat",
    "tradeDerived":        "biasedTrade",
    "reproductionDerived": "biasedReproduction",
    "lendingDerived":      "biasedLending",
}


# ─────────────────────────────────────────────────────────────────
# Model loading & effect vector computation
# ─────────────────────────────────────────────────────────────────

def load_coordinate_models(model_dir: str):
    models = {}
    for fname in os.listdir(model_dir):
        if fname.endswith(".pkl") and not fname.startswith("feature_scaler"):
            models[fname[:-4]] = joblib.load(os.path.join(model_dir, fname))
            
    scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        
    with open(os.path.join(model_dir, "normalization_constants.json")) as f:
        norm = json.load(f)
    return models, norm, scaler


def compute_effect_vector(state: np.ndarray, action: str,
                          models: dict, norm: dict, scaler=None) -> np.ndarray:
    """Predict E(a|s) = (I, D, C, P, X). Returns None if no model."""
    if f"intensity_{action}" not in models:
        return None
    X = state.reshape(1, -1)
    if scaler:
        X = scaler.transform(X)

    i_dist = models[f"intensity_{action}"].pred_dist(X)
    I = float(i_dist.loc[0])
    I_var = float(i_dist.scale[0] ** 2)

    D = float(np.clip(models[f"duration_{action}"].predict(X)[0], 0, 1))
    C = 1.0 / (1.0 + I_var)
    P = float(np.clip(models[f"propinquity_{action}"].predict(X)[0], 0, 1))

    ext_key = f"extent_{action}"
    if ext_key in models:
        proba = models[ext_key].predict_proba(X)[0]
        classes = models[ext_key].classes_
        Xc = float(sum(c * p for c, p in zip(classes, proba)))
    else:
        defaults = {"combat": -1.0, "trade": 1.0,
                    "reproduction": 1.0, "lending": 1.0}
        Xc = defaults.get(action, 0.0)

    return np.array([I, D, C, P, Xc])


# ─────────────────────────────────────────────────────────────────
# MaxEnt IRL (vectorized)
# ─────────────────────────────────────────────────────────────────

def maxent_irl_from_observations(chosen_vecs: np.ndarray,
                                 all_action_vecs: np.ndarray,
                                 n_iterations: int = 100,
                                 learning_rate: float = 0.02,
                                 print_progress: bool = False) -> np.ndarray:
    """Vectorized MaxEnt IRL.

    Args:
        chosen_vecs: (N, 5) — effect vectors of the chosen actions
        all_action_vecs: (N, K, 5) — effect vectors of ALL feasible
                         actions at each decision point (K = 4 usually)
        n_iterations: gradient ascent iterations
        learning_rate: step size

    Returns:
        theta: (5,) learned weight vector
    """
    N = chosen_vecs.shape[0]
    if N == 0:
        return np.zeros(5)

    # Empirical feature expectations (mean of chosen action vectors)
    emp = chosen_vecs.mean(axis=0)  # (5,)

    theta = np.zeros(5)
    for i in range(n_iterations):
        # Rewards: (N, K)
        rewards = all_action_vecs @ theta  # (N, K)

        # Softmax (numerically stable)
        rewards -= rewards.max(axis=1, keepdims=True)
        exp_r = np.exp(rewards)
        probs = exp_r / (exp_r.sum(axis=1, keepdims=True) + 1e-10)  # (N, K)

        expected = np.einsum('nk,nkj->nj', probs, all_action_vecs).mean(axis=0)

        gradient = emp - expected
        theta += learning_rate * gradient
        
        if print_progress and (i + 1) % 20 == 0:
            print(f"        [IRL Iteration {i+1:3d}/{n_iterations}] Grad Norm: {np.linalg.norm(gradient):.6f}")

    return theta


def normalize_vector(theta: np.ndarray) -> np.ndarray:
    m = np.abs(theta).max()
    return theta / m if m > 1e-10 else theta


# ─────────────────────────────────────────────────────────────────
# Build observation matrices from a DataFrame of agent records
# ─────────────────────────────────────────────────────────────────

def build_irl_matrices(df: pd.DataFrame, models: dict, norm: dict,
                       scaler=None, max_obs: int = 2000):
    """Convert DataFrame rows into vectorized IRL inputs.

    Returns (chosen_vecs, all_action_vecs) numpy arrays, or (None, None)
    if no discretionary actions found.
    """
    # Filter to discretionary actions only
    disc = df[df["action"].isin(DISCRETIONARY_ACTIONS)]
    if len(disc) == 0:
        return None, None

    # Subsample if too many (speed)
    if len(disc) > max_obs:
        disc = disc.sample(n=max_obs, random_state=42)

    chosen_list = []
    all_vecs_list = []

    for _, row in disc.iterrows():
        state = np.array([row.get(f, 0) for f in STATE_FEATURES], dtype=float)
        action = row["action"]

        chosen = compute_effect_vector(state, action, models, norm, scaler)
        if chosen is None:
            continue

        # Compute all 4 action vectors for this state
        action_vecs = []
        for a in DISCRETIONARY_ACTIONS:
            v = compute_effect_vector(state, a, models, norm, scaler)
            if v is not None:
                action_vecs.append(v)

        if len(action_vecs) == 0:
            continue

        # Pad to fixed width (4 actions) with zeros
        while len(action_vecs) < 4:
            action_vecs.append(np.zeros(5))

        chosen_list.append(chosen)
        all_vecs_list.append(np.stack(action_vecs))

    if not chosen_list:
        return None, None

    return np.array(chosen_list), np.array(all_vecs_list)


# ─────────────────────────────────────────────────────────────────
# Simulation helpers (for baseline conditions only)
# ─────────────────────────────────────────────────────────────────

def load_base_config(path: str) -> dict:
    with open(path) as f:
        full = json.load(f)
    return full.get("sugarscapeOptions", full)


def run_one_sim(args):
    cfg_path, python_alias, job_idx, total, counter, lock = args
    cmd = [python_alias, "sugarscape.py", "--conf", cfg_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with lock:
        counter.value += 1
        pct = counter.value / total * 100
        print(f"\r  [{counter.value:>3}/{total}] {pct:5.1f}%", end="", flush=True)


def safe_json_load(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def classify_action(record: dict) -> str:
    if record.get("preyKilled", False):
        return "combat"
    if record.get("tradePartners", 0) > 0:
        return "trade"
    if record.get("mates", 0) > 0:
        return "reproduction"
    if record.get("lendingPartners", 0) > 0:
        return "lending"
    return "none"


def agent_log_to_dataframe(agent_log_path: str) -> pd.DataFrame:
    """Convert a raw agent JSON log into a DataFrame matching focal-action CSV columns."""
    data = safe_json_load(agent_log_path)
    if not data:
        return pd.DataFrame()

    rows = []
    for r in data:
        rows.append({
            "agentID": r.get("ID", -1),
            "timestep": r.get("timestep", 0),
            "age": r.get("age", 0),
            "wealth": r.get("wealth", 0),
            "sugar": r.get("sugar", 0),
            "spice": r.get("spice", 0),
            "timeToLive": r.get("timeToLive", 0),
            "movement": r.get("movement", 0),
            "neighbors": r.get("neighbors", 0),
            "neighborsInTribe": r.get("neighborsInTribe", 0),
            "neighborsNotInTribe": r.get("neighborsNotInTribe", 0),
            "validMoves": r.get("validMoves", 0),
            "compositeHappiness": r.get("compositeHappiness", 0),
            "depression": int(r.get("depression", False)),
            "action": classify_action(r),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────

def main(args):
    pipeline_start = time.time()

    model_dir = args.models
    output_dir = args.output
    focal_csv = args.focal_csv
    config_path = args.config
    num_seeds = args.seeds
    timesteps = args.timesteps
    num_agents = args.agents
    num_cores = args.cores
    python_alias = args.python
    irl_iters = args.irl_iterations
    irl_lr = args.irl_lr

    os.makedirs(output_dir, exist_ok=True)
    max_cores = os.cpu_count() or 1
    if num_cores > max_cores:
        num_cores = max_cores

    print(f"\n{'='*60}")
    print(f"  Prioritization Vector Derivation (MaxEnt IRL)")
    print(f"{'='*60}")
    print(f"  Focal-action CSV:  {focal_csv}")
    print(f"  Baseline seeds:    {num_seeds}")
    print(f"  Baseline timesteps: {timesteps}")
    print(f"  IRL iterations:    {irl_iters}")
    print(f"{'='*60}\n")

    # ── 1. Load models ───────────────────────────────────────────
    print("  Loading coordinate models …")
    models, norm, scaler = load_coordinate_models(model_dir)
    print(f"    {len(models)} models loaded")
    if scaler:
        print("    Feature scaler loaded")
    print()

    all_vectors = {}
    all_thetas = {}

    # ── 2. Biased vectors from existing focal-action CSV ─────────
    print("  ── Deriving biased vectors from focal-action CSV ──\n")

    if os.path.exists(focal_csv):
        # Filter biased conditions if requested
        active_biased = BIASED_CONDITIONS
        if args.vectors:
            active_biased = {k: v for k, v in BIASED_CONDITIONS.items() if k in args.vectors}

        if active_biased:
            print(f"    Loading {focal_csv} in chunks (capping at 2640 discretionary per condition)...")
            chunks = []
            counts = {k: 0 for k in active_biased.values()}
            
            for chunk in pd.read_csv(focal_csv, chunksize=100000):
                for cond_key in counts.keys():
                    cond_chunk = chunk[(chunk["condition"] == cond_key) & (chunk["action"].isin(DISCRETIONARY_ACTIONS))]
                    needed = 2640 - counts[cond_key]
                    if needed > 0 and len(cond_chunk) > 0:
                        keep = cond_chunk.head(needed)
                        chunks.append(keep)
                        counts[cond_key] += len(keep)
                if all(c >= 2640 for c in counts.values()):
                    break
                    
            focal_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            print(f"    Loaded {len(focal_df):,} discretionary rows (chunked cap)")

            for vec_name, condition_key in active_biased.items():
                cond_df = focal_df[focal_df["condition"] == condition_key]
                disc_df = cond_df[cond_df["action"].isin(DISCRETIONARY_ACTIONS)]
                
                # Cap at 2640 to match coordinate training
                if len(disc_df) > 2640:
                    disc_df = disc_df.sample(n=2640, random_state=42)
                    
                disc_count = len(disc_df)
                print(f"\n    {vec_name}: {len(cond_df):,} total rows, "
                      f"{disc_count} discretionary used (capped at 2640)")

                if disc_count == 0:
                    print(f"      [skip] No discretionary actions")
                    continue

                t0 = time.time()
                chosen, all_vecs = build_irl_matrices(disc_df, models, norm, scaler)
                if chosen is None:
                    print(f"      [skip] Could not build matrices")
                    continue

                print(f"      IRL input: {chosen.shape[0]} observations × "
                      f"{all_vecs.shape[1]} actions\n")

                theta = maxent_irl_from_observations(
                    chosen, all_vecs, n_iterations=irl_iters, learning_rate=irl_lr, print_progress=True
                )
                p_vec = normalize_vector(theta)
                elapsed = time.time() - t0

                all_thetas[vec_name] = theta.tolist()
                all_vectors[vec_name] = p_vec.tolist()

                labels = ["I", "D", "C", "P", "X"]
                print(f"done ({elapsed:.1f}s)")
                print(f"      θ: {dict(zip(labels, [round(v, 4) for v in theta]))}")
                print(f"      P: {dict(zip(labels, [round(v, 4) for v in p_vec]))}")
        else:
            print("    [skip] No biased vectors requested")
    else:
        print(f"    [warn] Focal-action CSV not found: {focal_csv}")
        print(f"           Run 'make focal-action' first.")

    # ── 3. Baseline vectors from short derivation sims ───────────
    # Filter baseline conditions if requested
    active_baseline = BASELINE_CONDITIONS
    if args.vectors:
        active_baseline = {k: v for k, v in BASELINE_CONDITIONS.items() if k in args.vectors}

    if not active_baseline:
        print("\n  [skip] No baseline vectors requested")
    else:
        print(f"\n  ── Deriving baseline vectors ({num_seeds} seeds × "
              f"{timesteps} timesteps) ──\n")

        base_cfg = load_base_config(config_path)
        sim_dir = os.path.join(output_dir, "sim_logs")
        os.makedirs(sim_dir, exist_ok=True)

        random.seed(42)
        seeds = list(set(random.randint(0, sys.maxsize) for _ in range(num_seeds * 2)))[:num_seeds]

        # Build configs for all baseline conditions
        pending = []
        baseline_logs = {}  # cond_name -> list of agent_log_paths

        for cond_name, decision_models in active_baseline.items():
            cond_dir = os.path.join(sim_dir, cond_name)
            os.makedirs(cond_dir, exist_ok=True)
            baseline_logs[cond_name] = []
    
            for seed in seeds:
                cfg = dict(base_cfg)
                cfg["seed"] = seed
                cfg["agentDecisionModels"] = decision_models
                cfg["timesteps"] = timesteps
                cfg["startingAgents"] = num_agents
                cfg["startingDiseases"] = 0
                cfg["headlessMode"] = True
                cfg["debugMode"] = ["none"]
                cfg["keepAlivePostExtinction"] = False
                cfg["keepAliveAtEnd"] = False
                cfg["screenshots"] = False
                cfg["profileMode"] = False
                log_path = os.path.join(cond_dir, f"{cond_name}_{seed}.json")
                agent_log = os.path.join(cond_dir, f"{cond_name}_{seed}_agents.json")
                cfg["logfile"] = log_path
                cfg["agentLogfile"] = agent_log
                cfg["logfileFormat"] = "json"
    
                cfg_path = os.path.join(cond_dir, f"{cond_name}_{seed}.config")
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f)
    
                baseline_logs[cond_name].append(agent_log)
    
                # Skip if already completed
                if os.path.exists(agent_log):
                    existing = safe_json_load(agent_log)
                    if existing and len(existing) > 0:
                        continue
                pending.append(cfg_path)

    print(f"    Baseline sims: {len(seeds) * len(BASELINE_CONDITIONS)} total, "
          f"{len(pending)} need to run")

    if pending:
        manager = multiprocessing.Manager()
        counter = manager.Value('i', 0)
        lock = manager.Lock()
        worker_args = [
            (p, python_alias, i, len(pending), counter, lock)
            for i, p in enumerate(pending)
        ]
        print(f"    Running {len(pending)} simulations ({num_cores} cores) …")
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.map(run_one_sim, worker_args)
        print()

    # Derive baseline vectors
    for cond_name in BASELINE_CONDITIONS:
        print(f"\n    {cond_name}:")
        frames = []
        for agent_log in baseline_logs[cond_name]:
            df = agent_log_to_dataframe(agent_log)
            if len(df) > 0:
                frames.append(df)

        if not frames:
            print(f"      [skip] No data")
            continue

        cond_df = pd.concat(frames, ignore_index=True)
        disc_df = cond_df[cond_df["action"].isin(DISCRETIONARY_ACTIONS)]
        
        # Cap at 2640 to match coordinate training
        if len(disc_df) > 2640:
            disc_df = disc_df.sample(n=2640, random_state=42)
            
        disc_count = len(disc_df)
        print(f"      {len(cond_df):,} total rows, {disc_count} discretionary used (capped at 2640)")

        t0 = time.time()
        chosen, all_vecs = build_irl_matrices(disc_df, models, norm, scaler)
        if chosen is None:
            print(f"      [skip] No discretionary actions")
            continue

        print(f"      IRL: {chosen.shape[0]} obs\n")
        theta = maxent_irl_from_observations(
            chosen, all_vecs, n_iterations=irl_iters, learning_rate=irl_lr, print_progress=True
        )
        p_vec = normalize_vector(theta)
        elapsed = time.time() - t0

        all_thetas[cond_name] = theta.tolist()
        all_vectors[cond_name] = p_vec.tolist()

        labels = ["I", "D", "C", "P", "X"]
        print(f"done ({elapsed:.1f}s)")
        print(f"      θ: {dict(zip(labels, [round(v, 4) for v in theta]))}")
        print(f"      P: {dict(zip(labels, [round(v, 4) for v in p_vec]))}")

    # ── 4. Save ──────────────────────────────────────────────────
    vectors_path = os.path.join(output_dir, "prioritization_vectors.json")
    output_data = {
        "description": "Prioritization vectors derived via MaxEnt IRL",
        "coordinate_labels": ["intensity", "duration", "certainty",
                              "propinquity", "extent"],
        "vectors": all_vectors,
        "raw_thetas": all_thetas,
        "derivation_config": {
            "baseline_seeds": num_seeds,
            "baseline_timesteps": timesteps,
            "baseline_agents": num_agents,
            "irl_iterations": irl_iters,
            "irl_learning_rate": irl_lr,
            "focal_csv": focal_csv,
        },
    }
    with open(vectors_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # ── 5. Summary ───────────────────────────────────────────────
    elapsed = round(time.time() - pipeline_start, 2)

    print(f"\n{'='*60}")
    print(f"  {'Condition':<25} {'I':>8} {'D':>8} {'C':>8} {'P':>8} {'X':>8}")
    print(f"  {'-'*57}")
    for name, vec in all_vectors.items():
        vals = [f"{v:>8.4f}" for v in vec]
        print(f"  {name:<25} {''.join(vals)}")
    print(f"{'='*60}")
    print(f"  Saved → {vectors_path}")
    print(f"  {len(all_vectors)} vectors derived in {elapsed:.1f}s\n")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Derive prioritization vectors via MaxEnt IRL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c", "--config", default="config.json")
    p.add_argument("-m", "--models", default="fvdm_models")
    p.add_argument("-o", "--output", default="fvdm_vectors")
    p.add_argument("-f", "--focal-csv",
                   default="focal_action_results/results/focal_action_derivation.csv",
                   help="Path to existing focal-action derivation CSV.")
    p.add_argument("-s", "--seeds", type=int, default=3,
                   help="Seeds for baseline derivation sims.")
    p.add_argument("-a", "--agents", type=int, default=500)
    p.add_argument("-t", "--timesteps", type=int, default=1000,
                   help="Timesteps for baseline derivation sims.")
    p.add_argument("-j", "--cores", type=int, default=1)
    p.add_argument("--irl-iterations", type=int, default=100)
    p.add_argument("--irl-lr", type=float, default=0.02)
    p.add_argument("-v", "--vectors", nargs="+",
                   help="List of specific prioritization vectors to derive (e.g. combatDerived tradeDerived).")
    p.add_argument("--python", default="python3")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
