#!/usr/bin/env python3
"""
derive_vectors.py
-----------------
Prioritization Vector Derivation via Maximum Entropy IRL.

Derives the prioritization vectors P_i = (p_I, p_D, p_C, p_P, p_X) for
each baseline ethical condition by applying MaxEnt IRL to behavioral
trajectories.  The learned weight vector θ from R(s,a) = θᵀ E(a|s)
becomes the prioritization vector for that condition.

Vectors derived (per thesis §3.4.5 and §3.5):
  1. Raw Derived       — from rawSugarscape baseline trajectories
  2. Egoist Derived     — from egoist baseline trajectories
  3. Altruist Derived   — from altruist baseline trajectories
  4. Bentham Derived    — from bentham baseline trajectories
  5. Focus Derived      — from pooled biased focal-action trajectories
  6. Combat Derived     — from biasedCombat trajectories only
  7. Trade Derived      — from biasedTrade trajectories only
  8. Reproduction Derived — from biasedReproduction trajectories only

Pipeline:
  1. Run short derivation simulations WITH agent logs for each condition
  2. Load trained coordinate models from train_coordinates.py
  3. Compute felicific effect vectors for each (state, action) observation
  4. Apply MaxEnt IRL to learn θ per condition
  5. Normalize θ → prioritization vector P_i
  6. Save all vectors to JSON

Reference: Thesis Section 3.4.5 — Derivation of Prioritization Vectors.

Usage:
  .venv/bin/python derive_vectors.py [options]
"""

import argparse
import json
import math
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

# Conditions to derive vectors for.
# Key = vector name, Value = list of decisionModel strings for simulation.
DERIVATION_CONDITIONS = {
    "rawDerived":           ["rawSugarscape"],
    "egoistDerived":        ["egoist"],
    "altruistDerived":      ["altruist"],
    "benthamDerived":       ["bentham"],
    "combatDerived":        ["biasedCombat"],
    "tradeDerived":         ["biasedTrade"],
    "reproductionDerived":  ["biasedReproduction"],
    "lendingDerived":       ["biasedLending"],
}


# ─────────────────────────────────────────────────────────────────
# Simulation helpers (reused from run_focal_action.py)
# ─────────────────────────────────────────────────────────────────

def load_base_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        full = json.load(f)
    return full.get("sugarscapeOptions", full)


def make_derivation_config(base: dict, seed: int, decision_models: list,
                           timesteps: int, num_agents: int, output_dir: str,
                           condition_name: str) -> dict:
    cfg = dict(base)
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
    cfg["logfile"] = os.path.join(output_dir, f"{condition_name}_{seed}.json")
    cfg["agentLogfile"] = os.path.join(
        output_dir, f"{condition_name}_{seed}_agents.json"
    )
    cfg["logfileFormat"] = "json"
    return cfg


def write_run_config(cfg: dict, path: str):
    with open(path, "w") as f:
        json.dump(cfg, f)


def run_one_simulation(args):
    config_path, python_alias, job_idx, total_jobs, counter, lock = args
    cmd = [python_alias, "sugarscape.py", "--conf", config_path]
    start = time.time()
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    duration = time.time() - start
    with lock:
        counter.value += 1
        pct = (counter.value / total_jobs) * 100
        print(f"\r  [{counter.value:>4}/{total_jobs}]  {pct:5.1f}%  "
              f"last: {os.path.basename(config_path):<50}",
              end="", flush=True)
    return config_path, duration


def safe_json_load(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# Action classification (same as run_focal_action.py)
# ─────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────
# Felicific effect vector computation using trained models
# ─────────────────────────────────────────────────────────────────

def load_coordinate_models(model_dir: str) -> dict:
    """Load all trained .pkl models and normalization constants."""
    models = {}
    for fname in os.listdir(model_dir):
        if fname.endswith(".pkl"):
            name = fname[:-4]  # strip .pkl
            models[name] = joblib.load(os.path.join(model_dir, fname))

    norm_path = os.path.join(model_dir, "normalization_constants.json")
    with open(norm_path, "r") as f:
        norm_constants = json.load(f)

    return models, norm_constants


def compute_effect_vector(state_features: np.ndarray, action: str,
                          models: dict, norm_constants: dict) -> np.ndarray:
    """Predict E(a|s_i) = (I, D, C, P, X) for one state-action pair.

    Returns a 5-element numpy array, or None if the action has no model.
    """
    if f"intensity_{action}" not in models:
        return None

    X = state_features.reshape(1, -1)

    # Intensity
    i_dist = models[f"intensity_{action}"].pred_dist(X)
    I_pred = float(i_dist.loc[0])
    I_var = float(i_dist.scale[0] ** 2)

    # Duration
    D_pred = float(models[f"duration_{action}"].predict(X)[0])
    D_pred = max(0.0, min(1.0, D_pred))

    # Certainty (from Intensity variance)
    C_pred = 1.0 / (1.0 + I_var)

    # Propinquity
    P_pred = float(models[f"propinquity_{action}"].predict(X)[0])
    P_pred = max(0.0, min(1.0, P_pred))

    # Extent
    if f"extent_{action}" in models:
        ext_model = models[f"extent_{action}"]
        proba = ext_model.predict_proba(X)[0]
        classes = ext_model.classes_
        X_pred = float(sum(c * p for c, p in zip(classes, proba)))
    else:
        # Default extent by action type
        extent_defaults = {"combat": -1.0, "trade": 1.0,
                           "reproduction": 1.0, "lending": 1.0}
        X_pred = extent_defaults.get(action, 0.0)

    return np.array([I_pred, D_pred, C_pred, P_pred, X_pred])


# ─────────────────────────────────────────────────────────────────
# MaxEnt IRL
# ─────────────────────────────────────────────────────────────────

def maxent_irl(trajectories: list, n_features: int = 5,
               learning_rate: float = 0.01, n_iterations: int = 200,
               verbose: bool = False) -> np.ndarray:
    """Maximum Entropy Inverse Reinforcement Learning.

    Given a set of expert trajectories, each containing (state_features,
    action, effect_vector) tuples, learn the weight vector θ such that
    R(s,a) = θᵀ E(a|s) best explains the expert's behavior.

    Algorithm (Ziebart et al., 2008):
      1. Compute empirical feature expectations from expert trajectories
      2. Iteratively update θ to maximize the likelihood of observed
         actions under the MaxEnt policy

    Args:
        trajectories: list of lists, each inner list contains dicts with
                      'effect_vector' (np.array of length 5) and
                      'all_effect_vectors' (list of np.arrays for all
                      feasible actions)
        n_features: dimensionality of the effect vector (5 for FVDM)
        learning_rate: gradient ascent step size
        n_iterations: number of optimization iterations

    Returns:
        theta: learned weight vector (length n_features)
    """
    # ── Step 1: Empirical feature expectations ──
    # Average effect vector of the CHOSEN actions across all timesteps
    feature_sum = np.zeros(n_features)
    n_steps = 0
    for traj in trajectories:
        for step in traj:
            if step["effect_vector"] is not None:
                feature_sum += step["effect_vector"]
                n_steps += 1

    if n_steps == 0:
        return np.zeros(n_features)

    empirical_expectations = feature_sum / n_steps

    # ── Step 2: Gradient ascent on θ ──
    theta = np.zeros(n_features)

    for iteration in range(n_iterations):
        # Compute expected feature counts under current θ
        expected_features = np.zeros(n_features)
        n_decisions = 0

        for traj in trajectories:
            for step in traj:
                all_vecs = step.get("all_effect_vectors", [])
                if len(all_vecs) == 0:
                    continue

                # Compute softmax policy: P(a|s) ∝ exp(θᵀ E(a|s))
                rewards = np.array([theta.dot(v) for v in all_vecs])
                # Numerical stability
                rewards -= rewards.max()
                exp_rewards = np.exp(rewards)
                probs = exp_rewards / (exp_rewards.sum() + 1e-10)

                # Expected feature = Σ P(a|s) · E(a|s)
                for prob, vec in zip(probs, all_vecs):
                    expected_features += prob * vec
                n_decisions += 1

        if n_decisions > 0:
            expected_features /= n_decisions

        # Gradient = empirical - expected
        gradient = empirical_expectations - expected_features
        theta += learning_rate * gradient

        if verbose and (iteration + 1) % 50 == 0:
            grad_norm = np.linalg.norm(gradient)
            print(f"      Iteration {iteration+1}/{n_iterations}: "
                  f"||gradient|| = {grad_norm:.6f}")

    return theta


def normalize_to_prioritization_vector(theta: np.ndarray) -> np.ndarray:
    """Normalize θ to produce a prioritization vector on [-1, 1]^5.

    Each component is divided by the maximum absolute value so the
    strongest preference dimension has magnitude 1.
    """
    max_abs = np.abs(theta).max()
    if max_abs < 1e-10:
        return theta  # all zeros — no meaningful preference
    return theta / max_abs


# ─────────────────────────────────────────────────────────────────
# Trajectory construction
# ─────────────────────────────────────────────────────────────────

def build_trajectories_from_agent_log(agent_log_path: str,
                                      models: dict,
                                      norm_constants: dict) -> list:
    """Build MaxEnt IRL trajectories from an agent-level simulation log.

    Each trajectory corresponds to one agent's lifetime. Each step
    contains the effect vector of the chosen action and the effect
    vectors of ALL feasible discretionary actions (for the MaxEnt
    softmax denominator).
    """
    data = safe_json_load(agent_log_path)
    if data is None or len(data) == 0:
        return []

    # Group records by agent ID to form per-agent trajectories
    agent_records = {}
    for record in data:
        agent_id = record.get("ID", -1)
        if agent_id not in agent_records:
            agent_records[agent_id] = []
        agent_records[agent_id].append(record)

    trajectories = []
    for agent_id, records in agent_records.items():
        traj = []
        for record in records:
            action = classify_action(record)
            state = np.array([
                record.get(f, 0) for f in STATE_FEATURES
            ], dtype=float)

            # Compute effect vector for chosen action
            if action in DISCRETIONARY_ACTIONS:
                chosen_vec = compute_effect_vector(
                    state, action, models, norm_constants
                )
            else:
                chosen_vec = None

            # Compute effect vectors for ALL feasible discretionary actions
            # Feasibility heuristic from the log data:
            #   combat: preyKilled or aggressionFactor > 0 means it was feasible
            #   trade:  tradePartners >= 0 means trade was attempted
            #   reproduction: mates >= 0
            #   lending: lendingPartners >= 0
            # Since we can't know exact feasibility from logs, we compute
            # vectors for all 4 discretionary actions as candidates.
            all_vecs = []
            for a in DISCRETIONARY_ACTIONS:
                vec = compute_effect_vector(state, a, models, norm_constants)
                if vec is not None:
                    all_vecs.append(vec)

            traj.append({
                "action": action,
                "effect_vector": chosen_vec,
                "all_effect_vectors": all_vecs,
            })

        if len(traj) > 0:
            trajectories.append(traj)

    return trajectories


# ─────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────

def main(args):
    config_path = args.config
    model_dir = args.models
    output_dir = args.output
    num_seeds = args.seeds
    num_agents = args.agents
    timesteps = args.timesteps
    num_cores = args.cores
    python_alias = args.python
    irl_iterations = args.irl_iterations
    irl_lr = args.irl_lr

    os.makedirs(output_dir, exist_ok=True)

    max_cores = os.cpu_count() or 1
    if num_cores > max_cores:
        num_cores = max_cores

    pipeline_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Prioritization Vector Derivation (MaxEnt IRL)")
    print(f"{'='*60}")
    print(f"  Conditions:       {list(DERIVATION_CONDITIONS.keys())}")
    print(f"  Seeds:            {num_seeds}")
    print(f"  Timesteps:        {timesteps}")
    print(f"  Agents:           {num_agents}")
    print(f"  Model dir:        {model_dir}")
    print(f"  IRL iterations:   {irl_iterations}")
    print(f"  IRL learning rate: {irl_lr}")
    print(f"{'='*60}\n")

    # ── 1. Load trained coordinate models ────────────────────────
    print("  Loading trained coordinate models …")
    models, norm_constants = load_coordinate_models(model_dir)
    print(f"    Loaded {len(models)} models + normalization constants\n")

    # ── 2. Run derivation simulations ────────────────────────────
    base_cfg = load_base_config(config_path)
    base_cfg["timesteps"] = timesteps
    base_cfg["startingAgents"] = num_agents

    sim_dir = os.path.join(output_dir, "sim_logs")
    os.makedirs(sim_dir, exist_ok=True)

    random.seed(42)
    seeds = []
    while len(seeds) < num_seeds:
        s = random.randint(0, sys.maxsize)
        if s not in seeds:
            seeds.append(s)

    all_configs = {}  # condition -> list of (seed, cfg_path, agent_log_path)

    for cond_name, decision_models in DERIVATION_CONDITIONS.items():
        cond_dir = os.path.join(sim_dir, cond_name)
        os.makedirs(cond_dir, exist_ok=True)
        all_configs[cond_name] = []

        for seed in seeds:
            cfg = make_derivation_config(
                base=base_cfg, seed=seed,
                decision_models=decision_models,
                timesteps=timesteps, num_agents=num_agents,
                output_dir=cond_dir, condition_name=cond_name,
            )
            cfg_path = os.path.join(cond_dir, f"{cond_name}_{seed}.config")
            write_run_config(cfg, cfg_path)
            all_configs[cond_name].append((
                seed, cfg_path, cfg["agentLogfile"]
            ))

    # Filter already-completed runs
    pending = []
    for cond_name, configs in all_configs.items():
        for (seed, cfg_path, agent_log_path) in configs:
            if os.path.exists(agent_log_path):
                data = safe_json_load(agent_log_path)
                if data and len(data) > 0:
                    continue
            pending.append((cond_name, seed, cfg_path, agent_log_path))

    total_jobs = sum(len(v) for v in all_configs.values())
    skip_count = total_jobs - len(pending)
    print(f"  Total derivation runs:    {total_jobs}")
    print(f"  Already completed:        {skip_count}")
    print(f"  Queued to run:            {len(pending)}\n")

    if pending:
        start_time = time.time()
        manager = multiprocessing.Manager()
        counter = manager.Value('i', 0)
        lock = manager.Lock()

        worker_args = [
            (cfg_path, python_alias, i, len(pending), counter, lock)
            for i, (_, _, cfg_path, _) in enumerate(pending)
        ]

        print(f"  Running derivation simulations using {num_cores} core(s) …")
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.map(run_one_simulation, worker_args)
        elapsed = time.time() - start_time
        print(f"\n\n  Simulations completed in {elapsed:.1f}s\n")

    # ── 3. Build trajectories and derive vectors ─────────────────
    print("  Building trajectories and deriving prioritization vectors …\n")

    all_vectors = {}
    all_thetas = {}

    for cond_name, configs in all_configs.items():
        print(f"  ── {cond_name} ──")
        cond_start = time.time()

        # Build trajectories from all seeds
        all_trajectories = []
        total_steps = 0
        for (seed, cfg_path, agent_log_path) in configs:
            trajs = build_trajectories_from_agent_log(
                agent_log_path, models, norm_constants
            )
            all_trajectories.extend(trajs)
            for t in trajs:
                total_steps += len(t)

        print(f"    Trajectories: {len(all_trajectories)} agents, "
              f"{total_steps} total steps")

        if len(all_trajectories) == 0:
            print(f"    [skip] No trajectory data available")
            continue

        # Filter to trajectories that have at least one discretionary action
        meaningful_trajs = []
        for traj in all_trajectories:
            has_action = any(
                step["effect_vector"] is not None for step in traj
            )
            if has_action:
                meaningful_trajs.append(traj)

        print(f"    With discretionary actions: {len(meaningful_trajs)} agents")

        if len(meaningful_trajs) == 0:
            print(f"    [skip] No discretionary actions in trajectories")
            continue

        # Run MaxEnt IRL
        print(f"    Running MaxEnt IRL ({irl_iterations} iterations) …",
              end=" ", flush=True)
        theta = maxent_irl(
            meaningful_trajs,
            n_features=5,
            learning_rate=irl_lr,
            n_iterations=irl_iterations,
            verbose=False,
        )
        print("✓")

        # Normalize to prioritization vector
        p_vec = normalize_to_prioritization_vector(theta)

        cond_elapsed = round(time.time() - cond_start, 2)
        all_thetas[cond_name] = theta.tolist()
        all_vectors[cond_name] = p_vec.tolist()

        labels = ["I", "D", "C", "P", "X"]
        print(f"    θ (raw):  {dict(zip(labels, [round(v, 4) for v in theta]))}")
        print(f"    P (norm): {dict(zip(labels, [round(v, 4) for v in p_vec]))}")
        print(f"    Elapsed:  {cond_elapsed:.2f}s\n")

    # ── 4. Handle Focus Derived (pooled from all biased conditions) ──
    # The focusDerived vector was already handled above since we
    # configured it to run all 4 biased decision models together.
    # If it ran from separate agent logs (one per bias), the
    # trajectories are already pooled in the loop above.

    # ── 5. Save vectors ──────────────────────────────────────────
    vectors_path = os.path.join(output_dir, "prioritization_vectors.json")
    output_data = {
        "description": "Prioritization vectors derived via MaxEnt IRL",
        "coordinate_labels": ["intensity", "duration", "certainty",
                              "propinquity", "extent"],
        "vectors": all_vectors,
        "raw_thetas": all_thetas,
        "derivation_config": {
            "seeds": num_seeds,
            "timesteps": timesteps,
            "agents": num_agents,
            "irl_iterations": irl_iterations,
            "irl_learning_rate": irl_lr,
        },
    }
    with open(vectors_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"  Saved vectors → {vectors_path}")

    # ── 6. Print summary table ───────────────────────────────────
    pipeline_elapsed = round(time.time() - pipeline_start, 2)

    print(f"\n{'='*60}")
    print(f"  {'Condition':<25} {'I':>8} {'D':>8} {'C':>8} {'P':>8} {'X':>8}")
    print(f"  {'-'*57}")
    for cond_name, vec in all_vectors.items():
        vals = [f"{v:>8.4f}" for v in vec]
        print(f"  {cond_name:<25} {''.join(vals)}")
    print(f"{'='*60}")
    print(f"  Total elapsed time: {pipeline_elapsed:.2f}s")
    print(f"  {len(all_vectors)} vectors derived.\n")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive prioritization vectors via MaxEnt IRL for FVDM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", default="config.json",
        help="Path to master simulation config.",
    )
    parser.add_argument(
        "-m", "--models", default="fvdm_models",
        help="Directory containing trained coordinate models (.pkl).",
    )
    parser.add_argument(
        "-o", "--output", default="fvdm_vectors",
        help="Directory to save derivation outputs.",
    )
    parser.add_argument(
        "-s", "--seeds", type=int, default=10,
        help="Number of seeds per derivation condition.",
    )
    parser.add_argument(
        "-a", "--agents", type=int, default=250,
        help="Starting agents per derivation run.",
    )
    parser.add_argument(
        "-t", "--timesteps", type=int, default=5000,
        help="Timesteps per derivation run.",
    )
    parser.add_argument(
        "-j", "--cores", type=int, default=1,
        help="Parallel CPU cores for simulations.",
    )
    parser.add_argument(
        "--irl-iterations", type=int, default=200,
        help="Number of MaxEnt IRL gradient ascent iterations.",
    )
    parser.add_argument(
        "--irl-lr", type=float, default=0.01,
        help="MaxEnt IRL learning rate.",
    )
    parser.add_argument(
        "--python", default="python3",
        help="Python interpreter alias.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
