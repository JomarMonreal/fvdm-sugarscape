import sugarscape
import json
import os
import multiprocessing
import random
import math
import numpy as np
from functools import partial

def iter_batches(file_list, batch_size):
    for start_idx in range(0, len(file_list), batch_size):
        batch_files = file_list[start_idx:start_idx + batch_size]
        batch_data = []

        for filename in batch_files:
            filepath = os.path.join("observations", filename)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    action_obs = json.load(f)

                if not isinstance(action_obs, list):
                    print(f"  Warning: Skipping {filename} (not a valid observation list)")
                    continue

                batch_data.append((filename, action_obs))

            except (json.JSONDecodeError, ValueError, UnicodeDecodeError, OSError) as e:
                print(f"  Warning: Skipping {filename} (load error: {e})")
                continue

            except Exception as e:
                print(f"  Warning: Skipping {filename} (unexpected error: {e})")
                continue

        yield batch_data

class LinearRegressor:
    """Parametric Linear Regression using NumPy to capture OLS weights and Inverse Covariance."""
    def __init__(self):
        self.weights = None # beta
        self.inv_covariance = None # (X^T X)^-1
        self.mse = 0
        self.scaler = {"mean": [], "std": []}

    def fit(self, X, y):
        if not X: return
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        # 1. Scaling (Z-score)
        self.scaler["mean"] = X.mean(axis=0).tolist()
        self.scaler["std"] = X.std(axis=0).tolist()
        # Handle zero variance features
        std_safe = np.array(self.scaler["std"])
        std_safe[std_safe < 1e-12] = 1.0
        
        X_scaled = (X - np.array(self.scaler["mean"])) / std_safe
        
        # 2. Add intercept (1.0)
        X_design = np.column_stack([np.ones(n_samples), X_scaled])
            
        # 3. Normal Equation: beta = (X^T X)^-1 X^T y
        XTX = X_design.T @ X_design
        # Add small regularization to diagonal to prevent singularity
        XTX += np.eye(XTX.shape[0]) * 1e-6
            
        self.inv_covariance = np.linalg.inv(XTX)
        self.weights = self.inv_covariance @ X_design.T @ y
        
        # 4. MSE Calculation
        predictions = X_design @ self.weights
        self.mse = float(np.mean((y - predictions)**2))

    def to_dict(self):
        return {
            "weights": self.weights.tolist(),
            "inv_cov": self.inv_covariance.tolist(),
            "mse": self.mse,
            "scaler": self.scaler
        }

def worker(seed, config_path, fvdm_H, steps=200, burn_in=50, num_agents=None):
    conf = {}
    if os.path.exists("config.json"):
        with open("config.json", 'r') as f:
            try:
                data = json.load(f)
                conf = data.get("sugarscapeOptions", {})
            except: pass

    with open(config_path, 'r') as f:
        overrides = json.load(f)
        conf.update(overrides)
    
    if num_agents is not None:
        conf["startingAgents"] = num_agents
    
    conf["seed"] = seed
    conf["headlessMode"] = True # Force headless for parallel runs
    
    s = sugarscape.Sugarscape(conf)
    s.fvdm_observation_active = True
    s.fvdm_H = fvdm_H
    
    # Run simulation
    # Total steps = burn_in + observation_steps + fvdm_H (to resolve last emitted event)
    total_steps = burn_in + steps + fvdm_H
    
    for t in range(total_steps):
        # Burn-in: don't start recording until t >= burn_in
        # Observation: stop recording after steps
        if burn_in <= t < (burn_in + steps):
            s.fvdm_observation_active = True
        else:
            s.fvdm_observation_active = False
            
        s.doTimestep()

    return seed, s.fvdm_dataset

def calculate_coordinates(raw_obs):
    """
    Computes [I, D, C, P, X] from raw wealth snapshots.
    """
    w0_i, w1_i = raw_obs["w_ind_0"], raw_obs["w_ind_1"]
    wH_i = raw_obs["w_ind_H"]
    w0_p, w1_p = raw_obs["w_pop_0"], raw_obs["w_pop_1"]
    wH_p = raw_obs["w_pop_H"]
    
    # Safety check: ensure all wealth snapshots were resolved
    if any(v is None for v in [w1_i, wH_i, w1_p, wH_p]):
        return None
    
    eps = 0.001
    
    # Raw Deltas (normalized by initial individual wealth)
    denom = w0_i + 1 # Use +1 to avoid division by zero and dampen small wealth effects
    
    ind_imm = (w1_i - w0_i) / denom
    ind_fut = (wH_i - w0_i) / denom
    pop_imm = (w1_p - w0_p) / (w0_p + 1)
    pop_fut = (wH_p - w0_p) / (w0_p + 1)
    
    # 1. Intensity (I) - Clipping to [-1, 1]
    intensity = max(-1.0, min(1.0, ind_imm))
    
    # 2. Duration (D) - Persistence of effect
    # Defined as the average absolute delta at horizon H
    duration = max(0.0, min(1.0, (abs(ind_fut) + abs(pop_fut)) / 2))

    # 4. Propinquity (P) - Fraction of effect that is immediate
    total_abs_effect = abs(ind_imm) + abs(ind_fut) + eps
    propinquity = max(0.0, min(1.0, abs(ind_imm) / total_abs_effect))
    
    # 5. Extent (X) - Social spread (Directional)
    ind_delta = w1_i - w0_i
    pop_delta = w1_p - w0_p
    others_delta = pop_delta - ind_delta
    
    eps_val = 1e-4
    
    if ind_delta > eps_val and others_delta <= eps_val:
        # Benefits self, neutral/harms others (MOVE, COMBAT)
        extent = -1.0
    elif others_delta > eps_val and ind_delta <= eps_val:
        # Benefits others, neutral/harms self (MATE)
        extent = 1.0
    elif ind_delta > eps_val and others_delta > eps_val:
        # Mutual benefit (TRADE, CREDIT)
        extent = 0.0
    else:
        # Neutral or mutually destructive
        extent = 0.0
    
    return {
        "I": round(intensity, 4),
        "D": round(duration, 4),
        "P": round(propinquity, 4),
        "X": round(extent, 4),
        "action": raw_obs["action"],
        "s_i": raw_obs["s_i"]
    }

def collect_worker(args, fvdm_H, steps, burn_in, num_agents=None):
    seed, config_path = args
    action_name = config_path.split("_")[-1].split(".")[0].upper()
    _, seed_data = worker(seed, config_path, fvdm_H, steps, burn_in, num_agents=num_agents)
    return action_name, seed, seed_data

def collect_observations(num_seeds=5, demo=False, processes=None, num_agents=None):
    # 1. Load H*
    fvdm_H = 23 # Default
    if os.path.exists("horizon.json"):
        with open("horizon.json", 'r') as f:
            fvdm_H = json.load(f).get("H_star", 23)
    
    print(f"Using Evaluation Horizon H* = {fvdm_H}")
    
    bias_configs = [
        "configs/bias_move.json",
        "configs/bias_combat.json",
        "configs/bias_trade.json",
        "configs/bias_mate.json",
        "configs/bias_credit.json",
        "configs/bias_tagging.json"
    ]
    
    observation_steps = 50 if demo else 200
    burn_in = 20 if demo else 50
    
    all_tasks = []
    for config_path in bias_configs:
        seeds = [random.randint(0, 1000000) for _ in range(num_seeds)]
        for seed in seeds:
            all_tasks.append((seed, config_path))
    
    if processes is None:
        cpu_count = multiprocessing.cpu_count()
        num_workers = min(len(all_tasks), max(1, cpu_count - 1))
    else:
        num_workers = processes
        
    print(f"Using {num_workers} processes to collect {len(all_tasks)} observations across {len(bias_configs)} action types.")
    
    if not os.path.exists("observations"):
        os.makedirs("observations")
    
    pool = multiprocessing.Pool(processes=num_workers)
    func = partial(collect_worker, fvdm_H=fvdm_H, steps=observation_steps, burn_in=burn_in, num_agents=num_agents)
    
    completed = 0
    for action_name, seed, seed_data in pool.imap_unordered(func, all_tasks):
        output_file = f"observations/{action_name}_seed_{seed}.json"
        with open(output_file, 'w') as f:
            json.dump(seed_data, f)
        completed += 1
        if completed % 10 == 0 or completed == len(all_tasks):
            print(f"Progress: {completed}/{len(all_tasks)} simulations completed...")
            
    pool.close()
    pool.join()

def train_models(num_batches=1):
    # 2. Training Phase: Estimate Coordinate Functions via Linear Regression
    print("Training Phase: Estimate Coordinate Functions via Linear Regression...")
    
    if not os.path.exists("observations"):
        print("No observations found. Run collection phase first.")
        return
        
    bias_configs = [
        "configs/bias_move.json",
        "configs/bias_combat.json",
        "configs/bias_trade.json",
        "configs/bias_mate.json",
        "configs/bias_credit.json",
        "configs/bias_tagging.json"
    ]
    
    # Gather all observation files
    obs_files = sorted([f for f in os.listdir("observations") if f.endswith(".json")])
    total_files = len(obs_files)

    batch_size = math.ceil(total_files / num_batches)
    print(f"Processing {total_files} files in {num_batches} batch(es) (batch size ~{batch_size})...")

    # Accumulators per action: store running sums for Normal Equation components
    # For OLS: beta = (X^T X)^-1 (X^T y)
    # We accumulate X^T X and X^T y across batches, then solve at the end
    action_names = [c.split("_")[-1].split(".")[0].upper() for c in bias_configs]
    coords_to_train = ["I", "D", "P", "X"]
    
    # We need to know the feature dimension. It's 10 local state vars + 1 intercept = 11
    n_features = 11  # 10 state vars + intercept
    
    # Accumulators: per action, per coord
    accum = {}
    scaler_accum = {}  # For computing global mean/std
    for action in action_names:
        accum[action] = {}
        scaler_accum[action] = {"raw_X": [], "count": 0}
        for c in coords_to_train:
            accum[action][c] = {
                "XTX": np.zeros((n_features, n_features)),
                "XTy": np.zeros(n_features),
                "sum_y2": 0.0,
                "sum_y": 0.0,
                "n": 0
            }
    
    skipped_files = 0
    total_obs = 0
    
    # Pass 1: Collect raw X values across all batches to compute global scaler
    print("Pass 1: Computing global scalers...")
    for batch_idx, batch_data in enumerate(iter_batches(obs_files, batch_size)):
        print(f"  Scaler batch {batch_idx + 1}/{num_batches} ({len(batch_data)} loaded files)...")

        for filename, action_obs in batch_data:
            for raw in action_obs:
                coords = calculate_coordinates(raw)
                if coords:
                    action = coords["action"]
                    if action in scaler_accum:
                        scaler_accum[action]["raw_X"].append(coords["s_i"])
                        scaler_accum[action]["count"] += 1
                        total_obs += 1
    
    # Compute scalers
    scalers = {}
    for action in action_names:
        if scaler_accum[action]["count"] > 0:
            X_all = np.array(scaler_accum[action]["raw_X"])
            scalers[action] = {
                "mean": X_all.mean(axis=0).tolist(),
                "std": X_all.std(axis=0).tolist()
            }
        else:
            scalers[action] = {"mean": [0.0] * 10, "std": [1.0] * 10}
    
    # Free memory from scaler accumulation
    del scaler_accum
    
    print(f"  Total valid observations: {total_obs}")
    if skipped_files > 0:
        print(f"  Skipped {skipped_files} corrupted file(s).")
    
    # Pass 2: Accumulate X^T X and X^T y using the computed scalers
    print("Pass 2: Accumulating regression components...")

    for batch_idx, batch_data in enumerate(iter_batches(obs_files, batch_size)):
        print(f"  Training batch {batch_idx + 1}/{num_batches} ({len(batch_data)} loaded files)...")

        for filename, action_obs in batch_data:
            for raw in action_obs:
                coords = calculate_coordinates(raw)
                if not coords:
                    continue

                action = coords["action"]
                if action not in accum:
                    continue

                x = np.array(coords["s_i"])
                mean = np.array(scalers[action]["mean"])
                std = np.array(scalers[action]["std"])
                safe_std = np.where(np.array(std) < 1e-12, 1.0, std)

                x_scaled = (x - mean) / safe_std
                x_design = np.insert(x_scaled, 0, 1.0)

                for c in coords_to_train:
                    y_val = coords[c]
                    a = accum[action][c]

                    a["XTX"] += np.outer(x_design, x_design)
                    a["XTy"] += x_design * y_val
                    a["sum_y2"] += y_val ** 2
                    a["sum_y"] += y_val
                    a["n"] += 1

    # Pass 3: Solve for weights
    print("Solving regression models...")
    fvdm_models = {}
    
    for action in action_names:
        action_models = {}
        has_data = False
        
        for c in coords_to_train:
            a = accum[action][c]
            if a["n"] == 0:
                continue
            has_data = True
            
            XTX = a["XTX"] + np.eye(n_features) * 1e-6  # Regularization
            inv_cov = np.linalg.inv(XTX)
            weights = inv_cov @ a["XTy"]
            
            # MSE = (sum(y^2) - 2 * w^T * X^T * y + w^T * X^T * X * w) / n
            mse = (a["sum_y2"] - 2 * weights @ a["XTy"] + weights @ XTX @ weights) / a["n"]
            
            action_models[c] = {
                "weights": weights.tolist(),
                "inv_cov": inv_cov.tolist(),
                "mse": float(max(0, mse)),
                "scaler": scalers[action]
            }
        
        if has_data:
            print(f"  {action}: {accum[action]['I']['n']} samples")
            fvdm_models[action] = action_models
        else:
            print(f"  Warning: No observations for {action}")

    # Save models to results
    if not os.path.exists("results"):
        os.makedirs("results")
        
    output_path = "results/fvdm_weights.json"
    with open(output_path, 'w') as f:
        json.dump(fvdm_models, f, indent=2)
            
    print(f"Derivation complete. Regression models saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Felicific Derivation Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode with fewer seeds/steps")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds to run")
    parser.add_argument("--collect", action="store_true", help="Run collection phase")
    parser.add_argument("--train", action="store_true", help="Run training phase")
    parser.add_argument("--processes", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--batches", type=int, default=1, help="Number of batches for training phase")
    parser.add_argument("--agents", type=int, default=None, help="Number of starting agents")
    
    args = parser.parse_args()
    
    if args.demo and args.seeds == 20:
        args.seeds = 5

    if not args.collect and not args.train:
        args.collect = True
        args.train = True

    if args.collect:
        collect_observations(num_seeds=args.seeds, demo=args.demo, processes=args.processes, num_agents=args.agents)
    if args.train:
        train_models(num_batches=args.batches)
