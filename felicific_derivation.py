import sugarscape
import json
import os
import multiprocessing
import random
from functools import partial

def worker(seed, config_path, fvdm_H, steps=200, burn_in=50):
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

    return s.fvdm_dataset

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
    
    # 3. Certainty (C) - Placeholder here, computed by aggregator
    certainty = 0.0 
    
    # 4. Propinquity (P) - Fraction of effect that is immediate
    total_abs_effect = abs(ind_imm) + abs(ind_fut) + eps
    propinquity = max(0.0, min(1.0, abs(ind_imm) / total_abs_effect))
    
    # 5. Extent (X) - Social spread
    # (pop_delta - ind_delta) / total_delta
    total_spread = abs(pop_imm) + abs(ind_imm) + eps
    extent = max(-1.0, min(1.0, (pop_imm - ind_imm) / total_spread))
    
    return {
        "I": round(intensity, 4),
        "D": round(duration, 4),
        "P": round(propinquity, 4),
        "X": round(extent, 4),
        "action": raw_obs["action"],
        "s_i": raw_obs["s_i"]
    }

def run_derivation(num_seeds=5, demo=False):
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
    
    all_observations = []
    
    cpu_count = multiprocessing.cpu_count()
    num_workers = min(num_seeds, max(1, cpu_count - 1))
    
    for config_path in bias_configs:
        action_name = config_path.split("_")[-1].split(".")[0].upper()
        print(f"Rolling derivation for action: {action_name}...")
        
        seeds = [random.randint(0, 1000000) for _ in range(num_seeds)]
        
        pool = multiprocessing.Pool(processes=num_workers)
        func = partial(worker, config_path=config_path, fvdm_H=fvdm_H, steps=observation_steps, burn_in=burn_in)
        
        results = pool.map(func, seeds)
        pool.close()
        pool.join()
        
        action_obs = []
        for seed_data in results:
            action_obs.extend(seed_data)
        
        print(f"  Collected {len(action_obs)} raw observations for {action_name}")
        
        # Calculate coordinates
        for raw in action_obs:
            coords = calculate_coordinates(raw)
            if coords:
                all_observations.append(coords)

    # 2. Aggregating Certainty (C)
    # Group by (action, state_bin) to calculate positive outcome rate
    # For now, we'll use a very coarse binning for demo purposes
    # A real implementation would use a better state-binning strategy
    print("Calculating Certainty (C) across dataset...")
    for obs in all_observations:
        # Dummy Certainty calculation: likelihood of I > 0 in this action category
        # In a real run, this would be binned by s_i
        obs["C"] = 0.8 # Placeholder for demo consistency

    # 3. Save to results
    if not os.path.exists("results"):
        os.makedirs("results")
        
    output_path = "results/felicific_dataset.jsonl"
    with open(output_path, 'w') as f:
        for obs in all_observations:
            f.write(json.dumps(obs) + "\n")
            
    print(f"Derivation complete. {len(all_observations)} observations saved to {output_path}")

if __name__ == "__main__":
    import sys
    is_demo = "--demo" in sys.argv
    seeds = 5 if is_demo else 20
    for i, arg in enumerate(sys.argv):
        if arg == "--seeds" and i + 1 < len(sys.argv):
            seeds = int(sys.argv[i+1])
            break
            
    run_derivation(num_seeds=seeds, demo=is_demo)
