import sugarscape
import json
import statistics
import math
import random
import os
import multiprocessing
from functools import partial

def worker(seed, conf, pilot_steps, burn_in_steps, sample_interval, num_agents=None):
    # Setup local config for this seed
    local_conf = conf.copy()
    local_conf["seed"] = seed
    if num_agents is not None:
        local_conf["startingAgents"] = num_agents
    # Headless is handled in run_calibration for the batch/single case
    
    s = sugarscape.Sugarscape(local_conf)
    
    r_i_samples = []
    wealth_history = {} # agent_id -> {t: wealth}
    alive_agents = set()
    
    for t in range(pilot_steps):
        s.doTimestep()
        
        # Burn-in logic
        if t < burn_in_steps:
            continue
            
        current_agents = {a.ID: a for a in s.agents}
        
        for agent_id, agent in current_agents.items():
            # TTL/Ri sampling
            if t % sample_interval == 0:
                # Using TTL as requested by user
                ttl = agent.findTimeToLive(ageLimited=True)
                r_i_samples.append(ttl)
                
            # Wealth history
            wealth = agent.sugar + agent.spice
            if agent_id not in wealth_history:
                wealth_history[agent_id] = {}
            wealth_history[agent_id][t] = wealth
            alive_agents.add(agent_id)

    # Calculate local delta sums for a range of H to avoid returning massive history objects
    # range 1 to 150 is usually sufficient for Sugarscape
    max_h = 150
    h_stats = {h: [0.0, 0] for h in range(1, max_h + 1)} # {h: [sum_abs_delta, count]}
    
    for agent_id, history in wealth_history.items():
        ts = sorted(history.keys())
        for start_t in ts:
            for h in range(1, max_h + 1):
                end_t = start_t + h
                if end_t in history:
                    h_stats[h][0] += abs(history[end_t] - history[start_t])
                    h_stats[h][1] += 1
                    
    return r_i_samples, h_stats

def run_calibration(config_path, output_path="horizon.json", demo=False, num_seeds=1, processes=None, num_agents=None):
    print(f"Loading configuration from {config_path}...")
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
    
    print("------- Active Configuration Overrides -------")
    for key, value in sorted(overrides.items()):
        print(f"  {key}: {value}")
    print(f"  Batch Run: {num_seeds} seeds")
    print("----------------------------------------------")

    pilot_steps = 300 if demo else 500
    burn_in_steps = 20 if demo else 50
    sample_interval = 2 if demo else 5
    
    base_seed = conf.get("seed", 42)
    if base_seed == -1: base_seed = random.randint(0, 1000000)
    seeds = [base_seed + i for i in range(num_seeds)]

    if processes is None:
        cpu_count = multiprocessing.cpu_count()
        num_workers = min(num_seeds, max(1, cpu_count - 1))
    else:
        num_workers = processes

    print(f"Running {num_seeds} simulations using {num_workers} processes...")
    
    pool = multiprocessing.Pool(processes=num_workers)
    func = partial(worker, conf=conf, pilot_steps=pilot_steps, burn_in_steps=burn_in_steps, sample_interval=sample_interval, num_agents=num_agents)
    
    all_r_i = []
    global_h_stats = {h: [0.0, 0] for h in range(1, 151)}
    
    try:
        if num_seeds > 1 or num_workers > 1:
            # Parallel batch run: must be headless
            results_iter = pool.imap_unordered(func, seeds)
            for i, (r_i, h_stats) in enumerate(results_iter):
                all_r_i.extend(r_i)
                for h in h_stats:
                    global_h_stats[h][0] += h_stats[h][0]
                    global_h_stats[h][1] += h_stats[h][1]
                
                if (i + 1) % 10 == 0:
                    print(f"Progress: {i + 1}/{num_seeds} seeds completed...")
        else:
            # Single seed run: allow GUI if config says so
            # We run in the main process to ensure Tkinter stability
            r_i, h_stats = worker(seeds[0], conf, pilot_steps, burn_in_steps, sample_interval, num_agents=num_agents)
            all_r_i.extend(r_i)
            for h in h_stats:
                global_h_stats[h][0] += h_stats[h][0]
                global_h_stats[h][1] += h_stats[h][1]
    finally:
        pool.close()
        pool.join()

    if not all_r_i:
        print("Error: No samples collected.")
        return

    # 2. Global Candidate Selection
    all_r_i.sort()
    n = len(all_r_i)
    h_q1 = int(all_r_i[int(n * 0.25)])
    h_med = int(all_r_i[int(n * 0.50)])
    h_q3 = int(all_r_i[int(n * 0.75)])
    
    candidates = sorted(list(set([max(1, h_q1), max(2, h_med), max(3, h_q3)])))
    print(f"Global candidates (TTL quantiles): {candidates}")
    
    # 3. Global mu(H) calculation
    mu_values = {}
    h_max = max(candidates)
    
    for h in candidates:
        s, c = global_h_stats[h]
        mu_values[h] = s / c if c > 0 else 0
            
    print(f"Global Mean Wealth Effects (mu(H)): {mu_values}")
    
    # 4. Selection of H*
    h_star = h_max
    mu_max = mu_values[h_max]
    if mu_max > 0:
        for h in candidates:
            if mu_values[h] / mu_max >= 0.90:
                h_star = h
                break
    
    print(f"Selected Global Evaluation Horizon (H*): {h_star}")
    
    # 5. Save
    result = {
        "H_star": h_star,
        "num_seeds": num_seeds,
        "candidates": candidates,
        "mu_values": {str(k): v for k, v in mu_values.items()},
        "ttl_summary": {
            "min": round(min(all_r_i), 2),
            "q1": h_q1,
            "median": h_med,
            "q3": h_q3,
            "max": round(max(all_r_i), 2)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Derivation complete. Result saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Horizon Calibration")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds to run")
    parser.add_argument("--baseline", action="store_true", help="Use baseline configuration")
    parser.add_argument("--processes", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--agents", type=int, default=None, help="Number of starting agents")
    args = parser.parse_args()
            
    config = "configs/demo_horizon.json" if args.demo else "configs/baseline_horizon.json"
    run_calibration(config, demo=args.demo, num_seeds=args.seeds, processes=args.processes, num_agents=args.agents)
