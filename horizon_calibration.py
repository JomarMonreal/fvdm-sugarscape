import sugarscape
import json
import statistics
import math
import random
import os

def run_calibration(config_path, output_path="horizon.json", demo=False):
    print(f"Loading configuration from {config_path}...")
    # 0. Load defaults from global config.json if available
    conf = {}
    if os.path.exists("config.json"):
        with open("config.json", 'r') as f:
            try:
                data = json.load(f)
                conf = data.get("sugarscapeOptions", {})
            except:
                pass

    # Load user's overrides for the specific derivation run
    with open(config_path, 'r') as f:
        overrides = json.load(f)
        conf.update(overrides)
    
    # Show active overrides for transparency
    print("------- Active Configuration Overrides -------")
    for key, value in sorted(overrides.items()):
        print(f"  {key}: {value}")
    print("----------------------------------------------")

    # 1. Pilot Run: Collect TTL distribution
    print("Running pilot simulation to collect Time-To-Live (TTL) ratios...")
    # Respect headlessMode from overrides, default to True for calibration stability
    if "headlessMode" not in overrides:
        conf["headlessMode"] = True
    # Optimization: Use targeted step counts for calibration
    pilot_steps = 300 if demo else 500
    burn_in_steps = 20 if demo else 50
    sample_interval = 2 if demo else 5
    
    # Ensure seed is set for matched random runs
    if "seed" not in conf or conf["seed"] == -1:
        conf["seed"] = 42

    s = sugarscape.Sugarscape(conf)
    r_i_samples = []
    
    # We will also track wealth history for all agents to calculate mu(H) in one pass
    # wealth_history[agent_id] = {timestep: wealth}
    wealth_history = {}
    
    # Track which agents are alive at each step
    alive_agents = set()

    for t in range(pilot_steps):
        s.doTimestep()
        
        # Periodic sampling and burn-in check
        if t < burn_in_steps:
            continue
            
        if t % 50 == 0:
            print(f"Step {t}: {len(s.agents)} agents alive.")

        # Current living agents
        current_agents = {a.ID: a for a in s.agents}
        
        for agent_id, agent in current_agents.items():
            # TTL calculation: Sample periodically for candidate selection
            if t % sample_interval == 0:
                ttl = agent.findTimeToLive(ageLimited=True)
                r_i_samples.append(ttl)
                
            # Wealth history tracking: Record EVERY step to ensure delta calculation works
            wealth = agent.sugar + agent.spice
            if agent_id not in wealth_history:
                wealth_history[agent_id] = {}
            wealth_history[agent_id][t] = wealth
            alive_agents.add(agent_id)
        
        # For agents who died this step, mark their wealth as 0 for future steps in the window
        # (Though we'll only check deltas for agents that were alive at start of H)
        for agent_id in list(alive_agents):
            if agent_id not in current_agents and t not in wealth_history[agent_id]:
                # Agent just died
                wealth_history[agent_id][t] = 0

    if not r_i_samples:
        print("Error: No Ri samples collected. Simulation might have ended early or agents have no metabolism.")
        return

    # 2. Candidate Selection (Q1, Median, Q3)
    r_i_samples.sort()
    n = len(r_i_samples)
    h_q1 = int(r_i_samples[int(n * 0.25)])
    h_med = int(r_i_samples[int(n * 0.50)])
    h_q3 = int(r_i_samples[int(n * 0.75)])
    
    # Ensure candidates are unique, at least 1, and ordered
    candidates = sorted(list(set([max(1, h_q1), max(2, h_med), max(3, h_q3)])))
    print(f"Candidate horizons (Ri quantiles): {candidates}")
    
    # 3. Calculate mu(H) for each candidate
    # mu(H) = E[|W(t+H) - W(t)|]
    mu_values = {}
    h_max = max(candidates)
    
    for h in candidates:
        abs_deltas = []
        for agent_id, history in wealth_history.items():
            timesteps = sorted(history.keys())
            for start_t in timesteps:
                end_t = start_t + h
                if end_t in history:
                    # Agent was alive at start_t, and we have data for end_t (even if 0 due to death)
                    delta = abs(history[end_t] - history[start_t])
                    abs_deltas.append(delta)
        
        if abs_deltas:
            mu_values[h] = statistics.mean(abs_deltas)
        else:
            mu_values[h] = 0
            
    print(f"Mean Wealth Effects (mu(H)): {mu_values}")
    
    # 4. Selection of H* (Smallest H where mu(H) / mu(H_max) >= 0.90)
    h_star = h_max
    mu_max = mu_values[h_max]
    
    if mu_max > 0:
        for h in candidates:
            if mu_values[h] / mu_max >= 0.90:
                h_star = h
                break
    
    print(f"Selected Evaluation Horizon (H*): {h_star}")
    
    # 5. Save result to JSON
    result = {
        "H_star": h_star,
        "candidates": candidates,
        "mu_values": {str(k): v for k, v in mu_values.items()},
        "ttl_summary": {
            "min": round(min(r_i_samples), 2),
            "q1": h_q1,
            "median": h_med,
            "q3": h_q3,
            "max": round(max(r_i_samples), 2)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Derivation complete. Result saved to {output_path}")

if __name__ == "__main__":
    import sys
    # Default to demo if no args or if --demo passed
    is_demo = "--demo" in sys.argv or len(sys.argv) == 1
    config = "configs/demo_horizon.json" if is_demo else "configs/baseline_horizon.json"
    run_calibration(config, demo=is_demo)
