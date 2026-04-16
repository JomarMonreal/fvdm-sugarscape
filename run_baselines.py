import json
import os
import subprocess

seeds = [42, 101, 2026, 888, 9999]

with open("config.json", "r") as f:
    base_config = json.load(f)

if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/baseline_runs"):
    os.makedirs("data/baseline_runs")

for seed in seeds:
    config = json.loads(json.dumps(base_config))  # deep copy
    config["sugarscapeOptions"]["seed"] = seed
    config["sugarscapeOptions"]["logfile"] = f"data/baseline_runs/log_seed_{seed}.json"
    config["sugarscapeOptions"]["agentLogfile"] = f"data/baseline_runs/agent_log_seed_{seed}.json"
    
    config_file = f"config_seed_{seed}.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"Running baseline for seed {seed}...")
    import sys
    subprocess.run([sys.executable, "-u", "sugarscape.py", "--conf", config_file])
    print(f"Finished seed {seed}.")
    
    os.remove(config_file)
