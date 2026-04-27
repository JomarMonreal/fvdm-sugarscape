import json
import subprocess
import os

print("Running one seed of utilitarian-homo...")
subprocess.run(["python3", "run_experiments.py", "--seeds", "1", "--filter", "homo_fvdm_utilitarian", "--processes", "1", "--timesteps", "1000"])

# Check evaluation file
files = os.listdir("results/experiments/homo_fvdm_utilitarian/evaluation")
latest_eval = sorted(files)[-1]
path = os.path.join("results/experiments/homo_fvdm_utilitarian/evaluation", latest_eval)

with open(path) as f:
    d = json.load(f)

print("Total Loans:", d["behavioral_metrics"]["total_loans"])
