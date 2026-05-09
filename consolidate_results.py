#!/usr/bin/env python3
import os
import pandas as pd

def consolidate():
    results_root = "experiment_results/results"
    conditions = ["altruist", "bentham", "egoist", "heterogeneous", "rawSugarscape"]
    
    files_to_merge = ["per_timestep.csv", "per_seed_summary.csv", "condition_aggregates.csv"]
    
    for filename in files_to_merge:
        dataframes = []
        for cond in conditions:
            path = os.path.join(results_root, cond, filename)
            if os.path.exists(path):
                print(f"  Reading {path}...")
                dataframes.append(pd.read_csv(path))
            else:
                print(f"  [warn] {path} not found.")
        
        if dataframes:
            merged = pd.concat(dataframes, ignore_index=True)
            out_path = os.path.join(results_root, filename)
            merged.to_csv(out_path, index=False)
            print(f"  SUCCESS: Wrote merged file to {out_path}")
        else:
            print(f"  [error] No data found for {filename}")

if __name__ == "__main__":
    consolidate()
