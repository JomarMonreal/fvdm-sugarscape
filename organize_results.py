import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Organize simulation results by run type")
    parser.add_argument("--results", type=str, default="experiment_results/results", help="Path to results directory")
    args = parser.parse_args()

    results_dir = args.results
    
    # Check if files exist
    pts_path = os.path.join(results_dir, "per_timestep.csv")
    sum_path = os.path.join(results_dir, "per_seed_summary.csv")
    
    if not os.path.exists(pts_path) or not os.path.exists(sum_path):
        print(f"Results files not found in {results_dir}")
        return
        
    print(f"Loading data from {results_dir}...")
    df_ts = pd.read_csv(pts_path)
    df_sum = pd.read_csv(sum_path)
    
    # Define condition mappings
    # ts_1_baseline
    baseline_conds = ["Homo Egoist", "Homo Altruist", "Homo Bentham", "Hetero Base", "egoist", "altruist", "bentham", "heterogeneous", "rawSugarscape"]
    
    # ts_2_derived
    derived_conds = ["Homo Selfish Derived", "Homo Altruist Derived", "Homo Bentham Derived"]
    
    # ts_3_idealized
    idealized_conds = ["Homo Selfish Ideal", "Homo Altruist Ideal"]
    
    # ts_4_hetero
    hetero_conds = ["Hetero FVDM Derived", "Hetero FVDM Ideal"]
    
    # ts_5_mixed
    mixed_conds = ["Hetero Mixed (Ideal Altruist)", "Hetero Mixed (Ideal Selfish)", 
                   "Hetero Mixed (Derived Selfish)", "Hetero Mixed (Derived Altruist)", "Hetero Mixed (Derived Bentham)"]
                   
    groups = {
        "ts_1_baseline": baseline_conds,
        "ts_2_derived": derived_conds,
        "ts_3_idealized": idealized_conds,
        "ts_4_hetero": hetero_conds,
        "ts_5_mixed": mixed_conds
    }
    
    for group_name, conditions in groups.items():
        # Create group directory
        group_dir = os.path.join(results_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)
        
        # Filter DataFrames
        filtered_ts = df_ts[df_ts['condition'].isin(conditions)]
        filtered_sum = df_sum[df_sum['condition'].isin(conditions)]
        
        # Only save if there is data
        if not filtered_ts.empty:
            print(f"Saving {group_name} data ({len(filtered_ts)} rows)...")
            filtered_ts.to_csv(os.path.join(group_dir, "per_timestep.csv"), index=False)
            filtered_sum.to_csv(os.path.join(group_dir, "per_seed_summary.csv"), index=False)
            
            # Also organize sim_logs if they exist
            sim_logs_dir = os.path.join("experiment_results", "sim_logs")
            if os.path.exists(sim_logs_dir):
                target_logs_dir = os.path.join(sim_logs_dir, group_name)
                os.makedirs(target_logs_dir, exist_ok=True)
                
                # Move relevant log files
                for file in os.listdir(sim_logs_dir):
                    if file.endswith(".json"):
                        # Extract condition from filename (format: condition_seed.json)
                        # Be careful because condition name might have underscores
                        cond_match = next((c for c in conditions if file.startswith(c + "_")), None)
                        if cond_match:
                            src = os.path.join(sim_logs_dir, file)
                            dst = os.path.join(target_logs_dir, file)
                            os.rename(src, dst)
                            
    print("Done organizing results.")

if __name__ == "__main__":
    main()
