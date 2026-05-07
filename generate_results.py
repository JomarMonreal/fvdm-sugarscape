import os
import argparse
from run_experiments import parse_sugarscape_log, aggregate_seed_stats, write_csv

def main():
    parser = argparse.ArgumentParser(description="Generate results CSVs from sim_logs")
    parser.add_argument("--logs", type=str, default="experiment_results/sim_logs", help="Path to sim_logs directory")
    parser.add_argument("--output", type=str, default="experiment_results/results", help="Path to output results directory")
    parser.add_argument("--timesteps", type=int, default=5000, help="Number of timesteps per simulation")
    args = parser.parse_args()

    logs_dir = args.logs
    out_dir = args.output

    if not os.path.exists(logs_dir):
        print(f"Logs directory not found: {logs_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    all_per_timestep = []
    all_summaries = []
    condition_summaries = {}

    # Iterate over all JSON files in the logs directory and subdirectories
    print(f"Scanning {logs_dir} for log files...")
    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            if file.endswith(".json"):
                log_path = os.path.join(root, file)
                
                # Format is condition_seed.json
                basename = file[:-5]
                # Extract condition name and seed
                # Since condition name might have underscores, we can split by "_" and assume the last part is seed
                parts = basename.split("_")
                if len(parts) >= 2:
                    try:
                        seed = int(parts[-1])
                        condition = "_".join(parts[:-1])
                    except ValueError:
                        print(f"Skipping {file} - could not extract seed")
                        continue
                else:
                    print(f"Skipping {file} - unknown format")
                    continue
                
                pts, summary = parse_sugarscape_log(
                    log_path=log_path,
                    condition=condition,
                    seed=seed,
                    timesteps=args.timesteps,
                    duration=0.0
                )
                
                if pts is None:
                    print(f"Could not parse log {file}")
                    continue
                
                all_per_timestep.extend(pts)
                all_summaries.append(summary)
                
                if condition not in condition_summaries:
                    condition_summaries[condition] = []
                condition_summaries[condition].append(summary)

    print(f"Found {len(all_summaries)} valid logs.")
    
    if not all_summaries:
        print("No valid logs found. Exiting.")
        return

    # Write per_timestep.csv
    pts_path = os.path.join(out_dir, "per_timestep.csv")
    print(f"Writing {pts_path}...")
    write_csv(all_per_timestep, pts_path)

    # Write per_seed_summary.csv
    sum_path = os.path.join(out_dir, "per_seed_summary.csv")
    print(f"Writing {sum_path}...")
    write_csv(all_summaries, sum_path)

    # Write condition_aggregates.csv
    agg_rows = []
    for cname, summs in condition_summaries.items():
        if summs:
            agg = aggregate_seed_stats(summs)
            agg_rows.append(agg)
            
    agg_path = os.path.join(out_dir, "condition_aggregates.csv")
    print(f"Writing {agg_path}...")
    write_csv(agg_rows, agg_path)

    # Organize into per-condition subdirectories
    print("Organizing into per-condition directories...")
    for cname in condition_summaries:
        cond_dir = os.path.join(out_dir, cname)
        os.makedirs(cond_dir, exist_ok=True)

        cond_pts = [r for r in all_per_timestep if r["condition"] == cname]
        if cond_pts:
            write_csv(cond_pts, os.path.join(cond_dir, "per_timestep.csv"))

        cond_sums = [r for r in all_summaries if r["condition"] == cname]
        if cond_sums:
            write_csv(cond_sums, os.path.join(cond_dir, "per_seed_summary.csv"))

        cond_agg = [r for r in agg_rows if r.get("condition") == cname]
        if cond_agg:
            write_csv(cond_agg, os.path.join(cond_dir, "condition_aggregates.csv"))

        if cond_pts or cond_sums:
            print(f"  → {cname}/ ({len(cond_pts)} ts rows, {len(cond_sums)} seeds)")

    print("Generation complete.")

if __name__ == "__main__":
    main()
