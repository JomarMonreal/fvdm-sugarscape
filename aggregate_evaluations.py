import os
import json
import glob
import statistics
import csv

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def aggregate(results_dir):
    print("=== Stage 6: Outcome Evaluation ===")
    
    # Find all evaluation files
    eval_files = glob.glob(os.path.join(results_dir, "*_evaluation.json"))
    if not eval_files:
        print(f"No evaluation files found in {results_dir}")
        return

    # Group by condition
    # Filename format: {condition_name}_seed_{seed}_evaluation.json
    conditions = {}
    
    for fpath in eval_files:
        filename = os.path.basename(fpath)
        # Extract condition name
        condition_name = filename.split("_seed_")[0]
        
        with open(fpath, 'r') as f:
            try:
                data = json.load(f)
                flat_data = flatten_dict(data)
                
                if condition_name not in conditions:
                    conditions[condition_name] = []
                conditions[condition_name].append(flat_data)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {filename}")

    if not conditions:
        print("No valid evaluation data to aggregate.")
        return

    print(f"Aggregating {len(eval_files)} files across {len(conditions)} conditions...")

    # Determine all possible metrics (excluding non-numeric ones like log_file and categorical_end_state)
    all_metrics = set()
    for cond_data in conditions.values():
        for run in cond_data:
            for k, v in run.items():
                if isinstance(v, (int, float)):
                    all_metrics.add(k)
    
    all_metrics = sorted(list(all_metrics))
    
    # Calculate means and stddevs
    summary = []
    for condition_name, runs in conditions.items():
        row = {"Condition": condition_name, "N_Seeds": len(runs)}
        
        # Categorical counts
        extinct_count = sum(1 for r in runs if r.get("societal_metrics.categorical_end_state") == "Extinct")
        worse_count = sum(1 for r in runs if r.get("societal_metrics.categorical_end_state") == "Worse")
        better_count = sum(1 for r in runs if r.get("societal_metrics.categorical_end_state") == "Better")
        
        row["End_Extinct"] = extinct_count
        row["End_Worse"] = worse_count
        row["End_Better"] = better_count
        
        # Numeric metrics
        for metric in all_metrics:
            values = [r[metric] for r in runs if metric in r and isinstance(r[metric], (int, float))]
            if values:
                mean_val = statistics.mean(values)
                stdev_val = statistics.stdev(values) if len(values) > 1 else 0.0
                row[f"{metric}_mean"] = round(mean_val, 2)
                row[f"{metric}_stdev"] = round(stdev_val, 2)
            else:
                row[f"{metric}_mean"] = "N/A"
                row[f"{metric}_stdev"] = "N/A"
                
        summary.append(row)

    # Sort summary logically (Base first, then FVDM)
    summary.sort(key=lambda x: ("fvdm" in x["Condition"], x["Condition"]))

    # Write CSV
    csv_path = os.path.join(results_dir, "comparative_summary.csv")
    fieldnames = ["Condition", "N_Seeds", "End_Extinct", "End_Worse", "End_Better"]
    for metric in all_metrics:
        fieldnames.append(f"{metric}_mean")
        fieldnames.append(f"{metric}_stdev")
        
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
            
    print(f"\nSuccessfully created comparative summary at {csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate evaluation JSONs into a comparative summary.")
    parser.add_argument("results_dir", help="Directory containing the *_evaluation.json files")
    args = parser.parse_args()
    
    aggregate(args.results_dir)
