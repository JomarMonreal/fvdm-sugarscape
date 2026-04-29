import os
import subprocess
import sys

def aggregate_all():
    base_dir = "results/experiments"
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    # Use the same python executable that is running this script
    python_exe = sys.executable
    
    subfolders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])
    
    for folder in subfolders:
        condition_dir = os.path.join(base_dir, folder)
        logs_dir = os.path.join(condition_dir, "logs")
        eval_dir = os.path.join(condition_dir, "eval")
        
        print(f"--- Processing {folder} ---")
        
        # 1. Aggregate Timeseries
        if os.path.exists(logs_dir) and any(f.endswith('.json') for f in os.listdir(logs_dir)):
            print(f"  Aggregating timeseries...")
            try:
                subprocess.run([python_exe, "aggregate_timeseries.py", logs_dir], check=True)
            except subprocess.CalledProcessError:
                print(f"  Error aggregating timeseries for {folder}")
        else:
            print(f"  No logs found in {logs_dir}, skipping timeseries.")

        # 2. Aggregate Evaluations
        if os.path.exists(eval_dir) and any(f.endswith('_evaluation.json') for f in os.listdir(eval_dir)):
            print(f"  Aggregating evaluations...")
            try:
                subprocess.run([python_exe, "aggregate_evaluations.py", eval_dir], check=True)
            except subprocess.CalledProcessError:
                print(f"  Error aggregating evaluations for {folder}")
        else:
            print(f"  No evaluations found in {eval_dir}, skipping evaluations.")

    print("\nAll conditions processed.")

if __name__ == "__main__":
    aggregate_all()
