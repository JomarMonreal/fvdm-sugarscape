import os

def check_counts():
    base_dir = "results/experiments"
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    subfolders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])
    
    print(f"{'Condition':<30} | {'Total Files':<12} | {'Log-Eval Pairs':<15} | {'Orphans':<8}")
    print("-" * 75)
    
    for folder in subfolders:
        folder_path = os.path.join(base_dir, folder)
        logs_path = os.path.join(folder_path, "logs")
        eval_path = os.path.join(folder_path, "eval")
        
        # Collect logs and evals from their respective folders if they exist
        logs = []
        if os.path.exists(logs_path):
            logs = [f for f in os.listdir(logs_path) if f.endswith('.json')]
        
        evals = []
        if os.path.exists(eval_path):
            evals = [f for f in os.listdir(eval_path) if f.endswith('.json')]
            
        # Fallback to base folder if subfolders don't exist yet
        if not logs and not evals:
            files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            logs = [f for f in files if not f.endswith('_evaluation.json')]
            evals = [f for f in files if f.endswith('_evaluation.json')]
        
        # Determine pairs
        log_bases = set(f.replace('.json', '') for f in logs)
        eval_bases = set(f.replace('_evaluation.json', '') for f in evals)
        
        pairs = log_bases.intersection(eval_bases)
        orphans = (log_bases.union(eval_bases)) - pairs
        
        total_count = len(logs) + len(evals)
        print(f"{folder:<30} | {total_count:<12} | {len(pairs):<15} | {len(orphans):<8}")

if __name__ == "__main__":
    check_counts()
