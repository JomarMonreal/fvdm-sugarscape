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
        # Check both the folder itself and a potential logs/ subfolder (to be flexible)
        logs_path = os.path.join(folder_path, "logs")
        search_path = logs_path if os.path.exists(logs_path) else folder_path
        
        files = [f for f in os.listdir(search_path) if f.endswith('.json')]
        
        logs = [f for f in files if not f.endswith('_evaluation.json')]
        evals = [f for f in files if f.endswith('_evaluation.json')]
        
        # Determine pairs
        log_bases = set(f.replace('.json', '') for f in logs)
        eval_bases = set(f.replace('_evaluation.json', '') for f in evals)
        
        pairs = log_bases.intersection(eval_bases)
        orphans = (log_bases.union(eval_bases)) - pairs
        
        print(f"{folder:<30} | {len(files):<12} | {len(pairs):<15} | {len(orphans):<8}")

if __name__ == "__main__":
    check_counts()
