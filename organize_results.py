import os
import shutil
import json

def is_json_valid(file_path):
    """Checks if a JSON file is valid, with basic repair logic for raw logs."""
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
        if not content:
            return False
        
        # Repair logic for truncated JSON arrays (raw logs)
        if content.startswith('['):
            if not content.endswith(']'):
                last_brace = content.rfind('}')
                if last_brace == -1:
                    return False
                content = content[:last_brace + 1] + '\n]'
        
        json.loads(content)
        return True
    except Exception:
        return False

def cleanup_broken_files(base_dir):
    """Deletes invalid JSON files in base_dir and its subdirectories."""
    deleted_count = 0
    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith('.json'):
                path = os.path.join(root, filename)
                if not is_json_valid(path):
                    print(f"Deleting broken JSON: {path}")
                    os.remove(path)
                    deleted_count += 1
    return deleted_count

def cap_folder_to_500_pairs(folder_path):
    """Ensures exactly 500 complete log-evaluation pairs exist, deleting excess and orphans."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    # Map seed/base to its components
    pairs = {}
    for f in files:
        if f.endswith('_evaluation.json'):
            base = f.replace('_evaluation.json', '')
            if base not in pairs: pairs[base] = {}
            pairs[base]['eval'] = f
        else:
            base = f.replace('.json', '')
            if base not in pairs: pairs[base] = {}
            pairs[base]['log'] = f
            
    # Filter only complete pairs and sort for deterministic capping
    complete_bases = sorted([b for b, p in pairs.items() if 'log' in p and 'eval' in p])
    
    # Identify which to keep (first 500)
    to_keep_bases = set(complete_bases[:500])
    
    deleted_count = 0
    for f in files:
        base = f.replace('_evaluation.json', '').replace('.json', '')
        if base not in to_keep_bases:
            os.remove(os.path.join(folder_path, f))
            deleted_count += 1
    return deleted_count

def organize_results():
    base_dir = "results/experiments"
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    # 1. Cleanup broken files first
    print("Checking for broken JSON files...")
    broken_deleted = cleanup_broken_files(base_dir)
    if broken_deleted > 0:
        print(f"Deleted {broken_deleted} broken JSON files.")

    targets = [
        "homo_base_egoist",
        "homo_base_altruist",
        "homo_base_bentham",
        "homo_fvdm_selfish",
        "homo_fvdm_altruist",
        "homo_fvdm_utilitarian",
        "homo_fvdm_selfish2",
        "homo_fvdm_altruist2",
        "hetero_base",
        "hetero_fvdm_utilitarian1",
        "hetero_fvdm_utilitarian2",
        "hetero_mixed_egoist",
        "hetero_mixed_altruist",
        "hetero_selfish",
        "hetero_altruist"
    ]
    # Sort targets by length descending so that 'selfish2' is matched before 'selfish'
    targets.sort(key=len, reverse=True)

    # 2. Create target directories and move files
    for target in targets:
        target_dir = os.path.join(base_dir, target)
        os.makedirs(target_dir, exist_ok=True)

    moved_count = 0
    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)
        if os.path.isdir(file_path):
            continue

        for target in targets:
            if filename.startswith(target):
                dest_path = os.path.join(base_dir, target, filename)
                shutil.move(file_path, dest_path)
                moved_count += 1
                break

    print(f"Moved {moved_count} files into their respective folders.")

    # 3. Cap each target folder to 500 pairs
    print("Capping folders to 500 pairs...")
    for target in targets:
        target_dir = os.path.join(base_dir, target)
        if os.path.exists(target_dir):
            cap_deleted = cap_folder_to_500_pairs(target_dir)
            if cap_deleted > 0:
                print(f"  {target}: deleted {cap_deleted} excess/orphan files.")

if __name__ == "__main__":
    organize_results()
