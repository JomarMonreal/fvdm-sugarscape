import os
import shutil
import glob
from organize_results import organize_results
from group_logs_evals import group_files

def move_mixed_files(base_dir, mix_folder, pattern):
    """Moves files matching pattern from mix_folder and its subdirs back to base_dir"""
    mix_path = os.path.join(base_dir, mix_folder)
    
    if not os.path.exists(mix_path):
        return

    # Check the folder itself, plus logs/ and evaluation/ subfolders
    search_paths = [
        mix_path,
        os.path.join(mix_path, "logs"),
        os.path.join(mix_path, "evaluation")
    ]

    moved_count = 0
    for path in search_paths:
        if not os.path.exists(path):
            continue
            
        for file_path in glob.glob(os.path.join(path, f"*{pattern}*")):
            if os.path.isfile(file_path):
                filename = os.path.basename(file_path)
                dest_path = os.path.join(base_dir, filename)
                shutil.move(file_path, dest_path)
                moved_count += 1
                print(f"Recovered: {filename}")
                
    return moved_count

def fix_results():
    base_dir = "results/experiments"
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    print("Step 1: Extracting miscategorized files...")
    s2_count = move_mixed_files(base_dir, "homo_fvdm_selfish", "homo_fvdm_selfish2")
    a2_count = move_mixed_files(base_dir, "homo_fvdm_altruist", "homo_fvdm_altruist2")
    
    print(f"Found and extracted {s2_count} selfish2 files and {a2_count} altruist2 files.")
    
    print("\nStep 2: Re-running organize_results...")
    organize_results()
    
    print("\nStep 3: Re-grouping logs and evaluations...")
    # Get all subdirectories in results/experiments
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            group_files(item_path)
            
    print("\nDone! All files should be correctly sorted and grouped.")

if __name__ == "__main__":
    fix_results()
