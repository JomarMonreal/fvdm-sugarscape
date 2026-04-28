import os
import shutil

def organize_results():
    base_dir = "results/experiments"
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

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

    # Create target directories
    for target in targets:
        target_dir = os.path.join(base_dir, target)
        os.makedirs(target_dir, exist_ok=True)

    moved_count = 0
    # Iterate through all files in the base directory
    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Check if the filename starts with any of our target names
        for target in targets:
            if filename.startswith(target):
                # We have a match! Move the file.
                dest_path = os.path.join(base_dir, target, filename)
                shutil.move(file_path, dest_path)
                moved_count += 1
                break # Move to the next file

    print(f"Successfully moved {moved_count} files into their respective folders.")

if __name__ == "__main__":
    organize_results()
