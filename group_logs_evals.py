import os
import sys
import shutil
import argparse

def group_files(target_dir):
    if not os.path.isdir(target_dir):
        print(f"Error: '{target_dir}' is not a valid directory.")
        sys.exit(1)

    # Create subdirectories
    logs_dir = os.path.join(target_dir, "logs")
    eval_dir = os.path.join(target_dir, "evaluation")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    moved_logs = 0
    moved_evals = 0

    for filename in os.listdir(target_dir):
        file_path = os.path.join(target_dir, filename)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        if filename.endswith("_evaluation.json"):
            # Move to evaluation folder
            shutil.move(file_path, os.path.join(eval_dir, filename))
            moved_evals += 1
        elif filename.endswith(".json"):
            # Move to logs folder
            shutil.move(file_path, os.path.join(logs_dir, filename))
            moved_logs += 1

    print(f"Grouped files in '{target_dir}':")
    print(f"  - Moved {moved_logs} raw log files to 'logs/'")
    print(f"  - Moved {moved_evals} evaluation files to 'evaluation/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group log and evaluation files into separate subdirectories.")
    parser.add_argument("folder", help="The target directory containing the JSON files to group.")
    args = parser.parse_args()

    group_files(args.folder)
