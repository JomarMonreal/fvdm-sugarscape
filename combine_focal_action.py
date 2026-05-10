#!/usr/bin/env python3
"""
combine_focal_action.py
----------------------
Combines multiple focal_action_derivation.csv files into a single dataset.
Ensures column consistency and prints a summary of the combined data.

Usage:
  python combine_focal_action.py file1.csv file2.csv -o combined.csv
"""

import csv
import os
import sys
import argparse

def get_action_distribution(rows):
    """Compute per-action counts from observations."""
    counts = {"combat": 0, "trade": 0, "reproduction": 0, "lending": 0}
    for row in rows:
        action = row.get("action")
        if action in counts:
            counts[action] += 1
    return counts

def combine_csvs(input_paths, output_path):
    all_observations = []
    headers = None

    print(f"Reading {len(input_paths)} files...")
    
    for path in input_paths:
        if not os.path.exists(path):
            print(f"  [error] File not found: {path}")
            continue
        
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if headers is None:
                headers = reader.fieldnames
            else:
                if reader.fieldnames != headers:
                    # Check if they are at least compatible
                    missing = set(headers) - set(reader.fieldnames)
                    extra = set(reader.fieldnames) - set(headers)
                    if missing or extra:
                        print(f"  [warn] Header mismatch in {path}")
                        if missing: print(f"    Missing: {missing}")
                        if extra: print(f"    Extra: {extra}")
                        # We'll continue and use the union of headers if needed, 
                        # but for now let's assume they should be identical.
            
            rows = list(reader)
            all_observations.extend(rows)
            print(f"  Loaded {len(rows):>6} observations from {os.path.basename(path)}")

    if not all_observations:
        print("No observations found. Exiting.")
        return

    # Dynamically determine all headers in case they varied
    all_keys = []
    seen_keys = set()
    for row in all_observations:
        for k in row.keys():
            if k not in seen_keys:
                all_keys.append(k)
                seen_keys.add(k)

    print(f"\nWriting combined data to {output_path} ...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(all_observations)
    
    print(f"  Successfully wrote {len(all_observations)} observations.")

    # Print summary
    counts = get_action_distribution(all_observations)
    target = 2640
    print(f"\n{'='*40}")
    print(f"  Combined Data Summary")
    print(f"{'-'*40}")
    for action, count in counts.items():
        mark = "✓" if count >= target else "✗"
        print(f"    {mark} {action:<15} {count:>6} / {target}")
    print(f"{'='*40}\n")

def main():
    parser = argparse.ArgumentParser(description="Combine focal action derivation CSVs")
    parser.add_argument("inputs", nargs="+", help="Input CSV files")
    parser.add_argument("-o", "--output", default="combined_focal_action_derivation.csv", 
                        help="Output CSV file path")
    
    args = parser.parse_args()
    combine_csvs(args.inputs, args.output)

if __name__ == "__main__":
    main()
