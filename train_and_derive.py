#!/usr/bin/env python3
"""
train_and_derive.py
-------------------
FVDM Derivation Pipeline:
1. Trains felicific coordinate models (train_coordinates.py)
2. Derives prioritization vectors via MaxEnt IRL (derive_vectors.py)

This script automates the full FVDM derivation process using a combined 
focal-action dataset.

Usage:
  python3 train_and_derive.py --input combined_focal_action_derivation.csv
"""

import os
import sys
import argparse
import subprocess
import time

def run_command(cmd):
    """Run a shell command and print its output in real-time."""
    print(f"\n> Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"\n[error] Command failed with return code {process.returncode}")
        sys.exit(process.returncode)

def main():
    parser = argparse.ArgumentParser(description="FVDM Training and Vector Derivation Pipeline")
    parser.add_argument("-i", "--input", default="combined_focal_action_derivation.csv",
                        help="Path to the combined derivation CSV.")
    parser.add_argument("-m", "--models", default="fvdm_models",
                        help="Directory to save trained models.")
    parser.add_argument("-v", "--vectors", default="fvdm_vectors",
                        help="Directory to save derived prioritization vectors.")
    parser.add_argument("-c", "--config", default="config.json",
                        help="Path to master config.json (for baseline simulations).")
    parser.add_argument("-j", "--cores", type=int, default=1,
                        help="Number of CPU cores for parallel tasks.")
    # Detect default python (prefer .venv if it exists)
    default_python = "python3"
    if os.path.exists(".venv/bin/python"):
        default_python = ".venv/bin/python"
    elif os.path.exists("venv/bin/python"):
        default_python = "venv/bin/python"

    parser.add_argument("--python", default=default_python,
                        help="Python interpreter alias.")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline condition derivation (derives biased vectors only).")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[error] Input file not found: {args.input}")
        sys.exit(1)

    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"  FVDM DERIVATION PIPELINE")
    print(f"{'='*60}")
    print(f"  Input CSV:   {args.input}")
    print(f"  Models Dir:  {args.models}")
    print(f"  Vectors Dir: {args.vectors}")
    print(f"{'='*60}")

    # ── Phase 1: Train Coordinate Prediction Functions ──
    train_cmd = [
        args.python, "train_coordinates.py",
        "--input", args.input,
        "--output", args.models,
    ]
    run_command(train_cmd)

    # ── Phase 2: Derive Prioritization Vectors via MaxEnt IRL ──
    derive_cmd = [
        args.python, "derive_vectors.py",
        "--focal-csv", args.input,
        "--models", args.models,
        "--output", args.vectors,
        "--config", args.config,
        "--cores", str(args.cores),
        "--python", args.python,
    ]
    
    if args.skip_baseline:
        # Only derive the 4 biased vectors: combat, trade, reproduction, lending
        derive_cmd.extend(["--vectors", "combatDerived", "tradeDerived", "reproductionDerived", "lendingDerived"])

    run_command(derive_cmd)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total Elapsed Time: {total_time:.2f}s")
    print(f"  Models saved to:    {args.models}")
    print(f"  Vectors saved to:   {args.vectors}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
