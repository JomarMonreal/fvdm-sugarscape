#!/usr/bin/env python3
"""
bfe_from_baseline.py
--------------------
Derive FVDM prioritization profiles (BFE) from existing baseline homogeneous
simulation agent logs produced by run_experiments_baseline.py --agent-logs.

Each homogeneous condition provides clean, single-type training data:
  baseline_results/sim_logs/rawSugarscape/  -> rawSugarscape profile
  baseline_results/sim_logs/egoist/         -> egoist profile
  baseline_results/sim_logs/altruist/       -> altruist profile
  baseline_results/sim_logs/bentham/        -> bentham profile

This avoids the Bentham collapse problem seen in mixed-population derivation,
where Bentham agents (phi=0.5) are outcompeted and produce far fewer qualifying
observations than Egoist agents.

Usage:
  # First run the baseline experiment with agent logs saved:
  python run_experiments_baseline.py --seeds 37 --agent-logs --cores <N>

  # Then derive BFE profiles:
  python bfe_from_baseline.py --input baseline_results/sim_logs
  python bfe_from_baseline.py --input baseline_results/sim_logs --output fvdm_vectors --gamma 0.5
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from derive_vectors import (
    AGENT_TYPES,
    COORD_LABELS,
    DEFAULT_TARGET_SE,
    GAMMA_DEFAULT,
    agent_log_to_df,
    compute_per_seed_stats_from_accum,
    derive_profiles_from_accum,
    estimate_required_seeds,
    make_accum,
    print_variance_report,
    _accum_one_seed,
)

# Map condition directory name -> agent type key used by derive_vectors functions.
# rawSugarscape agents are logged as decisionModel="none" by sugarscape.py.
CNAME_TO_ATYPE = {
    "rawSugarscape": "rawSugarscape",
    "egoist":        "egoist",
    "altruist":      "altruist",
    "bentham":       "bentham",
}


def _seed_from_filename(cname: str, filename: str) -> str:
    """Extract seed string from '{cname}_{seed}_agents.json'."""
    base = os.path.basename(filename)
    # Remove cname prefix and _agents.json suffix
    middle = base[len(cname) + 1:]          # "{seed}_agents.json"
    return middle.replace("_agents.json", "")


def discover_logs(sim_logs_dir: str, cname: str) -> list[tuple[str, str]]:
    """Return list of (seed_tag, path) for all agent logs under sim_logs/cname/."""
    cdir = os.path.join(sim_logs_dir, cname)
    if not os.path.isdir(cdir):
        return []
    pattern = os.path.join(cdir, f"{cname}_*_agents.json")
    paths = sorted(glob.glob(pattern))
    return [(_seed_from_filename(cname, p), p) for p in paths]


def main(args):
    t0 = time.time()

    sim_logs_dir = args.input
    output_dir   = args.output
    gamma        = args.gamma
    target_se    = args.target_se
    max_seeds    = args.seeds  # None = use all available

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BFE Derivation from Baseline Homogeneous Logs")
    print(f"{'='*60}")
    print(f"  Input:    {sim_logs_dir}")
    print(f"  Output:   {output_dir}")
    print(f"  Gamma:    {gamma}")
    print(f"  Target SE: {target_se}")
    print(f"{'='*60}\n")

    # Discover available logs per condition
    accum = make_accum()
    total_files = 0
    missing_conditions = []

    for cname, atype in CNAME_TO_ATYPE.items():
        logs = discover_logs(sim_logs_dir, cname)
        if not logs:
            print(f"  [warn] No agent logs found for condition '{cname}' "
                  f"in {sim_logs_dir}/{cname}/")
            missing_conditions.append(cname)
            continue

        if max_seeds is not None:
            logs = logs[:max_seeds]

        print(f"  {cname:<16}  {len(logs)} log file(s) found")

        n_loaded = 0
        for seed_tag, path in logs:
            df = agent_log_to_df(path, seed_tag=seed_tag, slim=True)
            if df.empty:
                print(f"    [warn] Empty log: {path}")
                continue
            _accum_one_seed(df, gamma, accum)
            n_loaded += 1
            total_files += 1

        n_obs = accum[atype]["total_obs"]
        print(f"    → {n_loaded} seeds loaded, {n_obs:,} qualifying observations\n")

    if missing_conditions:
        print(f"  Missing conditions: {missing_conditions}")
        print(f"  Re-run baseline with --agent-logs to generate them.\n")

    if total_files == 0:
        print("  [error] No agent logs found. Aborting.")
        print(f"  Expected files at: {sim_logs_dir}/<condition>/<condition>_<seed>_agents.json")
        print(f"  Generate them with: python run_experiments_baseline.py --agent-logs")
        sys.exit(1)

    total_obs = sum(a["total_obs"] for a in accum.values())
    n_seeds_check = max(
        (len(a["seed_n_obs"]) for a in accum.values() if a["seed_n_obs"]),
        default=0,
    )
    print(f"  Total qualifying observations: {total_obs:,}  across ~{n_seeds_check} seed(s)\n")

    # Variance analysis
    print("  Variance analysis …\n")
    per_seed_stats  = compute_per_seed_stats_from_accum(accum)
    sample_size_rpt = estimate_required_seeds(per_seed_stats, target_se=target_se)
    global_n_req, _ = print_variance_report(sample_size_rpt, target_se)

    if global_n_req is not None and global_n_req > n_seeds_check:
        print(f"\n  [warn] Variance requires {global_n_req} seeds but only "
              f"{n_seeds_check} are available.")
        print(f"  Re-run baseline with --seeds {global_n_req} --agent-logs to expand.\n")
    elif global_n_req is not None:
        print(f"\n  Current {n_seeds_check} seed(s) satisfy target SE "
              f"(required {global_n_req}).\n")

    # Derive BFE profiles
    print("  Computing BFE profiles …\n")
    profiles = derive_profiles_from_accum(accum)

    if not profiles:
        print("  [error] No profiles derived. Aborting.")
        sys.exit(1)

    # φ-linear validation: Bentham ≈ 0.5·Egoist + 0.5·Altruist
    phi_bfs  = None
    phi_pred = None
    if all(k in profiles for k in ("egoist", "altruist", "bentham")):
        mu_e_imm = np.array(profiles["egoist"]["mu_imm"])
        mu_a_imm = np.array(profiles["altruist"]["mu_imm"])
        mu_e_fut = np.array(profiles["egoist"]["mu_fut"])
        mu_a_fut = np.array(profiles["altruist"]["mu_fut"])
        pred_imm = (0.5 * mu_e_imm + 0.5 * mu_a_imm).tolist()
        pred_fut = (0.5 * mu_e_fut + 0.5 * mu_a_fut).tolist()

        def _cos(u, v):
            u, v = np.array(u), np.array(v)
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            return float(np.dot(u, v) / (nu * nv)) if nu > 1e-10 and nv > 1e-10 else 0.0

        cos_imm = _cos(pred_imm, profiles["bentham"]["mu_imm"])
        cos_fut = _cos(pred_fut, profiles["bentham"]["mu_fut"])
        phi_bfs = round((cos_imm + cos_fut) / 2.0, 6)
        phi_pred = {
            "mu_imm":  [round(x, 6) for x in pred_imm],
            "mu_fut":  [round(x, 6) for x in pred_fut],
            "cos_imm": round(cos_imm, 6),
            "cos_fut": round(cos_fut, 6),
            "phi_bfs": phi_bfs,
        }
        print(f"\n  φ-Linear Validation (Bentham ≈ 0.5·Egoist + 0.5·Altruist)")
        print(f"  cos_imm = {cos_imm:.4f}   cos_fut = {cos_fut:.4f}   φ-BFS = {phi_bfs:.4f}")
        if phi_bfs >= 0.99:
            print("  → φ linearly parameterises felicific space (Bentham BFE is redundant).")
        else:
            print("  → argmin nonlinearity is material; Bentham BFE is not redundant.")

    # Save output
    out_path = os.path.join(output_dir, "bfe_profiles.json")
    seeds_used = {
        cname: len(discover_logs(sim_logs_dir, cname))
        for cname in CNAME_TO_ATYPE
        if discover_logs(sim_logs_dir, cname)
    }
    output = {
        "description": "BFE prioritization profiles derived from homogeneous baseline runs",
        "coordinate_labels": {
            "imm": ["I (intensity)", "D (duration)", "C (certainty)", "P=1 (propinquity)", "E=1/|V| (extent)"],
            "fut": ["J (future intensity)", "Df (future duration)", "C (certainty)", "P=gamma (propinquity)", "E=1/|V| (extent)"],
        },
        "derivation_config": {
            "source":     "baseline_homogeneous",
            "input_dir":  sim_logs_dir,
            "seeds_used": seeds_used,
            "gamma":      gamma,
            "filter":     "neighbors_gt_0",
            "agent_types": AGENT_TYPES,
        },
        "profiles":              {k: {kk: vv for kk, vv in v.items()} for k, v in profiles.items()},
        "phi_linear_validation": phi_pred,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = round(time.time() - t0, 1)
    print(f"\n{'='*60}")
    print(f"  {'Type':<20} {'n_obs':>8}  {'mu_imm[:2]':>22}  {'mu_fut[:2]':>22}")
    print(f"  {'-'*56}")
    for atype, prof in profiles.items():
        mi = prof["mu_imm"]
        mf = prof["mu_fut"]
        mi_str = f"I={mi[0]:.3f} D={mi[1]:.3f}"
        mf_str = f"J={mf[0]:.3f} Df={mf[1]:.3f}"
        print(f"  {atype:<20} {prof['n_obs']:>8}  {mi_str:>22}  {mf_str:>22}")
    if phi_bfs is not None:
        print(f"  {'φ-BFS (Bentham)':<20} {'':>8}  {'':>22}  {phi_bfs:>22.4f}")
    print(f"{'='*60}")
    print(f"  Saved → {out_path}")
    print(f"  Done in {elapsed}s\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Derive FVDM BFE profiles from baseline homogeneous agent logs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input",  default="baseline_results/sim_logs",
                   help="Directory containing per-condition subdirectories of agent logs.")
    p.add_argument("-o", "--output", default="fvdm_vectors",
                   help="Output directory for bfe_profiles.json.")
    p.add_argument("-g", "--gamma",  type=float, default=GAMMA_DEFAULT,
                   help="Lookahead discount factor (gamma) for future layer.")
    p.add_argument("--target-se",    type=float, default=DEFAULT_TARGET_SE,
                   help="Target standard error per vector component.")
    p.add_argument("-s", "--seeds",  type=int,   default=None,
                   help="Max seeds to use per condition (default: all available).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
