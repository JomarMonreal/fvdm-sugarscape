#!/usr/bin/env python3
"""
run_experiments_argmin_hetero.py
---------------------------------
Heterogeneous Egoist–Bentham sweep using consistent-argmin FVDM agents (Exp 6).

Sweeps fvdmArgminEgoist + fvdmArgminBentham proportions from 0% to 100% Bentham
in HETERO_STEP (default 20) percentage-point increments → 11 conditions.

Uses the same phi-corrected profiles (bfe_profiles_phi.json) as Exp 5
(run_experiments_fvdm_argmin.py), keeping the methodology consistent.

Thesis Objective 7: Examine whether FVDM-derived agents in a heterogeneous
Egoist–Bentham population replicate the societal outcome gradient.

Outputs (under results/argmin_hetero/):
  per_seed_summary.csv     — extinction, final pop, mean wealth, Gini, TTL per seed
  condition_aggregates.csv — mean ± sd across seeds
  hetero_bfs.csv           — BFS(argmin hetero, baseline hetero) per proportion × seed
  hetero_spearman.csv      — Spearman rho between pct_bentham and each outcome metric

Usage:
  python run_experiments_argmin_hetero.py
  python run_experiments_argmin_hetero.py -s 30 -t 5000 -a 250 -j 30
  python run_experiments_argmin_hetero.py --hetero-step 10 --force
"""

import argparse
import csv as csv_mod
import json
import math
import multiprocessing
import os
import random
import subprocess
import sys
import time
from math import gcd

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

HETERO_STEP  = 20
PROFILE_PATH = "fvdm_vectors/bfe_profiles_phi.json"
OUTPUT_DIR   = os.path.join("results", "argmin_hetero")

IMM_LABELS = ["I", "D", "C", "P", "E"]
FUT_LABELS = ["I", "D", "C", "P", "E"]


# ─────────────────────────────────────────────────────────────────────────────
# Condition construction
# ─────────────────────────────────────────────────────────────────────────────

def make_models_list(pct_bentham: int, total_agents: int = 250) -> list:
    if pct_bentham == 0:
        return ["fvdmArgminEgoist"]
    if pct_bentham == 100:
        return ["fvdmArgminBentham"]
    n_bentham = round(pct_bentham / 100 * total_agents)
    n_egoist  = total_agents - n_bentham
    g = gcd(n_bentham, n_egoist)
    return ["fvdmArgminEgoist"] * (n_egoist // g) + ["fvdmArgminBentham"] * (n_bentham // g)


def make_hetero_conditions(step: int = HETERO_STEP):
    """Returns (conds, base_pairs) dicts keyed by condition name."""
    conds      = {}
    base_pairs = {}
    for pct in range(0, 101, step):
        fname          = f"argminHeteroMix_p{pct:03d}"
        bname          = f"heteroMix_p{pct:03d}"
        conds[fname]   = make_models_list(pct)
        base_pairs[fname] = bname
    return conds, base_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Seed generation — matches all other experiment runners (seed=42)
# ─────────────────────────────────────────────────────────────────────────────

def generate_seeds(n: int, seed: int = 42) -> list:
    random.seed(seed)
    seeds = []
    while len(seeds) < n:
        s = random.randint(0, sys.maxsize)
        if s not in seeds:
            seeds.append(s)
    return seeds


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_base_config(path: str) -> dict:
    with open(path) as f:
        full = json.load(f)
    return full.get("sugarscapeOptions", full)


def make_run_config(base: dict, seed: int, decision_models: list,
                    timesteps: int, num_agents: int,
                    log_path: str, agent_log_path: str) -> dict:
    cfg = dict(base)
    cfg["seed"]                    = seed
    cfg["agentDecisionModels"]     = decision_models
    cfg["timesteps"]               = timesteps
    cfg["startingAgents"]          = num_agents
    cfg["startingDiseases"]        = 0
    cfg["headlessMode"]            = True
    cfg["debugMode"]               = ["none"]
    cfg["keepAlivePostExtinction"] = False
    cfg["keepAliveAtEnd"]          = False
    cfg["screenshots"]             = False
    cfg["profileMode"]             = False
    cfg["logfile"]                 = log_path
    cfg["agentLogfile"]            = agent_log_path
    cfg["logfileFormat"]           = "json"
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one_simulation(args):
    cfg_path, python_alias, counter, lock, total = args
    t = time.time()
    subprocess.run([python_alias, "sugarscape.py", "--conf", cfg_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dur = time.time() - t
    with lock:
        counter.value += 1
        n   = counter.value
        bar = "█" * int(30 * n / total) + "░" * (30 - int(30 * n / total))
        print(f"\r  [{bar}] {n:>4}/{total}  {n/total*100:5.1f}%  "
              f"{os.path.basename(cfg_path):<48}", end="", flush=True)
    return cfg_path, dur


# ─────────────────────────────────────────────────────────────────────────────
# Log parsing
# ─────────────────────────────────────────────────────────────────────────────

def safe_json_load(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def parse_sim_log(log_path: str, condition: str, seed: int, timesteps: int):
    sim_log = safe_json_load(log_path)
    if not sim_log:
        return None

    last      = sim_log[-1]
    final_ts  = int(last.get("timestep", 0))
    final_pop = int(last.get("population", 0))
    extinct   = int(final_ts < timesteps - 1 or final_pop == 0)

    wealth_vals = [float(e.get("meanWealth", 0))        for e in sim_log]
    gini_vals   = [float(e.get("giniCoefficient", 0))   for e in sim_log]
    ttl_vals    = [float(e.get("agentMeanTimeToLive", 0)) for e in sim_log
                   if e.get("agentMeanTimeToLive") is not None]

    return {
        "condition":        condition,
        "seed":             seed,
        "extinct":          extinct,
        "final_pop":        final_pop,
        "final_societal":   float(last.get("agentWealthTotal", 0)),
        "final_gini":       float(last.get("giniCoefficient", 0)),
        "final_ttl":        float(last.get("agentMeanTimeToLive", 0)),
        "mean_wealth":      float(np.mean(wealth_vals)) if wealth_vals else 0.0,
        "mean_gini":        float(np.mean(gini_vals))   if gini_vals  else 0.0,
        "mean_ttl":         float(np.mean(ttl_vals))    if ttl_vals   else 0.0,
    }


def parse_agent_log(agent_log_path: str, condition: str, seed: int):
    """Aggregate argmin_v_imm_* / argmin_v_fut_* over all rows with neighbours > 0."""
    data = safe_json_load(agent_log_path)
    if not data:
        return None

    imm_acc = np.zeros(5)
    fut_acc = np.zeros(5)
    n = 0

    rows = data if isinstance(data, list) else []
    for row in rows:
        if int(row.get("neighbors", row.get("neighbourhood", 0))) <= 0:
            continue
        try:
            imm = np.array([float(row.get(f"argmin_v_imm_{l}", 0.0)) for l in IMM_LABELS])
            fut = np.array([float(row.get(f"argmin_v_fut_{l}", 0.0)) for l in FUT_LABELS])
            imm_acc += imm
            fut_acc += fut
            n += 1
        except Exception:
            continue

    if n == 0:
        return None
    return {
        "condition": condition,
        "seed":      seed,
        "n_obs":     n,
        "mu_imm":    (imm_acc / n).tolist(),
        "mu_fut":    (fut_acc / n).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BFS
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def compute_bfs_vs_baseline(felicific_rows: list, base_pairs: dict,
                             baseline_hetero_csv: str) -> list:
    """BFS between observed argmin hetero vectors and matched baseline hetero vectors."""
    if not os.path.exists(baseline_hetero_csv):
        print(f"  [warn] Baseline hetero felicific CSV not found: {baseline_hetero_csv}")
        print(f"         Run run_experiments_baseline.py first to generate it.")
        return []

    base_rows = {}
    with open(baseline_hetero_csv, newline="") as f:
        for row in csv_mod.DictReader(f):
            key = (row["condition"], int(float(row["seed"])))
            base_rows[key] = row

    results = []
    for frow in felicific_rows:
        cond      = frow["condition"]
        seed      = frow["seed"]
        base_cond = base_pairs.get(cond)
        if not base_cond:
            continue
        brow = base_rows.get((base_cond, seed))
        if not brow:
            continue

        arg_imm  = np.array(frow["mu_imm"])
        arg_fut  = np.array(frow["mu_fut"])
        base_imm = np.array([float(brow.get(f"mean_v_imm_{l}", 0)) for l in IMM_LABELS])
        base_fut = np.array([float(brow.get(f"mean_v_fut_{l}", 0)) for l in FUT_LABELS])

        cos_imm = cosine_sim(arg_imm, base_imm)
        cos_fut = cosine_sim(arg_fut, base_fut)
        bfs     = (cos_imm + cos_fut) / 2.0

        pct = int(cond.split("_p")[-1])
        results.append({
            "argmin_condition":   cond,
            "baseline_condition": base_cond,
            "pct_bentham":        pct,
            "seed":               seed,
            "cosine_imm":         round(cos_imm, 6),
            "cosine_fut":         round(cos_fut, 6),
            "bfs":                round(bfs, 6),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Spearman correlation
# ─────────────────────────────────────────────────────────────────────────────

def spearman_r(x: list, y: list) -> float:
    n = len(x)
    if n < 2:
        return float("nan")

    def ranks(v):
        sv = sorted(range(n), key=lambda i: v[i])
        r  = [0.0] * n
        i  = 0
        while i < n:
            j = i
            while j < n - 1 and v[sv[j + 1]] == v[sv[j]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[sv[k]] = avg_rank
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mu_rx  = sum(rx) / n
    mu_ry  = sum(ry) / n
    num = sum((rx[i] - mu_rx) * (ry[i] - mu_ry) for i in range(n))
    den = math.sqrt(
        sum((rx[i] - mu_rx) ** 2 for i in range(n)) *
        sum((ry[i] - mu_ry) ** 2 for i in range(n))
    )
    return num / den if den > 1e-10 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# CSV writing
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(path: str, rows: list):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path}")


def aggregate_summaries(summary_rows: list) -> list:
    by_cond = {}
    for r in summary_rows:
        by_cond.setdefault(r["condition"], []).append(r)

    agg = []
    for cond in sorted(by_cond):
        rows = by_cond[cond]
        ext  = [r["extinct"]       for r in rows]
        pop  = [r["final_pop"]     for r in rows]
        soc  = [r["final_societal"] for r in rows]
        gin  = [r["final_gini"]    for r in rows]
        ttl  = [r["final_ttl"]     for r in rows]
        agg.append({
            "condition":              cond,
            "n_seeds":                len(rows),
            "extinction_rate":        round(float(np.mean(ext)),  4),
            "mean_final_pop":         round(float(np.mean(pop)),  2),
            "sd_final_pop":           round(float(np.std(pop)),   2),
            "mean_final_societal":    round(float(np.mean(soc)),  2),
            "mean_final_gini":        round(float(np.mean(gin)),  4),
            "sd_final_gini":          round(float(np.std(gin)),   4),
            "mean_final_ttl":         round(float(np.mean(ttl)),  4),
        })
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_dir = os.path.join(OUTPUT_DIR, "runs")
    os.makedirs(run_dir, exist_ok=True)

    base   = load_base_config("config.json")
    seeds  = generate_seeds(args.seeds)
    python = sys.executable

    conds, base_pairs = make_hetero_conditions(args.hetero_step)

    print(f"\n{'='*65}")
    print(f"  Argmin FVDM Heterogeneous Egoist–Bentham Sweep (Exp 6)")
    print(f"{'='*65}")
    print(f"  Agents: fvdmArgminEgoist + fvdmArgminBentham")
    print(f"  Profile: {PROFILE_PATH}")
    print(f"  Step: {args.hetero_step}%  Conditions: {len(conds)}")
    print(f"  Seeds: {args.seeds}   Timesteps: {args.timesteps}   Agents: {args.agents}")
    print(f"  Jobs: {args.jobs}")
    print(f"{'='*65}\n")

    # ── Build run list ────────────────────────────────────────────────────────
    run_meta  = []
    run_queue = []

    for cname, dm_list in conds.items():
        for seed in seeds:
            tag       = f"{cname}_{seed}"
            log_path  = os.path.join(run_dir, f"{tag}.json")
            alog_path = os.path.join(run_dir, f"{tag}_agents.json")
            cfg_path  = os.path.join(run_dir, f"{tag}_cfg.json")

            if not args.force and os.path.exists(log_path):
                d = safe_json_load(log_path)
                if d and len(d) > 0:
                    last_ts  = d[-1].get("timestep", 0)
                    last_pop = d[-1].get("population", -1)
                    if int(last_ts) >= args.timesteps - 1 or int(last_pop) == 0:
                        run_meta.append((cfg_path, cname, seed, log_path, alog_path))
                        continue

            cfg = make_run_config(base, seed, dm_list, args.timesteps,
                                  args.agents, log_path, alog_path)
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
            run_meta.append((cfg_path, cname, seed, log_path, alog_path))
            run_queue.append((cfg_path, python))

    total = len(run_queue)
    skipped = len(run_meta) - total
    print(f"  Total: {len(run_meta)}  |  Queued: {total}  |  Skipped: {skipped}\n")

    # ── Run simulations ───────────────────────────────────────────────────────
    if total > 0:
        mgr     = multiprocessing.Manager()
        counter = mgr.Value("i", 0)
        lock    = mgr.Lock()
        pool_args = [(cfg, py, counter, lock, total) for cfg, py in run_queue]

        t0 = time.time()
        with multiprocessing.Pool(processes=min(args.jobs, total)) as pool:
            for _ in pool.imap_unordered(run_one_simulation, pool_args):
                pass
        print(f"\n\n  Done in {time.time() - t0:.1f}s\n")
    else:
        print("  All runs already complete. Use --force to re-run.\n")

    # ── Parse results ─────────────────────────────────────────────────────────
    print("  Parsing logs ...")
    summary_rows   = []
    felicific_rows = []

    for cfg_path, cname, seed, log_path, alog_path in run_meta:
        row = parse_sim_log(log_path, cname, seed, args.timesteps)
        if row:
            summary_rows.append(row)
        frow = parse_agent_log(alog_path, cname, seed)
        if frow:
            felicific_rows.append(frow)

    agg = aggregate_summaries(summary_rows)

    # ── BFS vs baseline hetero ────────────────────────────────────────────────
    baseline_hetero_csv = os.path.join(
        args.baseline_dir, "hetero_results", "per_seed_felicific.csv"
    )
    bfs_rows = compute_bfs_vs_baseline(felicific_rows, base_pairs, baseline_hetero_csv)

    # ── Spearman correlations ─────────────────────────────────────────────────
    spearman_rows = []
    for metric, agg_key in [
        ("finalPopulation",     "mean_final_pop"),
        ("finalSocietalWealth", "mean_final_societal"),
        ("finalGini",           "mean_final_gini"),
        ("finalTTL",            "mean_final_ttl"),
        ("extinctionRate",      "extinction_rate"),
    ]:
        pct_vals, metric_vals = [], []
        for r in agg:
            pct_vals.append(int(r["condition"].split("_p")[-1]))
            metric_vals.append(r.get(agg_key, 0))
        rho = spearman_r(pct_vals, metric_vals)
        spearman_rows.append({
            "metric":  metric,
            "context": "argmin_hetero",
            "rho":     round(rho, 6) if not math.isnan(rho) else "",
            "n":       len(pct_vals),
        })

    # ── Write CSVs ────────────────────────────────────────────────────────────
    write_csv(os.path.join(OUTPUT_DIR, "per_seed_summary.csv"),     summary_rows)
    write_csv(os.path.join(OUTPUT_DIR, "condition_aggregates.csv"), agg)
    if bfs_rows:
        write_csv(os.path.join(OUTPUT_DIR, "hetero_bfs.csv"),       bfs_rows)
    write_csv(os.path.join(OUTPUT_DIR, "hetero_spearman.csv"),      spearman_rows)

    # ── Console summary ───────────────────────────────────────────────────────
    pct_levels = sorted({int(c.split("_p")[-1]) for c in conds})
    pct_bfs    = {}
    for r in bfs_rows:
        pct_bfs.setdefault(r["pct_bentham"], []).append(r["bfs"])

    print(f"\n  {'Pct':>5}  {'Condition':<28}  {'Seeds':>6}  "
          f"{'FinalPop':>9}  {'Gini':>7}  {'TTL':>7}  {'BFS':>8}")
    print(f"  {'-'*80}")
    for pct in pct_levels:
        fname    = f"argminHeteroMix_p{pct:03d}"
        arow     = next((r for r in agg if r["condition"] == fname), {})
        fp       = arow.get("mean_final_pop", 0)
        gi       = arow.get("mean_final_gini", 0)
        ttl      = arow.get("mean_final_ttl", 0)
        ns       = arow.get("n_seeds", 0)
        bfs_vals = pct_bfs.get(pct, [])
        bfs_str  = f"{np.mean(bfs_vals):.4f}" if bfs_vals else "   n/a"
        print(f"  {pct:>5}  {fname:<28}  {ns:>6}  "
              f"{fp:>9.1f}  {gi:>7.3f}  {ttl:>7.2f}  {bfs_str:>8}")

    print(f"\n  Spearman rho (pct_bentham vs metric):")
    for r in spearman_rows:
        rho_str = str(r["rho"]) if r["rho"] != "" else "n/a"
        print(f"    {r['metric']:<26}  rho = {rho_str}")

    print(f"\n  Output: {OUTPUT_DIR}/\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Argmin FVDM heterogeneous Egoist-Bentham sweep (Exp 6)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-s", "--seeds",        type=int, default=30)
    p.add_argument("-t", "--timesteps",    type=int, default=5000)
    p.add_argument("-a", "--agents",       type=int, default=250)
    p.add_argument("-j", "--jobs",         type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--hetero-step",        type=int, default=10,
                   help="Percentage-point increment for Bentham proportion sweep.")
    p.add_argument("--baseline-dir",       default="baseline_results",
                   help="Directory containing baseline_results/hetero_results/ "
                        "for BFS-vs-baseline computation.")
    p.add_argument("--force",              action="store_true",
                   help="Re-run simulations even if log files already exist.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
