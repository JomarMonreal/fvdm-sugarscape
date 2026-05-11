#!/usr/bin/env python3
"""
derive_vectors.py

Derives FVDM prioritization vectors for conditions 5–8 (Raw Sugarscape Derived,
Egoist Derived, Altruist Derived, Bentham Derived) via Behavioral Feature
Expectation (BFE) applied to observed decision trajectories from a
heterogeneous derivation population.

Algorithm (Abbeel & Ng, 2004):
  1. Run derivation simulations with all four baseline decision models present
     simultaneously (250 agents, round-robin assigned: none, egoist, altruist,
     bentham).  Heterogeneous populations trigger combat, trade, and reproduction
     across model types, producing behaviorally richer cell-selection trajectories.
  2. At every agent decision step where the chosen cell is occupied by a
     different-tribe agent (O=1, a contested move), record the combined
     feature vector keyed by the agent's decision model:
         f(c*) = E^imm(c*) + γ · E^fut(c*)
     where E^imm = (I, D, C, P, X) and E^fut = (I_f, D_f, C_f, P_f, X_f).
     Propinquity is the normalized intensity ratio:
         P   = I / (I + I_f)   (share of combined intensity that is immediate)
         P_f = I_f / (I + I_f) (complement; P + P_f = 1)
     Filtering to O=1 instances isolates the decisions where egoists actively
     seek occupied cells (combat loot) while altruists avoid them — producing
     meaningfully divergent feature expectations across model types.
  3. After all derivation runs, compute the per-type mean feature vector:
         μ_k = (1/N_k) Σ f_k(c*_t)   for each model type k
  4. Normalize each μ_k to unit length → P_i for condition k.

Justification:
  Under feature expectation matching (Abbeel & Ng, 2004), an agent's policy is
  characterized by the expected feature vector of its observed decisions. The
  mean feature vector μ(π) = E_π[f(c*)] is the simplest estimator of this
  quantity. Normalization to unit length places P_i on the same hypersphere as
  the felicific effect vectors, making Euclidean distance a consistent metric.

  Using a heterogeneous population rather than separate homogeneous runs ensures
  that agents face the full range of interaction types (combat, trade, lending,
  reproduction across model boundaries), producing cell-selection distributions
  that reflect each model's behaviour under realistic social pressure rather than
  under ecological isolation.

Output:
  data/baseline/prioritization_vectors.json   — 4 derived vectors + metadata

Usage:
    python derive_vectors.py [--seeds N] [--timesteps N] [--outdir PATH]
                             [--parallel N] [--log-interval N] [--force]
"""

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import agent as agent_module
import sugarscape as sugarscape_module
from run_baseline_conditions import BASE_CONFIG, compute_felicific_vectors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUT          = os.path.join(_SCRIPT_DIR, "data", "baseline")
DEFAULT_N_SEEDS      = 30
DEFAULT_TIMESTEPS    = 5000
DEFAULT_LOG_INTERVAL = 500
DEFAULT_PARALLEL     = 30
DEFAULT_PILOT        = 10

# All four model strings as assigned by sugarscape.py (rawSugarscape → "none")
MODEL_TO_CONDITION = {
    "none":     "rawSugarscape",
    "egoist":   "egoist",
    "altruist": "altruist",
    "bentham":  "bentham",
}

# Round-robin agent assignment in heterogeneous simulation
HETERO_MODELS = list(MODEL_TO_CONDITION.keys())

# ---------------------------------------------------------------------------
# Feature vector helper
# ---------------------------------------------------------------------------

def _feature_vector(ag, cell):
    """f(c*) = E^imm(c*) + γ · E^fut(c*), shape (5,). P = I/(I+I_f), P_f = I_f/(I+I_f)."""
    gamma = ag.decisionModelLookaheadDiscount if ag.decisionModelLookaheadFactor != 0 else 0.0
    E_imm, E_fut = compute_felicific_vectors(ag, cell)
    return np.array([
        E_imm["I"]  + gamma * E_fut["If"],
        E_imm["D"]  + gamma * E_fut["Df"],
        E_imm["C"]  + gamma * E_fut["Cf"],
        E_imm["P"]  + gamma * E_fut["Pf"],
        E_imm["X"]  + gamma * E_fut["Xf"],
    ])

# ---------------------------------------------------------------------------
# Per-process BFE state
# ---------------------------------------------------------------------------

# Per-type accumulator: model_str -> list of f(c*) arrays
_feature_accumulator = {m: [] for m in MODEL_TO_CONDITION}

# Progress tracking
_seed_idx        = 0
_n_seeds         = 0
_timesteps_total = 0
_log_interval    = DEFAULT_LOG_INTERVAL
_t0_seed         = 0.0
_t0_total        = 0.0

_orig_findBestCell = None
_orig_doTimestep   = None


def _patched_findBestCell(self):
    """Intercepts cell selection; records f(c*) only for contested moves (O=1).

    O=1: chosen cell is occupied by a different-tribe agent — the decisions
    where egoist/altruist/rawSugarscape behavioural differences emerge.
    """
    chosen_cell = _orig_findBestCell(self)
    if chosen_cell is not None:
        occupant = chosen_cell.agent
        if (occupant is not None
                and occupant is not self
                and occupant.tribe != self.tribe):
            model = self.decisionModel
            if model in _feature_accumulator:
                try:
                    _feature_accumulator[model].append(_feature_vector(self, chosen_cell))
                except Exception:
                    pass
    return chosen_cell


def _patched_doTimestep(self):
    """Wraps Sugarscape.doTimestep to print a per-type progress line."""
    _orig_doTimestep(self)

    t = self.timestep
    if t == 0 or (t % _log_interval != 0 and t != _timesteps_total):
        return

    now       = time.time()
    elapsed   = now - _t0_seed
    frac_seed = t / _timesteps_total if _timesteps_total > 0 else 1.0
    eta_seed  = (elapsed / frac_seed - elapsed) if frac_seed > 0 else 0.0

    seeds_done_frac = (_seed_idx + frac_seed) / _n_seeds if _n_seeds > 0 else 1.0
    total_elapsed   = now - _t0_total
    eta_total = (total_elapsed / seeds_done_frac - total_elapsed) if seeds_done_frac > 0 else 0.0

    pop = self.runtimeStats.get("population", "?")

    counts = {m: len(_feature_accumulator[m]) for m in MODEL_TO_CONDITION}
    counts_str = "  ".join(
        f"{MODEL_TO_CONDITION[m][:6]}={counts[m]:>7,}" for m in MODEL_TO_CONDITION
    )

    print(
        f"  seed {_seed_idx + 1}/{_n_seeds}"
        f"  t={t:>{len(str(_timesteps_total))}}/{_timesteps_total}"
        f"  pop={pop:>4}"
        f"  [{counts_str}]"
        f"  seed_eta={_fmt_time(eta_seed)}"
        f"  total_eta={_fmt_time(eta_total)}"
    )


def _fmt_time(seconds):
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def _apply_patches():
    global _orig_findBestCell, _orig_doTimestep
    _orig_findBestCell = agent_module.Agent.findBestCell
    _orig_doTimestep   = sugarscape_module.Sugarscape.doTimestep
    agent_module.Agent.findBestCell         = _patched_findBestCell
    sugarscape_module.Sugarscape.doTimestep = _patched_doTimestep


def _remove_patches():
    if _orig_findBestCell is not None:
        agent_module.Agent.findBestCell = _orig_findBestCell
    if _orig_doTimestep is not None:
        sugarscape_module.Sugarscape.doTimestep = _orig_doTimestep

# ---------------------------------------------------------------------------
# Seed-level workers (picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _setup_and_run_seed(seed, seed_idx, n_seeds, timesteps, log_interval):
    """Shared core: sets global state, patches, runs one heterogeneous seed."""
    global _feature_accumulator
    global _seed_idx, _n_seeds, _timesteps_total, _log_interval, _t0_seed, _t0_total

    _feature_accumulator = {m: [] for m in MODEL_TO_CONDITION}
    _seed_idx        = seed_idx
    _n_seeds         = n_seeds
    _timesteps_total = timesteps
    _log_interval    = log_interval
    _t0_seed         = time.time()
    _t0_total        = _t0_seed

    config = dict(BASE_CONFIG)
    config["agentDecisionModels"] = HETERO_MODELS
    config["experimentalGroup"]   = None
    config["timesteps"]           = timesteps
    config["headlessMode"]        = True
    config["logfile"]             = None
    config["agentLogfile"]        = None
    config["logfileFormat"]       = "csv"
    config["seed"]                = seed

    _apply_patches()
    try:
        random.seed(seed)
        sim = sugarscape_module.Sugarscape(config)
        try:
            sim.runSimulation(timesteps)
        except SystemExit:
            pass
    finally:
        _remove_patches()

    elapsed    = time.time() - _t0_seed
    counts     = {m: len(_feature_accumulator[m]) for m in MODEL_TO_CONDITION}
    counts_str = "  ".join(
        f"{MODEL_TO_CONDITION[m][:6]}={counts[m]:>7,}" for m in MODEL_TO_CONDITION
    )
    print(f"  seed {seed_idx + 1}/{n_seeds} complete  [{counts_str}]  ({_fmt_time(elapsed)})")


def _seed_worker(args):
    """Full-run worker: returns {model: (sum, count)} — compact for IPC."""
    seed, seed_idx, n_seeds, timesteps, log_interval = args
    _setup_and_run_seed(seed, seed_idx, n_seeds, timesteps, log_interval)
    result = {}
    for model, acc in _feature_accumulator.items():
        if acc:
            result[model] = (np.sum(acc, axis=0), len(acc))
        else:
            result[model] = (np.zeros(5), 0)
    return result


def _pilot_seed_worker(args):
    """Pilot worker: returns {model: (sum, sum_of_squares, count)} for variance."""
    seed, seed_idx, n_seeds, timesteps, log_interval = args
    _setup_and_run_seed(seed, seed_idx, n_seeds, timesteps, log_interval)
    result = {}
    for model, acc in _feature_accumulator.items():
        if acc:
            arr = np.array(acc)
            result[model] = (np.sum(arr, axis=0), np.sum(arr ** 2, axis=0), len(acc))
        else:
            result[model] = (np.zeros(5), np.zeros(5), 0)
    return result

# ---------------------------------------------------------------------------
# Pilot analysis
# ---------------------------------------------------------------------------

def _z_score(p):
    """Inverse normal CDF via statistics.NormalDist (Python ≥ 3.8)."""
    try:
        from statistics import NormalDist
        return NormalDist().inv_cdf(p)
    except ImportError:
        # Fallback hardcoded values for common α/(2k) quantiles
        table = {0.995: 2.576, 0.999: 3.090, 0.9995: 3.291}
        return min(table.items(), key=lambda kv: abs(kv[0] - p))[1]


def run_pilot(seeds, n_pilot, timesteps, log_interval, eps, alpha):
    """
    Run n_pilot heterogeneous seeds, estimate per-type per-dimension σ, and
    print the minimum --seeds required to satisfy the variance-based sample
    size bound at the given ε and α (Bonferroni-corrected over k=5 dimensions).

    Bound: N ≥ (z_{α/(2k)} · σ̂ / ε)²

    Returns the recommended number of seeds as an integer.
    """
    n_features    = 5
    feature_names = ["I", "D", "C", "P", "X"]
    z             = _z_score(1.0 - alpha / (2 * n_features))

    print(f"\nPilot mode: {n_pilot} seed(s)  |  ε={eps}  α={alpha}"
          f"  k={n_features}  z={z:.3f}\n")

    worker_args = [
        (seed, idx, n_pilot, timesteps, log_interval)
        for idx, seed in enumerate(seeds[:n_pilot])
    ]

    combined = {m: (np.zeros(5), np.zeros(5), 0) for m in MODEL_TO_CONDITION}
    for wa in worker_args:
        for model, (s, sq, n) in _pilot_seed_worker(wa).items():
            ps, psq, pn = combined[model]
            combined[model] = (ps + s, psq + sq, pn + n)

    W = 80
    print(f"\n{'═'*W}")
    print(f"  Pilot results  ({n_pilot} seed(s)  |  ε={eps}  α={alpha}  z={z:.3f})")
    print(f"{'─'*W}")
    print(f"  {'Type':<20}  {'dim':>3}  {'σ̂':>8}  {'N_req':>10}  {'n/seed':>8}  {'seeds_req':>10}")
    print(f"{'─'*W}")

    overall_max_seeds = 0

    for model, (total_sum, total_sq, total_n) in combined.items():
        cond_name    = MODEL_TO_CONDITION[model]
        avg_per_seed = total_n / n_pilot if n_pilot > 0 else 1.0
        type_max     = 0

        rows = []
        for j, fname in enumerate(feature_names):
            if total_n < 2:
                sigma = 0.0
            else:
                var   = (total_sq[j] - total_sum[j] ** 2 / total_n) / (total_n - 1)
                sigma = math.sqrt(max(var, 0.0))

            if sigma < 1e-10:
                n_req = seeds_req = 0
            else:
                n_req     = math.ceil((z * sigma / eps) ** 2)
                seeds_req = math.ceil(n_req / avg_per_seed) if avg_per_seed > 0 else 0

            type_max = max(type_max, seeds_req)
            rows.append((fname, sigma, n_req, seeds_req))

        for i, (fname, sigma, n_req, seeds_req) in enumerate(rows):
            prefix = f"  {cond_name:<20}" if i == 0 else f"  {'':20}"
            flag   = " ←" if seeds_req == type_max and seeds_req > 0 else ""
            print(f"{prefix}  {fname:>3}  {sigma:>8.4f}  {n_req:>10,}  "
                  f"{avg_per_seed:>8.0f}  {seeds_req:>10,}{flag}")

        print(f"  {'':20}  {'MAX':>3}  {'':>8}  {'':>10}  {'':>8}  {type_max:>10,}"
              f"  ← {cond_name}")
        print(f"{'─'*W}")
        overall_max_seeds = max(overall_max_seeds, type_max)

    print(f"{'═'*W}")
    print(f"  Recommendation:  --seeds {overall_max_seeds}")
    print(f"  (satisfies all types × all dims at ε={eps}, α={alpha},"
          f" Bonferroni over {n_features} dims)")
    print(f"{'═'*W}\n")

    return overall_max_seeds


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def _load_or_generate_seeds(seeds_path, n_seeds):
    if os.path.exists(seeds_path):
        with open(seeds_path) as f:
            seeds = json.load(f)
        if len(seeds) >= n_seeds:
            print(f"Reusing {n_seeds} seeds from {seeds_path}")
            return seeds[:n_seeds]
    seeds = []
    while len(seeds) < n_seeds:
        s = random.randint(0, sys.maxsize)
        if s not in seeds:
            seeds.append(s)
    with open(seeds_path, "w") as f:
        json.dump(seeds, f)
    print(f"Generated {n_seeds} derivation seeds → {seeds_path}")
    return seeds

# ---------------------------------------------------------------------------
# Save / merge results
# ---------------------------------------------------------------------------

def _save_vectors(vectors_path, condition_vectors, decision_counts, seeds, timesteps):
    existing = {}
    if os.path.exists(vectors_path):
        try:
            with open(vectors_path) as f:
                existing = json.load(f)
        except Exception:
            pass

    for condition_name, pi in condition_vectors.items():
        existing[condition_name] = {
            "vector":          [round(float(v), 6) for v in pi],
            "method":          "behavioral_feature_expectation_heterogeneous",
            "seeds_used":      len(seeds),
            "timesteps":       timesteps,
            "total_decisions": decision_counts.get(condition_name, 0),
        }

    with open(vectors_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Prioritization vectors saved → {vectors_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds",        type=int, default=DEFAULT_N_SEEDS,
                   help="Derivation seeds (default: %(default)s)")
    p.add_argument("--timesteps",    type=int, default=DEFAULT_TIMESTEPS,
                   help="Timesteps per derivation run (default: %(default)s)")
    p.add_argument("--outdir",       default=DEFAULT_OUT,
                   help="Output directory (default: %(default)s)")
    p.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL,
                   dest="log_interval",
                   help="Print progress every N timesteps (default: %(default)s)")
    p.add_argument("--parallel",     type=int, default=DEFAULT_PARALLEL,
                   help="Seeds to run in parallel (default: %(default)s)")
    p.add_argument("--pilot",        type=int, default=DEFAULT_PILOT, metavar="N",
                   help="Run N pilot seeds to estimate σ and print required sample "
                        "size; set to 0 to skip (default: %(default)s)")
    p.add_argument("--eps",          type=float, default=0.01,
                   help="Target mean-estimation error ε for sample-size bound (default: %(default)s)")
    p.add_argument("--alpha",        type=float, default=0.05,
                   help="Significance level α for sample-size bound (default: %(default)s)")
    p.add_argument("--force",        action="store_true",
                   help="Re-derive even if vectors already exist")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    seeds_path   = os.path.join(args.outdir, "seeds.json")
    vectors_path = os.path.join(args.outdir, "prioritization_vectors.json")
    seeds        = _load_or_generate_seeds(seeds_path, max(args.seeds, args.pilot))

    # Pilot phase: estimate σ and recommend seed count
    if args.pilot > 0:
        recommended = run_pilot(seeds, args.pilot, args.timesteps,
                                args.log_interval, args.eps, args.alpha)
        reply = input(f"Continue with full derivation using --seeds {args.seeds}? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted. Re-run with --seeds <N> as recommended.")
            return

    # Skip if already derived (unless forced)
    if not args.force and os.path.exists(vectors_path):
        try:
            with open(vectors_path) as f:
                saved = json.load(f)
            if all(c in saved and "vector" in saved[c]
                   for c in MODEL_TO_CONDITION.values()):
                print("All vectors already derived — skipping (use --force to re-derive).")
                for cname in MODEL_TO_CONDITION.values():
                    print(f"  {cname:<24}  {[round(v,4) for v in saved[cname]['vector']]}")
                return
        except Exception:
            pass

    n_workers = min(args.parallel, len(seeds))

    print(
        f"\nDerivation settings (BFE — heterogeneous population):"
        f"\n  method      : mean f(c*) per type, normalized (Abbeel & Ng, 2004)"
        f"\n  models      : {HETERO_MODELS}  (round-robin, 250 agents)"
        f"\n  seeds       : {args.seeds}"
        f"\n  timesteps   : {args.timesteps} (per seed)"
        f"\n  log_interval: every {args.log_interval} timesteps"
        f"\n  parallel    : {n_workers} seed(s) at a time"
        f"\n  output      : {vectors_path}\n"
    )

    worker_args = [
        (seed, idx, len(seeds), args.timesteps, args.log_interval)
        for idx, seed in enumerate(seeds)
    ]

    # Aggregate sums and counts across all seeds
    combined = {m: (np.zeros(5), 0) for m in MODEL_TO_CONDITION}
    t0_total = time.time()

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_seed_worker, a): a[0] for a in worker_args}
            for future in as_completed(futures):
                seed_result = future.result()
                for model, (s, n) in seed_result.items():
                    prev_s, prev_n = combined[model]
                    combined[model] = (prev_s + s, prev_n + n)
    else:
        global _t0_total
        _t0_total = t0_total
        for wa in worker_args:
            seed_result = _seed_worker(wa)
            for model, (s, n) in seed_result.items():
                prev_s, prev_n = combined[model]
                combined[model] = (prev_s + s, prev_n + n)

    # Compute per-condition normalized vectors
    condition_vectors = {}
    decision_counts   = {}
    for model, (total_sum, total_n) in combined.items():
        cond_name = MODEL_TO_CONDITION[model]
        decision_counts[cond_name] = total_n
        if total_n == 0:
            print(f"  WARNING: no decisions for '{model}' — defaulting to uniform")
            pi = np.full(5, 1.0 / math.sqrt(5))
        else:
            mu   = total_sum / total_n
            norm = np.linalg.norm(mu)
            pi   = mu / norm if norm > 1e-10 else np.full(5, 1.0 / math.sqrt(5))
        condition_vectors[cond_name] = pi

    _save_vectors(vectors_path, condition_vectors, decision_counts, seeds, args.timesteps)

    total_time = time.time() - t0_total
    print(f"\n{'═'*72}")
    print(f"  {'Condition':<24}  {'p_I':>7}  {'p_D':>7}  {'p_C':>7}  {'p_P':>7}  {'p_X':>7}  {'N':>12}")
    print(f"{'─'*72}")
    for cond_name, pi in condition_vectors.items():
        n = decision_counts[cond_name]
        vals = "  ".join(f"{v:+.4f}" for v in pi)
        print(f"  {cond_name:<24}  {vals}  {n:>12,}")
    print(f"{'═'*72}")
    print(f"  Total derivation time: {_fmt_time(total_time)}")
    print(f"  Vectors written to: {vectors_path}\n")


if __name__ == "__main__":
    main()
