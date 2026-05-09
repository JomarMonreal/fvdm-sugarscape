#!/usr/bin/env python3
"""
train_coordinates.py
--------------------
Felicific Coordinate Prediction Functions for FVDM.

Trains the prediction models that map (local state, action) → felicific
effect vector E(a|s_i) = (I, D, C, P, X).

Models:
  - NGBoost  → Intensity, Duration, Propinquity  (continuous, probabilistic)
  - Certainty is derived from NGBoost's predicted Intensity variance
  - Multinomial Gradient Boosting → Extent  (categorical {-1, 0, 1})

Pipeline:
  1. Load the derivation CSV produced by run_focal_action.py
  2. Compute ground-truth felicific coordinates from raw outcomes
  3. Filter to discretionary actions only (exclude "none")
  4. Train per-action NGBoost models for I, D, P
  5. Train per-action multinomial classifiers for X (extent)
  6. Evaluate with CRPS learning curves (sample sufficiency)
  7. Save trained models + normalization constants

Reference: Thesis Section 3.4.4 — Felicific Coordinate Prediction Functions.

Usage:
  .venv/bin/python train_coordinates.py [options]
"""

import argparse
import csv
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, log_loss

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────
# Feature columns (local state s_i)
# ─────────────────────────────────────────────────────────────────

STATE_FEATURES = [
    "age",
    "wealth",
    "sugar",
    "spice",
    "timeToLive",
    "movement",
    "neighbors",
    "neighborsInTribe",
    "neighborsNotInTribe",
    "validMoves",
    "compositeHappiness",
    "depression",
]

DISCRETIONARY_ACTIONS = ["combat", "trade", "reproduction", "lending"]


# ─────────────────────────────────────────────────────────────────
# Ground-truth coordinate computation
# ─────────────────────────────────────────────────────────────────

def compute_ground_truth_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ground-truth felicific coordinates from raw outcome data.

    Adds columns: gt_intensity_raw, gt_intensity, gt_duration,
                  gt_propinquity, gt_extent
    """
    df = df.copy()

    # ── Raw intensity = wealth change (before normalization) ──
    df["gt_intensity_raw"] = df["wealthGained"].astype(float)

    # ── Intensity: I(a,s_i) = ΔW / M_{I,a}  ──
    # M_{I,a} = max |ΔW| per action category
    max_abs_per_action = (
        df.groupby("action")["gt_intensity_raw"]
        .transform(lambda x: x.abs().max())
    )
    # Avoid division by zero
    max_abs_per_action = max_abs_per_action.replace(0, 1)
    df["gt_intensity"] = df["gt_intensity_raw"] / max_abs_per_action

    # ── Duration: D = max(0, min(1, |ΔW / (W_t + 1)|))  ──
    # W_t is wealth BEFORE the action = wealth - wealthGained
    wealth_before = df["wealth"].astype(float) - df["wealthGained"].astype(float)
    df["gt_duration"] = np.clip(
        np.abs(df["wealthGained"].astype(float) / (wealth_before + 1)),
        0, 1
    )

    # ── Propinquity: P = |ΔW_imm| / (|ΔW_imm| + |ΔW_fut| + ε) ──
    # Immediate effect = wealthGained (this timestep)
    # Future proxy = |timeToLiveDifference| (change in survival horizon)
    eps = 1e-8
    imm = df["wealthGained"].astype(float).abs()
    fut = df["timeToLiveDifference"].astype(float).abs()
    df["gt_propinquity"] = np.clip(imm / (imm + fut + eps), 0, 1)

    # ── Extent: X ∈ {-1, 0, 1}  ──
    # -1 = self-directed (combat: takes from others for self)
    #  0 = no relevant social effect (none / movement only)
    #  1 = other-directed (trade, reproduction, lending)
    extent_map = {
        "combat": -1,
        "trade": 1,
        "reproduction": 1,
        "lending": 1,
        "none": 0,
    }
    df["gt_extent"] = df["action"].map(extent_map).fillna(0).astype(int)

    return df


def compute_normalization_constants(df: pd.DataFrame) -> dict:
    """Extract M_{I,a} per action for use during inference."""
    constants = {}
    for action in DISCRETIONARY_ACTIONS:
        subset = df[df["action"] == action]
        if len(subset) > 0:
            constants[action] = float(subset["gt_intensity_raw"].abs().max())
        else:
            constants[action] = 1.0
    return constants


# ─────────────────────────────────────────────────────────────────
# CRPS scorer for NGBoost evaluation
# ─────────────────────────────────────────────────────────────────

def crps_normal(y_true, y_pred_dist_params):
    """Closed-form CRPS for a Normal distribution.

    For Normal(μ, σ):  CRPS = σ [z Φ(z) + φ(z) - 1/√π]
    where z = (y - μ) / σ
    """
    from scipy.stats import norm as scipy_norm

    mu = y_pred_dist_params[:, 0]
    sigma = np.abs(y_pred_dist_params[:, 1]) + 1e-8
    z = (y_true - mu) / sigma
    crps_vals = sigma * (
        z * (2 * scipy_norm.cdf(z) - 1)
        + 2 * scipy_norm.pdf(z)
        - 1 / np.sqrt(np.pi)
    )
    return np.mean(crps_vals)


# ─────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────

def train_ngboost_model(X: np.ndarray, y: np.ndarray,
                        n_estimators: int = 200,
                        learning_rate: float = 0.05,
                        verbose: bool = False) -> NGBRegressor:
    """Train an NGBoost regressor for a continuous felicific coordinate."""
    model = NGBRegressor(
        Dist=Normal,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        verbose=verbose,
        random_state=42,
    )
    model.fit(X, y)
    return model


def train_extent_classifier(X: np.ndarray, y: np.ndarray,
                            n_estimators: int = 200,
                            learning_rate: float = 0.1) -> GradientBoostingClassifier:
    """Train a multinomial gradient boosting classifier for Extent."""
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=4,
        random_state=42,
    )
    model.fit(X, y)
    return model


def evaluate_learning_curve(X: np.ndarray, y: np.ndarray,
                            fractions: list = None,
                            n_folds: int = 5) -> list:
    """Compute CRPS at increasing sample sizes to check sufficiency.

    Returns list of {"n": sample_size, "crps_mean": ..., "crps_std": ...}.
    """
    if fractions is None:
        fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

    results = []
    n_total = len(y)

    for frac in fractions:
        n = int(n_total * frac)
        if n < 50:
            continue

        # Subsample
        idx = np.random.RandomState(42).choice(n_total, size=n, replace=False)
        X_sub, y_sub = X[idx], y[idx]

        # Cross-validated CRPS
        kf = KFold(n_splits=min(n_folds, n), shuffle=True, random_state=42)
        fold_crps = []

        for train_idx, val_idx in kf.split(X_sub):
            model = NGBRegressor(
                Dist=Normal, n_estimators=100,
                learning_rate=0.05, verbose=False, random_state=42,
            )
            model.fit(X_sub[train_idx], y_sub[train_idx])
            y_dists = model.pred_dist(X_sub[val_idx])
            params = np.column_stack([y_dists.loc, y_dists.scale])
            fold_crps.append(crps_normal(y_sub[val_idx], params))

        results.append({
            "n": n,
            "crps_mean": round(float(np.mean(fold_crps)), 6),
            "crps_std": round(float(np.std(fold_crps)), 6),
        })

    return results


# ─────────────────────────────────────────────────────────────────
# Prediction functions (for use at inference time)
# ─────────────────────────────────────────────────────────────────

def predict_effect_vector(state_features: np.ndarray, action: str,
                          models: dict, norm_constants: dict) -> dict:
    """Predict the full felicific effect vector E(a|s_i) = (I, D, C, P, X).

    Args:
        state_features: 1D array of local state features for one agent.
        action: one of the discretionary action strings.
        models: dict with keys like "intensity_combat", "extent_combat", etc.
        norm_constants: M_{I,a} per action category.

    Returns:
        dict with keys I, D, C, P, X (all floats).
    """
    X = state_features.reshape(1, -1)

    # ── Intensity (NGBoost) ──
    intensity_model = models[f"intensity_{action}"]
    i_dist = intensity_model.pred_dist(X)
    I_pred = float(i_dist.loc[0])
    I_var = float(i_dist.scale[0] ** 2)

    # ── Duration (NGBoost) ──
    duration_model = models[f"duration_{action}"]
    D_pred = float(duration_model.predict(X)[0])

    # ── Propinquity (NGBoost) ──
    propinquity_model = models[f"propinquity_{action}"]
    P_pred = float(propinquity_model.predict(X)[0])

    # ── Certainty (from Intensity variance) ──
    C_pred = 1.0 / (1.0 + I_var)

    # ── Extent (multinomial classifier → expected value) ──
    extent_key = f"extent_{action}"
    if extent_key in models:
        extent_model = models[extent_key]
        proba = extent_model.predict_proba(X)[0]
        classes = extent_model.classes_
        X_pred = float(sum(c * p for c, p in zip(classes, proba)))
    else:
        # Single-class fallback: classifier was skipped during training
        extent_defaults = {"combat": -1.0, "trade": 1.0,
                           "reproduction": 1.0, "lending": 1.0}
        X_pred = extent_defaults.get(action, 0.0)

    return {
        "I": round(I_pred, 6),
        "D": round(np.clip(D_pred, 0, 1), 6),
        "C": round(C_pred, 6),
        "P": round(np.clip(P_pred, 0, 1), 6),
        "X": round(X_pred, 6),
    }


# ─────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────

# Minimum observations required PER ACTION (thesis §3.4.4)
# 22 predictors × 10 EPV × 3 (boosting adjustment) = 660, rounded up to 2,640
MIN_OBS_PER_ACTION = 2640


def main(args):
    input_csv = args.input
    output_dir = args.output
    n_estimators = args.estimators
    learning_rate = args.lr

    os.makedirs(output_dir, exist_ok=True)

    pipeline_start = time.time()

    # ── 1. Load and Filter derivation data in batches ────────────
    print(f"\n{'='*60}")
    print(f"  Felicific Coordinate Prediction — Model Training")
    print(f"{'='*60}\n")

    print(f"  Loading and filtering derivation data in chunks from: {input_csv}")
    
    # Use 100k row chunks to prevent RAM overflow
    chunk_size = 100000
    discretionary_chunks = []
    total_rows = 0
    
    try:
        reader = pd.read_csv(input_csv, chunksize=chunk_size)
        for i, chunk in enumerate(reader):
            total_rows += len(chunk)
            # Filter immediately: keep only Combat, Trade, Repro, Lending
            filtered = chunk[chunk["action"].isin(DISCRETIONARY_ACTIONS)].copy()
            discretionary_chunks.append(filtered)
            
            current_disc = sum(len(c) for c in discretionary_chunks)
            print(f"    Batch {i+1}: Processed {total_rows:,} rows... (Found {current_disc:,} discretionary)", end="\r")
        
        print(f"\n  Finished reading file.")
        df_disc = pd.concat(discretionary_chunks, ignore_index=True)
    except FileNotFoundError:
        print(f"\n  [error] Input file not found: {input_csv}")
        return

    # ── 2. Compute ground-truth coordinates on filtered subset ──
    print(f"  Computing ground-truth coordinates on {len(df_disc):,} observations …")
    df_disc = compute_ground_truth_coordinates(df_disc)

    n_discretionary = len(df_disc)
    print(f"  Discretionary observations: {n_discretionary}")

    action_counts = df_disc["action"].value_counts()
    per_action_counts = {}
    for act in DISCRETIONARY_ACTIONS:
        count = action_counts.get(act, 0)
        per_action_counts[act] = int(count)
        print(f"    {act:<15} {count:>6} observations")

    if n_discretionary == 0:
        print("\n  [error] No discretionary action observations. Aborting.\n")
        return

    # ── Sample-size guard (thesis §3.4.4) — per action ───────────
    short_actions = []
    for act in DISCRETIONARY_ACTIONS:
        count = per_action_counts.get(act, 0)
        if count < MIN_OBS_PER_ACTION:
            short_actions.append((act, count))

    if short_actions:
        print(f"\n  [warning] Some actions have fewer than {MIN_OBS_PER_ACTION} observations:")
        for act, count in short_actions:
            print(f"            {act}: {count} (need {MIN_OBS_PER_ACTION})")
        print(f"          Training will proceed with available data.")
        print(f"          Consider increasing --seeds or --timesteps for better results.\n")

    # ── Cap each action to exactly MIN_OBS_PER_ACTION ────────────
    capped_chunks = []
    for act in DISCRETIONARY_ACTIONS:
        act_df = df_disc[df_disc["action"] == act]
        if len(act_df) > MIN_OBS_PER_ACTION:
            act_df = act_df.sample(n=MIN_OBS_PER_ACTION, random_state=42)
        capped_chunks.append(act_df)

    df_disc = pd.concat(capped_chunks, ignore_index=True)
    n_discretionary = len(df_disc)

    print(f"\n  Using {n_discretionary:,} observations for training "
          f"(capped at {MIN_OBS_PER_ACTION} per action):")
    action_counts = df_disc["action"].value_counts()
    per_action_counts = {}
    for act in DISCRETIONARY_ACTIONS:
        count = action_counts.get(act, 0)
        per_action_counts[act] = int(count)
        print(f"    {act:<15} {count:>6} observations")

    # ── 4. Save normalization constants ──────────────────────────
    norm_constants = compute_normalization_constants(df_disc)
    norm_path = os.path.join(output_dir, "normalization_constants.json")
    with open(norm_path, "w") as f:
        json.dump(norm_constants, f, indent=2)
    print(f"\n  Normalization constants M_{{I,a}}:")
    for act, val in norm_constants.items():
        print(f"    {act:<15} {val:.4f}")

    # ── 5. Train per-action models ───────────────────────────────
    models = {}
    learning_curves = {}
    per_action_timing = {}

    for action in DISCRETIONARY_ACTIONS:
        subset = df_disc[df_disc["action"] == action]
        if len(subset) < 20:
            print(f"\n  [skip] {action}: only {len(subset)} samples (need ≥ 20)")
            continue

        X = subset[STATE_FEATURES].values.astype(float)
        y_intensity = subset["gt_intensity"].values.astype(float)
        y_duration = subset["gt_duration"].values.astype(float)
        y_propinquity = subset["gt_propinquity"].values.astype(float)
        y_extent = subset["gt_extent"].values.astype(int)

        print(f"\n  ── Training models for: {action} ({len(subset)} samples) ──")
        action_start = time.time()

        # Intensity (NGBoost)
        print(f"    Intensity (NGBoost) …", end=" ", flush=True)
        m_int = train_ngboost_model(X, y_intensity, n_estimators, learning_rate)
        models[f"intensity_{action}"] = m_int
        print("✓")

        # Duration (NGBoost)
        print(f"    Duration  (NGBoost) …", end=" ", flush=True)
        m_dur = train_ngboost_model(X, y_duration, n_estimators, learning_rate)
        models[f"duration_{action}"] = m_dur
        print("✓")

        # Propinquity (NGBoost)
        print(f"    Propinquity (NGBoost) …", end=" ", flush=True)
        m_prop = train_ngboost_model(X, y_propinquity, n_estimators, learning_rate)
        models[f"propinquity_{action}"] = m_prop
        print("✓")

        # Extent (Multinomial GBM)
        unique_classes = np.unique(y_extent)
        if len(unique_classes) > 1:
            print(f"    Extent (GBClassifier) …", end=" ", flush=True)
            m_ext = train_extent_classifier(X, y_extent, n_estimators)
            models[f"extent_{action}"] = m_ext
            print("✓")
        else:
            print(f"    Extent: single class ({unique_classes[0]}), skipping classifier")

        # CRPS learning curve for Intensity
        print(f"    Computing CRPS learning curve …", end=" ", flush=True)
        lc = evaluate_learning_curve(X, y_intensity)
        learning_curves[action] = lc
        print("✓")

        action_elapsed = round(time.time() - action_start, 2)
        per_action_timing[action] = action_elapsed
        print(f"    Elapsed: {action_elapsed:.2f}s")

        if lc:
            crps_max = lc[-1]["crps_mean"]
            print(f"    CRPS at full sample: {crps_max:.6f}")
            # Check 1.01× threshold (thesis §3.6.3)
            for point in lc:
                if point["crps_mean"] <= 1.01 * crps_max:
                    print(f"    Sufficient at n={point['n']} "
                          f"(CRPS={point['crps_mean']:.6f} ≤ "
                          f"1.01×{crps_max:.6f}={1.01*crps_max:.6f})")
                    break

    # ── 6. Save models ───────────────────────────────────────────
    print(f"\n  Saving models to: {output_dir}/")
    for name, model in models.items():
        model_path = os.path.join(output_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"    {name}.pkl")

    # ── 7. Save learning curves ──────────────────────────────────
    lc_path = os.path.join(output_dir, "learning_curves.json")
    with open(lc_path, "w") as f:
        json.dump(learning_curves, f, indent=2)
    print(f"    learning_curves.json")

    # ── 8. Demo prediction ───────────────────────────────────────
    print(f"\n  ── Sample predictions (first observation per action) ──")
    for action in DISCRETIONARY_ACTIONS:
        if f"intensity_{action}" not in models:
            continue
        subset = df_disc[df_disc["action"] == action]
        if len(subset) == 0:
            continue
        sample = subset.iloc[0]
        X_sample = sample[STATE_FEATURES].values.astype(float)
        vec = predict_effect_vector(X_sample, action, models, norm_constants)
        gt = {
            "I": round(float(sample["gt_intensity"]), 4),
            "D": round(float(sample["gt_duration"]), 4),
            "P": round(float(sample["gt_propinquity"]), 4),
            "X": int(sample["gt_extent"]),
        }
        print(f"    {action:<15} predicted={vec}  ground_truth={gt}")

    # ── 9. Save training report ───────────────────────────────────
    pipeline_elapsed = round(time.time() - pipeline_start, 2)

    report = {
        "input_csv": input_csv,
        "total_observations": int(total_rows),
        "discretionary_observations": int(n_discretionary),
        "minimum_required_per_action": MIN_OBS_PER_ACTION,
        "per_action_observations": per_action_counts,
        "per_action_training_time_seconds": per_action_timing,
        "total_training_time_seconds": pipeline_elapsed,
        "models_trained": len(models),
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "state_features": STATE_FEATURES,
        "normalization_constants": norm_constants,
        "learning_curves": learning_curves,
    }
    report_path = os.path.join(output_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"    training_report.json")

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Models saved:          {len(models)}")
    print(f"  Observations used:     {n_discretionary} ({MIN_OBS_PER_ACTION} per action cap)")
    print(f"  Total elapsed time:    {pipeline_elapsed:.2f}s")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train felicific coordinate prediction functions for FVDM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        default="focal_action_results/results/focal_action_derivation.csv",
        help="Path to the combined derivation CSV from run_focal_action.py.",
    )
    parser.add_argument(
        "-o", "--output",
        default="fvdm_models",
        help="Directory to save trained models and metadata.",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=200,
        help="Number of boosting estimators for NGBoost and GBClassifier.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate for NGBoost models.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
