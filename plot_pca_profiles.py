#!/usr/bin/env python3
"""
plot_pca_profiles.py
---------------------
PCA projection of the behavioral space to visually confirm that:
  1. Ethical types are distinct in 10-dimensional felicific space.
  2. FVDM argmin agents' observed vectors cluster near their matched profiles.

Data sources (any that exist are used):
  fvdm_vectors/bfe_profiles_phi.json         — 4 derived profiles (reference)
  results/argmin/per_seed_felicific.csv      — per-seed observed vectors, Exp 5
  results/argmin_hetero/per_seed_felicific.csv — hetero sweep (optional, Exp 6)

PCA is fitted on the 4 derived profiles, then all other points are projected
into the same space.  This keeps the reference frame stable regardless of how
many per-seed observations are loaded.

Output:
  plots/pca_profiles.pdf  (and .png)

Usage:
  python plot_pca_profiles.py
  python plot_pca_profiles.py --no-hetero --output plots/pca_profiles.pdf
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PROFILE_PATH        = "fvdm_vectors/bfe_profiles_phi.json"
ARGMIN_FELICIFIC    = "results/argmin/per_seed_felicific.csv"
HETERO_FELICIFIC    = "results/argmin_hetero/per_seed_felicific.csv"

IMM_LABELS = ["I", "D", "C", "P", "E"]
FUT_LABELS = ["I", "D", "C", "P", "E"]

PROFILE_ORDER = ["rawSugarscape", "egoist", "altruist", "bentham"]

COLORS = {
    "rawSugarscape": "#1f77b4",   # blue
    "egoist":        "#d62728",   # red
    "altruist":      "#2ca02c",   # green
    "bentham":       "#ff7f0e",   # orange
    # argmin condition names
    "argminRaw":      "#1f77b4",
    "argminEgoist":   "#d62728",
    "argminAltruist": "#2ca02c",
    "argminBentham":  "#ff7f0e",
}

LABELS = {
    "rawSugarscape": "Raw Sugarscape",
    "egoist":        "Egoist",
    "altruist":      "Altruist",
    "bentham":       "Bentham",
    "argminRaw":     "Raw (Argmin)",
    "argminEgoist":  "Egoist (Argmin)",
    "argminAltruist":"Altruist (Argmin)",
    "argminBentham": "Bentham (Argmin)",
}

ARGMIN_TO_PROFILE = {
    "argminRaw":      "rawSugarscape",
    "argminEgoist":   "egoist",
    "argminAltruist": "altruist",
    "argminBentham":  "bentham",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def profile_to_vec(profile: dict) -> np.ndarray:
    imm = [float(profile["mu_imm"][i]) for i in range(5)]
    fut = [float(profile["mu_fut"][i]) for i in range(5)]
    return np.array(imm + fut)


def load_profiles(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    profiles = data.get("profiles", {})
    return {k: profile_to_vec(v) for k, v in profiles.items()}


def load_felicific_csv(path: str) -> dict:
    """Returns {condition: np.ndarray of shape (n_seeds, 10)}."""
    if not os.path.exists(path):
        return {}
    by_cond = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row["condition"]
            imm  = [float(row.get(f"mu_imm_{l}", 0.0)) for l in IMM_LABELS]
            fut  = [float(row.get(f"mu_fut_{l}", 0.0)) for l in FUT_LABELS]
            vec  = imm + fut
            if any(v != 0.0 for v in vec):
                by_cond.setdefault(cond, []).append(vec)
    return {k: np.array(v) for k, v in by_cond.items()}


# ─────────────────────────────────────────────────────────────────────────────
# PCA (no sklearn dependency)
# ─────────────────────────────────────────────────────────────────────────────

def pca_fit(X: np.ndarray):
    """Fit PCA on X (n_samples × n_features). Returns (mean, components)."""
    mean = X.mean(axis=0)
    Xc   = X - mean
    cov  = Xc.T @ Xc / max(len(X) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    components = eigvecs[:, order].T   # shape (n_features, n_features)
    explained  = eigvals[order] / (eigvals.sum() + 1e-12)
    return mean, components, explained


def pca_transform(X: np.ndarray, mean: np.ndarray, components: np.ndarray,
                  n_components: int = 2) -> np.ndarray:
    return (X - mean) @ components[:n_components].T


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(args):
    if not os.path.exists(PROFILE_PATH):
        print(f"[error] {PROFILE_PATH} not found. Run derive_vectors_phi.py first.")
        sys.exit(1)

    profiles  = load_profiles(PROFILE_PATH)
    argmin_obs = load_felicific_csv(ARGMIN_FELICIFIC)

    has_argmin = bool(argmin_obs)
    if not has_argmin:
        print(f"  [info] {ARGMIN_FELICIFIC} not found — plotting derived profiles only.")
        print(f"         Transfer Exp 5 results and re-run, or re-parse on server.")

    # Fit PCA on the 4 derived profiles
    profile_keys  = [k for k in PROFILE_ORDER if k in profiles]
    profile_mat   = np.array([profiles[k] for k in profile_keys])  # (4, 10)
    mean, comps, expl = pca_fit(profile_mat)

    pc1_var = expl[0] * 100
    pc2_var = expl[1] * 100

    # Show at least 2 sig figs for small variances
    def fmt_var(v):
        if v >= 1.0:
            return f"{v:.1f}%"
        if v >= 0.01:
            return f"{v:.2f}%"
        return f"{v:.2e}%"

    # Project profiles
    proj_profiles = pca_transform(profile_mat, mean, comps)   # (4, 2)

    # Project argmin per-seed observations
    proj_argmin = {}
    for cond, mat in argmin_obs.items():
        proj_argmin[cond] = pca_transform(mat, mean, comps)   # (n, 2)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Per-seed argmin clouds (drawn first, behind profiles)
    for cond, pts in proj_argmin.items():
        profile_key = ARGMIN_TO_PROFILE.get(cond, cond)
        color = COLORS.get(profile_key, "#888888")
        ax.scatter(pts[:, 0], pts[:, 1],
                   color=color, alpha=0.25, s=25, marker="o", zorder=2)

    # Convex hulls / ellipses around each argmin cloud (optional: skip if < 3 pts)
    for cond, pts in proj_argmin.items():
        if len(pts) < 3:
            continue
        profile_key = ARGMIN_TO_PROFILE.get(cond, cond)
        color = COLORS.get(profile_key, "#888888")
        mu  = pts.mean(axis=0)
        cov = np.cov(pts.T)
        if cov.ndim < 2:
            continue
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        width  = 2 * 1.96 * np.sqrt(max(eigvals[0], 0))
        height = 2 * 1.96 * np.sqrt(max(eigvals[1], 0))
        angle  = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ell = matplotlib.patches.Ellipse(
            mu, width, height, angle=angle,
            edgecolor=color, facecolor="none", linewidth=1.2,
            linestyle="--", zorder=3, alpha=0.6
        )
        ax.add_patch(ell)

    # Derived profile reference points (large, bold)
    for i, k in enumerate(profile_keys):
        color = COLORS.get(k, "#888888")
        px, py = proj_profiles[i]
        ax.scatter(px, py, color=color, s=200, marker="D",
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.annotate(LABELS.get(k, k),
                    xy=(px, py), xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9, fontweight="bold", color=color)

    ax.set_xlabel(f"PC 1  ({pc1_var:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PC 2  ({pc2_var:.1f}% variance)", fontsize=10)
    ax.set_title("PCA Projection of Behavioral Space\n"
                 "(diamonds = derived profiles; clouds = argmin observed per seed)",
                 fontsize=10)
    ax.axhline(0, color="#cccccc", linewidth=0.5, zorder=0)
    ax.axvline(0, color="#cccccc", linewidth=0.5, zorder=0)
    ax.grid(True, alpha=0.2, zorder=0)

    # Legend
    legend_handles = []
    for k in profile_keys:
        color = COLORS.get(k, "#888888")
        legend_handles.append(
            Line2D([0], [0], marker="D", color="w", markerfacecolor=color,
                   markeredgecolor="black", markersize=9,
                   label=f"{LABELS.get(k, k)} (profile)")
        )
    if has_argmin:
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
                   markersize=6, alpha=0.5, label="Argmin observed (per seed)")
        )
    ax.legend(handles=legend_handles, fontsize=8, loc="best")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    png_path = args.output.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved → {args.output}")
    print(f"  Saved → {png_path}")
    print(f"\n  PC1 variance: {pc1_var:.1f}%")
    print(f"  PC2 variance: {pc2_var:.1f}%")
    print(f"  Total (PC1+PC2): {pc1_var + pc2_var:.1f}%\n")

    # ── Coordinate loadings report ────────────────────────────────────────────
    coord_names = [f"imm_{l}" for l in IMM_LABELS] + [f"fut_{l}" for l in FUT_LABELS]
    print("  PC1 loadings:")
    for name, load in sorted(zip(coord_names, comps[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {name:<12}  {load:+.4f}")
    print("  PC2 loadings:")
    for name, load in sorted(zip(coord_names, comps[1]), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {name:<12}  {load:+.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="PCA projection of FVDM behavioral space",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-o", "--output", default="plots/pca_profiles.pdf")
    return p.parse_args()


if __name__ == "__main__":
    plot(parse_args())
