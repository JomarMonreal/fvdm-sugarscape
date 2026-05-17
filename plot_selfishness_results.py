#!/usr/bin/env python3
"""
plot_selfishness_results.py
----------------------------
Visualizes the selfishness-factor sweep produced by
run_experiments_selfishness.py, replicating the outcome-gradient
analysis of Herman & Kremer (2024) Section VII-C (Figures 10–15).

Uses the per-seed data to compute median + Q1/Q3 quartile bands per φ level.

Reads:
  <results_dir>/per_seed_summary.csv     — one row per (condition, seed)
  <results_dir>/selfishness_spearman.csv — optional Spearman annotations

Produces (in --out directory):
  01_population.png
  02_societal_wealth.png
  03_mean_wealth.png
  04_ttl.png
  05_deaths_per_timestep.png
  06_age_at_death.png
  07_gini.png
  08_actions.png
  09_summary_panel.png   ← main replication figure

Usage:
  python plot_selfishness_results.py
  python plot_selfishness_results.py --results selfishness_results/results
  python plot_selfishness_results.py --results selfishness_results/results --out selfishness_plots
"""

import argparse
import csv
import os
import re
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy are required.  pip install matplotlib numpy")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
GRAY   = "#6B7280"
PURPLE = "#7C3AED"
AMBER  = "#D97706"
CYAN   = "#0891B2"


def load_csv(path):
    if not os.path.exists(path):
        print(f"  [warn] File not found: {path}")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def extract_phi(condition_name):
    """Parse φ from condition name like 'phi_050' → 0.50."""
    m = re.search(r"phi_(\d+)", condition_name)
    if m:
        return round(int(m.group(1)) / 100, 10)
    # Also accept raw float in a 'phi' column
    return None


def build_quartile_series(rows, metric):
    """
    Group per-seed rows by φ level.
    Returns (phi_list, median_list, q1_list, q3_list) sorted by φ.
    """
    phi_groups = {}
    for row in rows:
        phi = None
        if "phi" in row and row["phi"] not in ("", None):
            try:
                phi = round(float(row["phi"]), 10)
            except ValueError:
                pass
        if phi is None:
            phi = extract_phi(row.get("condition", ""))
        if phi is None:
            continue
        raw = row.get(metric, "")
        if raw in ("", None):
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        phi_groups.setdefault(phi, []).append(val)

    if not phi_groups:
        return [], [], [], []

    items = sorted(phi_groups.items())
    phis    = [x[0] for x in items]
    medians = [float(np.median(x[1])) for x in items]
    q1s     = [float(np.percentile(x[1], 25)) for x in items]
    q3s     = [float(np.percentile(x[1], 75)) for x in items]
    return phis, medians, q1s, q3s


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def styled_ax(ax, title, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Selfishness factor (φ)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xlim(-0.03, 1.03)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))


def add_reference_lines(ax, ys_for_anno=None):
    """Vertical markers at φ=0 (Altruist), φ=0.5 (Bentham), φ=1 (Egoist)."""
    markers = [(0.0, "Altruist\n(φ=0)", "left"),
               (0.5, "Bentham\n(φ=0.5)", "center"),
               (1.0, "Egoist\n(φ=1)", "right")]
    y_top = ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else None
    for xv, label, ha in markers:
        ax.axvline(xv, color=GRAY, linestyle=":", linewidth=1.2, alpha=0.7)
        ax.text(xv, ax.get_ylim()[1] * 0.97, label,
                ha=ha, va="top", fontsize=6.5, color=GRAY,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.6, edgecolor="none"))


def quartile_plot(ax, phis, medians, q1s, q3s, color):
    ax.plot(phis, medians, color=color, linewidth=2, marker="o", markersize=3.5,
            zorder=3, label="Median")
    ax.fill_between(phis, q1s, q3s, color=color, alpha=0.15, label="Q1–Q3")


def annotate_spearman(ax, spearman_lookup, metric):
    if metric not in spearman_lookup:
        return
    r_val = float(spearman_lookup[metric].get("spearman_r", 0))
    p_raw = spearman_lookup[metric].get("p_value", "")
    p_str = f"p={float(p_raw):.3f}" if p_raw not in ("", None) else ""
    ax.text(0.97, 0.05,
            f"Spearman r = {r_val:.3f}\n{p_str}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))


# ─────────────────────────────────────────────────────────────────────────────
# Individual plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_population(rows, spearman, out_dir):
    phis, meds, q1s, q3s = build_quartile_series(rows, "finalPopulation")
    if not phis:
        print("  [skip] finalPopulation data missing.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    quartile_plot(ax, phis, meds, q1s, q3s, BLUE)
    styled_ax(ax, "Final Population vs. Selfishness Factor", "Agents alive at end")
    add_reference_lines(ax)
    annotate_spearman(ax, spearman, "finalPopulation")
    ax.legend(fontsize=8, framealpha=0.5)
    save(fig, os.path.join(out_dir, "01_population.png"))


def plot_societal_wealth(rows, spearman, out_dir):
    phis, meds, q1s, q3s = build_quartile_series(rows, "finalSocietalWealth")
    if not phis:
        print("  [skip] finalSocietalWealth data missing.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    quartile_plot(ax, phis, meds, q1s, q3s, PURPLE)
    styled_ax(ax, "Total Societal Wealth vs. Selfishness Factor", "Total wealth")
    add_reference_lines(ax)
    annotate_spearman(ax, spearman, "finalSocietalWealth")
    ax.legend(fontsize=8, framealpha=0.5)
    save(fig, os.path.join(out_dir, "02_societal_wealth.png"))


def plot_mean_wealth(rows, spearman, out_dir):
    phis, meds, q1s, q3s = build_quartile_series(rows, "finalMeanWealth")
    if not phis:
        print("  [skip] finalMeanWealth data missing.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    quartile_plot(ax, phis, meds, q1s, q3s, GREEN)
    styled_ax(ax, "Mean Agent Wealth vs. Selfishness Factor", "Mean wealth per agent")
    add_reference_lines(ax)
    annotate_spearman(ax, spearman, "finalMeanWealth")
    ax.legend(fontsize=8, framealpha=0.5)
    save(fig, os.path.join(out_dir, "03_mean_wealth.png"))


def plot_ttl(rows, spearman, out_dir):
    phis, meds, q1s, q3s = build_quartile_series(rows, "finalMeanTimeToLive")
    if not phis:
        print("  [skip] finalMeanTimeToLive data missing.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    quartile_plot(ax, phis, meds, q1s, q3s, CYAN)
    styled_ax(ax, "Mean Time-to-Live vs. Selfishness Factor", "Mean TTL (timesteps)")
    add_reference_lines(ax)
    annotate_spearman(ax, spearman, "finalMeanTimeToLive")
    ax.legend(fontsize=8, framealpha=0.5)
    save(fig, os.path.join(out_dir, "04_ttl.png"))


def plot_deaths(rows, spearman, out_dir):
    phis, meds, q1s, q3s = build_quartile_series(rows, "meanDeathsPerTimestep")
    if not phis:
        print("  [skip] meanDeathsPerTimestep data missing.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    quartile_plot(ax, phis, meds, q1s, q3s, RED)
    styled_ax(ax, "Mean Deaths per Timestep vs. Selfishness Factor",
              "Mean deaths / timestep")
    add_reference_lines(ax)
    annotate_spearman(ax, spearman, "meanDeathsPerTimestep")
    ax.legend(fontsize=8, framealpha=0.5)
    save(fig, os.path.join(out_dir, "05_deaths_per_timestep.png"))


def plot_age_at_death(rows, spearman, out_dir):
    phis, meds, q1s, q3s = build_quartile_series(rows, "meanAgeAtDeath")
    if not phis:
        print("  [skip] meanAgeAtDeath data missing.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    quartile_plot(ax, phis, meds, q1s, q3s, AMBER)
    styled_ax(ax, "Mean Age at Death vs. Selfishness Factor",
              "Mean age at death (timesteps)")
    add_reference_lines(ax)
    annotate_spearman(ax, spearman, "meanAgeAtDeath")
    ax.legend(fontsize=8, framealpha=0.5)
    save(fig, os.path.join(out_dir, "06_age_at_death.png"))


def plot_gini(rows, spearman, out_dir):
    phis, meds, q1s, q3s = build_quartile_series(rows, "finalGini")
    if not phis:
        print("  [skip] finalGini data missing.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    quartile_plot(ax, phis, meds, q1s, q3s, AMBER)
    styled_ax(ax, "Wealth Inequality (Gini) vs. Selfishness Factor",
              "Gini coefficient")
    ax.set_ylim(0, 1.05)
    add_reference_lines(ax)
    annotate_spearman(ax, spearman, "finalGini")
    ax.legend(fontsize=8, framealpha=0.5)
    save(fig, os.path.join(out_dir, "07_gini.png"))


def plot_actions(rows, spearman, out_dir):
    pc, mc, qc1, qc3 = build_quartile_series(rows, "totalCombatActions")
    pt, mt, qt1, qt3 = build_quartile_series(rows, "totalTradeActions")
    pr, mr, qr1, qr3 = build_quartile_series(rows, "totalReproductionActions")

    if not pc and not pt and not pr:
        print("  [skip] Action data missing.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    if pc:
        ax.plot(pc, mc, color=RED,    marker="o", linewidth=2, markersize=3.5,
                label="Combat")
        ax.fill_between(pc, qc1, qc3, color=RED, alpha=0.10)
    if pt:
        ax.plot(pt, mt, color=BLUE,   marker="s", linewidth=2, markersize=3.5,
                label="Trade")
        ax.fill_between(pt, qt1, qt3, color=BLUE, alpha=0.10)
    if pr:
        ax.plot(pr, mr, color=GREEN,  marker="^", linewidth=2, markersize=3.5,
                label="Reproduction")
        ax.fill_between(pr, qr1, qr3, color=GREEN, alpha=0.10)
    styled_ax(ax, "Social Actions vs. Selfishness Factor", "Total action count")
    add_reference_lines(ax)
    ax.legend(fontsize=8, framealpha=0.5)
    save(fig, os.path.join(out_dir, "08_actions.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Summary panel (mirrors KH2024 Figs 10–15)
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_panel(rows, spearman, out_dir):
    metrics = [
        ("finalPopulation",      "Final Population",      BLUE,   False),
        ("finalSocietalWealth",  "Societal Wealth",        PURPLE, False),
        ("finalMeanWealth",      "Mean Agent Wealth",      GREEN,  False),
        ("finalMeanTimeToLive",  "Time-to-Live",           CYAN,   False),
        ("meanDeathsPerTimestep","Deaths / Timestep",      RED,    False),
        ("meanAgeAtDeath",       "Mean Age at Death",      AMBER,  False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "Selfishness Factor Sweep: Societal Outcome Gradients\n"
        "(Replication of Herman & Kremer 2024 — Section VII-C, Figures 10–15)\n"
        "Median ± IQR across seeds",
        fontsize=11, fontweight="bold", y=1.02
    )
    axes_flat = axes.flatten()

    for ax, (metric, label, color, _) in zip(axes_flat, metrics):
        phis, meds, q1s, q3s = build_quartile_series(rows, metric)
        if not phis:
            ax.set_visible(False)
            continue

        quartile_plot(ax, phis, meds, q1s, q3s, color)
        styled_ax(ax, label, label)

        # Reference lines (no text annotation to keep panel clean)
        for xv in [0.0, 0.5, 1.0]:
            ax.axvline(xv, color=GRAY, linestyle=":", linewidth=1, alpha=0.55)

        annotate_spearman(ax, spearman, metric)

    # Shared x-axis label
    fig.text(0.5, -0.02,
             "← Pure Altruist (φ=0)        Selfishness factor (φ)        Pure Egoist (φ=1) →",
             ha="center", fontsize=9, color=GRAY)

    plt.tight_layout()
    save(fig, os.path.join(out_dir, "09_summary_panel.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Plot selfishness-factor sweep results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results", default="selfishness_results/results",
                   help="Directory containing per_seed_summary.csv")
    p.add_argument("--out", default="selfishness_plots",
                   help="Output directory for plots")
    args = p.parse_args()

    summary_path  = os.path.join(args.results, "per_seed_summary.csv")
    spearman_path = os.path.join(args.results, "selfishness_spearman.csv")

    print(f"\nReading per-seed data from: {summary_path}")
    rows         = load_csv(summary_path)
    spearman_raw = load_csv(spearman_path)
    spearman     = {r["metric"]: r for r in spearman_raw}

    if not rows:
        print("\nNo data found. Run the sweep first:")
        print("  python run_experiments_selfishness.py --seeds 30 --cores <N>")
        sys.exit(0)

    phi_levels = sorted(set(
        round(float(r["phi"]), 2) for r in rows
        if r.get("phi") not in ("", None)
    ))
    print(f"Found {len(rows)} seed-condition rows | φ levels: {phi_levels}\n")

    os.makedirs(args.out, exist_ok=True)
    print(f"Writing plots to: {args.out}/\n")

    plot_population(rows, spearman, args.out)
    plot_societal_wealth(rows, spearman, args.out)
    plot_mean_wealth(rows, spearman, args.out)
    plot_ttl(rows, spearman, args.out)
    plot_deaths(rows, spearman, args.out)
    plot_age_at_death(rows, spearman, args.out)
    plot_gini(rows, spearman, args.out)
    plot_actions(rows, spearman, args.out)
    plot_summary_panel(rows, spearman, args.out)

    print(f"\nDone.  Main replication figure: {args.out}/09_summary_panel.png\n")


if __name__ == "__main__":
    main()
