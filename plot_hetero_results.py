#!/usr/bin/env python3
"""
plot_hetero_results.py
----------------------
Visualizes the Egoist-Bentham heterogeneous sweep results from
run_experiments_baseline.py, replicating the outcome-gradient
analysis of Kremer-Herman et al. (2024).

Reads:
  <baseline_dir>/hetero_results/condition_aggregates.csv
  <baseline_dir>/hetero_results/hetero_spearman.csv      (if present)

Produces:
  hetero_plots/01_extinction_rate.png
  hetero_plots/02_final_population.png
  hetero_plots/03_mean_wealth.png
  hetero_plots/04_societal_wealth.png
  hetero_plots/05_gini.png
  hetero_plots/06_ttl.png
  hetero_plots/07_actions.png
  hetero_plots/08_summary_panel.png    ← main replication figure

Usage:
  python plot_hetero_results.py
  python plot_hetero_results.py --baseline-dir baseline_results
  python plot_hetero_results.py --baseline-dir baseline_results --out hetero_plots
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
    print("ERROR: matplotlib and numpy are required. Install with:")
    print("  pip install matplotlib numpy")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path):
    if not os.path.exists(path):
        print(f"  [warn] File not found: {path}")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def extract_pct(condition_name):
    """Parse pct_bentham from condition name like 'heteroMix_p070' → 70."""
    m = re.search(r"_p(\d+)", condition_name)
    return int(m.group(1)) if m else None


def build_series(rows, metric):
    """
    Return (pct_list, value_list) sorted by pct_bentham.
    Skips rows where metric is missing or empty.
    """
    pts = []
    for row in rows:
        pct = extract_pct(row.get("condition", ""))
        if pct is None:
            continue
        raw = row.get(metric, "")
        if raw == "" or raw is None:
            continue
        try:
            pts.append((pct, float(raw)))
        except ValueError:
            continue
    pts.sort(key=lambda x: x[0])
    if not pts:
        return [], []
    xs, ys = zip(*pts)
    return list(xs), list(ys)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Style helpers
# ─────────────────────────────────────────────────────────────────────────────

BLUE  = "#2563EB"
RED   = "#DC2626"
GREEN = "#16A34A"
GRAY  = "#6B7280"

def styled_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xlim(-2, 102)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))


def add_inflection(ax, xs, ys, color=GRAY):
    """Mark the 40% and 60% inflection points noted in Herman & Kremer."""
    for xv in [40, 60]:
        if xv in xs:
            idx = xs.index(xv)
            ax.axvline(xv, color=color, linestyle=":", linewidth=1.2, alpha=0.7)
            ax.annotate(f"{xv}%", xy=(xv, ys[idx]),
                        xytext=(xv + 2, ys[idx]),
                        fontsize=7, color=color, va="center")


# ─────────────────────────────────────────────────────────────────────────────
# Individual plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_extinction(rows, out_dir):
    xs, ys = build_series(rows, "extinctionRate")
    if not xs:
        print("  [skip] extinctionRate data missing.")
        return
    ys_pct = [v * 100 for v in ys]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys_pct, color=RED, marker="o", linewidth=2, markersize=5)
    ax.fill_between(xs, ys_pct, alpha=0.08, color=RED)
    styled_ax(ax, "Extinction Rate vs. Bentham Proportion",
              "% Bentham agents in population", "Extinction rate (%)")
    ax.set_ylim(-2, 105)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    add_inflection(ax, xs, ys_pct)
    ax.annotate("← more Egoist | more Bentham →", xy=(50, -8),
                ha="center", fontsize=7, color=GRAY)
    save(fig, os.path.join(out_dir, "01_extinction_rate.png"))


def plot_final_population(rows, out_dir):
    xs, ys = build_series(rows, "mean_finalPopulation")
    if not xs:
        print("  [skip] mean_finalPopulation data missing.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, color=BLUE, marker="o", linewidth=2, markersize=5)
    ax.fill_between(xs, ys, alpha=0.08, color=BLUE)
    styled_ax(ax, "Final Population vs. Bentham Proportion",
              "% Bentham agents in population", "Mean final population (agents alive)")
    add_inflection(ax, xs, ys)
    save(fig, os.path.join(out_dir, "02_final_population.png"))


def plot_mean_wealth(rows, out_dir):
    xs, ys = build_series(rows, "mean_finalMeanWealth")
    if not xs:
        print("  [skip] mean_finalMeanWealth data missing.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, color=GREEN, marker="o", linewidth=2, markersize=5)
    ax.fill_between(xs, ys, alpha=0.08, color=GREEN)
    styled_ax(ax, "Mean Agent Wealth vs. Bentham Proportion",
              "% Bentham agents in population", "Mean final wealth per agent")
    add_inflection(ax, xs, ys)
    save(fig, os.path.join(out_dir, "03_mean_wealth.png"))


def plot_societal_wealth(rows, out_dir):
    xs, ys = build_series(rows, "mean_finalSocietalWealth")
    if not xs:
        print("  [skip] mean_finalSocietalWealth data missing.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, color="#7C3AED", marker="o", linewidth=2, markersize=5)
    ax.fill_between(xs, ys, alpha=0.08, color="#7C3AED")
    styled_ax(ax, "Total Societal Wealth vs. Bentham Proportion",
              "% Bentham agents in population", "Mean total societal wealth")
    add_inflection(ax, xs, ys)
    save(fig, os.path.join(out_dir, "04_societal_wealth.png"))


def plot_gini(rows, out_dir):
    xs, ys = build_series(rows, "mean_finalGini")
    if not xs:
        print("  [skip] mean_finalGini data missing.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, color="#D97706", marker="o", linewidth=2, markersize=5)
    ax.fill_between(xs, ys, alpha=0.08, color="#D97706")
    styled_ax(ax, "Wealth Inequality (Gini) vs. Bentham Proportion",
              "% Bentham agents in population", "Mean Gini coefficient (0 = equal, 1 = unequal)")
    ax.set_ylim(0, 1.05)
    add_inflection(ax, xs, ys)
    save(fig, os.path.join(out_dir, "05_gini.png"))


def plot_ttl(rows, out_dir):
    xs, ys = build_series(rows, "mean_finalMeanTimeToLive")
    if not xs:
        print("  [skip] mean_finalMeanTimeToLive data missing.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, color="#0891B2", marker="o", linewidth=2, markersize=5)
    ax.fill_between(xs, ys, alpha=0.08, color="#0891B2")
    styled_ax(ax, "Mean Agent Time-to-Live vs. Bentham Proportion",
              "% Bentham agents in population", "Mean final time-to-live (steps)")
    add_inflection(ax, xs, ys)
    save(fig, os.path.join(out_dir, "06_ttl.png"))


def plot_actions(rows, out_dir):
    xs_c, ys_c = build_series(rows, "mean_totalCombatActions")
    xs_t, ys_t = build_series(rows, "mean_totalTradeActions")
    xs_r, ys_r = build_series(rows, "mean_totalReproductionActions")

    if not xs_c and not xs_t and not xs_r:
        print("  [skip] Action data missing.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    if xs_c:
        ax.plot(xs_c, ys_c, color=RED,   marker="o", linewidth=2,
                markersize=4, label="Combat")
    if xs_t:
        ax.plot(xs_t, ys_t, color=BLUE,  marker="s", linewidth=2,
                markersize=4, label="Trade")
    if xs_r:
        ax.plot(xs_r, ys_r, color=GREEN, marker="^", linewidth=2,
                markersize=4, label="Reproduction")
    styled_ax(ax, "Social Actions vs. Bentham Proportion",
              "% Bentham agents in population", "Mean total action count")
    ax.legend(fontsize=8, framealpha=0.5)
    save(fig, os.path.join(out_dir, "07_actions.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Main replication panel (mirrors Herman & Kremer Table 5 structure)
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_panel(rows, spearman_rows, out_dir):
    metrics = [
        ("extinctionRate",            "Extinction Rate",    RED,      True),
        ("mean_finalPopulation",       "Final Population",   BLUE,     False),
        ("mean_finalMeanWealth",       "Mean Agent Wealth",  GREEN,    False),
        ("mean_finalSocietalWealth",   "Total Soc. Wealth",  "#7C3AED",False),
        ("mean_finalGini",             "Gini Coefficient",   "#D97706",False),
        ("mean_finalMeanTimeToLive",   "Time-to-Live",       "#0891B2",False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "Egoist–Bentham Heterogeneous Sweep: Societal Outcome Gradients\n"
        "(Replication of Kremer-Herman et al., 2024 — Table 5 structure)",
        fontsize=12, fontweight="bold", y=1.01
    )
    axes_flat = axes.flatten()

    for ax, (metric, label, color, is_pct) in zip(axes_flat, metrics):
        xs, ys = build_series(rows, metric)
        if not xs:
            ax.set_visible(False)
            continue

        plot_ys = [v * 100 for v in ys] if is_pct else ys
        ax.plot(xs, plot_ys, color=color, marker="o", linewidth=2, markersize=4)
        ax.fill_between(xs, plot_ys, alpha=0.08, color=color)

        # Inflection markers at 40% and 60%
        for xv in [40, 60]:
            ax.axvline(xv, color=GRAY, linestyle=":", linewidth=1, alpha=0.6)

        ylabel = f"{label} (%)" if is_pct else label
        styled_ax(ax, label, "% Bentham", ylabel)

        if is_pct:
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

        # Annotate Spearman r if available
        if spearman_rows:
            sr = {r["metric"]: r for r in spearman_rows}
            if metric in sr:
                r_val = float(sr[metric].get("spearman_r", 0))
                p_val = sr[metric].get("p_value", "")
                p_str = f"p={float(p_val):.3f}" if p_val != "" else ""
                ax.text(0.97, 0.05,
                        f"Spearman r = {r_val:.3f}\n{p_str}",
                        transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", alpha=0.7))

    fig.text(0.5, -0.02,
             "← 100% Egoist                    % Bentham agents                    100% Bentham →",
             ha="center", fontsize=9, color=GRAY)

    plt.tight_layout()
    save(fig, os.path.join(out_dir, "08_summary_panel.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Plot heterogeneous sweep results.")
    p.add_argument("--baseline-dir", default="baseline_results",
                   help="Directory produced by run_experiments_baseline.py")
    p.add_argument("--out", default="hetero_plots",
                   help="Output directory for plots")
    args = p.parse_args()

    agg_path      = os.path.join(args.baseline_dir, "hetero_results",
                                 "condition_aggregates.csv")
    spearman_path = os.path.join(args.baseline_dir, "hetero_results",
                                 "hetero_spearman.csv")

    print(f"\nReading aggregates from: {agg_path}")
    rows          = load_csv(agg_path)
    spearman_rows = load_csv(spearman_path)

    if not rows:
        print("\nNo data found. Run the sweep first:")
        print(f"  python run_experiments_baseline.py --seeds 30 --cores <N>")
        print(f"  (hetero sweep runs automatically unless --no-hetero is passed)\n")
        sys.exit(0)

    pcts = sorted(set(extract_pct(r["condition"]) for r in rows
                      if extract_pct(r["condition"]) is not None))
    print(f"Found {len(rows)} conditions | Bentham % levels: {pcts}\n")

    os.makedirs(args.out, exist_ok=True)
    print(f"Writing plots to: {args.out}/\n")

    plot_extinction(rows, args.out)
    plot_final_population(rows, args.out)
    plot_mean_wealth(rows, args.out)
    plot_societal_wealth(rows, args.out)
    plot_gini(rows, args.out)
    plot_ttl(rows, args.out)
    plot_actions(rows, args.out)
    plot_summary_panel(rows, spearman_rows, args.out)

    print(f"\nDone. All plots saved to '{args.out}/'")
    print("Main replication figure: 08_summary_panel.png\n")


if __name__ == "__main__":
    main()
