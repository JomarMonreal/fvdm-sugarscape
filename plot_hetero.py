#!/usr/bin/env python3
"""
plot_hetero.py
--------------
Replicates Figures 16–21 from Kremer-Herman et al. (2024) for the
heterogeneous Egoist–Bentham sweep.

For each outcome metric the script plots:
  - median   (solid line)
  - Q1 / Q3  (dotted lines)
across 30 seeds, with % Bentham on the x-axis (0–100).

Usage:
  python plot_hetero.py --input baseline_results/hetero_results
  python plot_hetero.py --input baseline_results/hetero_results --dpi 300
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pct_bentham(condition: str) -> int:
    """Extract the Bentham percentage from a condition name.

    'heteroMix_p020' -> 20
    'heteroMix_p100' -> 100
    """
    suffix = condition.split("_p")[-1]   # '020'
    return int(suffix)


def load_per_seed(input_dir: str) -> pd.DataFrame:
    path = os.path.join(input_dir, "per_seed_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"per_seed_summary.csv not found in {input_dir}\n"
            "Run: python run_experiments_baseline.py --seeds 30 --cores <N> "
            "--output baseline_results --hetero-step 20"
        )
    df = pd.read_csv(path)
    # Keep only heteroMix conditions
    df = df[df["condition"].str.startswith("heteroMix_")].copy()
    df["pctBentham"] = df["condition"].apply(pct_bentham)
    return df


def load_per_timestep(input_dir: str) -> pd.DataFrame:
    path = os.path.join(input_dir, "per_timestep.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df["condition"].str.startswith("heteroMix_")].copy()
    df["pctBentham"] = df["condition"].apply(pct_bentham)
    return df


def derive_extra_metrics(df_seed: pd.DataFrame, df_ts: pd.DataFrame | None):
    """Add deathsPerTimestep and meanAgeAtDeath to df_seed (in-place).

    deathsPerTimestep  — deaths per timestep as % of mean population,
                         matching Figure 20 in the paper.
                         If per_timestep data available: mean(agentDeaths/population)
                         over all timesteps; else falls back to totalDeaths/finalTimestep
                         divided by a population estimate.
    meanAgeAtDeath     — from per_timestep (proxy): deaths-weighted mean age
                         of living agents; falls back to NaN if unavailable.
    """
    if df_ts is not None and {"agentDeaths", "population", "condition", "seed"}.issubset(df_ts.columns):
        # Per-timestep death rate as fraction of population
        drate = (
            df_ts.groupby(["condition", "seed"])
            .apply(
                lambda g: (
                    (g["agentDeaths"] / g["population"].replace(0, np.nan))
                    .mean() * 100   # convert to percentage
                ),
                include_groups=False,
            )
            .reset_index(name="deathsPerTimestep")
        )
        df_seed = df_seed.merge(drate, on=["condition", "seed"], how="left")
    else:
        # Fallback: absolute deaths per timestep (less accurate)
        df_seed["deathsPerTimestep"] = (
            df_seed["totalDeaths"] / df_seed["finalTimestep"].replace(0, np.nan)
        )

    # Mean age at death proxy from per-timestep data
    if df_ts is not None and {"agentDeaths", "meanAge", "condition", "seed"}.issubset(df_ts.columns):
        groups = (
            df_ts.groupby(["condition", "seed"])
            .apply(
                lambda g: (
                    (g["agentDeaths"] * g["meanAge"]).sum()
                    / g["agentDeaths"].sum()
                    if g["agentDeaths"].sum() > 0
                    else np.nan
                ),
                include_groups=False,
            )
            .reset_index(name="meanAgeAtDeath")
        )
        df_seed = df_seed.merge(groups, on=["condition", "seed"], how="left")
    else:
        df_seed["meanAgeAtDeath"] = np.nan

    return df_seed


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

FIGURES = [
    # (column_in_df,        y_label,                  title)
    ("finalPopulation",      "Population",             "Population vs % Bentham"),
    ("finalSocietalWealth",  "Total Wealth",           "Total Wealth vs % Bentham"),
    ("finalMeanWealth",      "Mean Wealth",            "Mean Wealth vs % Bentham"),
    ("finalMeanTimeToLive",  "Mean Time to Live",      "Mean TTL vs % Bentham"),
    ("deathsPerTimestep",    "Mean Deaths/Timestep (%pop)", "Deaths per Timestep vs % Bentham"),
    ("meanAgeAtDeath",       "Mean Age at Death",      "Mean Age at Death vs % Bentham"),
]

MEDIAN_COLOR = "#c0006a"   # magenta — matches paper palette
Q_COLOR      = "#c0006a"


def plot_figure(df: pd.DataFrame, col: str, y_label: str, title: str,
                out_path: str, dpi: int = 150):
    levels = sorted(df["pctBentham"].unique())

    medians = []
    q1s     = []
    q3s     = []
    xs      = []

    for pct in levels:
        vals = df.loc[df["pctBentham"] == pct, col].dropna().values
        if len(vals) == 0:
            continue
        xs.append(pct)
        medians.append(np.median(vals))
        q1s.append(np.percentile(vals, 25))
        q3s.append(np.percentile(vals, 75))

    if not xs:
        print(f"  [skip] no data for {col}")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(xs, medians, color=MEDIAN_COLOR, linewidth=2, label="Median")
    ax.plot(xs, q1s,     color=Q_COLOR,      linewidth=1, linestyle="dotted", label="Q1 / Q3")
    ax.plot(xs, q3s,     color=Q_COLOR,      linewidth=1, linestyle="dotted")

    ax.set_xlabel("% Bentham (Utilitarian)", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(min(xs) - 2, max(xs) + 2)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Plot heterogeneous Egoist–Bentham sweep results (Figures 16–21)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", "-i", default="baseline_results/hetero_results",
                    help="Directory containing per_seed_summary.csv (and optionally "
                         "per_timestep.csv)")
    ap.add_argument("--dpi",   type=int, default=150)
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    args = ap.parse_args()

    fig_dir = os.path.join(args.input, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"\nLoading data from: {args.input}")
    df_seed = load_per_seed(args.input)
    df_ts   = load_per_timestep(args.input)

    if df_ts is not None:
        print(f"  per_timestep.csv loaded ({len(df_ts):,} rows) — computing deaths/ts and age at death")
    else:
        print("  per_timestep.csv not found — age at death will be NaN")

    df_seed = derive_extra_metrics(df_seed, df_ts)

    pct_levels = sorted(df_seed["pctBentham"].unique())
    print(f"  Conditions: {len(pct_levels)} levels  —  "
          f"{min(pct_levels)}% to {max(pct_levels)}% Bentham")
    print(f"  Seeds per condition: up to "
          f"{df_seed.groupby('pctBentham')['seed'].count().max()}\n")

    for i, (col, y_label, title) in enumerate(FIGURES, start=16):
        fname = f"fig{i:02d}_{col}.{args.format}"
        out_path = os.path.join(fig_dir, fname)
        plot_figure(df_seed, col, y_label, title, out_path, dpi=args.dpi)

    print(f"\nAll figures saved to: {fig_dir}/\n")


if __name__ == "__main__":
    main()
