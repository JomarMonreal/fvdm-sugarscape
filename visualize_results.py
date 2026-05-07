#!/usr/bin/env python3
"""
visualize_results.py
--------------------
Generates tables and graphs from the Digital Terrarium baseline replication
experiment results produced by run_experiments.py.

Outputs (saved to <results_dir>/figures/):
  Tables (CSV + printed to console):
    - summary_table.csv         : Condition-level aggregate stats
    - action_freq_table.csv     : Total action frequencies per condition

  Time-series graphs (mean across seeds ± 95% CI band):
    - ts_population.png
    - ts_wealth.png
    - ts_gini.png
    - ts_ttl.png
    - ts_deaths.png
    - ts_action_frequencies.png

  Bar / comparison graphs (per-condition final stats):
    - bar_extinction_rate.png
    - bar_final_population.png
    - bar_final_gini.png
    - bar_final_ttl.png
    - bar_total_actions.png
    - bar_action_breakdown.png

Usage:
  python3 visualize_results.py [--results <path>] [--output <path>]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ─────────────────────────────────────────────────────────────────
# Style & constants
# ─────────────────────────────────────────────────────────────────

CONDITION_ORDER = ["rawSugarscape", "egoist", "altruist", "bentham", "heterogeneous"]
CONDITION_LABELS = {
    "rawSugarscape":  "Raw Sugarscape",
    "egoist":         "Egoist",
    "altruist":       "Altruist",
    "bentham":        "Bentham",
    "heterogeneous":  "Heterogeneous",
}
CONDITION_COLORS = {
    "rawSugarscape":  "#6c757d",
    "egoist":         "#e63946",
    "altruist":       "#2a9d8f",
    "bentham":        "#e9c46a",
    "heterogeneous":  "#457b9d",
}
# Seaborn theme
sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def palette():
    return [CONDITION_COLORS[c] for c in CONDITION_ORDER if c in CONDITION_COLORS]


def label(condition):
    return CONDITION_LABELS.get(condition, condition)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────

def load_data(results_dir):
    pts_path = os.path.join(results_dir, "per_timestep.csv")
    sum_path = os.path.join(results_dir, "per_seed_summary.csv")
    agg_path = os.path.join(results_dir, "condition_aggregates.csv")

    for p in (pts_path, sum_path, agg_path):
        if not os.path.exists(p):
            print(f"  [error] Missing: {p}")
            sys.exit(1)

    pts = pd.read_csv(pts_path)
    summary = pd.read_csv(sum_path)
    agg = pd.read_csv(agg_path)

    # Enforce ordering
    pts["condition"] = pd.Categorical(pts["condition"], categories=CONDITION_ORDER, ordered=True)
    summary["condition"] = pd.Categorical(summary["condition"], categories=CONDITION_ORDER, ordered=True)
    agg["condition"] = pd.Categorical(agg["condition"], categories=CONDITION_ORDER, ordered=True)

    # Add friendly labels
    pts["Condition"] = pts["condition"].map(label)
    summary["Condition"] = summary["condition"].map(label)
    agg["Condition"] = agg["condition"].map(label)

    return pts, summary, agg


# ─────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────

def make_summary_table(agg, out_dir):
    cols = {
        "Condition":                      "Condition",
        "numSeeds":                       "Seeds",
        "extinctionRate":                 "Extinction Rate",
        "mean_finalPopulation":           "Mean Final Pop.",
        "std_finalPopulation":            "SD Final Pop.",
        "mean_finalMeanWealth":           "Mean Wealth (final)",
        "mean_finalGini":                 "Mean Gini (final)",
        "std_finalGini":                  "SD Gini",
        "mean_finalMeanTimeToLive":       "Mean TTL (final)",
        "std_finalMeanTimeToLive":        "SD TTL",
        "mean_totalDeaths":               "Mean Total Deaths",
        "mean_totalBorn":                 "Mean Total Born",
    }
    available = {k: v for k, v in cols.items() if k in agg.columns}
    tbl = agg[list(available.keys())].copy()
    tbl.columns = list(available.values())
    tbl = tbl.sort_values("Condition")

    path = os.path.join(out_dir, "summary_table.csv")
    tbl.to_csv(path, index=False)
    print(f"\n  Summary Table saved → {path}")
    print("\n" + tbl.to_string(index=False))
    return tbl


def make_action_table(agg, out_dir):
    cols = {
        "Condition":                        "Condition",
        "mean_totalCombatActions":          "Avg. Combat",
        "std_totalCombatActions":           "SD Combat",
        "mean_totalTradeActions":           "Avg. Trade",
        "std_totalTradeActions":            "SD Trade",
        "mean_totalReproductionActions":    "Avg. Reproduction",
        "std_totalReproductionActions":     "SD Reproduction",
        "mean_totalLendingActions":         "Avg. Lending",
        "std_totalLendingActions":          "SD Lending",
    }
    available = {k: v for k, v in cols.items() if k in agg.columns}
    tbl = agg[list(available.keys())].copy()
    tbl.columns = list(available.values())
    tbl = tbl.sort_values("Condition")

    path = os.path.join(out_dir, "action_freq_table.csv")
    tbl.to_csv(path, index=False)
    print(f"\n  Action Frequency Table saved → {path}")
    print("\n" + tbl.to_string(index=False))
    return tbl


# ─────────────────────────────────────────────────────────────────
# Time-series plots
# ─────────────────────────────────────────────────────────────────

def ts_mean_ci(pts, metric, title, ylabel, out_dir, fname):
    """Plot mean ± 95% CI ribbon per condition over timesteps."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for cond in CONDITION_ORDER:
        subset = pts[pts["condition"] == cond]
        if subset.empty:
            continue
        grp = subset.groupby("timestep")[metric]
        mn = grp.mean()
        se = grp.sem()
        n  = grp.count()
        ci = 1.96 * se  # 95% CI
        color = CONDITION_COLORS[cond]
        ax.plot(mn.index, mn.values, label=label(cond), color=color, linewidth=1.5)
        ax.fill_between(mn.index, (mn - ci).values, (mn + ci).values,
                        alpha=0.18, color=color)
    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    save(fig, os.path.join(out_dir, fname))


def plot_timeseries_population(pts, out_dir):
    ts_mean_ci(pts, "population", "Population Over Time", "Population",
               out_dir, "ts_population.png")


def plot_timeseries_wealth(pts, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, ylabel) in zip(axes, [
        ("meanWealth",    "Mean Agent Wealth"),
        ("societalWealth","Societal Wealth (Sum)"),
    ]):
        for cond in CONDITION_ORDER:
            sub = pts[pts["condition"] == cond]
            if sub.empty:
                continue
            grp = sub.groupby("timestep")[metric]
            mn = grp.mean()
            se = grp.sem()
            color = CONDITION_COLORS[cond]
            ax.plot(mn.index, mn.values, label=label(cond), color=color, linewidth=1.5)
            ax.fill_between(mn.index, (mn - 1.96*se).values, (mn + 1.96*se).values,
                            alpha=0.18, color=color)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " Over Time")
        ax.legend(fontsize=8)
    save(fig, os.path.join(out_dir, "ts_wealth.png"))


def plot_timeseries_gini(pts, out_dir):
    ts_mean_ci(pts, "giniCoefficient", "Gini Coefficient Over Time",
               "Gini Coefficient", out_dir, "ts_gini.png")


def plot_timeseries_ttl(pts, out_dir):
    ts_mean_ci(pts, "meanTimeToLive", "Mean Time-to-Live Over Time",
               "Mean TTL (timesteps)", out_dir, "ts_ttl.png")


def plot_timeseries_deaths(pts, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ("agentDeaths",           "Total Deaths per Timestep"),
        ("agentStarvationDeaths", "Starvation Deaths per Timestep"),
        ("agentCombatDeaths",     "Combat Deaths per Timestep"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        for cond in CONDITION_ORDER:
            sub = pts[pts["condition"] == cond]
            if sub.empty:
                continue
            grp = sub.groupby("timestep")[metric]
            mn = grp.mean()
            se = grp.sem()
            color = CONDITION_COLORS[cond]
            ax.plot(mn.index, mn.values, label=label(cond), color=color, linewidth=1.5)
            ax.fill_between(mn.index, (mn - 1.96*se).values, (mn + 1.96*se).values,
                            alpha=0.18, color=color)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Deaths")
        ax.set_title(title)
        ax.legend(fontsize=7)
    save(fig, os.path.join(out_dir, "ts_deaths.png"))


def plot_timeseries_actions(pts, out_dir):
    action_cols = ["movementActions", "combatActions", "tradeActions",
                   "reproductionActions", "lendingActions"]
    action_labels = ["Movement", "Combat", "Trade", "Reproduction", "Lending"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, (col, lbl) in enumerate(zip(action_cols, action_labels)):
        ax = axes[i]
        for cond in CONDITION_ORDER:
            sub = pts[pts["condition"] == cond]
            if sub.empty or col not in sub.columns:
                continue
            grp = sub.groupby("timestep")[col]
            mn = grp.mean()
            se = grp.sem()
            color = CONDITION_COLORS[cond]
            ax.plot(mn.index, mn.values, label=label(cond), color=color, linewidth=1.5)
            ax.fill_between(mn.index, (mn - 1.96*se).values, (mn + 1.96*se).values,
                            alpha=0.18, color=color)
        ax.set_title(f"{lbl} Actions Over Time")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("# Agents")
        ax.legend(fontsize=7)

    # Use last panel for action proportion stacked bar (mean across all timesteps)
    ax = axes[5]
    action_totals = pts.groupby("condition")[action_cols].mean()
    action_totals = action_totals.reindex([c for c in CONDITION_ORDER if c in action_totals.index])
    totals = action_totals.sum(axis=1).replace(0, 1)
    proportions = action_totals.div(totals, axis=0)
    proportions.index = [label(c) for c in proportions.index]
    proportions.columns = action_labels
    proportions.plot(kind="bar", stacked=True, ax=ax,
                     color=["#adb5bd","#e63946","#2a9d8f","#e9c46a","#457b9d"],
                     edgecolor="white", width=0.7)
    ax.set_title("Action Proportion (Final Timestep Avg.)")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=7, loc="upper right")

    save(fig, os.path.join(out_dir, "ts_action_frequencies.png"))


# ─────────────────────────────────────────────────────────────────
# Bar / comparison plots
# ─────────────────────────────────────────────────────────────────

def _bar_from_agg(agg, mean_col, std_col, title, ylabel, out_dir, fname,
                  pct=False, ylim=None):
    """Generic bar chart from aggregate data with error bars."""
    data = agg.set_index("condition").reindex(CONDITION_ORDER).dropna(subset=[mean_col])
    labels = [label(c) for c in data.index]
    means = data[mean_col].values
    stds  = data[std_col].values if std_col and std_col in data.columns else np.zeros(len(means))
    colors = [CONDITION_COLORS[c] for c in data.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, color=colors, edgecolor="white", width=0.55,
                  yerr=stds, error_kw={"elinewidth": 1.2, "capsize": 4, "ecolor": "#444"})
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if pct:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=20)
    save(fig, os.path.join(out_dir, fname))


def plot_bar_extinction(agg, out_dir):
    _bar_from_agg(agg, "extinctionRate", None,
                  "Extinction Rate by Condition", "Extinction Rate",
                  out_dir, "bar_extinction_rate.png", pct=True, ylim=(0, 1.05))


def plot_bar_final_population(agg, out_dir):
    _bar_from_agg(agg, "mean_finalPopulation", "std_finalPopulation",
                  "Mean Final Population by Condition", "Final Population",
                  out_dir, "bar_final_population.png")


def plot_bar_final_gini(agg, out_dir):
    _bar_from_agg(agg, "mean_finalGini", "std_finalGini",
                  "Mean Final Gini Coefficient by Condition", "Gini Coefficient",
                  out_dir, "bar_final_gini.png", ylim=(0, 1))


def plot_bar_final_ttl(agg, out_dir):
    _bar_from_agg(agg, "mean_finalMeanTimeToLive", "std_finalMeanTimeToLive",
                  "Mean Final Time-to-Live by Condition", "TTL (timesteps)",
                  out_dir, "bar_final_ttl.png")


def plot_bar_total_actions(agg, out_dir):
    """Grouped bar chart of total actions per condition."""
    action_cols = {
        "mean_totalCombatActions":       ("Combat",      "#e63946"),
        "mean_totalTradeActions":        ("Trade",       "#2a9d8f"),
        "mean_totalReproductionActions": ("Reproduction","#e9c46a"),
        "mean_totalLendingActions":      ("Lending",     "#457b9d"),
    }
    available = {k: v for k, v in action_cols.items() if k in agg.columns}
    data = agg.set_index("condition").reindex(CONDITION_ORDER).dropna()
    cond_labels = [label(c) for c in data.index]
    n_groups = len(cond_labels)
    n_actions = len(available)
    x = np.arange(n_groups)
    width = 0.18
    offsets = np.linspace(-(n_actions - 1) / 2, (n_actions - 1) / 2, n_actions) * width

    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, (col, (lbl, color)) in zip(offsets, available.items()):
        vals = data[col].values
        ax.bar(x + offset, vals, width, label=lbl, color=color, edgecolor="white")
    ax.set_title("Mean Total Action Counts by Condition")
    ax.set_ylabel("Total Actions (across all timesteps)")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=20)
    ax.legend()
    save(fig, os.path.join(out_dir, "bar_total_actions.png"))


def plot_bar_action_breakdown(summary, out_dir):
    """100% stacked bar of total actions proportions per condition, from per_seed data."""
    action_cols = ["totalCombatActions", "totalTradeActions",
                   "totalReproductionActions", "totalLendingActions"]
    action_labels = ["Combat", "Trade", "Reproduction", "Lending"]
    action_colors = ["#e63946", "#2a9d8f", "#e9c46a", "#457b9d"]

    data = summary.groupby("condition")[action_cols].mean()
    data = data.reindex([c for c in CONDITION_ORDER if c in data.index])
    totals = data.sum(axis=1).replace(0, 1)
    proportions = data.div(totals, axis=0)
    proportions.index = [label(c) for c in proportions.index]
    proportions.columns = action_labels

    fig, ax = plt.subplots(figsize=(8, 5))
    proportions.plot(kind="bar", stacked=True, ax=ax,
                     color=action_colors, edgecolor="white", width=0.6)
    ax.set_title("Action-Selection Proportions by Condition")
    ax.set_ylabel("Proportion of All Actions")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="upper right", fontsize=9)
    save(fig, os.path.join(out_dir, "bar_action_breakdown.png"))


def plot_births_deaths(pts, out_dir):
    """Side-by-side births vs. deaths over time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, title) in zip(axes, [
        ("agentsBorn",  "Agents Born per Timestep"),
        ("agentDeaths", "Agent Deaths per Timestep"),
    ]):
        for cond in CONDITION_ORDER:
            sub = pts[pts["condition"] == cond]
            if sub.empty:
                continue
            grp = sub.groupby("timestep")[metric]
            mn = grp.mean()
            se = grp.sem()
            color = CONDITION_COLORS[cond]
            ax.plot(mn.index, mn.values, label=label(cond), color=color, linewidth=1.5)
            ax.fill_between(mn.index, (mn - 1.96*se).values, (mn + 1.96*se).values,
                            alpha=0.15, color=color)
        ax.set_title(title)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Agents")
        ax.legend(fontsize=8)
    save(fig, os.path.join(out_dir, "ts_births_deaths.png"))


def plot_end_state_summary(summary, out_dir):
    """Summary of end states: Extinct, Better (Final > Initial), or Worse (0 < Final < Initial)."""
    df = summary.copy()
    
    # Calculate initial population if not explicitly provided
    # Final = Initial + Born - Deaths  =>  Initial = Final - Born + Deaths
    if "initialPopulation" not in df.columns:
        df["initialPopulation"] = df["finalPopulation"] - df["totalBorn"] + df["totalDeaths"]
    
    def get_state(row):
        if row["finalPopulation"] == 0:
            return "Extinct"
        elif row["finalPopulation"] > row["initialPopulation"]:
            return "Better"
        else:
            return "Worse"
            
    df["EndState"] = df.apply(get_state, axis=1)
    
    # Pivot to get counts per condition and state
    state_counts = df.groupby(["condition", "EndState"]).size().unstack(fill_value=0)
    
    # Ensure all states exist in columns
    for s in ["Extinct", "Better", "Worse"]:
        if s not in state_counts.columns:
            state_counts[s] = 0
            
    # Reorder columns and index
    state_counts = state_counts[["Extinct", "Worse", "Better"]]
    state_counts = state_counts.reindex([c for c in CONDITION_ORDER if c in state_counts.index])
    
    # Proportions for plotting
    props = state_counts.div(state_counts.sum(axis=1), axis=0)
    props.index = [label(c) for c in props.index]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e63946", "#f4a261", "#2a9d8f"] # Red (Extinct), Orange (Worse), Teal (Better)
    props.plot(kind="bar", stacked=True, color=colors, ax=ax, edgecolor="white", width=0.6)
    
    ax.set_title("Simulation End States by Condition")
    ax.set_ylabel("Proportion of Seeds")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="End State", loc="upper right")
    
    # Add counts as text on bars if feasible
    for i, (idx, row) in enumerate(state_counts.iterrows()):
        total = row.sum()
        if total == 0: continue
        current_y = 0
        for state in ["Extinct", "Worse", "Better"]:
            count = row[state]
            if count > 0:
                p = count / total
                ax.text(i, current_y + p/2, str(count), ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=9)
                current_y += p

    save(fig, os.path.join(out_dir, "bar_end_states.png"))
    
    # Save table
    tbl_path = os.path.join(out_dir, "end_state_table.csv")
    state_counts.index = [label(c) for c in state_counts.index]
    state_counts.to_csv(tbl_path)
    print(f"  End State Table saved → {tbl_path}")



def plot_wealth_inequality(pts, out_dir):
    """Min, mean, max wealth ribbons per condition."""
    fig, axes = plt.subplots(1, len(CONDITION_ORDER), figsize=(18, 4), sharey=True)
    for ax, cond in zip(axes, CONDITION_ORDER):
        sub = pts[pts["condition"] == cond]
        if sub.empty:
            ax.set_visible(False)
            continue
        grp = sub.groupby("timestep")
        mn_mean = grp["meanWealth"].mean()
        mn_min  = grp["minWealth"].mean()
        mn_max  = grp["maxWealth"].mean()
        color = CONDITION_COLORS[cond]
        ax.plot(mn_mean.index, mn_mean.values, color=color, linewidth=1.8, label="Mean")
        ax.fill_between(mn_mean.index, mn_min.values, mn_max.values,
                        alpha=0.25, color=color, label="Min–Max")
        ax.set_title(label(cond), fontsize=10)
        ax.set_xlabel("Timestep")
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("Wealth")
    axes[0].legend(fontsize=7)
    fig.suptitle("Wealth Spread Over Time (Min / Mean / Max)", fontsize=12)
    save(fig, os.path.join(out_dir, "ts_wealth_spread.png"))


# ─────────────────────────────────────────────────────────────────
# Per-tribe (intra-condition) plots
# ─────────────────────────────────────────────────────────────────

# Dynamic palette for tribe lines – extended so any new model gets a color
_TRIBE_COLORS = {
    "egoist":         "#e63946",
    "altruist":       "#2a9d8f",
    "bentham":        "#e9c46a",
    "rawSugarscape":  "#6c757d",
    "negativeBentham":"#9b59b6",
    "temperance":     "#f4a261",
    "asimov":         "#264653",
}
_FALLBACK_COLORS = ["#1d3557", "#bc6c25", "#606c38", "#dda15e", "#780000"]


def _detect_tribes(pts):
    """Return sorted list of tribe prefixes found in per-timestep columns."""
    tribes = set()
    for col in pts.columns:
        if col.endswith("_population"):
            prefix = col[:-len("_population")]
            if prefix:
                tribes.add(prefix)
    return sorted(tribes)


def _tribe_color(tribe, idx=0):
    if tribe in _TRIBE_COLORS:
        return _TRIBE_COLORS[tribe]
    return _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]


def _tribe_label(tribe):
    return CONDITION_LABELS.get(tribe, tribe.capitalize())


def _ts_tribe_metric(pts, metric_suffix, title, ylabel, out_dir, fname,
                     condition="heterogeneous"):
    """Plot a per-tribe metric over time for a specific condition."""
    sub = pts[pts["condition"] == condition]
    if sub.empty:
        return
    tribes = _detect_tribes(sub)
    if not tribes:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, tribe in enumerate(tribes):
        col = f"{tribe}_{metric_suffix}"
        if col not in sub.columns:
            continue
        grp = sub.groupby("timestep")[col]
        mn = grp.mean()
        se = grp.sem()
        color = _tribe_color(tribe, i)
        ax.plot(mn.index, mn.values, label=_tribe_label(tribe),
                color=color, linewidth=1.8)
        ax.fill_between(mn.index, (mn - 1.96*se).values, (mn + 1.96*se).values,
                        alpha=0.18, color=color)

    # Also plot global metric as dashed
    global_col = metric_suffix
    if global_col in sub.columns:
        grp = sub.groupby("timestep")[global_col]
        mn = grp.mean()
        ax.plot(mn.index, mn.values, label="Global", color="#333",
                linewidth=1.2, linestyle="--", alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    save(fig, os.path.join(out_dir, fname))


def plot_tribe_population(pts, out_dir):
    _ts_tribe_metric(pts, "population",
                     "Per-Tribe Population (Heterogeneous)", "Population",
                     out_dir, "tribe_population.png")


def plot_tribe_wealth(pts, out_dir):
    _ts_tribe_metric(pts, "meanWealth",
                     "Per-Tribe Mean Wealth (Heterogeneous)", "Mean Wealth",
                     out_dir, "tribe_mean_wealth.png")


def plot_tribe_ttl(pts, out_dir):
    _ts_tribe_metric(pts, "meanTimeToLive",
                     "Per-Tribe Mean Time-to-Live (Heterogeneous)", "Mean TTL",
                     out_dir, "tribe_ttl.png")


def plot_tribe_happiness(pts, out_dir):
    _ts_tribe_metric(pts, "meanHappiness",
                     "Per-Tribe Mean Happiness (Heterogeneous)", "Mean Happiness",
                     out_dir, "tribe_happiness.png")


def plot_tribe_deaths(pts, out_dir):
    """Per-tribe death breakdown over time."""
    sub = pts[pts["condition"] == "heterogeneous"]
    if sub.empty:
        return
    tribes = _detect_tribes(sub)
    if not tribes:
        return

    death_metrics = [
        ("agentDeaths",           "Total Deaths"),
        ("agentStarvationDeaths", "Starvation Deaths"),
        ("agentCombatDeaths",     "Combat Deaths"),
    ]

    fig, axes = plt.subplots(1, len(death_metrics), figsize=(5*len(death_metrics), 5))
    if len(death_metrics) == 1:
        axes = [axes]

    for ax, (metric, title) in zip(axes, death_metrics):
        for i, tribe in enumerate(tribes):
            col = f"{tribe}_{metric}"
            if col not in sub.columns:
                continue
            grp = sub.groupby("timestep")[col]
            mn = grp.mean()
            se = grp.sem()
            color = _tribe_color(tribe, i)
            ax.plot(mn.index, mn.values, label=_tribe_label(tribe),
                    color=color, linewidth=1.5)
            ax.fill_between(mn.index, (mn - 1.96*se).values, (mn + 1.96*se).values,
                            alpha=0.15, color=color)
        ax.set_title(f"Per-Tribe {title}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Deaths")
        ax.legend(fontsize=8)
    fig.suptitle("Per-Tribe Deaths (Heterogeneous)", fontsize=13)
    save(fig, os.path.join(out_dir, "tribe_deaths.png"))


def plot_tribe_actions(pts, out_dir):
    """Per-tribe action frequencies over time (one subplot per action type)."""
    sub = pts[pts["condition"] == "heterogeneous"]
    if sub.empty:
        return
    tribes = _detect_tribes(sub)
    if not tribes:
        return

    action_metrics = [
        ("movementActions", "Movement"),
        ("combatActions",   "Combat"),
        ("tradeActions",    "Trade"),
        ("reproductionActions", "Reproduction"),
        ("lendingActions",  "Lending"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(action_metrics):
        ax = axes[idx]
        for i, tribe in enumerate(tribes):
            col = f"{tribe}_{metric}"
            if col not in sub.columns:
                continue
            grp = sub.groupby("timestep")[col]
            mn = grp.mean()
            se = grp.sem()
            color = _tribe_color(tribe, i)
            ax.plot(mn.index, mn.values, label=_tribe_label(tribe),
                    color=color, linewidth=1.5)
            ax.fill_between(mn.index, (mn - 1.96*se).values, (mn + 1.96*se).values,
                            alpha=0.15, color=color)
        ax.set_title(f"{title} Actions")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("# Agents")
        ax.legend(fontsize=7)

    # Stacked bar for action proportions per tribe (last panel)
    ax = axes[5]
    action_cols = [f"_{m}" for m, _ in action_metrics]
    bar_data = {}
    for tribe in tribes:
        tribe_totals = []
        for metric, _ in action_metrics:
            col = f"{tribe}_{metric}"
            if col in sub.columns:
                tribe_totals.append(sub[col].sum())
            else:
                tribe_totals.append(0)
        bar_data[_tribe_label(tribe)] = tribe_totals

    bar_df = pd.DataFrame(bar_data, index=[lbl for _, lbl in action_metrics]).T
    total_per_tribe = bar_df.sum(axis=1).replace(0, 1)
    prop_df = bar_df.div(total_per_tribe, axis=0)
    prop_df.plot(kind="bar", stacked=True, ax=ax,
                 color=["#adb5bd", "#e63946", "#2a9d8f", "#e9c46a", "#457b9d"],
                 edgecolor="white", width=0.7)
    ax.set_title("Action Proportions by Tribe")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Per-Tribe Action Frequencies (Heterogeneous)", fontsize=13)
    save(fig, os.path.join(out_dir, "tribe_actions.png"))


def plot_tribe_summary_bars(summary, out_dir):
    """Bar chart comparing final metrics across tribes in heterogeneous condition."""
    sub = summary[summary["condition"] == "heterogeneous"]
    if sub.empty:
        return
    tribes = []
    for col in sub.columns:
        if col.endswith("_finalPopulation"):
            prefix = col[:-len("_finalPopulation")]
            if prefix:
                tribes.append(prefix)
    tribes = sorted(tribes)
    if not tribes:
        return

    metrics = [
        ("finalPopulation",     "Final Population"),
        ("finalMeanWealth",     "Final Mean Wealth"),
        ("finalMeanTimeToLive", "Final Mean TTL"),
        ("totalDeaths",         "Total Deaths"),
        ("totalCombatActions",  "Total Combat Actions"),
        ("totalTradeActions",   "Total Trade Actions"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        tribe_labels = [_tribe_label(t) for t in tribes]
        means = []
        stds = []
        colors = []
        for i, tribe in enumerate(tribes):
            col = f"{tribe}_{metric}"
            if col in sub.columns:
                means.append(sub[col].mean())
                stds.append(sub[col].std() if len(sub) > 1 else 0)
            else:
                means.append(0)
                stds.append(0)
            colors.append(_tribe_color(tribe, i))

        ax.bar(tribe_labels, means, color=colors, edgecolor="white", width=0.55,
               yerr=stds if any(s > 0 for s in stds) else None,
               error_kw={"elinewidth": 1.2, "capsize": 4, "ecolor": "#444"})
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Per-Tribe Summary (Heterogeneous)", fontsize=13)
    save(fig, os.path.join(out_dir, "tribe_summary_bars.png"))


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Digital Terrarium experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results", "-r",
        default="visualized_results",
        help="Path to the results/ directory containing the CSV files.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Where to save figures. Defaults to <results>/figures/.",
    )
    args = parser.parse_args()

    results_dir = args.results
    out_dir = args.output or os.path.join(results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  Loading results from: {results_dir}")
    pts, summary, agg = load_data(results_dir)

    num_seeds = int(agg["numSeeds"].max()) if "numSeeds" in agg.columns else "?"
    num_conds = len(agg)
    print(f"  Conditions: {num_conds}  |  Max seeds: {num_seeds}")
    print(f"  Saving figures to: {out_dir}\n")

    # ── Tables ────────────────────────────────────────────────────
    make_summary_table(agg, out_dir)
    make_action_table(agg, out_dir)

    # ── Time-series graphs ────────────────────────────────────────
    print("\n  Generating time-series plots …")
    plot_timeseries_population(pts, out_dir)
    plot_timeseries_wealth(pts, out_dir)
    plot_timeseries_gini(pts, out_dir)
    plot_timeseries_ttl(pts, out_dir)
    plot_timeseries_deaths(pts, out_dir)
    plot_timeseries_actions(pts, out_dir)
    plot_births_deaths(pts, out_dir)
    plot_wealth_inequality(pts, out_dir)

    # ── Bar charts ────────────────────────────────────────────────
    print("\n  Generating comparison bar charts …")
    plot_bar_extinction(agg, out_dir)
    plot_bar_final_population(agg, out_dir)
    plot_bar_final_gini(agg, out_dir)
    plot_bar_final_ttl(agg, out_dir)
    plot_bar_total_actions(agg, out_dir)
    plot_bar_action_breakdown(summary, out_dir)
    plot_end_state_summary(summary, out_dir)

    # ── Per-tribe plots (heterogeneous condition) ─────────────────
    tribes = _detect_tribes(pts)
    if tribes:
        print(f"\n  Generating per-tribe plots (detected: {', '.join(tribes)}) …")
        plot_tribe_population(pts, out_dir)
        plot_tribe_wealth(pts, out_dir)
        plot_tribe_ttl(pts, out_dir)
        plot_tribe_happiness(pts, out_dir)
        plot_tribe_deaths(pts, out_dir)
        plot_tribe_actions(pts, out_dir)
        plot_tribe_summary_bars(summary, out_dir)
    else:
        print("\n  No per-tribe columns detected — skipping tribe plots.")

    print(f"\n  All figures saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
