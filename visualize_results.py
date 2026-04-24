"""
visualize_results.py

Generates comprehensive tables and plots from aggregated evaluation data
and per-timestep timeseries data. Outputs to results/visualizations/.

Tables (CSV + Markdown):
  - Individual condition summaries
  - Cross-condition comparison tables (end states, societal, health, behavioral)
  - Per-model comparison for heterogeneous conditions

Plots (PNG):
  - Individual condition timeseries (population, wealth, TTL, deaths, age)
  - Cross-condition comparison timeseries
  - Heterogeneous per-model breakdowns
  - End-state bar charts
"""

import json
import os
import glob
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

MODEL_COLORS = {
    "fvdmSelfish":   "#e63946",
    "fvdmSelfish2":  "#ff6b6b",
    "fvdmAltruist":  "#2a9d8f",
    "fvdmAltruist2": "#52b788",
    "fvdmBentham":   "#457b9d",
}

CONDITION_COLORS = {
    "homo_fvdm_selfish":        "#e63946",
    "homo_fvdm_selfish2":       "#ff6b6b",
    "homo_fvdm_altruist":       "#2a9d8f",
    "homo_fvdm_altruist2":      "#52b788",
    "homo_fvdm_utilitarian":    "#457b9d",
    "hetero_fvdm_utilitarian1": "#f4a261",
    "hetero_fvdm_utilitarian2": "#e76f51",
}

CONDITION_LABELS = {
    "homo_fvdm_selfish":        "Homo Selfish (Derived)",
    "homo_fvdm_selfish2":       "Homo Selfish (Idealized)",
    "homo_fvdm_altruist":       "Homo Altruist (Derived)",
    "homo_fvdm_altruist2":      "Homo Altruist (Idealized)",
    "homo_fvdm_utilitarian":    "Homo Utilitarian",
    "hetero_fvdm_utilitarian1": "Hetero (Derived)",
    "hetero_fvdm_utilitarian2": "Hetero (Idealized)",
}

MODEL_LABELS = {
    "fvdmSelfish":   "Selfish (Derived)",
    "fvdmSelfish2":  "Selfish (Idealized)",
    "fvdmAltruist":  "Altruist (Derived)",
    "fvdmAltruist2": "Altruist (Idealized)",
    "fvdmBentham":   "Utilitarian",
}

OUT_DIR = "results/visualizations"
TABLE_DIR = os.path.join(OUT_DIR, "tables")
PLOT_DIR = os.path.join(OUT_DIR, "plots")


def ensure_dirs():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def label(condition):
    return CONDITION_LABELS.get(condition, condition)


def model_label(m):
    return MODEL_LABELS.get(m, m)


# ═══════════════════════════════════════════════════════════════════════════
#  TABLES
# ═══════════════════════════════════════════════════════════════════════════

def write_table(filename, headers, rows):
    """Write both CSV and Markdown versions of a table."""
    csv_path = os.path.join(TABLE_DIR, filename + ".csv")
    md_path = os.path.join(TABLE_DIR, filename + ".md")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

    with open(md_path, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(c) for c in row) + " |\n")


def create_individual_tables(all_agg):
    """One summary table per condition."""
    for d in all_agg:
        cond = d["condition"]
        headers = ["Metric", "Value"]
        rows = [
            ["Seeds", d["num_seeds"]],
            ["Extinct (%)", d["end_states"]["percentages"]["Extinct"]],
            ["Worse (%)", d["end_states"]["percentages"]["Worse"]],
            ["Better (%)", d["end_states"]["percentages"]["Better"]],
        ]
        for k, v in d["mean_societal_metrics"].items():
            rows.append([k, round(v, 2)])
        for k, v in d["mean_health_metrics"].items():
            rows.append([k, round(v, 2)])
        for k, v in d["mean_behavioral_metrics"].items():
            rows.append([k, round(v, 2)])
        write_table(f"individual_{cond}", headers, rows)
    print(f"  ✓ Individual tables: {len(all_agg)} conditions")


def create_comparison_tables(all_agg):
    """Cross-condition comparison tables."""
    # Sort for consistent ordering
    all_agg.sort(key=lambda d: d["condition"])

    # ── End States ──
    headers = ["Condition", "Seeds", "Extinct", "Worse", "Better"]
    rows = []
    for d in all_agg:
        e = d["end_states"]
        rows.append([
            label(d["condition"]),
            d["num_seeds"],
            f"{e['counts']['Extinct']} ({e['percentages']['Extinct']}%)",
            f"{e['counts']['Worse']} ({e['percentages']['Worse']}%)",
            f"{e['counts']['Better']} ({e['percentages']['Better']}%)",
        ])
    write_table("comparison_end_states", headers, rows)

    # ── Societal Metrics ──
    headers = ["Condition", "Mean Pop", "Final Pop", "End Wealth", "Mean Agent Wealth"]
    rows = []
    for d in all_agg:
        m = d["mean_societal_metrics"]
        rows.append([
            label(d["condition"]),
            round(m["mean_population"], 2),
            round(m["final_population"], 2),
            round(m["total_societal_wealth_end"], 2),
            round(m["mean_agent_wealth_overall"], 2),
        ])
    write_table("comparison_societal", headers, rows)

    # ── Health / Survival ──
    headers = ["Condition", "Mean TTL", "Mean Age@Death", "Deaths/Timestep",
               "Starvation", "Combat", "Aging"]
    rows = []
    for d in all_agg:
        h = d["mean_health_metrics"]
        rows.append([
            label(d["condition"]),
            round(h["mean_time_to_live"], 2),
            round(h["mean_age_at_death"], 2),
            round(h["mean_deaths_per_timestep"], 2),
            round(h["starvation_deaths"], 1),
            round(h["combat_deaths"], 1),
            round(h["aging_deaths"], 1),
        ])
    write_table("comparison_health", headers, rows)

    # ── Behavioral ──
    headers = ["Condition", "Reproductions", "Trades", "Loans", "Combats"]
    rows = []
    for d in all_agg:
        b = d["mean_behavioral_metrics"]
        rows.append([
            label(d["condition"]),
            round(b["total_reproductions"], 1),
            round(b["total_trades"], 1),
            round(b["total_loans"], 1),
            round(b["total_combats"], 1),
        ])
    write_table("comparison_behavioral", headers, rows)

    # ── Per-Model (hetero only) ──
    hetero = [d for d in all_agg if "hetero" in d["condition"]]
    if hetero:
        headers = ["Condition", "Model", "Mean Wealth", "Mean TTL",
                   "Mean Age@Death", "Total Deaths", "Combat Deaths"]
        rows = []
        for d in hetero:
            for m, s in d["mean_per_model_metrics"].items():
                rows.append([
                    label(d["condition"]),
                    model_label(m),
                    round(s.get("meanAgentWealth", 0), 2),
                    round(s.get("meanTimeToLive", 0), 2),
                    round(s.get("meanAgeAtDeath", 0), 2),
                    round(s.get("totalDeaths", 0), 1),
                    round(s.get("combatDeaths", 0), 1),
                ])
        write_table("comparison_hetero_per_model", headers, rows)

    print(f"  ✓ Comparison tables: 4 tables + {len(hetero)} hetero model tables")


# ═══════════════════════════════════════════════════════════════════════════
#  TIMESERIES HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def extract_series(ts_data, model, metric, max_t=None):
    """Extract (t_array, y_array) for a given model and metric from timeseries."""
    steps = ts_data["timeseries"].get(model, {})
    t_vals = sorted(int(t) for t in steps.keys())
    if max_t is not None:
        t_vals = [t for t in t_vals if t <= max_t]
    y_vals = [steps[str(t)][metric] for t in t_vals]
    return np.array(t_vals), np.array(y_vals)


METRIC_INFO = [
    ("mean_population",      "Mean Population",       "Population"),
    ("mean_societal_wealth", "Mean Societal Wealth",   "Wealth"),
    ("mean_agent_wealth",    "Mean Agent Wealth",      "Wealth per Agent"),
    ("mean_ttl",             "Mean Time-to-Live",      "TTL (timesteps)"),
    ("mean_deaths_per_pop",  "Deaths / Population",    "Death Rate"),
    ("mean_age_at_death",    "Mean Age at Death",      "Age (timesteps)"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL CONDITION PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_individual_timeseries(all_ts):
    """One figure per condition per metric — multiple models shown as separate lines."""
    for ts in all_ts:
        cond = ts["condition"]
        models = ts["models"]

        for metric_key, metric_title, y_label in METRIC_INFO:
            fig, ax = plt.subplots(figsize=(10, 5))

            for m in models:
                t, y = extract_series(ts, m, metric_key)
                color = MODEL_COLORS.get(m, None)
                ax.plot(t, y, label=model_label(m), color=color, linewidth=1.2)

            ax.set_title(f"{label(cond)} — {metric_title}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Timestep")
            ax.set_ylabel(y_label)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(PLOT_DIR, f"individual_{cond}_{metric_key}.png"))
            plt.close(fig)

    print(f"  ✓ Individual timeseries plots: {len(all_ts)} conditions × {len(METRIC_INFO)} metrics")


# ═══════════════════════════════════════════════════════════════════════════
#  CROSS-CONDITION COMPARISON PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_comparison_timeseries(all_ts):
    """
    For each metric, overlay all conditions on one plot.
    For homogeneous conditions we use the single model;
    for heterogeneous we sum populations across models.
    """
    for metric_key, metric_title, y_label in METRIC_INFO:
        fig, ax = plt.subplots(figsize=(12, 6))

        for ts in all_ts:
            cond = ts["condition"]
            models = ts["models"]
            color = CONDITION_COLORS.get(cond, None)

            if len(models) == 1:
                # Homogeneous — single model IS the condition
                t, y = extract_series(ts, models[0], metric_key)
                ax.plot(t, y, label=label(cond), color=color, linewidth=1.2)
            else:
                # Heterogeneous — aggregate across models
                # For population: sum; for rates/means: use overall mean
                all_t = set()
                per_model = {}
                for m in models:
                    t, y = extract_series(ts, m, metric_key)
                    per_model[m] = dict(zip(t, y))
                    all_t.update(t)
                t_sorted = sorted(all_t)

                if metric_key == "mean_population" or metric_key == "mean_societal_wealth":
                    # Sum across models
                    y_agg = [sum(per_model[m].get(t, 0) for m in models) for t in t_sorted]
                else:
                    # Average across models present at each timestep
                    y_agg = []
                    for t in t_sorted:
                        vals = [per_model[m][t] for m in models if t in per_model[m]]
                        y_agg.append(np.mean(vals) if vals else 0)

                ax.plot(t_sorted, y_agg, label=label(cond), color=color, linewidth=1.2)

        ax.set_title(f"Comparison — {metric_title}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Timestep")
        ax.set_ylabel(y_label)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"comparison_{metric_key}.png"))
        plt.close(fig)

    print(f"  ✓ Comparison timeseries plots: {len(METRIC_INFO)} metrics")


# ═══════════════════════════════════════════════════════════════════════════
#  HETEROGENEOUS PER-MODEL BREAKDOWN PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_hetero_model_breakdown(all_ts):
    """For each hetero condition, plot per-model lines side-by-side."""
    hetero = [ts for ts in all_ts if "hetero" in ts["condition"]]
    if not hetero:
        return

    for metric_key, metric_title, y_label in METRIC_INFO:
        fig, axes = plt.subplots(1, len(hetero), figsize=(7 * len(hetero), 5), sharey=True)
        if len(hetero) == 1:
            axes = [axes]

        for ax, ts in zip(axes, hetero):
            cond = ts["condition"]
            for m in ts["models"]:
                t, y = extract_series(ts, m, metric_key)
                color = MODEL_COLORS.get(m, None)
                ax.plot(t, y, label=model_label(m), color=color, linewidth=1.2)
            ax.set_title(label(cond), fontsize=11, fontweight="bold")
            ax.set_xlabel("Timestep")
            ax.legend(fontsize=8)

        axes[0].set_ylabel(y_label)
        fig.suptitle(f"Heterogeneous Model Breakdown — {metric_title}", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(PLOT_DIR, f"hetero_breakdown_{metric_key}.png"))
        plt.close(fig)

    print(f"  ✓ Hetero breakdown plots: {len(hetero)} conditions × {len(METRIC_INFO)} metrics")


# ═══════════════════════════════════════════════════════════════════════════
#  END-STATE BAR CHART
# ═══════════════════════════════════════════════════════════════════════════

def plot_end_state_bars(all_agg):
    """Stacked bar chart of end-state percentages."""
    all_agg_sorted = sorted(all_agg, key=lambda d: d["condition"])
    conds = [label(d["condition"]) for d in all_agg_sorted]
    extinct = [d["end_states"]["percentages"]["Extinct"] for d in all_agg_sorted]
    worse = [d["end_states"]["percentages"]["Worse"] for d in all_agg_sorted]
    better = [d["end_states"]["percentages"]["Better"] for d in all_agg_sorted]

    x = np.arange(len(conds))
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, extinct, width, label="Extinct", color="#e63946")
    ax.bar(x, worse, width, bottom=extinct, label="Worse", color="#f4a261")
    ax.bar(x, better, width, bottom=[e + w for e, w in zip(extinct, worse)], label="Better", color="#2a9d8f")

    ax.set_ylabel("Percentage (%)")
    ax.set_title("End-State Distribution by Condition", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "end_state_distribution.png"))
    plt.close(fig)
    print("  ✓ End-state bar chart")


# ═══════════════════════════════════════════════════════════════════════════
#  BEHAVIORAL METRICS BAR CHART
# ═══════════════════════════════════════════════════════════════════════════

def plot_behavioral_bars(all_agg):
    """Grouped bar chart of mean behavioral metrics."""
    all_agg_sorted = sorted(all_agg, key=lambda d: d["condition"])
    conds = [label(d["condition"]) for d in all_agg_sorted]
    metrics_keys = ["total_reproductions", "total_trades", "total_combats"]
    metric_labels = ["Reproductions", "Trades", "Combats"]
    colors = ["#457b9d", "#2a9d8f", "#e63946"]

    x = np.arange(len(conds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (key, mlabel, color) in enumerate(zip(metrics_keys, metric_labels, colors)):
        vals = [d["mean_behavioral_metrics"].get(key, 0) for d in all_agg_sorted]
        ax.bar(x + i * width, vals, width, label=mlabel, color=color)

    ax.set_ylabel("Mean Count per Seed")
    ax.set_title("Mean Behavioral Metrics by Condition", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "behavioral_comparison.png"))
    plt.close(fig)
    print("  ✓ Behavioral metrics bar chart")


# ═══════════════════════════════════════════════════════════════════════════
#  HEALTH METRICS BAR CHART
# ═══════════════════════════════════════════════════════════════════════════

def plot_health_bars(all_agg):
    """Grouped bar chart of mean health/survival metrics."""
    all_agg_sorted = sorted(all_agg, key=lambda d: d["condition"])
    conds = [label(d["condition"]) for d in all_agg_sorted]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ("mean_time_to_live", "Mean TTL", "#457b9d"),
        ("mean_age_at_death", "Mean Age at Death", "#2a9d8f"),
        ("mean_deaths_per_timestep", "Deaths / Timestep", "#e63946"),
    ]

    for ax, (key, title, color) in zip(axes, metrics):
        vals = [d["mean_health_metrics"].get(key, 0) for d in all_agg_sorted]
        x = np.arange(len(conds))
        ax.bar(x, vals, color=color, width=0.6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=35, ha="right", fontsize=7)

    fig.suptitle("Health & Survival Metrics by Condition", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(PLOT_DIR, "health_comparison.png"))
    plt.close(fig)
    print("  ✓ Health metrics bar chart")


# ═══════════════════════════════════════════════════════════════════════════
#  DERIVED vs IDEALIZED COMPARISON (Homo conditions)
# ═══════════════════════════════════════════════════════════════════════════

def plot_derived_vs_idealized(all_ts):
    """Side-by-side for Selfish vs Selfish2, Altruist vs Altruist2."""
    pairs = [
        ("homo_fvdm_selfish", "homo_fvdm_selfish2", "Selfish: Derived vs Idealized"),
        ("homo_fvdm_altruist", "homo_fvdm_altruist2", "Altruist: Derived vs Idealized"),
    ]

    ts_by_cond = {ts["condition"]: ts for ts in all_ts}

    for cond1, cond2, title_prefix in pairs:
        if cond1 not in ts_by_cond or cond2 not in ts_by_cond:
            continue

        for metric_key, metric_title, y_label in METRIC_INFO:
            fig, ax = plt.subplots(figsize=(10, 5))

            ts1, ts2 = ts_by_cond[cond1], ts_by_cond[cond2]
            m1, m2 = ts1["models"][0], ts2["models"][0]

            t1, y1 = extract_series(ts1, m1, metric_key)
            t2, y2 = extract_series(ts2, m2, metric_key)

            ax.plot(t1, y1, label=label(cond1), color=CONDITION_COLORS.get(cond1), linewidth=1.3)
            ax.plot(t2, y2, label=label(cond2), color=CONDITION_COLORS.get(cond2), linewidth=1.3,
                    linestyle="--")

            ax.set_title(f"{title_prefix} — {metric_title}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Timestep")
            ax.set_ylabel(y_label)
            ax.legend()
            fig.tight_layout()
            safe_title = title_prefix.split(":")[0].lower().strip()
            fig.savefig(os.path.join(PLOT_DIR, f"derived_vs_idealized_{safe_title}_{metric_key}.png"))
            plt.close(fig)

    print("  ✓ Derived vs Idealized comparison plots")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ensure_dirs()

    # Load all data
    agg_files = sorted(glob.glob("results/aggregated/*_aggregated.json"))
    ts_files = sorted(glob.glob("results/timeseries/*_timeseries.json"))

    all_agg = [load_json(f) for f in agg_files]
    all_ts = [load_json(f) for f in ts_files]

    print(f"Loaded {len(all_agg)} aggregated files, {len(all_ts)} timeseries files\n")

    print("Creating tables...")
    create_individual_tables(all_agg)
    create_comparison_tables(all_agg)

    print("\nCreating plots...")
    plot_end_state_bars(all_agg)
    plot_behavioral_bars(all_agg)
    plot_health_bars(all_agg)
    plot_individual_timeseries(all_ts)
    plot_comparison_timeseries(all_ts)
    plot_hetero_model_breakdown(all_ts)
    plot_derived_vs_idealized(all_ts)

    print(f"\n✅ All outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
