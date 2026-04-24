"""
aggregate_evaluations.py

Reads all *_evaluation.json files in a condition's evaluation/ subfolder,
aggregates categorical end states and mean metrics across all seeds,
and writes to results/aggregated/<condition>_aggregated.json.
"""

import os
import sys
import json
import glob
import argparse
from collections import defaultdict


def infer_condition(eval_dir):
    parent = os.path.dirname(os.path.normpath(eval_dir))
    return os.path.basename(parent)


def mean(values):
    return round(sum(values) / len(values), 4) if values else 0


def aggregate(eval_dir):
    condition = infer_condition(eval_dir)
    eval_files = sorted(glob.glob(os.path.join(eval_dir, "*_evaluation.json")))

    if not eval_files:
        print(f"No evaluation JSON files found in '{eval_dir}'.")
        sys.exit(1)

    # --- Accumulators ---
    end_state_counts = {"Extinct": 0, "Worse": 0, "Better": 0}
    num_seeds = 0

    # Societal metrics
    societal = defaultdict(list)

    # Health/survival metrics
    health = defaultdict(list)

    # Behavioral metrics
    behavioral = defaultdict(list)

    # Per-model metrics: model -> metric -> [values]
    per_model = defaultdict(lambda: defaultdict(list))

    for path in eval_files:
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Skipping (unreadable): {os.path.basename(path)} — {e}")
            continue

        num_seeds += 1

        # Categorical end state
        state = data.get("societal_metrics", {}).get("categorical_end_state", "Unknown")
        if state in end_state_counts:
            end_state_counts[state] += 1

        # Societal metrics
        sm = data.get("societal_metrics", {})
        for key in ("initial_population", "final_population", "mean_population",
                    "total_societal_wealth_end", "mean_agent_wealth_overall"):
            if key in sm:
                societal[key].append(sm[key])

        # Health/survival metrics
        hm = data.get("health_survival_metrics", {})
        for key in ("mean_time_to_live", "mean_age_at_death", "mean_deaths_per_timestep",
                    "starvation_deaths", "combat_deaths", "aging_deaths", "disease_deaths"):
            if key in hm:
                health[key].append(hm[key])

        # Behavioral metrics
        bm = data.get("behavioral_metrics", {})
        for key in ("total_reproductions", "total_trades", "total_loans", "total_combats"):
            if key in bm:
                behavioral[key].append(bm[key])

        # Per-model metrics
        for model, ms in data.get("per_model_metrics", {}).items():
            for key in ("meanSocietalWealth", "meanAgentWealth", "meanTimeToLive",
                        "meanAgeAtDeath", "totalDeaths", "starvationDeaths",
                        "combatDeaths", "agingDeaths", "diseaseDeaths"):
                if key in ms:
                    per_model[model][key].append(ms[key])

    # --- Finalize ---
    end_state_pct = {
        state: round((count / num_seeds) * 100, 2) if num_seeds > 0 else 0
        for state, count in end_state_counts.items()
    }

    output = {
        "condition": condition,
        "num_seeds": num_seeds,
        "end_states": {
            "counts": end_state_counts,
            "percentages": end_state_pct
        },
        "mean_societal_metrics": {k: mean(v) for k, v in societal.items()},
        "mean_health_metrics": {k: mean(v) for k, v in health.items()},
        "mean_behavioral_metrics": {k: mean(v) for k, v in behavioral.items()},
        "mean_per_model_metrics": {
            model: {k: mean(v) for k, v in metrics.items()}
            for model, metrics in per_model.items()
        }
    }

    out_dir = os.path.join("results", "aggregated")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{condition}_aggregated.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # --- Print summary ---
    print(f"Condition:  {condition}")
    print(f"Seeds:      {num_seeds}")
    print(f"\nEnd States:")
    for state, count in end_state_counts.items():
        print(f"  {state:8s}: {count:4d}  ({end_state_pct[state]:.1f}%)")

    print(f"\nMean Societal Metrics:")
    for k, v in output["mean_societal_metrics"].items():
        print(f"  {k}: {v}")

    print(f"\nMean Health Metrics:")
    for k, v in output["mean_health_metrics"].items():
        print(f"  {k}: {v}")

    print(f"\nMean Behavioral Metrics:")
    for k, v in output["mean_behavioral_metrics"].items():
        print(f"  {k}: {v}")

    if output["mean_per_model_metrics"]:
        print(f"\nMean Per-Model Metrics:")
        for model, metrics in output["mean_per_model_metrics"].items():
            print(f"  [{model}]")
            for k, v in metrics.items():
                print(f"    {k}: {v}")

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation JSONs from a condition's evaluation/ folder."
    )
    parser.add_argument(
        "eval_dir",
        help="Path to the evaluation/ subfolder (e.g. results/experiments/hetero_fvdm_utilitarian1/evaluation/)"
    )
    args = parser.parse_args()
    aggregate(args.eval_dir)
