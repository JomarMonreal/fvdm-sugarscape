"""
aggregate_timeseries.py

Reads all raw simulation log files in a condition's logs/ subfolder,
aggregates per-timestep metrics per decision model across all seeds,
and writes the result to results/timeseries/<condition_name>_timeseries.json.

Output structure:
{
  "condition": "<condition_name>",
  "num_seeds": <int>,
  "models": ["fvdmSelfish", ...],
  "timeseries": {
    "<model>": {
      "<timestep>": {
        "mean_population":        float,
        "mean_societal_wealth":   float,
        "mean_agent_wealth":      float,
        "mean_ttl":               float,
        "mean_deaths_per_pop":    float,
        "mean_age_at_death":      float,
        "seed_count":             int   # how many seeds had data at this timestep
      },
      ...
    },
    ...
  }
}
"""

import os
import sys
import json
import glob
import argparse
from collections import defaultdict

# ---------------------------------------------------------------------------
# Known condition → model mapping for OLD homogeneous logs (no statsByModel)
# ---------------------------------------------------------------------------
CONDITION_MODEL_MAP = {
    "homo_fvdm_selfish":      "fvdmSelfish",
    "homo_fvdm_selfish2":     "fvdmSelfish2",
    "homo_fvdm_altruist":     "fvdmAltruist",
    "homo_fvdm_altruist2":    "fvdmAltruist2",
    "homo_fvdm_utilitarian":  "fvdmBentham",
}


def safe_load(path):
    """Load a potentially truncated JSON array log file."""
    with open(path, "r") as f:
        content = f.read().strip()
    if not content.startswith("["):
        return None
    if not content.endswith("]"):
        last_brace = content.rfind("}")
        if last_brace == -1:
            return None
        content = content[:last_brace + 1] + "\n]"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def infer_condition(logs_dir):
    """Derive the condition name from the folder path."""
    # logs_dir is like  results/experiments/homo_fvdm_selfish/logs
    # or                results/experiments/homo_fvdm_selfish/logs/
    parent = os.path.dirname(os.path.normpath(logs_dir))
    return os.path.basename(parent)


def aggregate(logs_dir):
    condition = infer_condition(logs_dir)
    log_files = sorted(glob.glob(os.path.join(logs_dir, "*.json")))

    if not log_files:
        print(f"No JSON log files found in '{logs_dir}'.")
        sys.exit(1)

    # model → timestep → list of per-seed values
    # Each value dict: population, societal_wealth, agent_wealth, ttl, deaths, pop_for_death_rate, age_at_death
    accumulators = defaultdict(lambda: defaultdict(list))

    # Detect fallback model name for old homogeneous logs
    fallback_model = CONDITION_MODEL_MAP.get(condition)

    num_seeds = 0
    for log_path in log_files:
        data = safe_load(log_path)
        if not data:
            print(f"  Skipping (unreadable): {os.path.basename(log_path)}")
            continue

        num_seeds += 1

        for step in data:
            t = step.get("timestep", 0)
            stats_by_model = step.get("statsByModel", {})
            pop_by_model   = step.get("populationByModel", {})

            if stats_by_model:
                # ---- New log: per-model data available ----
                for model, s in stats_by_model.items():
                    pop = pop_by_model.get(model, s.get("count", 0))
                    deaths = s.get("deaths", 0)
                    accumulators[model][t].append({
                        "population":      pop,
                        "societal_wealth": s.get("wealthTotal", 0),
                        "agent_wealth":    s.get("meanWealth", 0),
                        "ttl":             s.get("meanTimeToLive", 0),
                        "deaths":          deaths,
                        "pop_for_rate":    pop if pop > 0 else 1,
                        "age_at_death":    s.get("meanAgeAtDeath", 0),
                        "has_deaths":      deaths > 0,
                    })
            else:
                # ---- Old log: homogeneous, use overall stats ----
                if fallback_model is None:
                    # Try to infer from first timestep's populationByModel if present
                    if pop_by_model:
                        model = list(pop_by_model.keys())[0]
                    else:
                        model = "unknown"
                else:
                    model = fallback_model

                pop    = step.get("population", 0)
                deaths = step.get("agentDeaths", 0)
                accumulators[model][t].append({
                    "population":      pop,
                    "societal_wealth": step.get("agentWealthTotal", 0),
                    "agent_wealth":    step.get("meanWealth", 0),
                    "ttl":             step.get("agentMeanTimeToLive", 0),
                    "deaths":          deaths,
                    "pop_for_rate":    pop if pop > 0 else 1,
                    "age_at_death":    step.get("meanAgeAtDeath", 0),
                    "has_deaths":      deaths > 0,
                })

    # Build final averaged timeseries
    timeseries = {}
    all_models = sorted(accumulators.keys())

    for model in all_models:
        timeseries[model] = {}
        for t in sorted(accumulators[model].keys()):
            records = accumulators[model][t]
            n = len(records)
            mean = lambda key: sum(r[key] for r in records) / n

            # Deaths per population: deaths / population at that timestep per seed, then average
            mean_deaths_per_pop = sum(
                r["deaths"] / r["pop_for_rate"] for r in records
            ) / n

            # Mean age at death: average only over seeds that had deaths at that step
            death_records = [r for r in records if r["has_deaths"]]
            mean_aad = (
                sum(r["age_at_death"] for r in death_records) / len(death_records)
                if death_records else 0
            )

            timeseries[model][str(t)] = {
                "mean_population":      round(mean("population"), 3),
                "mean_societal_wealth": round(mean("societal_wealth"), 3),
                "mean_agent_wealth":    round(mean("agent_wealth"), 3),
                "mean_ttl":             round(mean("ttl"), 3),
                "mean_deaths_per_pop":  round(mean_deaths_per_pop, 4),
                "mean_age_at_death":    round(mean_aad, 3),
                "seed_count":           n,
            }

    output = {
        "condition":  condition,
        "num_seeds":  num_seeds,
        "models":     all_models,
        "timeseries": timeseries,
    }

    # Save to results/timeseries/
    out_dir = os.path.join("results", "timeseries")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{condition}_timeseries.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Condition:  {condition}")
    print(f"Seeds read: {num_seeds}")
    print(f"Models:     {all_models}")
    print(f"Saved to:   {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate per-timestep model timeseries from raw simulation logs."
    )
    parser.add_argument(
        "logs_dir",
        help="Path to the logs/ subfolder of a condition (e.g. results/experiments/homo_fvdm_selfish/logs/)"
    )
    args = parser.parse_args()
    aggregate(args.logs_dir)
