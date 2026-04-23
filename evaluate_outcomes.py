import json
import argparse
import sys
import os

def evaluate_outcomes(log_file):
    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found.")
        return
        
    try:
        with open(log_file, 'r') as f:
            content = f.read().strip()

        # Gracefully repair incomplete JSON arrays (e.g. from OOM kills or crashes).
        # The log is a JSON array like: [\n\t{...},\n\t{...},\n\t{...}\n]
        # If killed mid-write, it may end with: ',', '{...', or just have no ']'.
        if not content.startswith('['):
            print(f"Error: Could not decode JSON from '{log_file}'. Make sure the simulation finished cleanly.")
            return

        if not content.endswith(']'):
            # Truncate to the last complete JSON object
            last_brace = content.rfind('}')
            if last_brace == -1:
                print(f"Error: Could not decode JSON from '{log_file}'. File contains no complete records.")
                return
            content = content[:last_brace + 1] + '\n]'

        log_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{log_file}'. Make sure the simulation finished cleanly. ({e})")
        return

    if not log_data or not isinstance(log_data, list):
        print("Error: Log file is empty or improperly formatted.")
        return

    first_step = log_data[0]
    last_step = log_data[-1]

    initial_population = first_step.get("population", 0)
    final_population = last_step.get("population", 0)
    
    # Calculate means over all timesteps
    total_timesteps = len(log_data)
    mean_population = sum(step.get("population", 0) for step in log_data) / total_timesteps
    mean_wealth = sum(step.get("meanWealth", 0) for step in log_data) / total_timesteps
    mean_time_to_live = sum(step.get("agentMeanTimeToLive", 0) for step in log_data) / total_timesteps
    
    # Death statistics
    total_starvation = sum(step.get("agentStarvationDeaths", 0) for step in log_data)
    total_combat_deaths = sum(step.get("agentCombatDeaths", 0) for step in log_data)
    total_age_deaths = sum(step.get("agentAgingDeaths", 0) for step in log_data)
    total_disease_deaths = sum(step.get("agentDiseaseDeaths", 0) for step in log_data)
    total_deaths = total_starvation + total_combat_deaths + total_age_deaths + total_disease_deaths
    
    mean_deaths_per_timestep = total_deaths / total_timesteps
    
    # Age at death
    mean_age_at_death = sum(step.get("meanAgeAtDeath", 0) for step in log_data) / total_timesteps if any("meanAgeAtDeath" in step for step in log_data) else "N/A"

    # Behavioral metrics: use the standard runtimeStats keys logged every timestep
    total_reproductions = sum(step.get("agentsBorn", 0) for step in log_data)
    total_trades = sum(step.get("tradeVolume", 0) for step in log_data)
    total_loans = sum(step.get("lendingVolume", 0) for step in log_data)
    total_combats = sum(step.get("agentCombatDeaths", 0) for step in log_data)

    # Per-model aggregation across all timesteps
    model_accumulators = {}
    for step in log_data:
        stats_by_model = step.get("statsByModel", {})
        for model, s in stats_by_model.items():
            if model not in model_accumulators:
                model_accumulators[model] = {
                    "wealthTotalSum": 0, "meanWealthSum": 0, "meanTTLSum": 0,
                    "totalDeaths": 0, "starvationDeaths": 0, "combatDeaths": 0,
                    "agingDeaths": 0, "diseaseDeaths": 0,
                    "meanAgeAtDeathSum": 0, "ageAtDeathCount": 0, "timestepCount": 0
                }
            acc = model_accumulators[model]
            acc["wealthTotalSum"] += s.get("wealthTotal", 0)
            acc["meanWealthSum"] += s.get("meanWealth", 0)
            acc["meanTTLSum"] += s.get("meanTimeToLive", 0)
            acc["totalDeaths"] += s.get("deaths", 0)
            acc["starvationDeaths"] += s.get("starvationDeaths", 0)
            acc["combatDeaths"] += s.get("combatDeaths", 0)
            acc["agingDeaths"] += s.get("agingDeaths", 0)
            acc["diseaseDeaths"] += s.get("diseaseDeaths", 0)
            if s.get("meanAgeAtDeath", 0) > 0:
                acc["meanAgeAtDeathSum"] += s.get("meanAgeAtDeath", 0)
                acc["ageAtDeathCount"] += 1
            acc["timestepCount"] += 1

    per_model_summary = {}
    for model, acc in model_accumulators.items():
        tc = acc["timestepCount"]
        per_model_summary[model] = {
            "meanSocietalWealth": round(acc["wealthTotalSum"] / tc, 2) if tc > 0 else 0,
            "meanAgentWealth": round(acc["meanWealthSum"] / tc, 2) if tc > 0 else 0,
            "meanTimeToLive": round(acc["meanTTLSum"] / tc, 2) if tc > 0 else 0,
            "totalDeaths": acc["totalDeaths"],
            "starvationDeaths": acc["starvationDeaths"],
            "combatDeaths": acc["combatDeaths"],
            "agingDeaths": acc["agingDeaths"],
            "diseaseDeaths": acc["diseaseDeaths"],
            "meanAgeAtDeath": round(acc["meanAgeAtDeathSum"] / acc["ageAtDeathCount"], 2) if acc["ageAtDeathCount"] > 0 else 0
        }

    wealth_timeseries = []
    for step in log_data:
        wealth_timeseries.append({
            "timestep": step.get("timestep", 0),
            "agentWealthTotal": round(step.get("agentWealthTotal", 0), 2),
            "meanWealth": round(step.get("meanWealth", 0), 2)
        })

    # Categorical Population End State
    end_state = "Better"
    if final_population == 0:
        end_state = "Extinct"
    elif final_population < initial_population:
        end_state = "Worse"
    
    print(f"=== Outcome Evaluation: {os.path.basename(log_file)} ===")
    print(f"Total Timesteps Run: {total_timesteps}")
    print("\n--- Societal Metrics ---")
    print(f"Initial Population: {initial_population}")
    print(f"Final Population: {final_population}")
    print(f"Mean Population: {mean_population:.2f}")
    print(f"Categorical End State: {end_state}")
    print(f"Total Societal Wealth (End): {last_step.get('agentWealthTotal', 0):.2f}")
    print(f"Mean Agent Wealth (Overall): {mean_wealth:.2f}")
    
    print("\n--- Health & Survival Metrics ---")
    print(f"Mean Time To Live: {mean_time_to_live:.2f}")
    print(f"Mean Age at Death: {mean_age_at_death}")
    print(f"Mean Deaths / Timestep: {mean_deaths_per_timestep:.2f}")
    print(f"  - Starvation: {total_starvation}")
    print(f"  - Combat: {total_combat_deaths}")
    print(f"  - Aging: {total_age_deaths}")
    print(f"  - Disease: {total_disease_deaths}")

    print("\n--- Behavioral Metrics (Interaction Frequencies) ---")
    print(f"Total Reproductions: {total_reproductions}")
    print(f"Total Trades: {total_trades}")
    print(f"Total Loans: {total_loans}")
    print(f"Total Combats: {total_combats}")

    print(f"\n--- Wealth Time Series ---")
    print(f"Recorded {len(wealth_timeseries)} timesteps of agentWealthTotal and meanWealth")

    if per_model_summary:
        print(f"\n--- Per-Model Statistics ---")
        for model, s in per_model_summary.items():
            print(f"  [{model}]")
            print(f"    Mean Societal Wealth:  {s['meanSocietalWealth']:.2f}")
            print(f"    Mean Agent Wealth:     {s['meanAgentWealth']:.2f}")
            print(f"    Mean Time To Live:     {s['meanTimeToLive']:.2f}")
            print(f"    Mean Age At Death:     {s['meanAgeAtDeath']:.2f}")
            print(f"    Total Deaths:          {s['totalDeaths']}")
            print(f"      - Starvation:        {s['starvationDeaths']}")
            print(f"      - Combat:            {s['combatDeaths']}")
            print(f"      - Aging:             {s['agingDeaths']}")
            print(f"      - Disease:           {s['diseaseDeaths']}")

    # Population breakdown by decision model and tribe
    if "populationByModel" in first_step:
        print(f"\n--- Population by Decision Model ---")
        print(f"Initial: {first_step.get('populationByModel', {})}")
        print(f"Final:   {last_step.get('populationByModel', {})}")
    
    # Save the report to a JSON file
    report = {
        "log_file": os.path.basename(log_file),
        "total_timesteps": total_timesteps,
        "societal_metrics": {
            "initial_population": initial_population,
            "final_population": final_population,
            "mean_population": round(mean_population, 2),
            "categorical_end_state": end_state,
            "total_societal_wealth_end": round(last_step.get('agentWealthTotal', 0), 2),
            "mean_agent_wealth_overall": round(mean_wealth, 2)
        },
        "health_survival_metrics": {
            "mean_time_to_live": round(mean_time_to_live, 2),
            "mean_age_at_death": mean_age_at_death if isinstance(mean_age_at_death, str) else round(mean_age_at_death, 2),
            "mean_deaths_per_timestep": round(mean_deaths_per_timestep, 2),
            "starvation_deaths": total_starvation,
            "combat_deaths": total_combat_deaths,
            "aging_deaths": total_age_deaths,
            "disease_deaths": total_disease_deaths
        },
        "behavioral_metrics": {
            "total_reproductions": total_reproductions,
            "total_trades": total_trades,
            "total_loans": total_loans,
            "total_combats": total_combats
        },
        "per_model_metrics": per_model_summary
    }
    
    output_filename = os.path.splitext(log_file)[0] + "_evaluation.json"
    try:
        with open(output_filename, 'w') as out_f:
            json.dump(report, out_f, indent=4)
        print(f"\nReport successfully saved to: {output_filename}")
    except IOError as e:
        print(f"\nError saving report to JSON: {e}")

    # Save wealth time series to a separate folder
    timeseries_dir = os.path.join(os.path.dirname(log_file), "wealth_timeseries")
    os.makedirs(timeseries_dir, exist_ok=True)
    timeseries_filename = os.path.join(timeseries_dir, os.path.splitext(os.path.basename(log_file))[0] + "_wealth_timeseries.json")
    try:
        with open(timeseries_filename, 'w') as ts_f:
            json.dump(wealth_timeseries, ts_f)
        print(f"Wealth time series saved to: {timeseries_filename}")
    except IOError as e:
        print(f"Error saving wealth time series: {e}")

    print("========================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate simulation outcomes from JSON logs.")
    parser.add_argument("log_files", nargs="+", help="Path to one or more JSON log files.")
    args = parser.parse_args()

    for log_file in args.log_files:
        evaluate_outcomes(log_file)