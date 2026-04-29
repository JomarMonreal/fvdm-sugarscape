import argparse
import json
import os
import random
import multiprocessing
import fnmatch
import gc
import math
import sys
import hashlib
from functools import partial
import sugarscape

CONFIGS = {
    "homo_base_egoist": "configs/test_homo_base_selfish.json",
    "homo_base_altruist": "configs/test_homo_base_altruist.json",
    "homo_base_bentham": "configs/test_homo_base_utilitarian.json",
    "homo_fvdm_selfish": "configs/test_homo_fvdm_selfish.json",
    "homo_fvdm_altruist": "configs/test_homo_fvdm_altruist.json",
    "homo_fvdm_utilitarian": "configs/test_homo_fvdm_utilitarian.json",
    "homo_fvdm_selfish2": "configs/test_homo_fvdm_selfish2.json",
    "homo_fvdm_altruist2": "configs/test_homo_fvdm_altruist2.json",
    "hetero_base": "configs/test_hetero_base.json",
    "hetero_fvdm": "configs/test_hetero_fvdm.json",
    "hetero_fvdm_utilitarian1": "configs/test_hetero_fvdm_utilitarian1.json",
    "hetero_fvdm_utilitarian2": "configs/test_hetero_fvdm_utilitarian2.json",
    "hetero_mixed_egoist": "configs/test_hetero_mixed_egoist.json",
    "hetero_mixed_altruist": "configs/test_hetero_mixed_altruist.json",
    "hetero_mixed_bentham": "configs/test_hetero_mixed_bentham.json",
    "hetero_selfish": "configs/test_hetero_selfish.json",
    "hetero_altruist": "configs/test_hetero_altruist.json"
}

def worker(args):
    seed, condition_name, config_path, output_dir, timesteps = args
    
    # Load the specific config
    with open(config_path, 'r') as f:
        conf = json.load(f)
        
    # ENFORCE BASELINE PARAMETERS
    conf["seed"] = seed
    conf["timesteps"] = timesteps
    conf["startingAgents"] = 250
    conf["startingDiseases"] = 50
    conf["headlessMode"] = True
    conf["environmentHeight"] = 50
    conf["environmentWidth"] = 50
    conf["agentTagging"] = True
    conf["agentInheritancePolicy"] = "children"
    conf["agentStartingSugar"] = [10, 40]
    conf["agentStartingSpice"] = [10, 40]
    conf["keepAlivePostExtinction"] = False  # Always end on extinction
    # Ensure seasons and pollution are disabled
    conf["environmentSeasonInterval"] = 0
    conf["environmentPollutionDiffusionDelay"] = 0
    conf["environmentSpiceConsumptionPollutionFactor"] = 0
    conf["environmentSugarConsumptionPollutionFactor"] = 0
    
    # Set the logfile specifically for this run
    log_filename = f"{condition_name}_seed_{seed}.json"
    conf["logfile"] = os.path.join(output_dir, log_filename)
    
    # Ensure all required environment and agent keys exist
    defaults = {
        "agentAggressionFactor": [1, 1],
        "agentBaseInterestRate": [0.05, 0.10],
        "agentDecisionModel": "egoist",
        "agentDecisionModelFactor": [1.0, 1.0],
        "agentDecisionModelLookaheadDiscount": [0.5, 0.5],
        "agentDecisionModelLookaheadFactor": [0.5, 0.5],
        "agentDecisionModels": ["egoist"],
        "agentDecisionModelTribalFactor": [-1, -1],
        "agentDepressionPercentage": 0,
        "agentLogfile": None,
        "agentDiseaseProtectionChance": [0.0, 0.0],
        "agentDynamicSelfishnessFactor": [0.0, 0.0],
        "agentDynamicTemperanceFactor": [0, 0],
        "agentFemaleFertilityAge": [12, 15],
        "agentFemaleInfertilityAge": [40, 50],
        "agentFertilityFactor": [1, 1],
        "agentImmuneSystemLength": 35,
        "agentInheritancePolicy": "children",
        "agentLeader": False,
        "agentLendingFactor": [1, 1],
        "agentLoanDuration": [5, 5],
        "agentLookaheadFactor": [1, 10],
        "agentMaleFertilityAge": [12, 15],
        "agentMaleInfertilityAge": [40, 50],
        "agentMaleToFemaleRatio": 1.0,
        "agentMaxAge": [60, 100],
        "agentMaxFriends": [0, 0],
        "agentMovement": [1, 6],
        "agentMovementMode": "cardinal",
        "agentReplacements": 0,
        "agentSelfishnessFactor": [-1, 1],
        "agentSpiceMetabolism": [1, 4],
        "agentStartingSpice": [10, 40],
        "agentStartingSugar": [10, 40],
        "agentSugarMetabolism": [1, 4],
        "agentTagging": False,
        "agentTagPreferences": [0, 0],
        "agentTagStringLength": 11,
        "agentTemperanceFactor": [0, 0],
        "agentTradeFactor": [1, 1],
        "agentUniversalSpice": [0, 0],
        "agentUniversalSugar": [0, 0],
        "agentVision": [1, 1],
        "agentVisionMode": "cardinal",
        "aggressionFactor": 0,
        "baseInterestRate": 0,
        "debugMode": ["none"],
        "decisionModel": "egoist",
        "decisionModelFactor": 1.0,
        "decisionModelLookaheadDiscount": 0,
        "decisionModelLookaheadFactor": 0,
        "decisionModelTribalFactor": 0,
        "depressionFactor": 0,
        "diseaseAggressionPenalty": [0, 0],
        "diseaseFertilityPenalty": [0, 0],
        "diseaseFriendlinessPenalty": [0, 0],
        "diseaseHappinessPenalty": [0, 0],
        "diseaseIncubationPeriod": [0, 0],
        "diseaseList": [],
        "diseaseMovementPenalty": [0, 0],
        "diseaseProtectionChance": 0,
        "diseaseSpiceMetabolismPenalty": [0, 0],
        "diseaseSugarMetabolismPenalty": [0, 0],
        "diseaseTagStringLength": [0, 0],
        "diseaseTimeframe": [0, 0],
        "diseaseTransmissionChance": [0, 0],
        "diseaseVisionPenalty": [0, 0],
        "dynamicSelfishnessFactor": 0,
        "dynamicTemperanceFactor": 0,
        "environmentEquator": -1,
        "environmentFile": None,
        "environmentHeight": 50,
        "environmentMaxCombatLoot": 1,
        "environmentMaxSpice": 4,
        "environmentMaxSugar": 4,
        "environmentMaxTribes": 1,
        "environmentPollutionDiffusionDelay": 0,
        "environmentPollutionDiffusionTimeframe": [0, 0],
        "environmentPollutionTimeframe": [0, 0],
        "environmentQuadrantSizeFactor": 1.0,
        "environmentSeasonalGrowbackDelay": 0,
        "environmentSeasonInterval": 0,
        "environmentSpiceConsumptionPollutionFactor": 0,
        "environmentSpicePeaks": [[25, 25, 4, 1]],
        "environmentSpiceProductionPollutionFactor": 0,
        "environmentSpiceRegrowRate": 1,
        "environmentStartingQuadrants": [1, 2, 3, 4],
        "environmentSugarConsumptionPollutionFactor": 0,
        "environmentSugarPeaks": [[25, 25, 4, 1]],
        "environmentSugarProductionPollutionFactor": 0,
        "environmentSugarRegrowRate": 1,
        "environmentTribePerQuadrant": False,
        "environmentUniversalSpiceIncomeInterval": 0,
        "environmentUniversalSugarIncomeInterval": 0,
        "environmentWidth": 50,
        "environmentWraparound": True,
        "experimentalGroup": None,
        "fertilityAge": [12, 15],
        "fertilityFactor": [1, 1],
        "follower": False,
        "headlessMode": True,
        "immuneSystem": [1, 1],
        "infertilityAge": [40, 50],
        "inheritancePolicy": "children",
        "interfaceHeight": 1000,
        "interfaceWidth": 900,
        "keepAliveAtEnd": False,
        "keepAlivePostExtinction": False,
        "lendingFactor": 0,
        "loanDuration": 0,
        "logfile": None,
        "logfileFormat": "json",
        "lookaheadFactor": 0,
        "maxAge": [60, 100],
        "maxFriends": 0,
        "movement": 1,
        "movementMode": "cardinal",
        "neighborhoodMode": "vonNeumann",
        "profileMode": False,
        "screenshots": False,
        "seed": seed,
        "selfishnessFactor": 0,
        "sex": "any",
        "spice": [50, 100],
        "startingAgents": 250,
        "startingDiseases": 0,
        "startingDiseasesPerAgent": [0, 0],
        "sugar": [50, 100],
        "tagging": False,
        "tagPreferences": [0, 0],
        "tags": [0, 0],
        "temperanceFactor": 0,
        "timesteps": 5000,
        "tradeFactor": 0,
        "universalSpice": 0,
        "universalSugar": 0,
        "vision": 1,
        "visionMode": "cardinal"
    }
    
    for k, v in defaults.items():
        if k not in conf:
            conf[k] = v
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize and run the simulation
        s = sugarscape.Sugarscape(conf)
        # startLog writes the JSON array opening bracket and first stats entry.
        # This is normally called by s.runSimulation(), but we drive the loop manually.
        s.updateRuntimeStats()
        s.startLog(s.log)
        for t in range(conf["timesteps"]):
            s.doTimestep()
            # Stop immediately on extinction (s.end is set internally by sugarscape
            # when it detects len(agents) == 0 and keepAlivePostExtinction is False)
            if len(s.agents) == 0 or s.end:
                break
                
        # Properly close the JSON logs since we bypassed s.run().
        # Guard against the log already being closed (e.g. via toggleEnd on extinction).
        try:
            if s.log and not s.log.closed:
                s.endLog(s.log)
        except Exception:
            pass
                
        # Evaluate immediately and clean up raw log to save disk space
        import evaluate_outcomes
        evaluate_outcomes.evaluate_outcomes(conf["logfile"])
        
        # Keep the raw per-timestep JSON log for downstream analysis
        # Explicitly trigger garbage collection to prevent memory bloat in long-running workers
        s = None
        gc.collect()
        
        return condition_name, seed, True, None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return condition_name, seed, False, str(e)

def run_experiments(num_seeds, processes, filter_name=None, timesteps=5000):
    print(f"=== Starting Stage 5: Comparison Runs ===")
    print(f"Generating {num_seeds} random seeds...")
    
    seeds = [random.randint(0, 1000000) for _ in range(num_seeds)]
    output_dir = "results/experiments"
    
    tasks = []
    for seed in seeds:
        for condition_name, config_path in CONFIGS.items():
            if filter_name:
                # Use fnmatch to support exact matches and wildcards (e.g. '*fvdm*')
                if not fnmatch.fnmatch(condition_name, filter_name):
                    continue
            tasks.append((seed, condition_name, config_path, output_dir, timesteps))
            
    num_workers = processes if processes is not None else multiprocessing.cpu_count()
    print(f"Executing {len(tasks)} total simulations using {num_workers} processes.")
    print("This will take a significant amount of time. Please wait...\n")
    
    # maxtasksperchild=10 ensures worker processes are periodically restarted,
    # which clears any memory leaks or resource fragmentation in long-running experiments.
    pool = multiprocessing.Pool(processes=num_workers, maxtasksperchild=10)
    
    completed = 0
    successful = 0
    failed = 0
    
    for condition_name, seed, success, result in pool.imap_unordered(worker, tasks):
        completed += 1
        if success:
            successful += 1
            print(f"[{completed}/{len(tasks)}] DONE: {condition_name} (Seed {seed}) -> {result}")
        else:
            failed += 1
            print(f"[{completed}/{len(tasks)}] FAILED: {condition_name} (Seed {seed}) -> Error: {result}")
            
    pool.close()
    pool.join()
    
    print("\n=== Stage 5 Complete ===")
    print(f"Total: {completed} | Successful: {successful} | Failed: {failed}")
    print(f"Logs saved to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comparison experiments across identical seeds.")
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeds to run")
    parser.add_argument("--processes", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--filter", type=str, default=None, help="Filter configurations by name")
    parser.add_argument("--timesteps", type=int, default=5000, help="Number of timesteps to run")
    args = parser.parse_args()
    
    run_experiments(args.seeds, args.processes, filter_name=args.filter, timesteps=args.timesteps)
