import sugarscape
import json
import os
import multiprocessing
import random
import fvdm
from functools import partial

def worker(seed, condition, steps=500):
    # Load baseline
    with open('configs/baseline_horizon.json', 'r') as f:
        conf = json.load(f)
    
    # Apply condition-specific overrides
    mapping = {
        "selfish": "egoist",
        "altruist": "altruist",
        "utilitarian": "bentham"
    }
    conf["agentDecisionModels"] = [mapping[condition]]
    conf["decisionModelFactor"] = 1.0  # Force benchmark logic
    conf["seed"] = seed
    conf["headlessMode"] = True
    conf["timesteps"] = steps
    
    # Ensure all required environment and agent keys exist
    defaults = {
        "agentAggressionFactor": [0, 0],
        "agentBaseInterestRate": [0, 0],
        "agentDecisionModel": mapping[condition],
        "agentDecisionModelFactor": [1.0, 1.0],
        "agentDecisionModelLookaheadDiscount": [0, 0],
        "agentDecisionModelLookaheadFactor": [0, 0],
        "agentDecisionModels": [mapping[condition]],
        "agentDecisionModelTribalFactor": [0, 0],
        "agentDepressionPercentage": 0,
        "agentLogfile": None,
        "agentDiseaseProtectionChance": [0, 0],
        "agentDynamicSelfishnessFactor": [0, 0],
        "agentDynamicTemperanceFactor": [0, 0],
        "agentFemaleFertilityAge": [12, 15],
        "agentFemaleInfertilityAge": [40, 50],
        "agentFertilityFactor": [1, 1],
        "agentImmuneSystemLength": 11,
        "agentInheritancePolicy": "children",
        "agentLeader": False,
        "agentLendingFactor": [0, 0],
        "agentLoanDuration": [0, 0],
        "agentLookaheadFactor": [0, 0],
        "agentMaleFertilityAge": [12, 15],
        "agentMaleInfertilityAge": [40, 50],
        "agentMaleToFemaleRatio": 1.0,
        "agentMaxAge": [60, 100],
        "agentMaxFriends": [0, 0],
        "agentMovement": [1, 1],
        "agentMovementMode": "cardinal",
        "agentReplacements": 0,
        "agentSelfishnessFactor": [0, 0],
        "agentSpiceMetabolism": [1, 1],
        "agentStartingSpice": [50, 100],
        "agentStartingSugar": [50, 100],
        "agentSugarMetabolism": [1, 1],
        "agentTagging": False,
        "agentTagPreferences": [0, 0],
        "agentTagStringLength": 11,
        "agentTemperanceFactor": [0, 0],
        "agentTradeFactor": [0, 0],
        "agentUniversalSpice": [0, 0],
        "agentUniversalSugar": [0, 0],
        "agentVision": [1, 1],
        "agentVisionMode": "cardinal",
        "aggressionFactor": 0,
        "baseInterestRate": 0,
        "debugMode": ["none"],
        "decisionModel": mapping[condition],
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
        "logfileFormat": "csv",
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
        "timesteps": steps,
        "tradeFactor": 0,
        "universalSpice": 0,
        "universalSugar": 0,
        "vision": 1,
        "visionMode": "cardinal"
    }
    for k, v in defaults.items():
        if k not in conf:
            conf[k] = v

    s = sugarscape.Sugarscape(conf)
    s.prioritization_observation_active = True
    s.prioritization_event_log = [] # Ensure it's empty
    
    # Run simulation
    for t in range(steps):
        s.doTimestep()
        if len(s.agents) == 0:
            break
            
    return s.prioritization_event_log

def run_derivation(num_seeds=5, demo=False):
    print(f"Starting Prioritization Vector Derivation (seeds={num_seeds})...")
    
    conditions = {
        "selfish": "configs/prior_selfish.json",
        "altruist": "configs/prior_altruist.json",
        "utilitarian": "configs/prior_utilitarian.json"
    }
    
    # Load the Coordinate Function Store
    store = fvdm.FelicificCoordinateStore()
    store.load()
    if not store.models:
        print("Error: FelicificCoordinateStore is empty. Run 'make derive-felicific' first.")
        return

    results = {}
    
    for condition in conditions.keys():
        print(f"Running condition: {condition}...")
        
        seeds = [random.randint(0, 1000000) for _ in range(num_seeds)]
        
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        func = partial(worker, condition=condition, steps=250 if demo else 500)
        logs = pool.map(func, seeds)
        pool.close()
        pool.join()
        
        # Flatten logs
        all_events = [event for log in logs for event in log]
        print(f"  Collected {len(all_events)} decision events for {condition}")
        
        # Calculate predicted vectors for all events and average them
        sums = [0.0] * 5 # I, D, C, P, X
        count = 0
        
        for event in all_events:
            action_str = event["action"]
            state_vec = event["s_i"]
            
            # Map string to ActionType enum
            try:
                action_type = fvdm.ActionType[action_str]
            except KeyError:
                continue
                
            # Create a dummy LocalState for prediction
            state = fvdm.LocalState(*state_vec)
            
            # Predict
            pred = store.predict(action_type, state)
            
            # Update sums
            sums[0] += pred.intensity
            sums[1] += pred.duration
            sums[2] += pred.certainty
            sums[3] += pred.propinquity
            sums[4] += pred.extent
            count += 1
            
        if count > 0:
            results[condition] = [s / count for s in sums]
            print(f"  Mean Vector ({condition}): {results[condition]}")
        else:
            print(f"  Warning: No valid decisions recorded for {condition}")

    # Save to results
    if not os.path.exists("results"):
        os.makedirs("results")
        
    output_path = "results/prioritization_vectors.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
            
    print(f"Derivation complete. Prioritization vectors saved to {output_path}")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    
    run_derivation(num_seeds=args.seeds, demo=args.demo)
