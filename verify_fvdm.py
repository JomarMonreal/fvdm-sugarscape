import fvdm
import agent
import cell
import environment
import json

class MockSugarscape:
    def __init__(self):
        self.configuration = {
            "agentAggressionFactor": [0, 0],
            "agentBaseInterestRate": [0, 0],
            "agentDecisionModelFactor": [0, 0],
            "agentDecisionModelLookaheadDiscount": [0, 0],
            "agentDecisionModelLookaheadFactor": [0, 0],
            "agentDecisionModelTribalFactor": [0, 0],
            "agentDiseaseProtectionChance": [0, 0],
            "agentDynamicSelfishnessFactor": [0, 0],
            "agentDynamicTemperanceFactor": [0, 0],
            "agentFemaleFertilityAge": [0, 0],
            "agentFemaleInfertilityAge": [0, 0],
            "agentFertilityFactor": [1, 1],
            "agentLeader": False,
            "agentImmuneSystemLength": 0,
            "agentInheritancePolicy": "none",
            "agentLendingFactor": [0, 0],
            "agentLoanDuration": [0, 0],
            "agentLookaheadFactor": [0, 0],
            "agentMaleFertilityAge": [0, 0],
            "agentMaleInfertilityAge": [0, 0],
            "agentMaleToFemaleRatio": 0,
            "agentMaxAge": [-1, -1],
            "agentMaxFriends": [0, 0],
            "agentMovement": [1, 1],
            "agentMovementMode": "cardinal",
            "neighborhoodMode": "vonNeumann",
            "agentSelfishnessFactor": [0, 0],
            "agentSpiceMetabolism": [0, 0],
            "agentStartingSpice": [10, 10],
            "agentStartingSugar": [10, 10],
            "agentSugarMetabolism": [0, 0],
            "agentTagPreferences": [0, 0],
            "agentTagging": False,
            "agentTemperanceFactor": [0, 0],
            "agentTradeFactor": [0, 0],
            "agentUniversalSpice": [0, 0],
            "agentUniversalSugar": [0, 0],
            "agentVision": [1, 1],
            "agentVisionMode": "cardinal",
            "environmentWraparound": False,
            "environmentMaxCellDistance": 10,
            "startingDiseases": 0,
            "diseaseVisionPenalty": [0, 0],
            "diseaseMovementPenalty": [0, 0],
            "agentVision": [1, 1],
            "agentMovement": [1, 1],
            "agentVisionMode": "cardinal",
            "agentMovementMode": "cardinal",
            "environmentMaxTribes": 1,
            "agentTagStringLength": 1
        }
        self.debug = ["none"]
        self.timestep = 0

def test_fvdm():
    print("Testing FVDM Integration...")
    
    # 1. Setup minimal environment
    mock_ss = MockSugarscape()
    env_config = {
        "equator": -1, "globalMaxSpice": 100, "globalMaxSugar": 100,
        "maxCombatLoot": 10, "neighborhoodMode": "vonNeumann",
        "pollutionDiffusionDelay": 0, "pollutionDiffusionTimeframe": [0,0],
        "pollutionTimeframe": [0,0], "seasonalGrowbackDelay": 0,
        "seasonInterval": 0, "spiceConsumptionPollutionFactor": 0,
        "spiceProductionPollutionFactor": 0, "spiceRegrowRate": 1,
        "sugarConsumptionPollutionFactor": 0, "sugarProductionPollutionFactor": 0,
        "sugarRegrowRate": 1, "sugarscapeSeed": 0,
        "universalSpiceIncomeInterval": 0, "universalSugarIncomeInterval": 0,
        "wraparound": False
    }
    env = environment.Environment(10, 10, mock_ss, env_config)
    
    # Setup cells
    for i in range(10):
        for j in range(10):
            env.setCell(cell.Cell(i, j, env), i, j)
    env.findCellNeighbors()
    env.findCellRanges()
    
    # 2. Setup Agent
    c = env.findCell(5, 5)
    agent_config = {
        "aggressionFactor": 0, "baseInterestRate": 0, "decisionModel": "none",
        "decisionModelFactor": 0, "decisionModelLookaheadDiscount": 0,
        "decisionModelLookaheadFactor": 0, "decisionModelTribalFactor": 0,
        "depressionFactor": 0, "diseaseProtectionChance": 0,
        "dynamicSelfishnessFactor": 0, "dynamicTemperanceFactor": 0,
        "fertilityAge": 0, "fertilityFactor": 1, "follower": False,
        "immuneSystem": None, "infertilityAge": 100, "inheritancePolicy": "none",
        "lendingFactor": 0, "loanDuration": 0, "lookaheadFactor": 0,
        "maxAge": -1, "maxFriends": 0, "movement": 1, "movementMode": "cardinal",
        "neighborhoodMode": "vonNeumann", "seed": 0, "selfishnessFactor": 0,
        "sex": "male", "spice": 10, "spiceMetabolism": 0, "sugar": 10,
        "sugarMetabolism": 0, "tagging": False, "tagPreferences": None,
        "tags": None, "temperanceFactor": 0, "tradeFactor": 0,
        "universalSpice": 0, "universalSugar": 0, "vision": 1, "visionMode": "cardinal"
    }
    a = agent.Agent(1, 0, c, agent_config)
    c.agent = a
    
    # Initialize agent's surroundings
    a.findCellsInRange()
    a.updateNeighbors()
    
    # 3. Test Local State Extraction
    print("Extracting Local State...")
    state = a.get_fvdm_local_state()
    print(f"Wealth: {state.wealth}, Metabolism: {state.metabolism}")
    print(f"Empty Cells: {state.emptyCells}, Near Agent Count: {state.nearAgentCount}")
    
    # 4. Test Feasibility Checks
    print("\n--- Scenario 1: Isolated Agent ---")
    brain = fvdm.FVDMDecisionModel(fvdm.FelicificVector([1, 1, 1]))
    actions = brain.determine_feasible_actions(a, state)
    print(f"Feasible Actions: {[action.name for action in actions]}")
    assert fvdm.ActionType.STAY in actions
    assert fvdm.ActionType.MOVE in actions

    # 5. Test Combat Feasibility
    print("\n--- Scenario 2: Enemy in Range (Combat) ---")
    enemy_cell = env.findCell(5, 6)
    enemy_config = agent_config.copy()
    enemy_config["tags"] = [1 - t for t in (a.tags or [0])] # Different tribe
    enemy = agent.Agent(2, 0, enemy_cell, enemy_config)
    enemy_cell.agent = enemy
    a.updateNeighbors()
    a.findCellsInRange()
    state = a.get_fvdm_local_state()
    actions = brain.determine_feasible_actions(a, state)
    print(f"Feasible Actions: {[action.name for action in actions]}")
    assert fvdm.ActionType.COMBAT in actions

    # 6. Test Mating Feasibility
    print("\n--- Scenario 3: Compatible Mate (Mate) ---")
    mate_cell = env.findCell(4, 5)
    mate_config = agent_config.copy()
    mate_config["sex"] = "female"
    mate = agent.Agent(3, 0, mate_cell, mate_config)
    mate_cell.agent = mate
    a.updateNeighbors()
    state = a.get_fvdm_local_state()
    actions = brain.determine_feasible_actions(a, state)
    print(f"Feasible Actions: {[action.name for action in actions]}")
    assert fvdm.ActionType.MATE in actions

    # 7. Test Trade Feasibility
    print("\n--- Scenario 4: Trade Partner (Trade) ---")
    # Change MRS to trigger trade
    a.marginalRateOfSubstitution = 2.0
    mate.marginalRateOfSubstitution = 0.5
    actions = brain.determine_feasible_actions(a, state)
    print(f"Feasible Actions: {[action.name for action in actions]}")
    assert fvdm.ActionType.TRADE in actions

    # 8. Test Credit/Tagging
    print("\n--- Scenario 5: Lending & Tagging ---")
    a.tagging = True
    a.lendingFactor = 0.5 # Make agent a lender
    a.sugar = 20 # Increase wealth above startingSugar to satisfy isLender
    a.spice = 20
    actions = brain.determine_feasible_actions(a, state)
    print(f"Feasible Actions: {[action.name for action in actions]}")
    assert fvdm.ActionType.TAGGING in actions
    assert fvdm.ActionType.CREDIT in actions

    print("\nAll Feasibility Scenarios Successful!")

if __name__ == "__main__":
    try:
        test_fvdm()
    except Exception as e:
        print(f"Verification Failed: {e}")
        import traceback
        traceback.print_exc()
