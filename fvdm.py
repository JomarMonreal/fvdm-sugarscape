import math
import os
import json
import numpy as np
from enum import Enum, auto

class ActionType(Enum):
    MOVE = auto()
    STAY = auto()
    COMBAT = auto()
    TRADE = auto()
    MATE = auto()
    CREDIT = auto()
    TAGGING = auto()

class LocalState:
    def __init__(self, wealth, metabolism, nearSugarAmt, nearSugarDist, 
                 nearSpiceAmt, nearSpiceDist, emptyCells, nearAgentCount, 
                 nearStrongDist, nearWeakDist):
        self.wealth = wealth
        self.metabolism = metabolism
        self.nearSugarAmt = nearSugarAmt
        self.nearSugarDist = nearSugarDist
        self.nearSpiceAmt = nearSpiceAmt
        self.nearSpiceDist = nearSpiceDist
        self.emptyCells = emptyCells
        self.nearAgentCount = nearAgentCount
        self.nearStrongDist = nearStrongDist
        self.nearWeakDist = nearWeakDist

    def to_vector(self):
        return [
            self.wealth, self.metabolism, 
            self.nearSugarAmt, self.nearSugarDist,
            self.nearSpiceAmt, self.nearSpiceDist,
            self.emptyCells, self.nearAgentCount,
            self.nearStrongDist, self.nearWeakDist
        ]

class FelicificEffectVector:
    def __init__(self, intensity, duration, certainty, propinquity, extent):
        """
        Represents a position in the five-dimensional felicific effect space.
        Bounds: I in [-1,1], D,C,P in [0,1], X in [-1,1]
        """
        self.intensity = max(-1.0, min(1.0, float(intensity)))
        self.duration = max(0.0, min(1.0, float(duration)))
        self.certainty = max(0.0, min(1.0, float(certainty)))
        self.propinquity = max(0.0, min(1.0, float(propinquity)))
        self.extent = max(-1.0, min(1.0, float(extent)))
        
        self.values = [self.intensity, self.duration, self.certainty, self.propinquity, self.extent]

    def distance_to(self, other, metric="euclidean"):
        if metric == "euclidean":
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.values, other.values)))
        return 0

    def to_list(self):
        return self.values

    def __repr__(self):
        return (f"FelicificEffectVector(I={self.intensity:.2f}, D={self.duration:.2f}, "
                f"C={self.certainty:.2f}, P={self.propinquity:.2f}, X={self.extent:.2f})")

class FVDMDecisionModel:
    def __init__(self, prioritization_vector, coordinate_store=None):
        self.prioritization_vector = prioritization_vector
        self.coordinate_store = coordinate_store or FelicificCoordinateStore()

class FelicificCoordinateStore:
    def __init__(self):
        self.models = {} # action_name -> {coord -> {weights, inv_cov, mse, scaler}}

    def load(self, path="results/fvdm_weights.json"):
        """Loads the parametric regression models."""
        if not os.path.exists(path):
            return
        with open(path, 'r') as f:
            try:
                self.models = json.load(f)
            except: pass

    def predict(self, action, state):
        """Predicts the FelicificEffectVector using Linear Regression weights."""
        action_name = action.name
        if action_name not in self.models:
            return FelicificEffectVector(0, 0, 0, 0, 0)

        m = self.models[action_name]
        x = np.array(state.to_vector())
        
        preds = {}
        # 1. Predict I, D, P, X using their respective linear models
        for c in ["I", "D", "P", "X"]:
            model = m[c]
            w = np.array(model["weights"])
            mean = np.array(model["scaler"]["mean"])
            std = np.array(model["scaler"]["std"])
            
            # Scale input for this specific model, replacing 0s with 1s to prevent division by zero
            safe_std = np.where(std == 0, 1.0, std)
            x_scaled = (x - mean) / safe_std
            # Add intercept term
            x_design = np.insert(x_scaled, 0, 1.0)
            
            preds[c] = float(np.dot(x_design, w))
            
        # 2. Dynamic Certainty (C) Calculation
        # Computed at runtime based on the Prediction Variance of the Intensity (I) model.
        # This captures both global noise (MSE) and state-space distance (via Inverse Covariance).
        model_i = m["I"]
        inv_cov_i = np.array(model_i["inv_cov"])
        mse_i = model_i["mse"]
        mean_i = np.array(model_i["scaler"]["mean"])
        std_i = np.array(model_i["scaler"]["std"])
        
        safe_std_i = np.where(std_i == 0, 1.0, std_i)
        x_scaled_i = (x - mean_i) / safe_std_i
        x_design_i = np.insert(x_scaled_i, 0, 1.0)
        
        # Pred Var = sigma^2 * (1 + x^T * (X^T X)^-1 * x)
        pred_var = mse_i * (1.0 + x_design_i.T @ inv_cov_i @ x_design_i)
        
        # Map variance to Certainty [0, 1]
        preds["C"] = 1.0 / (1.0 + pred_var)
        
        return FelicificEffectVector(preds["I"], preds["D"], preds["C"], preds["P"], preds["X"])

class FVDMDecisionModel:
    def __init__(self, prioritization_vector, coordinate_store=None):
        self.prioritization_vector = prioritization_vector
        self.coordinate_store = coordinate_store or FelicificCoordinateStore()

    def predict_effects(self, agent, state, action):
        return self.coordinate_store.predict(action, state)

    def determine_feasible_actions(self, agent, state):
        """
        Determines which deliberate actions are currently feasible for the agent
        based on the local state and simulation rules.
        """
        feasible = [ActionType.STAY]
        
        # 1. MOVE: Feasible if there are reachable empty cells
        if state.emptyCells > 0:
            feasible.append(ActionType.MOVE)
            
        # 2. COMBAT: Feasible if there's a valid target in range
        # Criteria: non-tribe, weaker/equal wealth, no retaliation (proximity logic)
        for cell in agent.cellsInRange:
            target = cell.agent
            if target and target != agent:
                if target.findTribe() != agent.findTribe():
                    # Research excludes sites controlled by richer outsiders
                    if (target.sugar + target.spice) <= (agent.sugar + agent.spice):
                        # Simple proximity check for "retaliation" could be added here
                        feasible.append(ActionType.COMBAT)
                        break # Found at least one valid target
        
        # 3. TRADE: Feasible if a neighbor has a different MRS and trade is mutually beneficial
        for neighbor in agent.neighbors:
            if neighbor.marginalRateOfSubstitution != agent.marginalRateOfSubstitution:
                # canTradeWithNeighbor handles parity checks
                if agent.canTradeWithNeighbor(neighbor) != False:
                    feasible.append(ActionType.TRADE)
                    break
                    
        # 4. MATE: Feasible if there's a compatible fertile neighbor and space for offspring
        if agent.isFertile():
            for neighbor in agent.neighbors:
                if neighbor.sex != agent.sex and neighbor.isFertile():
                    # Requires at least one adjacent empty cell for the child
                    if len(agent.findEmptyNeighborCells()) > 0 or len(neighbor.findEmptyNeighborCells()) > 0:
                        feasible.append(ActionType.MATE)
                        break
                        
        # 5. CREDIT: Feasible if agent is a lender/borrower and has neighbors
        if agent.isLender() or agent.isBorrower():
            if len(agent.neighbors) > 0:
                feasible.append(ActionType.CREDIT)
                
        # 6. TAGGING: Feasible if tagging is enabled and neighbor exists
        if agent.tagging and len(agent.neighbors) > 0:
            feasible.append(ActionType.TAGGING)

        return feasible

    def predict_effects(self, agent, state, action):
        """
        Predicts the FelicificEffectVector for a given action using the 
        Coordinate Function Store.
        """
        return self.coordinate_store.predict(action, state)

    def select_best_action(self, agent):
        state = agent.get_fvdm_local_state()
        feasible_actions = self.determine_feasible_actions(agent, state)
        
        best_action = None
        min_distance = float('inf')
        
        for action in feasible_actions:
            predicted_effect = self.predict_effects(agent, state, action)
            distance = predicted_effect.distance_to(self.prioritization_vector)
            
            if distance < min_distance:
                min_distance = distance
                best_action = action
                
        return best_action

class PrioritizationVectorStore:
    def __init__(self, filename="results/prioritization_vectors.json"):
        self.filename = filename
        self.vectors = {}
        
    def load(self):
        if not os.path.exists(self.filename):
            print(f"Warning: {self.filename} not found. Using default vectors.")
            self.vectors = {
                "selfish": [1.0, 0.2, 0.8, 1.0, 0.0],
                "altruist": [0.5, 0.5, 0.8, 0.2, 1.0],
                "utilitarian": [0.8, 0.8, 0.8, 0.8, 0.5]
            }
            return
            
        with open(self.filename, 'r') as f:
            data = json.load(f)
            # Convert lists to FelicificEffectVector objects
            for condition, values in data.items():
                self.vectors[condition] = FelicificEffectVector(*values)
        print(f"Loaded {len(self.vectors)} prioritization vectors from {self.filename}")

    def get_vector(self, condition):
        # Map condition names to the keys in the JSON
        mapping = {
            "Egoist": "selfish",
            "Altruist": "altruist",
            "Benthamite": "utilitarian"
        }
        key = mapping.get(condition, "utilitarian")
        return self.vectors.get(key, FelicificEffectVector(0.8, 0.8, 0.8, 0.8, 0.5))
