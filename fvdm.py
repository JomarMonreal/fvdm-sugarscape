import math
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
    def __init__(self, prioritization_vector):
        self.prioritization_vector = prioritization_vector

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
        Predicts the FelicificEffectVector for a given action.
        Currently returns a placeholder zero-vector.
        """
        # Placeholder logic for effect prediction
        return FelicificEffectVector(0, 0, 0, 0, 0)

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
