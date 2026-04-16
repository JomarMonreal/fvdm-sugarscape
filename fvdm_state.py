from dataclasses import dataclass

@dataclass(frozen=True)
class LocalState:
    # Internal state
    current_sugar: float
    current_spice: float
    total_wealth: float
    vision: int
    sugar_metabolism: float
    spice_metabolism: float
    estimated_time_to_live: float

    # Resource opportunity
    nearest_visible_sugar_amount: float
    distance_to_nearest_visible_sugar: float
    nearest_visible_spice_amount: float
    distance_to_nearest_visible_spice: float
    number_of_reachable_empty_cells: int

    # Social / risk / interaction
    number_of_nearby_agents: int
    distance_to_nearest_stronger_nearby_agent: float
    distance_to_nearest_weaker_nearby_agent: float
    mean_wealth_of_nearby_agents: float
    
    # action opportunities indicators
    can_trade: bool
    can_mate: bool
    can_credit: bool
    can_combat: bool
    can_tag: bool

def build_local_state(agent, world) -> LocalState:
    MAX_SENTINEL = 9999.0
    
    current_sugar = agent.sugar
    current_spice = agent.spice
    total_wealth = current_sugar + current_spice
    vision = agent.findVision()
    sugar_metabolism = agent.findSugarMetabolism()
    spice_metabolism = agent.findSpiceMetabolism()
    # Handle TTL calculation
    ttl = agent.findTimeToLive(ageLimited=False) if hasattr(agent, 'findTimeToLive') else (agent.maxAge - agent.age if agent.maxAge != -1 else MAX_SENTINEL)
    
    nearest_visible_sugar_amount = 0.0
    distance_to_nearest_visible_sugar = MAX_SENTINEL
    nearest_visible_spice_amount = 0.0
    distance_to_nearest_visible_spice = MAX_SENTINEL
    
    number_of_reachable_empty_cells = 0
    number_of_nearby_agents = 0
    distance_to_nearest_stronger_nearby_agent = MAX_SENTINEL
    distance_to_nearest_weaker_nearby_agent = MAX_SENTINEL
    sum_nearby_wealth = 0.0
    
    cells_in_range = agent.cellsInRange if hasattr(agent, 'cellsInRange') and agent.cellsInRange else agent.findCellsInRange()
    for distance, cells in cells_in_range.items():
        for cell in cells:
            if cell.agent is None:
                number_of_reachable_empty_cells += 1
                if cell.sugar > nearest_visible_sugar_amount and distance < distance_to_nearest_visible_sugar:
                    nearest_visible_sugar_amount = cell.sugar
                    distance_to_nearest_visible_sugar = distance
                if cell.spice > nearest_visible_spice_amount and distance < distance_to_nearest_visible_spice:
                    nearest_visible_spice_amount = cell.spice
                    distance_to_nearest_visible_spice = distance
            else:
                neighbor = cell.agent
                if neighbor.isAlive():
                    number_of_nearby_agents += 1
                    neighbor_wealth = neighbor.sugar + neighbor.spice
                    sum_nearby_wealth += neighbor_wealth
                    
                    if neighbor_wealth > total_wealth:
                        if distance < distance_to_nearest_stronger_nearby_agent:
                            distance_to_nearest_stronger_nearby_agent = distance
                    else:
                        if distance < distance_to_nearest_weaker_nearby_agent:
                            distance_to_nearest_weaker_nearby_agent = distance

    mean_wealth_of_nearby_agents = (sum_nearby_wealth / number_of_nearby_agents) if number_of_nearby_agents > 0 else 0.0
    
    return LocalState(
        current_sugar=current_sugar,
        current_spice=current_spice,
        total_wealth=total_wealth,
        vision=vision,
        sugar_metabolism=sugar_metabolism,
        spice_metabolism=spice_metabolism,
        estimated_time_to_live=ttl,
        nearest_visible_sugar_amount=nearest_visible_sugar_amount,
        distance_to_nearest_visible_sugar=distance_to_nearest_visible_sugar,
        nearest_visible_spice_amount=nearest_visible_spice_amount,
        distance_to_nearest_visible_spice=distance_to_nearest_visible_spice,
        number_of_reachable_empty_cells=number_of_reachable_empty_cells,
        number_of_nearby_agents=number_of_nearby_agents,
        distance_to_nearest_stronger_nearby_agent=distance_to_nearest_stronger_nearby_agent,
        distance_to_nearest_weaker_nearby_agent=distance_to_nearest_weaker_nearby_agent,
        mean_wealth_of_nearby_agents=mean_wealth_of_nearby_agents,
        can_trade=False, # Re-evaluated externally
        can_mate=False,
        can_credit=False,
        can_combat=False,
        can_tag=False
    )
