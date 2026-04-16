from fvdm_action import ActionType, ActionCandidate
from fvdm_state import LocalState

def enumerate_feasible_actions(agent, world, state: LocalState) -> list[ActionCandidate]:
    actions = []
    
    # 1. STAY is always a deliberate option
    actions.append(ActionCandidate(action_type=ActionType.STAY))
    
    # 2. MOVE 
    # Enumerate legal target cells in vision range.
    cells_in_range_dict = agent.findCellsInRange()
    for distance, cells in cells_in_range_dict.items():
        for cell in cells:
            if cell.agent is None:
                actions.append(ActionCandidate(action_type=ActionType.MOVE, target_cell_coords=(cell.x, cell.y)))

    # 3. COMBAT
    # Valid targets using old combat rules (isNeighborValidPrey -> checks different tribe and enough wealth)
    if agent.findAggression() > 0:
        for distance, cells in cells_in_range_dict.items():
            for cell in cells:
                if cell.agent is not None and cell.agent != agent and cell.agent.isAlive():
                    if agent.isNeighborValidPrey(cell.agent):
                        actions.append(ActionCandidate(action_type=ActionType.COMBAT, target_cell_coords=(cell.x, cell.y), target_agent_id=cell.agent.ID))

    # Neighbors are frequently used for Social Actions
    neighbors = [cell.agent for cell in agent.cell.neighbors.values() if cell.agent is not None and cell.agent.isAlive()]
                        
    # 4. TRADE
    # Enumerate neighbors where trade is possible under old MRS rules
    if agent.tradeFactor > 0:
        # We need to make sure MRS is updated if necessary, but old logic did it once per turn anyway.
        agent.findMarginalRateOfSubstitution() 
        for neighbor in neighbors:
            if neighbor.marginalRateOfSubstitution is None:
                neighbor.findMarginalRateOfSubstitution()
            if agent.canTradeWithNeighbor(neighbor) != False:
                actions.append(ActionCandidate(action_type=ActionType.TRADE, target_agent_id=neighbor.ID))
                
    # 5. MATE
    # Opposite sex, both fertile, at least one open site
    if agent.isFertile():
        for neighbor in neighbors:
            if agent.isNeighborReproductionCompatible(neighbor):
                # Check for empty cells adjacent to either
                empty_cells = agent.findEmptyNeighborCells() + neighbor.findEmptyNeighborCells()
                if len(empty_cells) > 0:
                    actions.append(ActionCandidate(action_type=ActionType.MATE, target_agent_id=neighbor.ID))
                    
    # 6. CREDIT
    # Enumerate neighbors satisfying lender/borrower conditions
    if agent.isLender():
        for neighbor in neighbors:
            if neighbor.isBorrower():
                # Simplified representation: detailed logic checks sugar/spice balances independently, 
                # but feasibility means "agent is nearby and capable of borrowing". Act execution calculates amounts.
                if agent.sugar > agent.startingSugar or agent.spice > agent.startingSpice:
                    actions.append(ActionCandidate(action_type=ActionType.CREDIT, target_agent_id=neighbor.ID))
                    
    # 7. TAGGING
    # Picks neighboring agent and tag position
    if agent.tags is not None and agent.tagging:
        for neighbor in neighbors:
            for tag_index in range(len(agent.tags)):
                actions.append(ActionCandidate(action_type=ActionType.TAGGING, target_agent_id=neighbor.ID, tag_index=tag_index))

    return actions
