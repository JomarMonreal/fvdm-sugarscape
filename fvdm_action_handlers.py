import math

def do_targeted_move(agent, target_cell):
    agent.gotoCell(target_cell)

def do_stay(agent):
    # Staying means no movement occurs
    pass

def do_targeted_combat(agent, target_cell):
    agent.doCombat(target_cell)

def do_targeted_mate(agent, mate, empty_cell=None):
    if empty_cell is None:
        empty_cells = agent.findEmptyNeighborCells() + mate.findEmptyNeighborCells()
        if not empty_cells:
            return
        empty_cell = empty_cells[0]
        
    if mate not in agent.socialNetwork["mates"]:
        agent.socialNetwork["mates"].append(mate)
        
    childEndowment = agent.findChildEndowment(mate)
    child = agent.addChildToCell(mate, empty_cell, childEndowment)
    child.findCellsInRange()
    child.findNeighborhood()
    agent.socialNetwork["children"].append(child)
    agent.addAgentToSocialNetwork(child)
    mate.addAgentToSocialNetwork(child)
    
    mate.updateTimesVisitedWithAgent(agent, agent.lastMovedTimestep)
    mate.updateTimesReproducedWithAgent(agent, agent.lastMovedTimestep)
    agent.updateTimesReproducedWithAgent(mate, agent.lastMovedTimestep)
    
    sugarCost = agent.startingSugar / (agent.fertilityFactor * 2)
    spiceCost = agent.startingSpice / (agent.fertilityFactor * 2)
    mateSugarCost = mate.startingSugar / (mate.fertilityFactor * 2)
    mateSpiceCost = mate.startingSpice / (mate.fertilityFactor * 2)
    
    agent.sugar -= sugarCost
    agent.spice -= spiceCost
    mate.sugar -= mateSugarCost
    mate.spice -= mateSpiceCost
    
    agent.lastReproducedTimestep = agent.timestep
    agent.lastMates += 1

def do_targeted_trade(agent, neighbor):
    # Isolated trade attempt
    tradeFlag = True
    transactions = 0
    while tradeFlag:
        neighborMRS = neighbor.marginalRateOfSubstitution
        if agent.canTradeWithNeighbor(neighbor) == False:
            tradeFlag = False
            continue

        if neighborMRS > agent.marginalRateOfSubstitution:
            spiceSeller = neighbor
            sugarSeller = agent
        else:
            spiceSeller = agent
            sugarSeller = neighbor

        spiceSellerMRS = spiceSeller.marginalRateOfSubstitution
        sugarSellerMRS = sugarSeller.marginalRateOfSubstitution

        if spiceSellerMRS < 0 or sugarSellerMRS < 0:
            break

        tradePrice = math.sqrt(spiceSellerMRS * sugarSellerMRS)
        sugarPrice = 0
        spicePrice = 0
        if tradePrice < 1:
            spicePrice = 1
            sugarPrice = tradePrice
        else:
            spicePrice = tradePrice
            sugarPrice = 1

        if spiceSeller.spice - spicePrice < spiceSeller.spiceMetabolism or sugarSeller.sugar - sugarPrice < sugarSeller.sugarMetabolism:
            tradeFlag = False
            continue

        spiceSellerNewMRS = spiceSeller.findNewMarginalRateOfSubstitution(spiceSeller.sugar + sugarPrice, spiceSeller.spice - spicePrice)
        sugarSellerNewMRS = sugarSeller.findNewMarginalRateOfSubstitution(sugarSeller.sugar - sugarPrice, sugarSeller.spice + spicePrice)

        betterSpiceSellerMRS = abs(1 - spiceSellerMRS) > abs(1 - spiceSellerNewMRS)
        betterSugarSellerMRS = abs(1 - sugarSellerMRS) > abs(1 - sugarSellerNewMRS)
        betterSpiceSellerWelfare = spiceSeller.findWelfare(sugarPrice, (-1 * spicePrice)) >= spiceSeller.findWelfare(0, 0)
        betterSugarSellerWelfare = sugarSeller.findWelfare((-1 * sugarPrice), spicePrice) >= sugarSeller.findWelfare(0, 0)

        betterForSpiceSeller = betterSpiceSellerMRS or betterSpiceSellerWelfare
        betterForSugarSeller = betterSugarSellerMRS or betterSugarSellerWelfare

        checkForMRSCrossing = spiceSellerNewMRS < sugarSellerNewMRS
        
        if betterForSpiceSeller and betterForSugarSeller and not checkForMRSCrossing:
            spiceSeller.sugar += sugarPrice
            spiceSeller.spice -= spicePrice
            sugarSeller.sugar -= sugarPrice
            sugarSeller.spice += spicePrice
            spiceSeller.findMarginalRateOfSubstitution()
            sugarSeller.findMarginalRateOfSubstitution()
            transactions += 1
        else:
            tradeFlag = False
            
    if transactions > 0:
        agent.tradeVolume += transactions
        agent.lastTradeTimestep = agent.timestep
        agent.lastTradePartners += 1

def do_targeted_credit(agent, borrower):
    interestRate = min(1, agent.lendingFactor * agent.baseInterestRate)
    maxSugarLoan = agent.sugar / 2
    maxSpiceLoan = agent.spice / 2
    if agent.isFertile():
        maxSugarLoan = max(0, agent.sugar - agent.startingSugar)
        maxSpiceLoan = max(0, agent.spice - agent.startingSpice)

    sugarLoanNeed = max(0, borrower.startingSugar - borrower.sugar)
    spiceLoanNeed = max(0, borrower.startingSpice - borrower.spice)

    sugarLoanPrincipal = min(maxSugarLoan, sugarLoanNeed)
    spiceLoanPrincipal = min(maxSpiceLoan, spiceLoanNeed)
    
    sugarLoanAmount = sugarLoanPrincipal + (sugarLoanPrincipal * interestRate)
    spiceLoanAmount = spiceLoanPrincipal + (spiceLoanPrincipal * interestRate)

    if (sugarLoanNeed == 0 and spiceLoanNeed == 0) or (sugarLoanAmount == 0 and spiceLoanAmount == 0):
        return

    agentSugarMetab = agent.findSugarMetabolism()
    agentSpiceMetab = agent.findSpiceMetabolism()
    if agent.sugar - sugarLoanPrincipal <= agentSugarMetab or agent.spice - spiceLoanPrincipal <= agentSpiceMetab:
        return

    if borrower.isCreditWorthy(sugarLoanAmount, spiceLoanAmount, agent.loanDuration):
        agent.addLoanToAgent(borrower, agent.lastMovedTimestep, sugarLoanPrincipal, sugarLoanAmount, spiceLoanPrincipal, spiceLoanAmount, agent.loanDuration)
        agent.lastLendedTimestep = agent.timestep
        agent.lastLoans += 1

def do_targeted_tagging(agent, neighbor, tag_index):
    if agent.tags is not None and len(agent.tags) > tag_index:
        neighbor.flipTag(tag_index, agent.tags[tag_index])
        neighbor.tribe = neighbor.findTribe()
