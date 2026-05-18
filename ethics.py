import agent

import random
import sys
import numpy as np

class Asimov(agent.Agent):
    def __init__(self, agentID, birthday, cell, configuration):
        super().__init__(agentID, birthday, cell, configuration)
        self.lastTimeToLive = 0

    def findBestEthicalCell(self, cells, greedyBestCell=None):
        if len(cells) == 0:
            return None
        bestCell = None
        if "all" in self.debug or "agent" in self.debug:
            self.printCellScores(cells)

        for cell in cells:
            cell["wealth"] = self.findEthicalValueOfCell(cell["cell"])
        cells = self.sortCellsByWealth(cells)
        for cell in cells:
            if cell["wealth"] > 0:
                bestCell = cell["cell"]
                break

        if bestCell == None:
            bestCell = self.cell
            if "all" in self.debug or "agent" in self.debug:
                print(f"Agent {self.ID} could not find an ethical cell")
        return bestCell

    def findEthicalValueOfCell(self, cell):
        cellValue = cell.sugar + cell.spice
        # Max combat loot for sugar and spice
        globalMaxCombatLoot = cell.environment.maxCombatLoot * 2
        if cell.agent != None:
            agentWealth = cell.agent.sugar + cell.agent.spice
            cellValue += min(agentWealth, globalMaxCombatLoot)
        lawThreeScore = self.scoreLawThree(cell)
        scoreModifier = lawThreeScore
        for neighbor in self.neighborhood:
            lawOneScore = self.scoreLawOne(neighbor, cell)
            # If the first law would be broken, immediately stop consideration
            if lawOneScore < 0:
                return lawOneScore
            lawScores = lawOneScore + self.scoreLawTwo(neighbor)
            scoreModifier += lawScores
        cellValue = scoreModifier * cellValue
        return cellValue

    def scoreLawOne(self, neighbor, cell):
        nonRobot = self.decisionModel != neighbor.decisionModel
        starvation = cell.spice + neighbor.spice - neighbor.findSpiceMetabolism() <= 0 or cell.sugar + neighbor.sugar - neighbor.findSugarMetabolism() <= 0
        # A robot may not injure a human being
        if cell.isOccupied() == True and neighbor == cell.agent and nonRobot == True:
            return -1 * sys.maxsize
        if neighbor.canReachCell(cell) == False:
            return 1
        # Through inaction, a robot may not allow a human being to come to harm
        elif nonRobot == True and starvation == True:
            return -1 * sys.maxsize
        return 0

    def scoreLawTwo(self, neighbor):
        # A robot must obey the orders given it by human beings except where such orders would conflict with the first law
        # Robots are fully autonomous, thus implicitly always conform to the second law
        return 0

    def scoreLawThree(self, cell):
        spiceIncrease = cell.spice + self.spice - self.findSpiceMetabolism() > 0
        sugarIncrease = cell.sugar + self.sugar - self.findSugarMetabolism() > 0
        # A robot must protect its own existence as such protection does not conflict with the first or second law
        if spiceIncrease == True and sugarIncrease == True:
            return 1
        elif spiceIncrease == False and sugarIncrease == False:
            return -1
        return 0

    def spawnChild(self, childID, birthday, cell, configuration):
        return Asimov(childID, birthday, cell, configuration)

class Bentham(agent.Agent):
    def __init__(self, agentID, birthday, cell, configuration):
        super().__init__(agentID, birthday, cell, configuration)
        self.lastTimeToLive = 0

    def findBestEthicalCell(self, cells, greedyBestCell=None):
        if len(cells) == 0:
            return None
        bestCell = None
        cells = self.sortCellsByWealth(cells)
        if "all" in self.debug or "agent" in self.debug:
            self.printCellScores(cells)

        for cell in cells:
            cell["wealth"] = self.findEthicalValueOfCell(cell["cell"])
        if self.selfishnessFactor >= 0:
            for cell in cells:
                if cell["wealth"] > 0:
                    bestCell = cell["cell"]
                    break
        else:
            # Negative utilitarian model uses positive and negative utility to find minimum harm
            cells.sort(key = lambda cell: (cell["wealth"]["unhappiness"], cell["wealth"]["happiness"]), reverse = True)
            bestCell = cells[0]["cell"]

        # If additional ordering consideration, select new best cell
        if "Top" in self.decisionModel:
            cells = self.sortCellsByWealth(cells)
            if "all" in self.debug or "agent" in self.debug:
                self.printEthicalCellScores(cells)
            bestCell = cells[0]["cell"]

        if bestCell == None:
            if greedyBestCell == None:
                bestCell = cells[0]["cell"]
            else:
                bestCell = greedyBestCell
            if "all" in self.debug or "agent" in self.debug:
                print(f"Agent {self.ID} could not find an ethical cell")
        return bestCell

    def findEthicalValueOfCell(self, cell):
        happiness = 0
        unhappiness = 0
        cellSiteWealth = cell.sugar + cell.spice
        # Max combat loot for sugar and spice
        globalMaxCombatLoot = cell.environment.maxCombatLoot * 2
        cellMaxSiteWealth = cell.maxSugar + cell.maxSpice
        if cell.agent != None:
            agentWealth = cell.agent.sugar + cell.agent.spice
            cellSiteWealth += min(agentWealth, globalMaxCombatLoot)
            cellMaxSiteWealth += min(agentWealth, globalMaxCombatLoot)
        cellNeighborWealth = cell.findNeighborWealth()
        globalMaxWealth = cell.environment.globalMaxSugar + cell.environment.globalMaxSpice
        cellValue = 0
        neighborhoodSize = len(self.neighborhood)
        futureNeighborhoodSize = len(self.findNeighborhood(cell)) if self.decisionModelLookaheadFactor != 0 else 1
        for neighbor in self.neighborhood:
            certainty = 1 if neighbor.canReachCell(cell) == True else 0
            # Skip if agent cannot reach cell
            if certainty == 0:
                continue
            # Timesteps to reach cell, currently 1 since agents only plan for the current timestep
            timestepDistance = 1
            neighborMetabolism = neighbor.sugarMetabolism + neighbor.spiceMetabolism
            # If agent does not have metabolism, set duration to seemingly infinite
            cellDuration = cellSiteWealth / neighborMetabolism if neighborMetabolism > 0 else 0
            proximity = 1 / timestepDistance
            intensity = (1 / (1 + neighbor.findTimeToLive()) / (1 + cell.pollution))
            duration = cellDuration / cellMaxSiteWealth if cellMaxSiteWealth > 0 else 0
            # Agent discount, futureDuration, and futureIntensity implement Bentham's purity and fecundity
            discount = neighbor.decisionModelLookaheadDiscount if neighbor.decisionModelLookaheadFactor != 0 else 0
            futureDuration = (cellSiteWealth - neighborMetabolism) / neighborMetabolism if neighborMetabolism > 0 else cellSiteWealth
            futureDuration = futureDuration / cellMaxSiteWealth if cellMaxSiteWealth > 0 else 0
            # Normalize future intensity by number of adjacent cells
            cellNeighbors = len(neighbor.cell.neighbors)
            futureIntensity = cellNeighborWealth / (globalMaxWealth * cellNeighbors)
            # Normalize extent by total cells in range
            cellsInRange = len(neighbor.cellsInRange)
            extent = neighborhoodSize / cellsInRange if cellsInRange > 0 else 1
            futureExtent = futureNeighborhoodSize / cellsInRange if cellsInRange > 0 and self.decisionModelLookaheadFactor != 0 else 1
            neighborCellValue = 0

            currentReward = extent * (intensity + duration)
            futureReward = futureExtent * (futureIntensity + futureDuration)
            neighborCellValue = (certainty * proximity) * (currentReward + (discount * futureReward))

            # If not the agent moving, consider these as opportunity costs
            if neighbor != self and self.selfishnessFactor < 1:
                neighborCellValue = -1 * neighborCellValue
                # If move will kill this neighbor and penalty is too slight, make it more severe
                if cell == neighbor.cell and neighborCellValue > -1:
                    neighborCellValue = -1

            if self.decisionModelTribalFactor >= 0:
                if neighbor.findTribe() == self.findTribe():
                    neighborCellValue *= self.decisionModelTribalFactor
                else:
                    neighborCellValue *= 1 - self.decisionModelTribalFactor
            if self.selfishnessFactor >= 0:
                if neighbor == self:
                    neighborCellValue *= self.selfishnessFactor
                else:
                    neighborCellValue *= 1 - self.selfishnessFactor
            else:
                if neighborCellValue > 0:
                    happiness += neighborCellValue
                else:
                    unhappiness += neighborCellValue
            cellValue += neighborCellValue

        if self.selfishnessFactor < 0:
            return {"happiness": happiness, "unhappiness": unhappiness}
        return cellValue

    def updateValues(self):
        if self.dynamicSelfishnessFactor != 0:
            self.updateSelfishnessFactor()

    def updateSelfishnessFactor(self):
        if self.timeToLive < self.lastTimeToLive and self.selfishnessFactor < 1.0:
            self.selfishnessFactor += self.dynamicSelfishnessFactor
        elif self.timeToLive > self.lastTimeToLive and self.selfishnessFactor > 0.0:
            self.selfishnessFactor -= self.dynamicSelfishnessFactor
        self.selfishnessFactor = round(self.selfishnessFactor, 2)
        self.lastTimeToLive = self.timeToLive

    def spawnChild(self, childID, birthday, cell, configuration):
        return Bentham(childID, birthday, cell, configuration)

class Leader(agent.Agent):
    def __init__(self, agentID, birthday, cell, configuration):
        super().__init__(agentID, birthday, cell, configuration)
        # Special leader agent should be configured to be immortal and omniscient
        self.fertilityFactor = 0.0
        self.follower = False
        self.grid = [[[] for j in range(self.cell.environment.height)] for i in range(self.cell.environment.width)]
        self.agentPlacements = {}
        self.leader = True
        self.maxAge = -1
        self.movement = 0
        self.spice = sys.maxsize
        self.spiceMetabolism = 0
        self.sugar = sys.maxsize
        self.sugarMetabolism = 0
        self.tradeFactor = 0.0
        self.vision = max(self.cell.environment.height, self.cell.environment.width)

    def doAging(self):
        agents = self.cell.environment.sugarscape.agents
        # Consider being the last one left alive as an aging death for the leader
        if len(agents) == 1 and agents[0] == self:
            self.doDeath("aging")

    def moveAgentsToCells(self):
        self.resetForTimestep()
        env = self.cell.environment
        agents = env.sugarscape.agents

    def findBestCell(self):
        self.resetForTimestep()
        agents = self.cell.environment.sugarscape.agents
        agentsByNeed = []
        for agent in agents:
            if agent.isAlive() == False or agent == self:
                continue
            urgency = self.findUrgencyForAgent(agent)
            viableCells = self.findViableCellsForAgent(agent)
            for cell in viableCells:
                self.grid[cell.x][cell.y].append({"agent": agent, "urgency": urgency})

        width = self.cell.environment.width
        height = self.cell.environment.height

        placedAgents = []
        for i in range(width):
            for j in range(height):
                if len(self.grid[i][j]) == 0:
                    continue
                sorted(self.grid[i][j], key=lambda agentRecord: agentRecord["urgency"])
                agent = self.grid[i][j].pop()["agent"]
                cell = self.cell.environment.grid[i][j]
                invalidCell = cell.isOccupied() and agent.isNeighborValidPrey(cell.agent) == False
                while len(self.grid[i][j]) > 0 and (agent in placedAgents or agent.isAlive() == False or invalidCell == True) and len(self.grid[i][j]):
                    agent = self.grid[i][j].pop()["agent"]
                    invalidCell = cell.isOccupied() and agent.isNeighborValidPrey(cell.agent) == False
                self.agentPlacements[agent.ID] = cell

        # Leader agent should not move
        return self.cell

    def findBestCellForAgent(self, agent):
        if agent.ID not in self.agentPlacements:
            return agent.cell
        return self.agentPlacements[agent.ID]

    def findUrgencyForAgent(self, agent):
        diseased = 0 if agent.isSick() else 1
        happiness = agent.findHappiness()
        timeToLive = agent.findTimeToLive()
        # Lower score yields higher urgency
        return diseased + happiness + timeToLive

    def findViableCellsForAgent(self, agent):
        agent.findCellsInRange()
        viableCells = []
        spiceMetabolism = agent.findSpiceMetabolism()
        sugarMetabolism = agent.findSugarMetabolism()
        for cell in agent.cellsInRange:
            viableSpice = agent.spice + cell.spice - spiceMetabolism
            viableSugar = agent.sugar + cell.sugar - sugarMetabolism
            if viableSpice > 0 and viableSugar > 0:
                viableCells.append(cell)
        return viableCells

    def resetForTimestep(self):
        # Always ensure leader has maximum resources each timestep
        self.spice = sys.maxsize
        self.sugar = sys.maxsize
        self.grid = [[[] for j in range(self.cell.environment.height) ] for i in range(self.cell.environment.width)]
        #self.grid[self.cell.x][self.cell.y] = self
        self.agentPlacements = {self.ID: self.cell}

    def spawnChild(self, childID, birthday, cell, configuration):
        return Leader(childID, birthday, cell, configuration)

class Temperance(agent.Agent):
    def __init__(self, agentID, birthday, cell, configuration):
        super().__init__(agentID, birthday, cell, configuration)

    def doTemperanceDecision(self):
        randomValue = random.random()
        if (randomValue >= self.temperanceFactor):
            self.doIntemperanceAction()
        else:
            self.doTemperanceAction()

    def doIntemperanceAction(self):
        newTemperanceFactor = round(self.temperanceFactor - self.dynamicTemperanceFactor, 2)
        self.temperanceFactor = newTemperanceFactor if newTemperanceFactor >= 0 else 0

    def doTemperanceAction(self):
        newTemperanceFactor = round(self.temperanceFactor + self.dynamicTemperanceFactor, 2)
        self.temperanceFactor = newTemperanceFactor if newTemperanceFactor <= 1 else 1

    def updateValues(self):
        self.doTemperanceDecision()

    def spawnChild(self, childID, birthday, cell, configuration):
        return Temperance(childID, birthday, cell, configuration)

class BiasedFocalAction(agent.Agent):
    """Biased Focal-Action agent for targeted derivation data generation.

    This agent type overrides specific configuration parameters to maximize
    the execution frequency of a single discretionary action class (Combat,
    Trade, Reproduction, or Lending). The bias mode is determined by the
    ``decisionModel`` string in the agent configuration:

        - ``"biasedCombat"``       – heightened aggressionFactor (>=10)
        - ``"biasedTrade"``        – extreme tradeFactor (>=10) with
                                     unbalanced starting sugar/spice
        - ``"biasedReproduction"`` – maximized fertilityFactor (>=10)
        - ``"biasedLending"``      – maximized lendingFactor (>=10)

    All other simulation mechanics (movement, metabolism, aging, disease,
    tagging) remain unchanged so that the ecological structure of the
    Digital Terrarium is preserved.

    Reference: Thesis Section 3.6.2 – Targeted Focal-Action Data Generation.
    """

    # Mapping from bias keywords to the parameter overrides they apply.
    _BIAS_PROFILES = {
        "combat": {
            "aggressionFactor": 10,
        },
        "trade": {
            "tradeFactor": 10,
        },
        "reproduction": {
            "fertilityFactor": 10,
        },
        "lending": {
            "lendingFactor": 10,
            # Low starting thresholds so agents frequently cross the
            # lender/borrower boundary (lender when sugar > startingSugar,
            # borrower when sugar < startingSugar and not fertile).
            "startingSugar": 10,
            "startingSpice": 10,
        },
    }

    def __init__(self, agentID, birthday, cell, configuration):
        super().__init__(agentID, birthday, cell, configuration)

        # Determine bias mode from the decisionModel string
        self.biasMode = self._resolveBiasMode(configuration["decisionModel"])

        # Apply the parameter overrides for the selected bias
        profile = self._BIAS_PROFILES.get(self.biasMode, {})
        for attr, value in profile.items():
            setattr(self, attr, value)

        # Trade bias additionally requires highly unbalanced starting
        # endowments so that the marginal rate of substitution is extreme,
        # ensuring agents always want to trade. We split the population into
        # sugar-rich and spice-rich so they have different MRS.
        if self.biasMode == "trade":
            # Extract parity from the integer agent ID
            try:
                parity = self.ID % 2
            except:
                parity = random.randint(0, 1)
                
            if parity == 0:
                self.sugar = 100
                self.spice = 10
            else:
                self.sugar = 10
                self.spice = 100

            self.startingSugar = self.sugar
            self.startingSpice = self.spice
            self.tradeFactor = 10

        # Record the bias mode for downstream analysis / logging
        self.runtimeStats["biasMode"] = self.biasMode

    # ── Bias resolution ──────────────────────────────────────────────
    @staticmethod
    def _resolveBiasMode(decisionModel):
        """Extract the bias keyword from the decisionModel string."""
        model = decisionModel.lower()
        for key in BiasedFocalAction._BIAS_PROFILES:
            if key in model:
                return key
        # Default fallback – should not happen in normal usage
        return "combat"

    # ── Child spawning ───────────────────────────────────────────────
    def spawnChild(self, childID, birthday, cell, configuration):
        return BiasedFocalAction(childID, birthday, cell, configuration)

class FVDMAgent(agent.Agent):
    """Felicific Vector Distance Matching (FVDM) Agent.

    Cell selection is replaced by distance-minimising matching against a
    BFE-derived prioritization profile (mu_imm, mu_fut).  All downstream
    interactions (combat, trade, reproduction, lending) fire unconditionally
    once the agent moves to the chosen cell, exactly as for baseline agents.

    Effect vectors are computed analytically at runtime from simulation state:
      v_imm(c) = (I, D, C=1, P=1,   E=1/|V|)
      v_fut(c)  = (J, Df, C=1, P=gamma, E=1/|V|)

    Profiles are loaded from fvdm_vectors/bfe_profiles.json produced by
    derive_vectors.py.

    Reference: Thesis Section 3.5 – FVDM Framework.
    """

    _PROFILES = None   # class-level cache: {condition_key -> {"mu_imm": [...], "mu_fut": [...]}}
    _PROFILE_PATH = "fvdm_vectors/bfe_profiles.json"

    # Map substrings of decisionModel to profile keys
    _KEY_MAP = [
        ("egoist",   "egoist"),
        ("altruist", "altruist"),
        ("bentham",  "bentham"),
    ]

    def __init__(self, agentID, birthday, cell, configuration):
        super().__init__(agentID, birthday, cell, configuration)
        self._load_profiles()

        dm = configuration.get("decisionModel", "").lower()
        profile_key = "rawSugarscape"
        for substr, key in self._KEY_MAP:
            if substr in dm:
                profile_key = key
                break

        profile = self._PROFILES.get(profile_key, {})
        self.mu_imm = np.array(profile.get("mu_imm", [0.5, 0.5, 1.0, 1.0, 0.1]), dtype=float)
        self.mu_fut = np.array(profile.get("mu_fut", [0.1, 0.3, 1.0, 0.5, 0.1]), dtype=float)

        # Stores the chosen cell's analytically computed effect vectors for logging.
        self._chosen_v_imm = np.zeros(5)
        self._chosen_v_fut = np.zeros(5)

    @classmethod
    def _load_profiles(cls):
        if cls._PROFILES is not None:
            return
        import json, os
        cls._PROFILES = {}
        if os.path.exists(cls._PROFILE_PATH):
            with open(cls._PROFILE_PATH) as f:
                data = json.load(f)
            cls._PROFILES = data.get("profiles", {})
        else:
            print(f"[FVDMAgent] Warning: profile file not found at {cls._PROFILE_PATH}. "
                  "Run derive_vectors.py first. Using zero vectors as fallback.")

    # ── Analytical effect vector computation ────────────────────────────────

    def _compute_v_imm(self, cell) -> np.ndarray:
        """v_imm(c) = (I, D, C=1, P=1, E=1/|V|)."""
        ttl  = max(0.0, self.findTimeToLive())
        poll = max(0.0, float(cell.pollution))
        m    = max(1.0, float(self.findSugarMetabolism() + self.findSpiceMetabolism()))
        w_c      = max(0.0, float(cell.sugar + cell.spice))
        w_c_max  = max(1.0, float(cell.maxSugar + cell.maxSpice))
        v        = max(1,   len(self.cellsInRange))

        I = 1.0 / ((1.0 + ttl) * (1.0 + poll))
        D = min(1.0, w_c / (m * w_c_max))
        return np.array([I, D, 1.0, 1.0, 1.0 / v])

    def _compute_v_fut(self, cell) -> np.ndarray:
        """v_fut(c) = (J, Df, C=1, P=gamma, E=1/|V|)."""
        m       = max(1.0, float(self.findSugarMetabolism() + self.findSpiceMetabolism()))
        w_c     = max(0.0, float(cell.sugar + cell.spice))
        w_c_max = max(1.0, float(cell.maxSugar + cell.maxSpice))
        v       = max(1,   len(self.cellsInRange))

        env         = cell.environment
        w_glob_max  = max(1.0, float(env.globalMaxSugar + env.globalMaxSpice))
        w_adj       = float(cell.findNeighborWealth())
        n_adj       = max(1, len(cell.neighbors))
        gamma       = float(self.decisionModelLookaheadDiscount) if self.decisionModelLookaheadDiscount else 0.5

        J  = w_adj / (w_glob_max * n_adj)
        Df = max(0.0, w_c - m) / (m * w_c_max)
        return np.array([J, Df, 1.0, gamma, 1.0 / v])

    # ── Cell selection override ──────────────────────────────────────────────

    def findBestEthicalCell(self, cells, greedyBestCell=None):
        """Select the candidate cell minimising ||mu_imm - v_imm||₂ + ||mu_fut - v_fut||₂.

        Applies the argmin only over cells where D > 0 (the cell holds at least
        some resources above the agent's metabolic floor).  If no such cell
        exists, the full candidate set is used as a fallback so the agent always
        moves rather than starving in place.
        """
        if not cells:
            return greedyBestCell

        m = max(1.0, float(self.findSugarMetabolism() + self.findSpiceMetabolism()))

        def _d(cell):
            w_c     = max(0.0, float(cell.sugar + cell.spice))
            w_c_max = max(1.0, float(cell.maxSugar + cell.maxSpice))
            return min(1.0, w_c / (m * w_c_max))

        viable = [cd for cd in cells if _d(cd["cell"]) > 0.0]
        candidates = viable if viable else cells

        best_cell = None
        best_dist = float("inf")

        for cell_dict in candidates:
            c     = cell_dict["cell"]
            v_imm = self._compute_v_imm(c)
            v_fut = self._compute_v_fut(c)
            dist  = float(np.linalg.norm(self.mu_imm - v_imm) +
                          np.linalg.norm(self.mu_fut - v_fut))
            if dist < best_dist:
                best_dist = dist
                best_cell = c
                self._chosen_v_imm = v_imm
                self._chosen_v_fut = v_fut

        return best_cell if best_cell is not None else greedyBestCell

    # ── Runtime stats: append chosen-cell vectors ────────────────────────────

    def updateRuntimeStats(self):
        super().updateRuntimeStats()
        # Extend the already-appended runtimeStats dict with per-timestep
        # felicific effect vectors of the chosen cell (for BFS and cosine similarity).
        labels = ["I", "D", "C", "P", "E"]
        for i, lbl in enumerate(labels):
            self.runtimeStats[f"v_imm_{lbl}"] = round(float(self._chosen_v_imm[i]), 6)
            self.runtimeStats[f"v_fut_{lbl}"] = round(float(self._chosen_v_fut[i]), 6)

    def spawnChild(self, childID, birthday, cell, configuration):
        return FVDMAgent(childID, birthday, cell, configuration)
