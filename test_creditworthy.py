import sugarscape
import json
import logging

class MockAgent:
    def __init__(self):
        self.sugarMeanIncome = 5
        self.spiceMeanIncome = 5
        self.sugarMetabolism = 2
        self.spiceMetabolism = 2
        self.sugar = 10
        self.spice = 10
        self.socialNetwork = {"creditors": []}
    
    def findSpiceMetabolism(self): return self.spiceMetabolism
    def findSugarMetabolism(self): return self.sugarMetabolism
    def findCurrentSugarDebt(self): return 0
    def findCurrentSpiceDebt(self): return 0
    
    def isCreditWorthy(self, sugarLoanAmount, spiceLoanAmount, loanDuration):
        if loanDuration == 0:
            return False
        spiceMetabolism = self.findSpiceMetabolism()
        sugarMetabolism = self.findSugarMetabolism()
        sugarLoanCostPerTimestep = sugarLoanAmount / loanDuration
        spiceLoanCostPerTimestep = spiceLoanAmount / loanDuration
        sugarIncomePerTimestep = ((self.sugarMeanIncome - sugarMetabolism) - self.findCurrentSugarDebt()) - sugarLoanCostPerTimestep
        spiceIncomePerTimestep = ((self.spiceMeanIncome - spiceMetabolism) - self.findCurrentSpiceDebt()) - spiceLoanCostPerTimestep
        print(f"Income: {self.sugarMeanIncome}, Metab: {sugarMetabolism}, Cost: {sugarLoanCostPerTimestep} => Net: {sugarIncomePerTimestep}")
        if sugarIncomePerTimestep >= 0 and spiceIncomePerTimestep >= 0:
            return True
        return False

agent = MockAgent()
print(agent.isCreditWorthy(10, 10, 10))
