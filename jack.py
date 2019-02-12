import numpy as np
from math import *

MAX_CARS = 20
MAX_MOVE_OF_CARS = 5
EXPECTED_FIRST_LOC_REQUESTS = 3
EXPECTED_SECOND_LOC_REQUESTS = 4
EXPECTED_FIRST_LOC_RETURNS = 3
EXPECTED_SECOND_LOC_RETURNS = 2
DISCOUNT_RATE = 0.9
RENTAL_CREDIT = 10
COST_OF_MOVING = 2

policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
stateVal = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

states = []
for i in range(MAX_CARS + 1):
    for j in range(MAX_CARS + 1):
        states.append([i, j])

actions = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1)

pBackup = dict()
def poisson(x, lam):
    global pBackup
    key = x * 10 + lam
    if key not in pBackup.keys():
        pBackup[key] = np.exp(-lam) * pow(lam, x) / factorial(x)
    return pBackup[key]

POISSON_UPPER_BOUND = 11

newStateVal = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
improvePolicy = False



def expectedReturn(state, action, stateValue):
    # Initiate and populate returns with cost associated with moving cars
    returns = 0.0
    returns -= COST_OF_MOVING * np.absolute(action)
    # Number of cars to start the day
    carsLoc1 = int(min(state[0] - action, MAX_CARS))
    carsLoc2 = int(min(state[1] + action, MAX_CARS))
    # Iterate over Rental Rates
    for rentalsLoc1 in range(0, POISSON_UPPER_BOUND):
        for rentalsLoc2 in range(0, POISSON_UPPER_BOUND):
            # Rental Probabilities
            rentalsProb = poisson(rentalsLoc1, EXPECTED_FIRST_LOC_REQUESTS) * poisson(rentalsLoc2, EXPECTED_SECOND_LOC_REQUESTS)
            # Total Rentals
            totalRentalsLoc1 = min(carsLoc1, rentalsLoc1)
            totalRentalsLoc2 = min(carsLoc2, rentalsLoc2)
            # Total Rewards
            rewards = (totalRentalsLoc1 + totalRentalsLoc2) * RENTAL_CREDIT
            # Iterate over Return Rates
            for returnsLoc1 in range(0, POISSON_UPPER_BOUND):
                for returnsLoc2 in range(0, POISSON_UPPER_BOUND):
                    # Return Rate Probabilities
                    prob = poisson(returnsLoc1, EXPECTED_FIRST_LOC_RETURNS) * poisson(returnsLoc2, EXPECTED_SECOND_LOC_RETURNS) * rentalsProb
                    # Number of cars at the end of the day
                    carsLoc1_prime = min(carsLoc1 - totalRentalsLoc1 + returnsLoc1, MAX_CARS)
                    carsLoc2_prime = min(carsLoc2 - totalRentalsLoc2 + returnsLoc2, MAX_CARS)
                    # Number of cars at the end of the day
                    returns += prob * (rewards + DISCOUNT_RATE * stateValue[carsLoc1_prime, carsLoc2_prime])
    return returns

#policy evaluation -------------------------------------------------------#

for i, j in states:
    newStateVal[i, j] = expectedReturn([i, j], policy[i, j], stateVal)

if np.sum(np.absolute(newStateVal - stateVal)) < 1e-4:
    stateVal = newStateVal.copy()
    improvePolicy = True
    continue

#policy improvement ------------------------------------------------------#
if improvePolicy == True:
    newPolicy = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    for i, j in states:
        actionReturns = []
        for action in actions:
            if ((action >= 0 and i >= action) or (action < 0 and j >= np.absolute(action))):
                actionReturns.append(expectedReturn([i, j], action, stateValue))
            else:
                actionReturns.append(-float('inf'))
        bestAction = np.argmax(actionReturns)
        newPolicy[i, j] = actions[bestAction]
    policyChanges = np.sum(newPolicy != policy)
    if policyChanges == 0:
        policy = newPolicy
        break
    policy = newPolicy
    improvePolicy = False
