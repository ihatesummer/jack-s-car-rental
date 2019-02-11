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