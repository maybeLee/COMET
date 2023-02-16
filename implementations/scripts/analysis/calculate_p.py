import numpy as np

SEED_NUM = 8


def pr(p, k):
    return ((1 - p) ** (k - 1)) * p


def check_p(p):
    epsilon = 0.001
    return epsilon < ((1 - p) ** (SEED_NUM - 1)) * p


for p in np.arange(0.001, 1, 0.001):
    total_sum = 0
    for k in range(1, SEED_NUM + 1):
        total_sum += pr(p, k)
    if total_sum >= 0.95 and total_sum <= 1 and check_p(p):
        print("Find p: ", p)
