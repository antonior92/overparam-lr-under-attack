import numpy as np


def frac2int(proportion, denominator):
    return max(int(proportion * denominator), 1)


frac2int_vec = np.vectorize(frac2int)