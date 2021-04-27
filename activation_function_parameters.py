import scipy.stats as stats
import numpy as np

if __name__ == "__main__":
    # relu
    fn = lambda x: np.max(0, x)
    # Check https://en.wikipedia.org/wiki/Half-normal_distribution assume that std = 1
    # and divide everything by 2
    mu0 = 1 / np.sqrt(2 * np.pi)
    mu1 = 1 / 2 - 1 / np.pi
    mus = np.sqrt(mu1 - mu0**1 +mu1)