import argparse
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np  # numpy > 1.10 so we can use np.linalg.norm(...,axis=axis, keepdims=keepdims)
import random
import scipy.linalg as linalg
from ..adversarial_attack import compute_adv_attack


def generate_features(n_samples, n_features, rng, kind, off_diag):
    # Get random components
    z = rng.randn(n_samples, n_features)
    if kind == 'isotropic':
        return z
    elif kind == 'equicorrelated':
        s = (n_features, n_features)
        cov = (1-off_diag) * np.eye(*s) + off_diag * np.ones(s)
        # take square root
        u, s, vh = np.linalg.svd(cov)
        cov_sqr = np.dot(u * np.sqrt(s), vh)
        # return
        return z @ cov_sqr
    else:
        raise ValueError('Invalid kind of feature generation')


def train_and_evaluate(n_samples, n_features, noise_std, parameter_norm, epsilon, ord,
                       n_test_samples, kind, off_diag, datagen_parameter, seed):
    # Get state
    rng = np.random.RandomState(seed)

    # Get parameter
    if datagen_parameter == 'gaussian_prior':
        beta = parameter_norm / np.sqrt(n_features) * rng.randn(n_features)
    elif datagen_parameter == 'constant':
        beta = parameter_norm / np.sqrt(n_features) * np.ones(n_features)

    # Generate training data
    # Get X matrix
    X = generate_features(n_samples, n_features, rng, kind, off_diag)
    # Get error
    e = rng.randn(n_samples)
    # Compute output
    y = X @ beta + noise_std * e

    # Train
    beta_hat, _resid, _rank, _s = linalg.lstsq(X, y)

    # Test data
    # Get X matrix
    X_test = generate_features(n_test_samples, n_features, rng, kind, off_diag)
    # Get error
    e_test = rng.randn(n_test_samples)
    # Compute output
    y_test = X_test @ beta + noise_std * e_test

    # Generate adversarial disturbance
    pnorms = {}
    pnorms['norm-2.0'] = np.linalg.norm(beta_hat, ord=2)

    # Compute error = y_pred - y_test
    risk = {}
    test_error = X_test @ beta_hat - y_test
    risk['predrisk'] = np.mean(test_error ** 2)
    for p in ord:
        # Compute ord
        if p != np.Inf and p > 1:
            q = p / (p - 1)
        elif p == 1:
            q = np.Inf
        else:
            q = 1
        pnorms['norm-{:.1f}'.format(p)] = np.linalg.norm(beta_hat, ord=q)

        jac = beta_hat
        delta_x = compute_adv_attack(test_error, jac, ord=p)
        for e in epsilon:
            # Estimate adversarial arisk
            delta_X = e * delta_x
            r = np.mean((y_test - (X_test + delta_X) @ beta_hat) ** 2)
            risk['advrisk-{:.1f}-{:.1f}'.format(p, e)] = r
    return risk, pnorms
