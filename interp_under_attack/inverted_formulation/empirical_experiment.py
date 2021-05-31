import argparse
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np  # numpy > 1.10 so we can use np.linalg.norm(...,axis=axis, keepdims=keepdims)
import scipy.linalg as linalg
from ..adversarial_attack import compute_adv_attack


def ridge_regression(X, y, ridge=1e-8):
    """Compute the solution to `min ||y - X b||^2 + r||b||^2`"""
    # SVD implementation of ridge regression
    u, s, vh = linalg.svd(X, full_matrices=False, compute_uv=True)
    prod_aux = s / (ridge + s ** 2)  # If S = diag(s) => P = inv(S.T S + ridge * I) S.T => prod_aux = diag(P)
    estim_param = (prod_aux * (y @ u)) @ vh  # here estim_param = V P U.T
    return estim_param


def train_and_evaluate(n_samples, n_features, feature_scaling, feature_std, epsilon, ord, n_test_samples, seed):
    # Get state
    rng = np.random.RandomState(seed)

    # Get training data
    y = rng.randn(n_samples)
    X = feature_std / feature_scaling * rng.randn(n_samples, n_features) + 1 / feature_scaling * y[:, None]

    # Train
    beta_hat = ridge_regression(X, y)

    # Test data
    # Get X matrix
    y_test = rng.randn(n_test_samples)
    X_test = feature_std / feature_scaling * rng.randn(n_test_samples, n_features) + 1 / feature_scaling * y_test[:, None]

    # Generate adversarial disturbance
    l2_param_norm = np.linalg.norm(beta_hat, ord=2)
    if ord != np.Inf and ord > 1:
        q = ord / (ord - 1)
    elif ord == 1:
        q = np.Inf
    else:
        q = 1
    lq_param_norm = np.linalg.norm(beta_hat, ord=q)

    # Compute error = y_pred - y_test
    test_error = X_test @ beta_hat - y_test
    jac = beta_hat
    delta_x = compute_adv_attack(test_error, jac, ord=ord)
    risk = []
    for e in epsilon:
        # Estimate adversarial arisk
        delta_X = e * delta_x
        r = np.mean((y_test - (X_test + delta_X) @ beta_hat) ** 2)
        risk.append(r)
    return risk, l2_param_norm, lq_param_norm
