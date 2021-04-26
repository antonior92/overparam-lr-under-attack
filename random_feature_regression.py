import numpy as np
import scipy.linalg as linalg


def uniform_distribution_over_the_sphere(n_samples: int, dimension: int, rng):
    """Generate i.i.d. samples. Each uniformly distributed over the sphere."""
    sphere_radius = np.sqrt(dimension)
    before_normalization = rng.randn(n_samples, dimension)
    X = sphere_radius * before_normalization / np.linalg.norm(before_normalization, ord=2, axis=-1, keepdims=True)
    return X


def ridge_regression(X, y, ridge):
    """Compute the solution to `min ||y - X b||^2 + r||b||^2`"""
    # SVD implementation of ridge regression
    u, s, vh = linalg.svd(X, full_matrices=False, compute_uv=True)
    prod_aux = s / (ridge + s ** 2)  # If S = diag(s) => P = inv(S.T S + ridge * I) S.T => prod_aux = diag(P)
    estim_param = (prod_aux * (y @ u)) @ vh  # here estim_param = V P U.T
    return estim_param


def train_and_evaluate(n_samples, n_features, input_dim, noise_std, snr, n_test_samples, activation_function,
                       regularization, seed):
    # Get state
    rng = np.random.RandomState(seed)

    # Get parameter
    beta = np.sqrt(snr) * noise_std / np.sqrt(input_dim) * rng.randn(input_dim)

    # Generate training data
    # Get inputs
    X = uniform_distribution_over_the_sphere(n_samples, input_dim, rng)
    # Get error
    e = rng.randn(n_samples)
    # Compute output
    y = X @ beta + noise_std * e

    # Train
    # Get random features matrix
    Theta = uniform_distribution_over_the_sphere(n_features, input_dim, rng)
    # Get Features
    Z = activation_function(1/np.sqrt(input_dim) * X @ Theta.T)
    # estimate parameter using ridge regularization (See Eq.(2) in the Mei and Montanari paper)
    estim_param = ridge_regression(Z, y, ridge=n_features * n_samples * regularization / input_dim)

    # Generate test data
    # Get inputs
    X_test = uniform_distribution_over_the_sphere(n_test_samples, input_dim, rng)
    # Get error
    e_test = rng.randn(n_test_samples)
    # Compute output
    y_test = X_test @ beta + noise_std * e_test

    # Test
    # Get Features
    Z_test = activation_function(1 / np.sqrt(input_dim) * X_test @ Theta.T)
    test_error = Z_test @ estim_param - y_test

    # Get mean square test error
    mse = np.mean(test_error ** 2)
    # Get parameter norm
    estim_param_l2norm = np.linalg.norm(estim_param, ord=2)
    return mse, estim_param_l2norm


if __name__ == '__main__':
    input_dim = 10
    n_features = 100
    n_samples = 200
    n_test_samples = 100
    seed = 1
    snr = 1
    noise_std = 1
    activation_function = np.tanh
    regularization = 1e-7

    mse, estim_param_l2norm = \
    train_and_evaluate(n_samples, n_features, input_dim, noise_std, snr, n_test_samples, activation_function,
                       regularization, seed)
    print(mse, estim_param_l2norm)

