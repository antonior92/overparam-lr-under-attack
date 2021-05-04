import numpy as np
import scipy.linalg as linalg
from scipy.optimize import root
from activation_function_parameters import *
from analitic_functions_v import AnaliticalVFunctions


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
    beta = snr * noise_std / np.sqrt(input_dim) * rng.randn(input_dim)

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


def compute_asymptotics(features_over_input_dim, samples_over_input_dim, activation_params,
                        regularization, snr, noise_std, compute_vs):
    # As defined in Eq (8) of Mei and Montanari
    mu_star = np.sqrt(activation_params['E{fn(G)**2}'] -
                      activation_params['E{fn(G)}']**2 - activation_params['E{G*fn(G)}']**2)

    zeta = activation_params['E{G*fn(G)}']**2 / mu_star**2
    corrected_regularizaton = regularization / mu_star **2
    xi_imag = np.sqrt(features_over_input_dim * samples_over_input_dim * corrected_regularizaton)
    rho = snr**2
    psi1 = features_over_input_dim  # as used in Mei and Montanari - to make equations bellow easier to read!
    psi2 = samples_over_input_dim  # as used in Mei and Montanari - to make equations bellow easier to read!

    vf, vs = compute_vs(psi1, psi2, zeta, 1j*xi_imag)

    # Implements Eq (16) of Mei and Montanari
    # I am assuming here that chi is a real number. And that, except for numerical errors
    # imag(vs * vf) was supposed to be zero
    chi = -np.imag(vf) * np.imag(vs)

    def m(p, q):
        """implement chi zeta monomial in compact format."""
        return chi ** p * zeta ** q

    # ---- Compute prediction risk ---- #
    # Implements eq (17) of Mei and Montanari
    E0 = - m(5, 6) + 3 * m(4, 4) + (psi1*psi2 - psi1 - psi2 + 1) * m(3, 6) - 2 * m(3, 4) - 3 * m(3, 2) + \
        (psi1 + psi2 - 3 * psi1 * psi2 + 1) * m(2, 4) + 2 * m(2, 2) + m(2, 0) + \
        3 * psi1 * psi2 * m(1, 2) - psi1 * psi2
    E1 = psi2 * m(3, 4) - psi2 * m(2, 2) + psi1 * psi2 * m(1, 2) - psi1 * psi2
    E2 = m(5, 6) - 3 * m(4, 4) + (psi1 - 1) * m(3, 6) + 2 * m(3, 4) + 3 * m(3, 2) + \
         (-psi1 - 1) * m(2, 4) - 2 * m(2, 2) - m(2, 0)

    B = E1 / E0  # Implements Eq (18) of Mei and Montanari
    V = E2 / E0  # Implements Eq (19) of Mei and Montanari
    R = rho / (1 + rho) * B + 1 / (1 + rho) * V  # Implements Eq (20) of Mei and Montanari
    predicted_risk = noise_std ** 2 * ((snr**2 + 1) * R + 1)  # Implements LHS of Eq (5) of Mei and Montanari

    # ---- Compute parameter norm ---- #
    # Implements Eq (48) of Mei and Montanari
    A1 = - rho / (1 + rho) * m(2, 0) * ((1 - psi2) * m(1, 4) - m(1, 2) + (1 + psi2) * m(0, 2) + 1) + \
         + 1 / (1 + rho) * m(2, 0) * (m(1, 2) - 1) * (m(2, 4) - 2 * m(1, 2) + m(0, 2) + 1)
    A0 = E0
    A = A1 / A0
    parameter_norm = noise_std * np.sqrt((snr**2 + 1) * A) / mu_star

    return predicted_risk, parameter_norm


if __name__ == '__main__':
    import itertools
    from tqdm import tqdm
    import pandas as pd

    input_dim = 400
    n_samples = 300
    n_test_samples = 500
    seed = 2
    snr = 4
    noise_std = 1
    activation_function = get_activation('relu')
    activation_params = activation_function_parameters('relu')
    regularization = 1e-7
    repetitions = 4
    lower_proportion = -0.99
    upper_proportion = 1
    num_points = 60

    # Compute performance for varying number of features
    df = pd.DataFrame(columns=['proportion', 'seed', 'l2_param_norm', 'risk'])
    proportions = np.logspace(lower_proportion, upper_proportion, num_points)
    run_instances = list(itertools.product(range(repetitions), proportions))
    for seed, proportion in tqdm(run_instances, smoothing=0.03):
        n_features = max(int(proportion * n_samples), 1)
        mse, estim_param_l2norm = \
            train_and_evaluate(n_samples, n_features, input_dim, noise_std, snr, n_test_samples, activation_function,
                               regularization, seed)
        df = df.append({'proportion': proportion, 'seed': seed,
                        'l2_param_norm': estim_param_l2norm, 'risk': mse}, ignore_index=True)

    # Compute bounds
    features_over_input_dim = n_features / input_dim
    samples_over_input_dim = n_samples / input_dim
    compute_vs = AnaliticalVFunctions()

    proportions2 = np.logspace(lower_proportion, upper_proportion, 100)
    predicted_risk = []
    parameter_norm = []
    for proportion in tqdm(proportions2):
        n_features = max(int(proportion * n_samples), 1)
        r, n = compute_asymptotics(n_features/input_dim, n_samples/input_dim,
                                   activation_params, regularization, snr, noise_std,
                                   compute_vs)
        predicted_risk.append(r)
        parameter_norm.append(n)

    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(df['proportion'], df['risk'], '*')
    ax.plot(proportions2, predicted_risk)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('prediction risk')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(df['proportion'], df['l2_param_norm'], '*')
    ax.plot(proportions2, np.array(parameter_norm))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('parameter norm')
    plt.show()
