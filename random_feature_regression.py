import numpy as np
import scipy.linalg as linalg
from scipy.optimize import root
from activation_function_parameters import *

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


def compute_asymptotics(features_over_input_dim, samples_over_input_dim, activation_params,
                        regularization, snr, noise_std):
    # As defined in Eq (8) of Mei and Montanari
    mu_star = np.sqrt(activation_params['E{fn(G)**2}'] -
                      activation_params['E{fn(G)}']**2 - activation_params['E{G*fn(G)}']**2)

    zeta = activation_params['E{G*fn(G)}'] / mu_star
    corrected_regularizaton = regularization / mu_star ** 2
    xi = np.imag(np.sqrt(features_over_input_dim * samples_over_input_dim * corrected_regularizaton))

    def analytical_function(inp):
        """Implement equation (15) from Mei and Montanary"""
        # Get input
        vs_real, vs_imag, vf_real, vf_imag = inp
        # Convert to complex number
        vf = complex(vf_real, vf_imag)  # v1 in eq (15)
        vs = complex(vs_real, vs_imag)  # v2 in eq (15)
        # Do complex calculations
        den = 1 - zeta ** 2 * vs * vf
        eq1 = vf - features_over_input_dim * (-xi - vs - zeta ** 2 * vs / den)
        eq2 = vs - samples_over_input_dim * (-xi - vf - zeta ** 2 * vf / den)
        # Return real and imaginary parts
        return np.array([eq1.real, eq1.imag, eq2.real, eq2.imag])

    sol = root(analytical_function, [0, 1000, 0, 1000])

    vs = complex(sol['x'][0], sol['x'][1])
    vf = complex(sol['x'][2], sol['x'][3])
    chi = vs * vf  # Implements Eq (16) of Mei and Montanari
    psi1 = features_over_input_dim  # as used in Mei and Montanari - to make equations bellow easier to read!
    psi2 = samples_over_input_dim  # as used in Mei and Montanari - to make equations bellow easier to read!

    def m(p, q):
        """implement chi zeta monomial in compact format."""
        return chi ** p * zeta ** q

    # Implements eq (17) of Mei and Montanari
    E0 = - m(5, 6) + 3 * m(4, 4) + (psi1*psi2 - psi1 - psi2 + 1) * m(3, 6) +\
        (psi1 + psi2 - 3 * psi1 * psi2 + 1) * m(2, 4) + 2 * m(2, 2) + m(2, 0) + \
        3 * psi1 * psi2 * m(1, 2) - psi1 * psi2
    E1 = psi2 * m(3, 4) - psi2 * m(2, 2) + psi1 * psi2 * m(1, 2) - psi1 * psi2
    E2 = m(5, 6) + 3 * m(4, 4) + (psi1 - 1) * m(3, 6) + 2 * m(3, 4) + 3 * m(3, 2) + \
         (-psi1 - 1) * m(2, 4) - 2 * m(2, 2) - m(2, 0)

    B = E1 / E0  # Implements Eq (18) of Mei and Montanari
    V = E2 / E0  # Implements Eq (19) of Mei and Montanari
    R = snr / (1 + snr) * B + 1 / (1 + snr) * V  # Implements Eq (20) of Mei and Montanari

    predicted_risk = noise_std ** 2 * (snr ** 2 + 1) * R  # Implements LHS of Eq (5) of Mei and Montanari
    return predicted_risk


if __name__ == '__main__':
    import itertools
    from tqdm import tqdm
    import pandas as pd

    input_dim = 10
    n_samples = 200
    n_test_samples = 100
    seed = 1
    snr = 1
    noise_std = 1
    activation_function = get_activation('tanh')
    activation_params = activation_function_parameters('tanh')
    regularization = 1e-7
    repetitions = 3
    lower_proportion = -1
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

    proportions2 = np.logspace(lower_proportion, upper_proportion, 100)
    predicted_risk = []
    for proportion in proportions2:
        n_features = max(int(proportion * n_samples), 1)
        r = compute_asymptotics(n_features/input_dim, n_samples/input_dim,
                                activation_params, regularization, snr, noise_std)
        predicted_risk.append(r)

    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(df['proportion'], df['risk'], '*')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()
