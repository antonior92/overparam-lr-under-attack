import scipy.linalg as linalg
import numpy as np
from activation_function_parameters import get_activation, implemented_activations


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


if __name__ == '__main__':
    import itertools
    from tqdm import tqdm
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser(description='Double descent for l-2 adversarial attack')
    parser.add_argument('-o', '--output', default='./performance.csv',
                        help='output csv file.')
    parser.add_argument('--num_train_samples', type=int, default=300,
                        help='number of samples in the experiment.')
    parser.add_argument('--num_test_samples', type=int, default=500,
                        help='number of samples in the experiment.')
    parser.add_argument('--input_dim', type=int, default=50,
                        help='dimension of the input vector.')
    parser.add_argument('-r', '--repetitions', type=int, default=4,
                        help='number of times each experiment is repeated')
    parser.add_argument('-p', '--ord', type=float, default=2.0,
                        help='ord is p norm of the adversarial attack.')
    parser.add_argument('-n', '--num_points', default=60, type=int,
                        help='number of points')
    parser.add_argument('-l', '--lower_proportion', default=-1, type=float,
                        help='the lowest value for the proportion (n features / n samples) is 10^l.')
    parser.add_argument('-u', '--upper_proportion', default=1, type=float,
                        help='the upper value for the proportion (n features / n samples) is 10^u.')
    parser.add_argument('-e', '--epsilon', default=[0, 0.1, 0.5, 1, 2], type=float, nargs='+',
                        help='the epsilon values used when computing the adversarial ttack')
    parser.add_argument('-s', '--noise_std', type=float, default=1.0,
                        help='standard deviation of the additive noise added.')
    parser.add_argument('--regularization', type=float, default=1e-7,
                        help='type of ridge regularization.')
    parser.add_argument('--activation', choices=implemented_activations, default='relu',
                        help='activations function')
    parser.add_argument('--snr', type=float, default=2.0,
                        help='signal-to-noise ratio `snr = |signal| / |noise|')
    args, unk = parser.parse_known_args()

    activation_function = get_activation(args.activation)
    regularization = 1e-2
    repetitions = 4
    lower_proportion = -0.99
    upper_proportion = 1
    num_points = 60

    # Compute performance for varying number of features
    tqdm.write("Estimating performance as a function of proportion...")
    df = pd.DataFrame(columns=['proportion', 'seed', 'n_features', 'l2_param_norm', 'risk'])
    proportions = np.logspace(args.lower_proportion, args.upper_proportion, args.num_points)
    run_instances = list(itertools.product(range(repetitions), proportions))
    for seed, proportion in tqdm(run_instances, smoothing=0.03):
        n_features = max(int(proportion * args.num_train_samples), 1)
        mse, estim_param_l2norm = \
            train_and_evaluate(args.num_train_samples, n_features, args.input_dim, args.noise_std, args.snr,
                               args.num_test_samples, activation_function, args.regularization, seed)
        df = df.append({'proportion': proportion, 'seed': seed, 'n_features': n_features,
                        'l2_param_norm': estim_param_l2norm, 'risk': mse, **vars(args)}, ignore_index=True)
        df.to_csv(args.output, index=False)
    tqdm.write("Done")