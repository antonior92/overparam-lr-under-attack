import argparse
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import scipy.linalg as linalg


def train_and_evaluate(n_samples, n_features, noise_std, snr, epsilon, seed):
    # Get state
    rng = np.random.RandomState(seed)

    # Get parameter
    param_norm = np.sqrt(snr) * args.noise_std
    beta = param_norm / np.sqrt(n_features) * rng.randn(n_features)

    # Generate training data
    # Get X matrix
    X = np.random.randn(n_samples, n_features)
    # Get error
    e = np.random.randn(n_samples)
    # Compute output
    y = X @ beta + noise_std * e

    # Train
    beta_hat, _resid, _rank, _s = linalg.lstsq(X, y)

    # Test data
    # Get X matrix
    X_test = np.random.randn(n_samples, n_features)
    # Get error
    e_test = np.random.randn(n_samples)
    # Compute output
    y_test = X_test @ beta + noise_std * e_test

    # Generate adversarial disturbance
    estim_param_norm = np.linalg.norm(beta_hat, ord=2)

    risk = []
    for e in epsilon:
        # Estimate adversarial risk
        delta_X = - beta_hat[None, :] /estim_param_norm * e * np.sign(y_test - X_test @ beta_hat)[:, None]
        r = np.mean((y_test - (X_test + delta_X) @ beta_hat) ** 2)
        risk.append(r)
    return risk, estim_param_norm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Double descent for l-2 adversarial attack')
    parser.add_argument('-o', '--output', default='./performance.csv',
                        help='output csv file.')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='number of samples in the experiment')
    parser.add_argument('-r', '--repetitions', type=int, default=4,
                        help='number of times each experiment is repeated')
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
    parser.add_argument('--snr', type=float, default=2.0,
                         help='signal-to-noise ratio `snr = |signal|^2 / |noise|^2')
    args, unk = parser.parse_known_args()

    tqdm.write("Estimating performance as a function of proportion...")
    list_dict = []
    underp = np.logspace(args.lower_proportion, 0, args.num_points // 2)
    overp = np.logspace(0.00001, args.upper_proportion, args.num_points - args.num_points // 2)
    proportions = np.concatenate((underp, overp))
    run_instances = list(itertools.product(range(args.repetitions), proportions))

    # Some of the executions are computationally heavy and others are not. We shuffle the configurations
    # so the progress bar can give a more accurate notion of the time to completion
    random.shuffle(run_instances)
    prev_mdl = None  # used only if reuse_weights is True
    df = pd.DataFrame(columns=['proportion', 'seed', 'norm'] + ['risk-{}'.format(e) for e in args.epsilon])
    for seed, proportion in tqdm(run_instances, smoothing=0.03):
        n_features = max(int(proportion * args.num_samples), 1)
        risk, estim_param_norm = train_and_evaluate(args.num_samples, n_features, args.noise_std, args.snr, args.epsilon, seed)
        dict1 = {'proportion': proportion, 'seed': seed, 'norm': estim_param_norm, 'snr': args.snr, 'noise_std': args.noise_std}
        dict_risks = {'risk-{}'.format(e): r for e, r in zip(args.epsilon, risk)}
        df = df.append({**dict1, **dict_risks}, ignore_index=True)
        df.to_csv(args.output, index=False)
    tqdm.write("Done")