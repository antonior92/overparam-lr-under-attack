import argparse
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np  # numpy > 1.10 so we can use np.linalg.norm(...,axis=axis, keepdims=keepdims)
import random
import scipy.linalg as linalg
from adversarial_attack import compute_adv_attack


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
        # Estimate adversarial risk
        delta_X = e * delta_x
        r = np.mean((y_test - (X_test + delta_X) @ beta_hat) ** 2)
        risk.append(r)
    return risk, l2_param_norm, lq_param_norm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Double descent for l-2 adversarial attack')
    parser.add_argument('-o', '--output', default='./performance.csv',
                        help='output csv file.')
    parser.add_argument('--num_train_samples', type=int, default=300,
                       help='number of samples in the experiment')
    parser.add_argument('--num_test_samples', type=int, default=100,
                       help='number of samples in the experiment')
    parser.add_argument('-r', '--repetitions', type=int, default=1,
                        help='number of times each experiment is repeated')
    parser.add_argument('-p', '--ord', type=float, default='inf',
                        help='ord is p norm of the adversarial attack. By default l_infty attacks')
    parser.add_argument('-n', '--num_points', default=60, type=int,
                        help='number of points')
    parser.add_argument('-l', '--lower_proportion', default=-1, type=float,
                        help='the lowest value for the proportion (n features / n samples) is 10^l.')
    parser.add_argument('-u', '--upper_proportion', default=2, type=float,
                        help='the upper value for the proportion (n features / n samples) is 10^u.')
    parser.add_argument('-e', '--epsilon', default=[0, 0.01], type=float, nargs='+',
                        help='the epsilon values used when computing the adversarial ttack')
    parser.add_argument('--feature_std', type=float, default=0.01,
                         help='variance of the features used in this experiment.')
    parser.add_argument('--feature_scaling_growth', type=float, default=1.0,
                         help='here it is determined how the feature scaling grow with the number of features'
                              '`feature_scaling = n_features ** (feature_scaling_growth / 2)`')
    args, unk = parser.parse_known_args()
    print(args)

    tqdm.write("Estimating performance as a function of proportion...")
    list_dict = []
    proportions = np.logspace(args.lower_proportion, args.upper_proportion, args.num_points)
    run_instances = list(itertools.product(range(args.repetitions), proportions))

    # Some of the executions are computationally heavy and others are not. We shuffle the configurations
    # so the progress bar can give a more accurate notion of the time to completion
    #random.shuffle(run_instances)
    prev_mdl = None  # used only if reuse_weights is True
    df = pd.DataFrame(columns=['proportion', 'seed', 'l2_param_norm', 'lq_param_norm'] + ['risk-{}'.format(e) for e in args.epsilon])
    for seed, proportion in tqdm(run_instances, smoothing=0.03):
        n_features = max(int(proportion * args.num_train_samples), 1)
        feature_scaling = n_features ** (args.feature_scaling_growth / 2)
        risk, l2_param_norm, lq_param_norm = train_and_evaluate(args.num_train_samples, n_features, feature_scaling,
                                                                args.feature_std, args.epsilon, args.ord,
                                                                args.num_test_samples, seed)
        dict1 = {'proportion': proportion, 'feature_scaling_growth': args.feature_scaling_growth,
                 'feature_std': args.feature_std, 'n_train': args.num_train_samples,
                 'n_test': args.num_test_samples, 'ord': args.ord, 'seed': seed,
                 'l2_param_norm': l2_param_norm, 'lq_param_norm': lq_param_norm}
        dict_risks = {'risk-{}'.format(e): r for e, r in zip(args.epsilon, risk)}
        df = df.append({**dict1, **dict_risks}, ignore_index=True)
        df.to_csv(args.output, index=False)
    tqdm.write("Done")