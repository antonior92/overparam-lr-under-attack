import argparse
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np  # numpy > 1.10 so we can use np.linalg.norm(...,axis=axis, keepdims=keepdims)
import random
import scipy.linalg as linalg
from adversarial_attack import compute_adv_attack


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


def train_and_evaluate(n_samples, n_features, noise_std, parameter_norm, epsilon, ord, n_test_samples, kind, off_diag, seed):
    # Get state
    rng = np.random.RandomState(seed)

    # Get parameter
    beta = parameter_norm / np.sqrt(n_features) * rng.randn(n_features)

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
    parser.add_argument('--num_train_samples', type=int, default=100,
                       help='number of samples in the experiment')
    parser.add_argument('--num_test_samples', type=int, default=100,
                       help='number of samples in the experiment')
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
    parser.add_argument('-f', '--features_kind', choices=['isotropic', 'equicorrelated'], default='isotropic',
                        help='how the features are generated')
    parser.add_argument('--off_diag', default=0.5, type=float,
                        help='value of diagonal values. Default is 0.5. Only take effect when '
                             'features_kind = equicorrelated.')
    parser.add_argument('--signal_amplitude', type=float, default=1.0,
                         help='signal amplitude. I.e. \|beta*\|_2')
    args, unk = parser.parse_known_args()
    print(args)

    tqdm.write("Estimating performance as a function of proportion...")
    list_dict = []
    proportions = np.logspace(args.lower_proportion, args.upper_proportion, args.num_points)
    run_instances = list(itertools.product(range(args.repetitions), proportions))

    # Some of the executions are computationally heavy and others are not. We shuffle the configurations
    # so the progress bar can give a more accurate notion of the time to completion
    random.shuffle(run_instances)
    prev_mdl = None  # used only if reuse_weights is True
    df = pd.DataFrame(columns=['proportion', 'seed', 'l2_param_norm', 'lq_param_norm'] + ['risk-{}'.format(e) for e in args.epsilon])
    for seed, proportion in tqdm(run_instances, smoothing=0.03):
        n_features = max(int(proportion * args.num_train_samples), 1)
        risk, l2_param_norm, lq_param_norm = train_and_evaluate(args.num_train_samples, n_features, args.noise_std, args.signal_amplitude,
                                                    args.epsilon, args.ord, args.num_test_samples, args.features_kind,
                                                    args.off_diag, seed)
        dict1 = {'proportion': proportion, 'n_features': n_features, 'n_train':args.num_train_samples,
                 'n_test': args.num_test_samples, 'ord': args.ord, 'features_kind': args.features_kind, 'seed': seed,
                 'l2_param_norm': l2_param_norm, 'lq_param_norm': lq_param_norm,
                 'signal_amplitude': args.signal_amplitude, 'noise_std': args.noise_std}
        if args.features_kind =='equicorrelated':
            dict1['off_diag'] = args.off_diag
        dict_risks = {'risk-{}'.format(e): r for e, r in zip(args.epsilon, risk)}
        df = df.append({**dict1, **dict_risks}, ignore_index=True)
        df.to_csv(args.output, index=False)
    tqdm.write("Done")