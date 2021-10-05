import argparse
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np  # numpy > 1.10 so we can use np.linalg.norm(...,axis=axis, keepdims=keepdims)
import random
from interp_under_attack. adversarial_attack import compute_adv_attack
import json
import scipy.linalg as linalg

def generate_features(n_samples, n_features, rng, kind, off_diag):
    # Get random components
    z = rng.randn(n_samples, n_features)
    s = (n_features, n_features)
    if kind == 'isotropic':
        return z
    elif kind == 'equicorrelated':  # Significant faster implementation then the naive one
        # For convenience, let u be a vector of ones
        u = np.ones(n_features)
        # The equicorrelated matrix can be written as:
        # C = off_diag * np.outer(u, u) + (1 - off_diag) * np.eye(n_features)
        # Here we compute the decomposition of
        # np.outer(u, u) = v S v.T  using: https://math.stackexchange.com/q/704238
        # S = diag(s_rankone)
        w = u.copy()
        w[0] += np.linalg.norm(u)
        # We could just define v = np.eye(n_features) - 2 * np.outer(w, w) / np.dot(w, w)
        # instead we define the v_dot(z) = z @ v for efficiency
        def v_dot(z):
            """Compute z @ v for v = np.eye(n_features) - 2 * np.outer(w, w) / np.dot(w, w).

            where z has shape (n_samples, n_features) and the return also has shape (n_samples, n_features).
            """
            return z - 2 * np.outer((z @ w), w) / np.dot(w, w)

        s_rankone = np.array([np.dot(u, u)] + [0] * (n_features - 1))
        # Using this decomposition, we can write
        # C = V (off_diag * S + (1 - off_diag) * I) V.T
        s = off_diag * s_rankone + (1-off_diag)
        return v_dot(v_dot(z) * np.sqrt(s))
    else:
        #  For general cov matrices we could just use
        #         u, s, vh = np.linalg.svd(cov)
        #         cov_sqr = np.dot(u * np.sqrt(s), vh)
        #         return z @ cov_sqr
        # TODO: add latter...
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

    # Compute distance
    l2distance = np.linalg.norm(beta_hat - beta, ord=2)
    return risk, pnorms, l2distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Double descent for l-2 adversarial attack')
    parser.add_argument('-o', '--output', default='./performance',
                        help='output file.')
    parser.add_argument('--num_train_samples', type=int, default=100,
                       help='number of samples in the experiment')
    parser.add_argument('--num_test_samples', type=int, default=100,
                       help='number of samples in the experiment')
    parser.add_argument('-r', '--repetitions', type=int, default=4,
                        help='number of times each experiment is repeated')
    parser.add_argument('-p', '--ord',  default=[2.0], type=float, nargs='+',
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
    parser.add_argument('--datagen_parameter', choices=['gaussian_prior', 'constant'], default='gaussian_prior',
                        help='how the features are generated')
    parser.add_argument('--off_diag', default=0.5, type=float,
                        help='value of diagonal values. Default is 0.5. Only take effect when '
                             'features_kind = equicorrelated.')
    parser.add_argument('--signal_amplitude', type=float, default=1.0,
                         help='signal amplitude. I.e. \|beta*\|_2')
    args, unk = parser.parse_known_args()
    print(args)

    with open(args.output + '.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    tqdm.write("Estimating performance as a function of proportion...")
    list_dict = []
    proportions = np.logspace(args.lower_proportion, args.upper_proportion, args.num_points)
    run_instances = list(itertools.product(range(args.repetitions), proportions))

    # Some of the executions are computationally heavy and others are not. We shuffle the configurations
    # so the progress bar can give a more accurate notion of the time to completion
    random.shuffle(run_instances)
    prev_mdl = None  # used only if reuse_weights is True
    df = pd.DataFrame(columns=['proportion', 'seed'] + ['norm-{:.1f}'.format(p) for p in args.ord] +
                              ['advrisk-{:.1f}-{:.1f}'.format(p, e) for p, e in itertools.product(args.ord, args.epsilon)])
    for seed, proportion in tqdm(run_instances, smoothing=0.03):
        n_features = max(int(proportion * args.num_train_samples), 1)
        risk, pnorms, l2distance = train_and_evaluate(
            args.num_train_samples, n_features, args.noise_std, args.signal_amplitude,
            args.epsilon, args.ord, args.num_test_samples, args.features_kind,
            args.off_diag, args.datagen_parameter, seed)
        dict1 = {'proportion': proportion, 'n_features': n_features, 'seed': seed, 'l2distance': l2distance}
        df = df.append({**risk, **pnorms, **dict1}, ignore_index=True)
        df.to_csv(args.output + '.csv', index=False)
    tqdm.write("Done")

