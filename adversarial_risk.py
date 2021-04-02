import argparse
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np  # numpy > 1.10 so we can use np.linalg.norm(...,axis=axis, keepdims=keepdims)
import random
import scipy.linalg as linalg


def compute_adv_attack(error, jac, ord = 2.0):
    """Compute adversarial atack with unitary p-norm.

    :param error:
        A numpy array of shape = (n_points,) containing (y_pred - y_true)
    :param jac:
        A numpy array of shape = (n_points, n_parameters) giving the Jacobian matrix.
        I.e. the derivative of the error in relation to the parameters. For linear model
        the Jacobian should be the same for all points. In this case, just use
        shape = (1, n_parameters) for the same result with less computation.
    :param ord:
        The p-norm is bounded in the adversarial attack. `ord` gives which p-norm is used
        ord = 2 is the euclidean norm. `ord` can a float value grater then 1 or np.inf,
        (for the infinity norm).
    :return:
        An array containing `delta_x` of shape = (n_points, n_parameters)
        which should perturbate the input. The p-norm of each row is equal to 1.
        In order to obtain the adversarial attack bounded by `e` just multiply it
        `delta_x`.
    """
    p = ord
    if p < 1:
        raise ValueError('`ord` is float value. 1<=ord<=np.inf.'
                         'ord = {} is not valid'.format(p))

    # Given p compute q
    if p == np.inf:
        magnitude = np.ones_like(jac)
    elif p == 1:
        magnitude = np.array(np.max(jac, axis=-1, keepdims=True) == jac, dtype=np.float)
    else:
        # Compute magnitude (this follows from the case the holder inequality hold:
        # i.e. see Ash p. 96 section 2.4 exercise 4)
        q = p / (p - 1)
        magnitude = np.abs(jac) ** (q / p)
    dx = np.sign(jac) * magnitude
    # rescale
    dx = dx / np.linalg.norm(dx, ord=p, axis=-1, keepdims=True)
    # Compute delta_x
    delta_x = dx * np.sign(error)[:, None]

    return delta_x


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


def train_and_evaluate(n_samples, n_features, noise_std, snr, epsilon, ord, n_test_samples, kind, off_diag, seed):
    # Get state
    rng = np.random.RandomState(seed)

    # Get parameter
    param_norm = np.sqrt(snr) * args.noise_std
    beta = param_norm / np.sqrt(n_features) * rng.randn(n_features)

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
    q = ord / (ord - 1) if ord != np.Inf else 1
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
    df = pd.DataFrame(columns=['proportion', 'seed', 'l2_param_norm', 'lq_param_norm'] + ['risk-{}'.format(e) for e in args.epsilon])
    for seed, proportion in tqdm(run_instances, smoothing=0.03):
        n_features = max(int(proportion * args.num_train_samples), 1)
        risk, l2_param_norm, lq_param_norm = train_and_evaluate(args.num_train_samples, n_features, args.noise_std, args.snr,
                                                    args.epsilon, args.ord, args.num_test_samples, args.features_kind,
                                                    args.off_diag, seed)
        dict1 = {'proportion': proportion, 'n_features': n_features, 'n_train':args.num_train_samples,
                 'n_test': args.num_test_samples, 'ord': args.ord, 'features_kind': args.features_kind, 'seed': seed,
                 'l2_param_norm': l2_param_norm, 'lq_param_norm': lq_param_norm,
                 'snr': args.snr, 'noise_std': args.noise_std}
        if args.features_kind =='equicorrelated':
            dict1['off_diag'] = args.off_diag
        dict_risks = {'risk-{}'.format(e): r for e, r in zip(args.epsilon, risk)}
        df = df.append({**dict1, **dict_risks}, ignore_index=True)
        df.to_csv(args.output, index=False)
    tqdm.write("Done")