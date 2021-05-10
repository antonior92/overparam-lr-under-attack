import scipy.linalg as linalg
import numpy as np
from activation_function_parameters import get_activation, implemented_activations, get_activation_grad
from adversarial_attack import compute_pgd_attack
from uniform_distribution_over_the_sphere import uniform_distribution_over_the_sphere


def ridge_regression(X, y, ridge):
    """Compute the solution to `min ||y - X b||^2 + r||b||^2`"""
    # SVD implementation of ridge regression
    u, s, vh = linalg.svd(X, full_matrices=False, compute_uv=True)
    prod_aux = s / (ridge + s ** 2)  # If S = diag(s) => P = inv(S.T S + ridge * I) S.T => prod_aux = diag(P)
    estim_param = (prod_aux * (y @ u)) @ vh  # here estim_param = V P U.T
    return estim_param


class Mdl(object):
    def __init__(self, Theta, estim_param, activation):
        self.Theta = Theta
        self.estim_param = estim_param
        self.activation_function = get_activation(activation)
        self.activation_function_grad = get_activation_grad(activation)

    # Function that computes prediction and derivative. It will be used to compute the adversarial attack.
    # Since the function is simple, we just compute the derivative by hand. For more complex models it would
    # make sense to use an autograd tool...
    def __call__(self, x):
        # Compute prediction
        input_dim = self.Theta.shape[0]
        a = 1 / np.sqrt(input_dim) * x @ self.Theta.T
        z = self.activation_function(a)
        y_pred = z @ self.estim_param
        # Compute derivative
        jac = 1 / np.sqrt(input_dim) * np.einsum('ij,jk,j->ik', self.activation_function_grad(z),
                                                 self.Theta, self.estim_param)
        return y_pred, jac


def train_and_evaluate(n_samples, n_features, input_dim, noise_std, snr, n_test_samples, activation,
                       regularization, ord, epsilon, seed):
    # Get state
    rng = np.random.RandomState(seed)
    # Get parameter
    beta = snr * noise_std / np.sqrt(input_dim) * rng.randn(input_dim)
    # Get activation
    activation_function = get_activation(activation)
    activation_function_grad = get_activation_grad(activation)  # To be used by the adversarial attack

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

    # Get parameter norm
    estim_param_l2norm = np.linalg.norm(estim_param, ord=2)
    if ord != np.Inf and ord > 1:
        q = ord / (ord - 1)
    elif ord == 1:
        q = np.Inf
    else:
        q = 1
    estim_param_lqnorm = np.linalg.norm(estim_param, ord=q)

    # Estimate risk
    mdl = Mdl(Theta, estim_param, activation)
    risk = []
    for e in epsilon:
        # Estimate adversarial risk
        if e > 0:
            delta_X = compute_pgd_attack(X_test, y_test, mdl, max_perturb=e)
            X_adv = X_test + delta_X
        else:
            X_adv = X_test  # i.e. there is no disturbance
        z = activation_function(1 / np.sqrt(input_dim) * X_adv @ Theta.T)
        r = np.mean((y_test - z @ estim_param) ** 2)
        risk.append(r)

    return risk, estim_param_l2norm, estim_param_lqnorm


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
    parser.add_argument('-r', '--repetitions', type=int, default=1,
                        help='number of times each experiment is repeated')
    parser.add_argument('-p', '--ord', type=float, default=2.0,
                        help='ord is p norm of the adversarial attack.')
    parser.add_argument('-n', '--num_points', default=60, type=int,
                        help='number of points')
    parser.add_argument('-l', '--lower_proportion', default=-1, type=float,
                        help='the lowest value for the proportion (n features / n samples) is 10^l.')
    parser.add_argument('-u', '--upper_proportion', default=1, type=float,
                        help='the upper value for the proportion (n features / n samples) is 10^u.')
    parser.add_argument('-s', '--noise_std', type=float, default=1.0,
                        help='standard deviation of the additive noise added.')
    parser.add_argument('--regularization', type=float, default=1e-7,
                        help='type of ridge regularization.')
    parser.add_argument('--activation', choices=implemented_activations, default='relu',
                        help='activations function')
    parser.add_argument('-e', '--epsilon', default=[0, 0.5], type=float, nargs='+',
                        help='the epsilon values used when computing the adversarial attack')
    parser.add_argument('--snr', type=float, default=2.0,
                        help='signal-to-noise ratio `snr = |signal| / |noise|')
    args, unk = parser.parse_known_args()

    # Compute performance for varying number of features
    tqdm.write("Estimating performance as a function of proportion...")
    df = pd.DataFrame(columns=['proportion', 'seed', 'n_features', 'l2_param_norm', 'risk'])
    proportions = np.logspace(args.lower_proportion, args.upper_proportion, args.num_points)
    run_instances = list(itertools.product(range(args.repetitions), proportions))
    for seed, proportion in tqdm(run_instances, smoothing=0.03):
        n_features = max(int(proportion * args.num_train_samples), 1)
        risk, estim_param_l2norm, estim_param_lq_norm = \
            train_and_evaluate(args.num_train_samples, n_features, args.input_dim, args.noise_std, args.snr,
                               args.num_test_samples, args.activation, args.regularization, args.ord,
                               args.epsilon, seed)
        dict1 = {'proportion': proportion, 'seed': seed, 'n_features': n_features,
                        'l2_param_norm': estim_param_l2norm, 'lq_param_norm': estim_param_lq_norm}
        dict_risks = {'risk-{}'.format(e): r for e, r in zip(args.epsilon, risk)}
        df = df.append({**dict1, **dict_risks, **vars(args)}, ignore_index=True)
        df.to_csv(args.output, index=False)
    tqdm.write("Done")