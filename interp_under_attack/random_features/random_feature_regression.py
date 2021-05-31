import scipy.linalg as linalg
import numpy as np

from ..util import frac2int
from .activation_function_parameters import get_activation, implemented_activations, get_activation_grad
from ..adversarial_attack import compute_pgd_attack
from .uniform_distribution_over_the_sphere import uniform_distribution_over_the_sphere


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
        input_dim = self.Theta.shape[1]
        a = 1 / np.sqrt(input_dim) * x @ self.Theta.T
        z = self.activation_function(a)
        y_pred = z @ self.estim_param
        # Compute derivative
        jac = 1 / np.sqrt(input_dim) * np.einsum('ij,jk,j->ik', self.activation_function_grad(z),
                                                 self.Theta, self.estim_param)
        return y_pred, jac


def train_and_evaluate(n_samples, n_features, input_dim, noise_std, parameter_norm, n_test_samples, activation,
                       regularization, ord, epsilon, n_adv_steps, datagen_parameter,  seed):
    # Get state
    rng = np.random.RandomState(seed)
    # Get parameter
    # Get parameter
    if datagen_parameter == 'gaussian_prior':
        beta = parameter_norm / np.sqrt(input_dim) * rng.randn(input_dim)
    elif datagen_parameter == 'constant':
        beta = parameter_norm / np.sqrt(input_dim) * np.ones(input_dim)
    # Get activation
    activation_function = get_activation(activation)

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
    pnorms = {}
    pnorms['norm-2.0'] = np.linalg.norm(estim_param, ord=2)

    # Estimate arisk
    risk = {}
    mdl = Mdl(Theta, estim_param, activation)
    for p in ord:
        if p != np.Inf and p > 1:
            q = p / (p - 1)
        elif p == 1:
            q = np.Inf
        else:
            q = 1
        pnorms['norm-{:.1f}'.format(p)] = np.linalg.norm(estim_param, ord=q)

        for e in epsilon:
            # Estimate adversarial arisk
            if e > 0:
                delta_X = compute_pgd_attack(X_test, y_test, mdl, max_perturb=e, ord=p, steps=n_adv_steps)
                X_adv = X_test + delta_X
            else:
                X_adv = X_test  # i.e. there is no disturbance
            z = activation_function(1 / np.sqrt(input_dim) * X_adv @ Theta.T)
            r = np.mean((y_test - z @ estim_param) ** 2)
            risk['advrisk-{:.1f}-{:.1f}'.format(p, e)] = r

    return risk, pnorms
