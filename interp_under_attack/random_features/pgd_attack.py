import numpy as np
from activation_function_parameters import get_activation
from interp_under_attack.random_features import compute_pgd_attack
from uniform_distribution_over_the_sphere import uniform_distribution_over_the_sphere
from random_feature_regression import ridge_regression, Mdl

# Test example of the PGD attack..
if __name__ == "__main__":
    n_samples = 100
    n_features = 50
    input_dim = 100
    noise_std = 0.1
    snr = 2
    n_test_samples = 300
    activation = 'relu'
    regularization = 1e-7
    ord = 2.0
    seed = 1

    # Get state
    rng = np.random.RandomState(seed)
    # Get parameter
    beta = snr * noise_std / np.sqrt(input_dim) * rng.randn(input_dim)
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
    Z = activation_function(1 / np.sqrt(input_dim) * X @ Theta.T)
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

    mdl = Mdl(Theta, estim_param, activation)
    delta_X = compute_pgd_attack(X_test, y_test, mdl, max_perturb=10,
                                verbose=True)
    X_adv = X_test + delta_X

    z = activation_function(1 / np.sqrt(input_dim) * X_adv @ Theta.T)
    r = np.mean((y_test - z @ estim_param) ** 2)
    print(r)