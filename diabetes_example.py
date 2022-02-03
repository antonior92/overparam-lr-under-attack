# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets
from sklearn import linear_model
import tqdm
import os
from interp_under_attack.adversarial_training import adversarial_training

colors = cycle(["b", "r", "g", "c", "k"])


def plot_coefs(alphas, coefs, name):
    for coef_l, c in zip(coefs, colors):
        plt.semilogx(1/alphas, coef_l, c=c)

    plt.xlabel("$$1/\delta$$")
    plt.ylabel("coefficients")

    if args.save:
        plt.savefig(os.path.join(args.save,'diabetes_{}.pdf'.format(name)))
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot parameters profile')
    parser.add_argument('--plot_style', nargs='*', default=[],
                        help='plot styles to be used')
    parser.add_argument('--save', default='',
                        help='save plot in the given folder (do not write extension). By default just show it.')
    args, unk = parser.parse_known_args()
    if args.plot_style:
        plt.style.use(args.plot_style)

    X, y = datasets.load_diabetes(return_X_y=True)

    X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)

    # Compute lasso paths
    eps = 1e-5  # the smaller it is the longer is the path
    print("Computing regularization path using the lasso...")
    alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps=eps)

    # Compute ridge paths
    n_alphas = 200
    alphas_ridge = np.logspace(-1, 4, n_alphas)
    coefs_ridge_ = []
    for a in tqdm.tqdm(alphas_ridge):
        ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
        ridge.fit(X, y)
        coefs_ridge_.append(ridge.coef_)
    coefs_ridge = np.stack((coefs_ridge_)).T


    n_alphas = 200
    alphas_adv = np.logspace(-5, 0, n_alphas)
    coefs_advtrain_l2_ = []
    coefs_advtrain_linf_ = []
    for a in tqdm.tqdm(alphas_adv):
        coefs_advtrain_l2_.append(adversarial_training(X, y, p=2, eps=a))
        coefs_advtrain_linf_.append(adversarial_training(X, y, p=1000, eps=a)) # p = infty seems ill conditioned
    coefs_advtrain_l2 = np.stack((coefs_advtrain_l2_)).T
    coefs_advtrain_linf = np.stack((coefs_advtrain_linf_)).T

    # Display results
    plot_coefs(alphas_lasso, coefs_lasso, 'lasso')
    plot_coefs(alphas_ridge, coefs_ridge, 'ridge')
    plot_coefs(alphas_adv, coefs_advtrain_l2, 'advtrain_l2')
    plot_coefs(alphas_adv, coefs_advtrain_linf, 'advtrain_linf')