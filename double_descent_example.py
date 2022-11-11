import numpy as np
from sklearn.kernel_approximation import RBFSampler
from scipy.linalg import lstsq
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import scipy.signal as sgn
import numpy as np
import itertools
import random
import pandas as pd
import tqdm
import scipy.linalg as linalg

from sklearn import datasets


n_train = 300
sd = 1
cutoff_freq = 0.7
max_delay = 2
n_experiments = 50
repetitions = 10

class RandomFourierFeatures(object):
    def __init__(self, n_features: int = 20, gamma: float = 1.0, random_state: int = 0, ridge: float = 0.0):
        self.gamma = gamma
        self.rbf_feature = RBFSampler(n_components=n_features, gamma=gamma, random_state=random_state)
        self.n_features = n_features
        self.random_state = random_state
        self.estim_param = None
        self.ridge = ridge

    def fit(self, X, y):
        X = np.atleast_2d(X)
        Xf = self.rbf_feature.fit_transform(X)
        if self.ridge <= 0:  # min norm solution
            self.estim_param, _resid, _rank, _s = linalg.lstsq(Xf, y)
        else:  # SVD implementation of ridge regression
            u, s, vh = linalg.svd(Xf, full_matrices=False, compute_uv=True)
            prod_aux = s / (
                        self.ridge + s ** 2)  # If S = diag(s) => P = inv(S.T S + ridge * I) S.T => prod_aux = diag(P)
            self.estim_param = (prod_aux * (y @ u)) @ vh  # here estim_param = V P U.T

        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        Xf = self.rbf_feature.transform(X)
        return Xf @ self.estim_param

    @property
    def param_norm(self):
        return np.linalg.norm(self.estim_param)

    def get_condition_number(self, X):
        X = np.atleast_2d(X)
        Xf = self.rbf_feature.transform(X)
        s = np.linalg.svd(Xf, compute_uv=False)
        return s[1] / s[-1]

X, y = datasets.load_diabetes(return_X_y=True)
X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)

X_train = X[:n_train, :]
X_test = X[n_train:, :]
y_train = y[:n_train]
y_test = y[n_train:]

mse_train = np.zeros(n_experiments)
mse_test = np.zeros(n_experiments)
param_norm = np.zeros(n_experiments)
conditining = np.zeros(n_experiments)

# Repeat experiments for different number of parameters
df = pd.DataFrame(columns=['proportion', 'seed', 'train_mse', 'test_mse', 'param_norm', 'conditioning'])
proportions = np.logspace(-1, 2, n_experiments)
run_instances = list(itertools.product(range(repetitions), proportions))
random.shuffle(run_instances)

i = 0
for seed, proportion in tqdm.tqdm(run_instances, smoothing=0.03):
    # initialize and train model
    n_features = int(proportion * n_train)
    mdl = RandomFourierFeatures(n_features=n_features, gamma=0.6, random_state=seed)
    mdl.fit(X_train, y_train)
    y_train_pred = mdl.predict(X_train)

    # Save results
    y_test_pred = mdl.predict(X_test)
    df.loc[i, 'seed'] = seed
    df.loc[i, 'proportion'] = proportion
    df.loc[i, 'train_mse'] =  np.mean((y_train_pred - y_train)**2)
    df.loc[i, 'test_mse'] = np.mean((y_test_pred - y_test) ** 2)
    df.loc[i, 'param_norm'] = mdl.param_norm
    df.loc[i, 'conditioning'] = mdl.get_condition_number(X_train)
    i += 1


def get_quantiles(xaxis, r, quantileslower=0.05, quantilesupper=0.95):
    new_xaxis, inverse, counts = np.unique(xaxis, return_inverse=True, return_counts=True)

    r_values = np.zeros([len(new_xaxis), max(counts)])
    secondindex = np.zeros(len(new_xaxis), dtype=int)
    for n in range(len(xaxis)):
        i = inverse[n]
        j = secondindex[i]
        r_values[i, j] = r[n]
        secondindex[i] += 1
    m = np.median(r_values, axis=1)
    lerr = m - np.quantile(r_values, quantileslower, axis=1)
    uerr = np.quantile(r_values, quantilesupper, axis=1) - m
    return new_xaxis, m, lerr, uerr


def plot_errorbar(ax, x, y, lerr, uerr, color, lbl=''):
    l, = ax.plot(x, y, '-o', ms=4, color=color, label=lbl)
    ax.plot(x, y - lerr, alpha=0.6, color=l.get_color() )
    ax.plot(x, y + uerr, alpha=0.6, color=l.get_color())
    ax.fill_between(x.astype(float), y - lerr, y + uerr, alpha=0.3, color=l.get_color())


if __name__ == '__main__':
    new_xaxis, m_test, lerr_test, uerr_test = get_quantiles(df['proportion'], df['test_mse'])
    new_xaxis, m_train, lerr_train, uerr_train = get_quantiles(df['proportion'], df['train_mse'])
    new_xaxis, m_param, lerr_param, uerr_param = get_quantiles(df['proportion'], df['param_norm'])

    plt.style.use(['plot_style_files/mystyle.mplsty',
                   'plot_style_files/one_half2.mplsty',
                   'plot_style_files/mycolors.mplsty',
                   'plot_style_files/mylegend.mplsty'])
    fig, ax = plt.subplots()
    plot_errorbar(ax, new_xaxis, m_test, lerr_test,  uerr_test, color='red', lbl='test')
    ax.axvline(1, ls='--', color='black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$$m / n$$')
    ax.set_ylabel('MSE')
    plt.savefig('out/figures/dd_mse.pdf')

    fig, ax = plt.subplots()
    plot_errorbar(ax, new_xaxis, m_param, lerr_param,  uerr_param, color='green')
    ax.axvline(1, ls='--', color='black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$$m / n$$')
    ax.set_ylabel('Parameter Norm')
    plt.savefig('out/figures/dd_paramnorm.pdf')


