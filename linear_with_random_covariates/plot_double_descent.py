import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

markers = ['*', 'o', 's', '<', '>', 'h']

def asymptotic_risk(proportion, signal_amplitude, features_kind, off_diag, noise_std=1.0):
    # This follows from Hastie Thm.1 (p.7) and is the same regardless of the covariance matrix

    # The variance term
    v_underparametrized = proportion / (1 - proportion)
    v_overparametrized = 1 / (proportion - 1)
    v = (proportion < 1) * v_underparametrized + (proportion > 1) * v_overparametrized

    # The bias term
    b_underparametrized = 0
    if features_kind == 'isotropic':
        b_overparametrized = (1 - 1 / proportion)
    elif features_kind == 'equicorrelated':
        b_overparametrized = (1 - off_diag) * (1 - 1 / proportion)
    b = (proportion < 1) * b_underparametrized + (proportion > 1) * b_overparametrized

    return noise_std ** 2 * v + signal_amplitude ** 2 * b


def assymptotic_l2_norm_squared(proportion, signal_amplitude, features_kind, off_diag, noise_std=1.0):

    if features_kind == 'isotropic':
        v_underparametrized = proportion / (1 - proportion)
        v_overparametrized = 1 / (proportion - 1)
    elif features_kind == 'equicorrelated':
        v_underparametrized = proportion / ((1 - proportion) * (1 - off_diag))
        v_overparametrized = 1 / ((proportion - 1)* (1 - off_diag))
    v = (proportion < 1) * v_underparametrized + (proportion > 1) * v_overparametrized

    b_underparametrized = 1
    b_overparametrized = 1 / proportion
    b = (proportion < 1) * b_underparametrized + (proportion > 1) * b_overparametrized

    return noise_std ** 2 * v + signal_amplitude ** 2 * b


def assymptotic_lp_norm_squared(arisk, anorm, ord, n_features, signal_amplitude, datagen_parameter):
    # Generalize to other norms,
    # using https://math.stackexchange.com/questions/218046/relations-between-p-norms
    if ord == np.inf:
        factor = n_features ** 1/2
    else:
        factor = n_features ** (1/2-1/ord)

    lfactor = 1 if ord >= 2 else factor
    ufactor = 1 if ord <= 2 else factor

    lower_bound = anorm * lfactor ** 2
    upper_bound = anorm * ufactor ** 2

    if datagen_parameter == 'constant':
        n = signal_amplitude * n_features ** (1/2-1/ord)
        lower_bound = np.maximum((n - lfactor * np.sqrt(arisk)) ** 2, lower_bound)

    return lower_bound, upper_bound


def adversarial_bounds(arisk, anorm, noise_std, signal_amplitude, eps, ord, n_features, datagen_parameter):
    lqnorm_lb, lqnorm_ub = assymptotic_lp_norm_squared(arisk, anorm, ord, n_features, signal_amplitude, datagen_parameter)

    upper_bound = (np.sqrt(arisk) + eps * np.sqrt(lqnorm_ub))**2 + noise_std ** 2
    lower_bound = arisk + eps**2 * lqnorm_lb + noise_std ** 2

    return lower_bound, upper_bound


def plot_risk_per_ord(ax, p):
    i = 0
    risk = []
    for e in epsilon:
        r = df['advrisk-{:.1f}-{:.1f}'.format(p, e)]
        risk.append(r)
        # Plot empirical value
        l, = ax.plot(proportion, r, markers[i], ms=4, label='${}$'.format(e))
        # Plot upper bound
        lb, ub = adversarial_bounds(arisk, anorm, noise_std, signal_amplitude, e, p, proportions_for_bounds * n_train,  datagen_parameter)
        if e == 0:
            ax.plot(proportions_for_bounds, ub, '-', color=l.get_color(), lw=2)
        else:
            ax.fill_between(proportions_for_bounds, lb, ub, color=l.get_color(), alpha=0.2)
            ax.plot(proportions_for_bounds, ub, '-', color=l.get_color(), lw=1)
            ax.plot(proportions_for_bounds, lb, '-', color=l.get_color(), lw=1)
        i += 1


def plot_risk_per_eps(ax, e):
    markers = ['<', '>', 'h', '*', 'o', 's']
    i = 0
    risk = []
    for p in ord:
        r = df['advrisk-{:.1f}-{:.1f}'.format(p, e)]
        risk.append(r)
        # Plot empirical value

        l, = ax.plot(proportion, r, markers[i], ms=4, label='$\\ell_{}$'.format('\\infty' if p == np.Inf else int(p)))
        # Plot upper bound
        lb, ub = adversarial_bounds(arisk, anorm, noise_std, signal_amplitude, e, p,
                                    proportions_for_bounds * n_train, datagen_parameter)
        if e == 0:
            ax.plot(proportions_for_bounds, ub, '-', color=l.get_color(), lw=2)
        else:
            ax.fill_between(proportions_for_bounds, lb, ub, color=l.get_color(), alpha=0.2)
            ax.plot(proportions_for_bounds, ub, '-', color=l.get_color(), lw=1)
            ax.plot(proportions_for_bounds, lb, '-', color=l.get_color(), lw=1)

        # Increment
        i += 1


def plot_norm(ax):
    for p in ord:
        pnorm = df['norm-{:.1f}'.format(p)]
        l, = ax.plot(proportion, pnorm, 'o', ms=4, label=p)
        if p == 2:
            ax.plot(proportions_for_bounds, np.sqrt(anorm), '-', color=l.get_color(), lw=2)
        else:
            lb, ub = assymptotic_lp_norm_squared(arisk, anorm, p, proportions_for_bounds * n_train, signal_amplitude, datagen_parameter)
            ax.fill_between(proportions_for_bounds, np.sqrt(lb), np.sqrt(ub), color=l.get_color(), alpha=0.2)
            ax.plot(proportions_for_bounds, np.sqrt(ub), '-', color=l.get_color(), lw=1)
            ax.plot(proportions_for_bounds, np.sqrt(lb), '-', color=l.get_color(), lw=1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot performance as a function of the proportion '
                                                 'n features / n samples rate.')
    parser.add_argument('--file', default='performance.csv',
                        help='input csv.')
    parser.add_argument('--plot_type', choices=['risk_per_ord', 'risk_per_eps', 'norm'], default='risk_per_ord',
                        help='plot styles to be used')
    parser.add_argument('--ord', type=float,
                        help='ord norm')
    parser.add_argument('--eps', type=float,
                        help='eps for the adver')
    parser.add_argument('--plot_style', nargs='*', default=[],
                        help='plot styles to be used')
    parser.add_argument('-n', '--num_points', default=1000, type=int,
                        help='number of points')
    parser.add_argument('--y_min', default=None, type=float,
                        help='inferior limit to y-axis in the plot.')
    parser.add_argument('--remove_ylabel', action='store_true',
                        help='don include ylable')
    parser.add_argument('--remove_legend', action='store_true',
                        help='don include legend')
    parser.add_argument('--y_max', default=None, type=float,
                        help='superior limit to y-axis in the plot.')
    parser.add_argument('--save', default='',
                        help='save plot in the given file (do not write extension). By default just show it.')
    args, unk = parser.parse_known_args()
    if args.plot_style:
        plt.style.use(args.plot_style)

    df = pd.read_csv(args.file)

    ord, epsilon = zip(*[(float(k.split('-')[1]), float(k.split('-')[2])) for k in df.keys() if 'advrisk-' in k])
    epsilon = np.unique(epsilon)
    ord = np.unique(ord)
    proportion = np.array(df['proportion'])

    seed = np.array(df['seed'])
    signal_amplitude = np.array(df['signal_amplitude'])[0]  # assuming all snr are the same
    n_train = np.array(df['n_train'])[0]  # assuming a fixed n_train
    noise_std = np.array(df['noise_std'])[0]  # assuming all noise_std are the same
    features_kind = np.array(df['features_kind'])[0]  # assuming all features_kind are the
    datagen_parameter = np.array(df['datagen_parameter'])[0]  # assuming all features_kind are the same

    # assuming all off_diag are the same
    off_diag = np.array(df['off_diag'])[0] if features_kind == 'equicorrelated' else None
    proportions_for_bounds = np.logspace(np.log10(min(proportion)), np.log10(max(proportion)), args.num_points)
    snr = signal_amplitude / noise_std

    # compute standard arisk
    arisk = asymptotic_risk(proportions_for_bounds, signal_amplitude, features_kind, off_diag)
    anorm = assymptotic_l2_norm_squared(proportions_for_bounds, snr, features_kind, off_diag)

    # Plot arisk (one subplot per order)
    fig, ax = plt.subplots()
    if args.plot_type == 'risk_per_ord':
        p = args.ord if args.ord is not None else ord[0]
        plot_risk_per_ord(ax, p)
        if not args.remove_ylabel:
            ax.set_ylabel('Risk')
    elif args.plot_type == 'risk_per_eps':
        e = args.eps if args.eps is not None else epsilon[0]
        plot_risk_per_eps(ax, e)
        if not args.remove_ylabel:
            ax.set_ylabel('Risk')
    elif args.plot_type == 'norm':
        plot_norm(ax)
        if not args.remove_ylabel:
            ax.set_ylabel('Norm')
    # Labels
    # Plot vertical line at the interpolation threshold
    ax.axvline(1, ls='--')
    ax.set_xlabel('$m/n$')
    if args.y_max:
        ax.set_ylim((10**args.y_min, 10**args.y_max))
    ax.set_xscale('log')
    ax.set_yscale('log')
    if not args.remove_legend:
        plt.legend()
    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()





