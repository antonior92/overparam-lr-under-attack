import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

from interp_under_attack.linear_with_random_covariates.asymptotics import asymptotic_risk, assymptotic_l2_norm_squared, \
    assymptotic_lp_norm_squared, adversarial_bounds

markers = ['*', 'o', 's', '<', '>', 'h']


def plot_risk_per_ord(ax, p):
    i = 0
    risk = []
    for e in epsilon:
        r = df['advrisk-{:.1f}-{:.1f}'.format(p, e)]
        risk.append(r)
        # Plot empirical value
        l, = ax.plot(proportion, r, markers[i], ms=4, label='$\delta={}$'.format(e))
        # Plot upper bound
        lb, ub = adversarial_bounds(arisk, anorm, noise_std, signal_amplitude, e, p, proportions_for_bounds * n_train, datagen_parameter)
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
                                    proportions_for_bounds * n_train, off_diag, datagen_parameter)
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
            lb, ub = assymptotic_lp_norm_squared(arisk, anorm, p, proportions_for_bounds * n_train, signal_amplitude,
                                                 off_diag, datagen_parameter)
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





