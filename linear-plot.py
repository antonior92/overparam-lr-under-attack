import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json as json
import numpy as np

###########################
# Asymptotic computations #
###########################
def asymptotic_risk(proportion, signal_amplitude, noise_std, features_kind, off_diag):
    # This follows from Hastie Thm.1 (p.7) and is the same regardless of the covariance matrix
    # Does not account for the noise

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
    else:
        raise ValueError
    b = (proportion < 1) * b_underparametrized + (proportion > 1) * b_overparametrized

    return noise_std ** 2 * v + signal_amplitude ** 2 * b


def assymptotic_l2_norm(proportion, signal_amplitude, noise_std, features_kind, off_diag):
    if features_kind == 'isotropic':
        v_underparametrized = proportion / (1 - proportion)
        v_overparametrized = 1 / (proportion - 1)
    elif features_kind == 'equicorrelated':
        v_underparametrized = proportion / ((1 - proportion) * (1 - off_diag))
        v_overparametrized = 1 / ((proportion - 1)* (1 - off_diag))
    else:
        raise ValueError
    v = (proportion < 1) * v_underparametrized + (proportion > 1) * v_overparametrized

    b_underparametrized = 1
    b_overparametrized = 1 / proportion
    b = (proportion < 1) * b_underparametrized + (proportion > 1) * b_overparametrized

    return np.sqrt(noise_std ** 2 * v + signal_amplitude ** 2 * b)


def assymptotic_l2_distance(proportion, signal_amplitude, noise_std, features_kind, off_diag):
    if features_kind == 'isotropic':
        v_underparametrized = proportion / (1 - proportion)
        v_overparametrized = 1 / (proportion - 1)
    elif features_kind == 'equicorrelated':
        v_underparametrized = proportion / ((1 - proportion) * (1 - off_diag))
        v_overparametrized = 1 / ((proportion - 1)* (1 - off_diag))
    else:
        raise ValueError
    v = (proportion < 1) * v_underparametrized + (proportion > 1) * v_overparametrized

    b_underparametrized = 0
    b_overparametrized = 1 - 1 / proportion
    b = (proportion < 1) * b_underparametrized + (proportion > 1) * b_overparametrized

    return np.sqrt(noise_std ** 2 * v + signal_amplitude ** 2 * b)


def lp_norm_bounds(ord, sz):
    # Generalize to other norms,
    # using https://math.stackexchange.com/questions/218046/relations-between-p-norms
    if ord == np.inf:
        factor = sz ** 1/2
    else:
        factor = sz ** (1/2-1/ord)

    lfactor = 1 if ord >= 2 else factor
    ufactor = 1 if ord <= 2 else factor

    return lfactor, ufactor


def adversarial_bounds(arisk, anorm, noise_std, eps, ord, n_features):
    lb, ub = lp_norm_bounds(ord, n_features)

    upper_bound = (np.sqrt(arisk) + eps * lb * anorm)**2 + noise_std ** 2
    lower_bound = arisk + (eps * ub * anorm) ** 2 + noise_std ** 2

    return lower_bound, upper_bound


#########
# Plots #
#########
def plot_risk_and_bounds(lbl, p, e):
    r = df['advrisk-{:.1f}-{:.1f}'.format(p, e)]
    # Plot empirical value
    l, = ax.plot(df['proportion'], r, markers[i], ms=4, label=lbl)
    if args.remove_bounds:
        return
    # Plot upper bound
    lb, ub = adversarial_bounds(arisk, anorm, config['noise_std'], e, p,
                                proportions_for_bounds * config['num_train_samples'])
    if e == 0:
        ax.plot(proportions_for_bounds, ub, '-', color=l.get_color(), lw=2)
    else:
        ax.fill_between(proportions_for_bounds, lb, ub, color=l.get_color(), alpha=0.2)
        ax.plot(proportions_for_bounds, ub, '-', color=l.get_color(), lw=1)
        ax.plot(proportions_for_bounds, lb, '-', color=l.get_color(), lw=1)


def plot_norm(ax, p):
    pnorm = df['norm-{:.1f}'.format(p)]
    l, = ax.plot(df['proportion'], pnorm, markers[i], ms=4, label=p)
    if args.remove_bounds:
        return
    if p == 2:
        ax.plot(proportions_for_bounds, anorm, '-', color=l.get_color(), lw=2)
    else:
        lb, ub = lp_norm_bounds(p, proportions_for_bounds * config['num_train_samples'])
        ax.fill_between(proportions_for_bounds, lb * anorm, ub * anorm, color=l.get_color(), alpha=0.2)
        ax.plot(proportions_for_bounds, ub * anorm, '-', color=l.get_color(), lw=1)
        ax.plot(proportions_for_bounds, lb * anorm, '-', color=l.get_color(), lw=1)


def plot_distance(ax):
    distance = df['l2distance']
    l, = ax.plot(df['proportion'], distance, markers[i], ms=4)
    if args.remove_bounds:
        return
    ax.plot(proportions_for_bounds, adistance, '-', color=l.get_color(), lw=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot performance as a function of the proportion '
                                                 'n features / n samples rate.')
    parser.add_argument('file', help='input file.')
    parser.add_argument('--plot_type', choices=['risk_per_ord', 'risk_per_eps', 'norm', 'distance'],
                        default='risk_per_ord', help='plot styles to be used')
    parser.add_argument('--ord', type=float,
                        help='ord norm')
    parser.add_argument('--eps', type=float,
                        help='eps for the adversarial attack')
    parser.add_argument('--plot_style', nargs='*', default=[],
                        help='plot styles to be used')
    parser.add_argument('-n', '--num_points', default=1000, type=int,
                        help='number of points')
    parser.add_argument('--y_min', default=None, type=float,
                        help='inferior limit to y-axis in the plot.')
    parser.add_argument('--y_max', default=None, type=float,
                        help='superior limit to y-axis in the plot.')
    parser.add_argument('--remove_ylabel', action='store_true',
                        help='don include ylable')
    parser.add_argument('--remove_legend', action='store_true',
                        help='don include legend')
    parser.add_argument('--remove_bounds', action='store_true',
                        help='don include legend')
    parser.add_argument('--second_marker_set', action='store_true',
                        help='don include ylable')
    parser.add_argument('--save', default='',
                        help='save plot in the given file (do not write extension). By default just show it.')
    args, unk = parser.parse_known_args()
    if args.plot_style:
        plt.style.use(args.plot_style)
    markers = ['*', 'o', 's', '<', '>', 'h']
    if args.second_marker_set:
        markers = ['<', '>', 'h', '*', 'o', 's']
    # Read files
    df = pd.read_csv(args.file + '.csv')
    with open(args.file + '.json') as f:
        config = json.load(f)
    proportions_for_bounds = np.logspace(config['lower_proportion'], config['upper_proportion'], args.num_points)

    # compute standard arisk
    arisk = asymptotic_risk(proportions_for_bounds, config['signal_amplitude'], config['noise_std'],
                            config['features_kind'], config['off_diag'])
    anorm = assymptotic_l2_norm(proportions_for_bounds, config['signal_amplitude'], config['noise_std'],
                                config['features_kind'], config['off_diag'])
    adistance = assymptotic_l2_distance(proportions_for_bounds, config['signal_amplitude'], config['noise_std'],
                                        config['features_kind'], config['off_diag'])

    # Plot arisk (one subplot per order)
    fig, ax = plt.subplots()
    if args.plot_type == 'risk_per_ord':
        p = args.ord if args.ord is not None else config['ord'][0]
        for i, e in enumerate(config['epsilon']):
            lbl = '$\delta={}$'.format(e)
            plot_risk_and_bounds(lbl, p, e)
        if not args.remove_ylabel:
            ax.set_ylabel('Risk')
    elif args.plot_type == 'risk_per_eps':
        e = args.eps if args.eps is not None else config['epsilon'][-1]
        for i, p in enumerate(config['ord']):
            lbl = '$\\ell_{}$'.format('\\infty' if p == np.Inf else r'{' + str(int(p)) + r'}')
            plot_risk_and_bounds(lbl, p, e)
        if not args.remove_ylabel:
            ax.set_ylabel('Risk')
    elif args.plot_type == 'norm':
        for i, p in enumerate(config['ord']):
            plot_norm(ax, p)
        if not args.remove_ylabel:
            ax.set_ylabel('Norm')
    elif args.plot_type == 'distance':
        plot_distance(ax)
        if not args.remove_ylabel:
            ax.set_ylabel('l2_distance')
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





