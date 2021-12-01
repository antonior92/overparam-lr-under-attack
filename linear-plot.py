import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json as json
import numpy as np
from linear import *


#########
# Plots #
#########
def plot_risk_and_exactbounds(ax, i, df, lbl, p, e, xaxis):
    r = df['advrisk-{:.1f}-{:.1f}'.format(p, e)]
    pred_risk = df['predrisk']
    lqnorm = df['norm-{:.1f}'.format(p)]
    # Plot empirical value
    l, = ax.plot(xaxis, r, markers[i], ms=3, label=lbl)
    if args.remove_bounds:
        return
    # Plot upper bound
    ub = (np.sqrt(pred_risk) + e * lqnorm) ** 2
    lb = pred_risk + (e * lqnorm) ** 2

    ax.plot(xaxis, ub, '.', color='black', ms=2)
    ax.plot(xaxis, lb, 's', color='black', ms=2)


def plot_risk_and_bounds(ax, i, df, lbl, p, e, xaxis, xaxis_for_bounds, n_features_for_bounds, anorm, arisk):
    r = df['advrisk-{:.1f}-{:.1f}'.format(p, e)]
    # Plot empirical value
    l, = ax.plot(xaxis, r, markers[i], ms=4, label=lbl)
    if args.remove_bounds:
        return
    # Plot upper bound
    lb, ub = adversarial_bounds(arisk, anorm, e, p, n_features_for_bounds)

    lb_2, ub_2 = adversarial_bounds(arisk, anorm, e, 2, n_features_for_bounds)
    if e == 0 or args.fillbetween == 'only-show-ub':
        ax.plot(xaxis_for_bounds, ub, '-', color=l.get_color(), lw=1.5)
    else:
        if args.fillbetween == 'closest-l2bound':
            if p < 2:
                ax.fill_between(xaxis_for_bounds, lb, lb_2, color=l.get_color(), alpha=0.3)
                ax.plot(xaxis_for_bounds, lb, '-', color=l.get_color(), lw=1)
            if p > 2:
                ax.fill_between(xaxis_for_bounds, ub_2, ub, color=l.get_color(), alpha=0.3)
                ax.plot(xaxis_for_bounds, ub, '-', color=l.get_color(), lw=1)
            if p == 2:
                ax.fill_between(xaxis_for_bounds, lb, ub, color=l.get_color(), alpha=0.3)
                ax.plot(xaxis_for_bounds, ub, '-', color=l.get_color(), lw=1)
                ax.plot(xaxis_for_bounds, lb, '-', color=l.get_color(), lw=1)
        elif args.fillbetween == 'from-lower-to-upper':
            ax.fill_between(xaxis_for_bounds, lb, ub, color=l.get_color(), alpha=0.3)
            ax.plot(xaxis_for_bounds, ub, '-', color=l.get_color(), lw=1)
            ax.plot(xaxis_for_bounds, lb, '-', color=l.get_color(), lw=1)


def plot_norm(ax, i, df, lbl, p, xaxis, xaxis_for_bounds, n_features_for_bounds, anorm):
    pnorm = df['norm-{:.1f}'.format(p)]
    l, = ax.plot(xaxis, pnorm, markers[i], ms=4, label=lbl)
    if args.remove_bounds:
        return
    if p == 2:
        ax.plot(xaxis_for_bounds, anorm, '-', color=l.get_color(), lw=2)
    else:
        lb, ub = lp_norm_bounds(p, n_features_for_bounds)
        ax.fill_between(xaxis_for_bounds, lb * anorm, ub * anorm, color=l.get_color(), alpha=0.2)
        ax.plot(xaxis_for_bounds, ub * anorm, '-', color=l.get_color(), lw=1)
        ax.plot(xaxis_for_bounds, lb * anorm, '-', color=l.get_color(), lw=1)


def plot_distance(ax, i, df, xaxis, xaxis_for_bounds, adistance):
    distance = df['l2distance']
    l, = ax.plot(xaxis, distance, markers[i], ms=4)
    if args.remove_bounds:
        return
    ax.plot(xaxis_for_bounds, adistance, '-', color=l.get_color(), lw=2)


def plot_fn(ax, df, config, ii):
    # Get proportion for bounds
    proportions_for_bounds = np.logspace(config['lower_proportion'], config['upper_proportion'], args.num_points)
    # Adjust scaling
    if config['swep_over'] == 'num_features':
        n_features_for_bounds = config["num_train_samples"] * proportions_for_bounds
        n_train_for_bounds = config["num_train_samples"] * np.ones_like(n_features_for_bounds)
    elif config['swep_over'] == 'num_train_samples':
        n_features_for_bounds = config["num_train_samples"]
        n_train_for_bounds = config["num_train_samples"] * (1 / proportions_for_bounds)
    else:
        raise ValueError
    # compute standard arisk
    if not args.remove_bounds:
        proportion_latent = config['num_latent'] / config['num_train_samples'] if config['features_kind'] == 'latent' else 0
        arisk = asymptotic_risk(proportions_for_bounds, config['signal_amplitude'], config['noise_std'],
                                config['features_kind'], config['off_diag'],
                                config['mispec_factor'], proportion_latent)
        anorm = assymptotic_l2_norm(proportions_for_bounds, config['signal_amplitude'], config['noise_std'],
                                    config['features_kind'], config['off_diag'], config['mispec_factor'],
                                    proportion_latent)
        adistance = assymptotic_l2_distance(proportions_for_bounds, config['signal_amplitude'], config['noise_std'],
                                            config['features_kind'], config['off_diag'], config['mispec_factor'],
                                            proportion_latent)
        if config['scaling'] == 'sqrt':
            anorm *= np.sqrt(n_features_for_bounds)
        elif config['scaling'] == 'sqrtlog':
            anorm *= np.sqrt(np.log(n_features_for_bounds))
        elif config['scaling'] == 'none':
            pass
        else:
            raise ValueError
    else:
        arisk, anorm, adistance = None, None, None

    # Define what is on the x-axis
    if args.xaxis == 'm-over-n':
        xaxis = df['proportion']
        xaxis_for_bounds = proportions_for_bounds
        xlabel = '$m/n$'
    elif args.xaxis == 'n-over-m':
        xaxis = 1 / df['proportion']
        xaxis_for_bounds = 1 / proportions_for_bounds
        xlabel = '$n/m$'
    elif args.xaxis == 'm':
        xaxis = df['n_features']
        xaxis_for_bounds = n_features_for_bounds
        xlabel = '$m$'
    elif args.xaxis == 'n':
        xaxis = df['n_train']
        xaxis_for_bounds = n_train_for_bounds
        xlabel = '$n$'

    # Plot arisk (one subplot per order)
    if args.plot_type == 'risk_per_ord':
        p = args.ord[0] if args.ord is not None else config['ord'][0]
        for i, e in enumerate(config['epsilon']):
            lbl = '$\delta={}$'.format(e)
            plot_risk_and_bounds(ax, i, df, lbl, p, e, xaxis, xaxis_for_bounds, n_features_for_bounds, anorm, arisk)
        if not args.remove_ylabel:
            ax.set_ylabel('Risk')
    elif args.plot_type == 'risk_per_eps':
        e = args.eps[0] if args.eps is not None else config['epsilon'][-1]
        for i, p in enumerate(config['ord']):
            lbl = '$\\ell_{}$'.format('\\infty' if p == np.Inf else r'{' + str(int(p)) + r'}')
            plot_risk_and_bounds(ax, i, df, lbl, p, e, xaxis, xaxis_for_bounds, n_features_for_bounds, anorm, arisk)
        if not args.remove_ylabel:
            ax.set_ylabel('Risk')
    elif args.plot_type == 'advrisk':
        e = args.eps[ii] if args.eps is not None else config['epsilon'][-1]
        p = args.ord[ii] if args.ord is not None else config['ord'][0]
        lbl = args.labels[ii] if args.labels is not None else ''
        plot_risk_and_bounds(ax, ii, df,  lbl, p, e, xaxis, xaxis_for_bounds, n_features_for_bounds, anorm, arisk)
        if not args.remove_ylabel:
            ax.set_ylabel('Risk')
    elif args.plot_type == 'norm':
        p = args.ord[ii] if args.ord is not None else config['ord'][0]
        lbl = args.labels[ii] if args.labels is not None else ''
        plot_norm(ax, ii, df, lbl, p, xaxis, xaxis_for_bounds, n_features_for_bounds, anorm)
        if not args.remove_ylabel:
            ax.set_ylabel('Norm')
    elif args.plot_type == 'l2distance':
        plot_distance(ax, ii, df, xaxis, xaxis_for_bounds, adistance)
        if not args.remove_ylabel:
            ax.set_ylabel('l2_distance')
    elif args.plot_type == 'train_mse':
        ax.plot(xaxis, df['train_mse'], markers[ii], ms=4)
    # Labels
    # Plot vertical line at the interpolation threshold
    if args.xaxis == 'n-over-m' or args.xaxis == 'm-over-n':
        ax.axvline(1, ls='--')
    if not args.remove_xlabel:
        ax.set_xlabel(xlabel)
    if args.y_max:
        ax.set_ylim((10**args.y_min, 10**args.y_max))
    if args.xaxis == 'n-over-m' or args.xaxis == 'm-over-n':
        ax.set_xscale('log')
        ax.set_yscale('log')
    if not args.remove_legend:
        plt.legend()
    plt.grid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot performance as a function of the proportion '
                                                 'n features / n samples rate.')
    parser.add_argument('files', help='input file.', nargs='+')
    parser.add_argument('--plot_type', choices=['risk_per_ord', 'risk_per_eps', 'advrisk', 'norm', 'l2distance', 'train_mse'],
                        default='risk_per_ord', help='plot styles to be used')
    parser.add_argument('--ord', type=float, nargs='+',
                        help='ord norm')
    parser.add_argument('--eps', type=float, nargs='+',
                        help='eps for the adversarial attack')
    parser.add_argument('--plot_style', nargs='*', default=[],
                        help='plot styles to be used')
    parser.add_argument('--labels', nargs='*', default=None,
                        help='labels to be used in the legend')
    parser.add_argument('-n', '--num_points', default=1000, type=int,
                        help='number of points')
    parser.add_argument('--xaxis', choices=['m-over-n', 'n-over-m', 'n', 'm'], default='m-over-n',
                        help='what to show in the x-axis')
    parser.add_argument('--y_min', default=None, type=float,
                        help='inferior limit to y-axis in the plot.')
    parser.add_argument('--y_max', default=None, type=float,
                        help='superior limit to y-axis in the plot.')
    parser.add_argument('--remove_ylabel', action='store_true',
                        help='don include ylable')
    parser.add_argument('--remove_xlabel', action='store_true',
                        help='don include ylable')
    parser.add_argument('--remove_legend', action='store_true',
                        help='don include legend')
    parser.add_argument('--remove_bounds', action='store_true',
                        help='don include legend')
    parser.add_argument('--second_marker_set', action='store_true',
                        help='don include ylabel')
    parser.add_argument('--fillbetween', choices=['from-lower-to-upper', 'closest-l2bound', 'only-show-ub'],
                        default='from-lower-to-upper',  help='don show fill between')
    parser.add_argument('--save', default='',
                        help='save plot in the given file (do not write extension). By default just show it.')
    args, unk = parser.parse_known_args()
    if args.plot_style:
        plt.style.use(args.plot_style)
    markers = ['*', 'o', 's', '<', '>', 'h']
    if args.second_marker_set:
        markers = ['<', '>', 'h', '*', 'o', 's']
    # Read files
    fig, ax = plt.subplots()
    for ii, file in enumerate(args.files):
        df = pd.read_csv(file + '.csv')
        with open(file + '.json') as f:
            config = json.load(f)
        plot_fn(ax, df, config, ii)
    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()





