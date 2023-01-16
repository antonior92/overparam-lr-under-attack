import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json as json
import numpy as np
from linear import *


def get_quantiles(xaxis, r, quantileslower=0.25, quantilesupper=0.75):
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


#########
# Plots #
#########
def plot_experiments(xaxis, r, ithplot, lbl, plt_type='markers'):
    if plt_type == 'markers':
        l, = ax.plot(xaxis, r, 'o', ms=4, label=lbl)
        return l
    else:
        new_xaxis, m, lerr, uerr = get_quantiles(xaxis, r)
        if plt_type == 'error_bars':
            ec = ax.errorbar(x=new_xaxis, y=m, yerr=[lerr, uerr], capsize=3.5, alpha=0.8,
                             marker='o', markersize=3.5,  ls='', label=lbl)
            return ec.lines[0]
        elif plt_type == 'error_bars_connected':
            ec = ax.errorbar(x=new_xaxis, y=m, yerr=[lerr, uerr], capsize=3.5, alpha=0.8,
                             marker='o', markersize=3.5, ls=':', label=lbl)
            return ec.lines[0]
        elif plt_type == 'median_line':
            l, = ax.plot(new_xaxis, m, '-'+ markers[ithplot], label=lbl)
            return l
        elif plt_type == 'fill_between':
            l, = ax.plot(new_xaxis, m - lerr, alpha=0.6)
            ax.plot(new_xaxis, m + uerr, color=l.get_color(), alpha=0.6)
            ax.fill_between(new_xaxis, m-lerr, m+uerr, alpha=0.3, color=l.get_color())


def plot_empiricalbounds(ax, df, p, e, xaxis, color):
    ord_dict= {np.float64(ss.split('-')[1]): ss for ss in df.keys() if 'norm-' in ss}
    lqnorm = df[ord_dict[p]]
    pred_risk = df['predrisk']
    # Plot upper bound
    ub = (np.sqrt(pred_risk) + e * lqnorm) ** 2
    lb = pred_risk + (e * lqnorm) ** 2

    new_xaxis, mu, _, uerr = get_quantiles(xaxis, ub, quantilesupper=1)
    _, ml, lerr, _ = get_quantiles(xaxis, lb, quantileslower=0)
    ax.plot(new_xaxis, ml - lerr, color=color, lw=1)
    ax.plot(new_xaxis, mu + uerr, color=color, lw=1)
    ax.fill_between(new_xaxis, ml - lerr, mu + uerr, alpha=0.3, color=color)


def plot_risk_and_bounds(ax, i, df, lbl, p, e, xaxis, xaxis_for_bounds, n_features_for_bounds, anorm, arisk):
    ord_eps_dict = {(np.float64(ss.split('-')[1]), np.float64(ss.split('-')[2])): ss for ss in df.keys() if 'advrisk' in ss}
    r = df[ord_eps_dict[(p, e)]]
    # Plot empirical value
    l = plot_experiments(xaxis, r, i, lbl, args.experiment_plot)
    if args.empirical_bounds:
        plot_empiricalbounds(ax, df, p, e, xaxis, l.get_color())
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
    l = plot_experiments(xaxis, pnorm, i, lbl, args.experiment_plot)
    if args.remove_bounds:
        return
    if p == 2:
        ax.plot(xaxis_for_bounds, anorm, '-', color=l.get_color(), lw=2)
    else:
        lb, ub = lp_norm_bounds(p, n_features_for_bounds)
        if args.fillbetween == 'only-show-ub':
            ax.plot(xaxis_for_bounds, ub * anorm, '-', color=l.get_color(), lw=1.5)
        else:
            ax.fill_between(xaxis_for_bounds, lb * anorm, ub * anorm, color=l.get_color(), alpha=0.2)
            ax.plot(xaxis_for_bounds, ub * anorm, '-', color=l.get_color(), lw=1)
            ax.plot(xaxis_for_bounds, lb * anorm, '-', color=l.get_color(), lw=1)


def plot_distance(ax, i, df, xaxis, xaxis_for_bounds, adistance):
    distance = df['l2distance']
    l = plot_experiments(xaxis, distance, i, lbl, args.experiment_plot)
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
        arisk = asymptotic_risk(proportions_for_bounds, config['signal_amplitude'], config.get('noise_std', 0),
                                config.get('features_kind', 'latent'), config.get('off_diag', 0),
                                config.get('mispec_factor', 1), proportion_latent)
        anorm = assymptotic_l2_norm(proportions_for_bounds, config['signal_amplitude'], config.get('noise_std', 0),
                                config.get('features_kind', 'latent'), config.get('off_diag', 0),
                                config.get('mispec_factor', 1), proportion_latent)
        adistance = assymptotic_l2_distance(proportions_for_bounds, config['signal_amplitude'], config.get('noise_std', 0),
                                config.get('features_kind', 'latent'), config.get('off_diag', 0),
                                config.get('mispec_factor', 1), proportion_latent)
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
            ax.set_ylabel('Adv. Risk')
    elif args.plot_type == 'risk_per_eps':
        e = args.eps[0] if args.eps is not None else config['epsilon'][-1]
        for i, p in enumerate(config['ord']):
            lbl = '$\\ell_{}$-'.format('\\infty' if p == np.Inf else r'{' + str(int(p)) + r'}')
            plot_risk_and_bounds(ax, i, df, lbl, p, e, xaxis, xaxis_for_bounds, n_features_for_bounds, anorm, arisk)
        if not args.remove_ylabel:
            ax.set_ylabel('Adv. Risk')
    elif args.plot_type == 'advrisk':
        e = args.eps[ii] if args.eps is not None else config['epsilon'][-1]
        p = args.ord[ii] if args.ord is not None else config['ord'][0]
        lbl = args.labels[ii] if args.labels is not None else ''
        plot_risk_and_bounds(ax, ii, df,  lbl, p, e, xaxis, xaxis_for_bounds, n_features_for_bounds, anorm, arisk)
        if not args.remove_ylabel:
            if args.ylabel is None:
                ax.set_ylabel('Risk')
            else:
                ax.set_ylabel(args.ylabel)
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
        lbl = args.labels[ii] if args.labels is not None else ''
        plot_experiments(xaxis, df['train_mse'], ii, lbl, args.experiment_plot)
        if not args.remove_ylabel:
            ax.set_ylabel('Train. MSE')
    # Labels
    # Plot vertical line at the interpolation threshold
    if args.xaxis == 'n-over-m' or args.xaxis == 'm-over-n':
        ax.axvline(1, ls='--', color='black')
    if not args.remove_xlabel:
        ax.set_xlabel(xlabel)
    if args.y_max:
        ax.set_ylim((10**args.y_min, 10**args.y_max))
    if args.xaxis == 'n-over-m' or args.xaxis == 'm-over-n':
        ax.set_xscale('log')
        if args.y_scale == 'log':
            ax.set_yscale('log')
    if not args.remove_legend:
        if args.out_legend:
            plt.subplots_adjust(right=0.8)
            plt.legend(bbox_to_anchor=(args.out_legend_bbox_x, args.out_legend_bbox_y), loc='upper left')
        else:
            plt.legend()


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
    parser.add_argument('--ylabel', default=None,
                        help='ylabel')
    parser.add_argument('--xaxis', choices=['m-over-n', 'n-over-m', 'n', 'm'], default='m-over-n',
                        help='what to show in the x-axis')
    parser.add_argument('--y_min', default=None, type=float,
                        help='inferior limit to y-axis in the plot.')
    parser.add_argument('--y_max', default=None, type=float,
                        help='superior limit to y-axis in the plot.')
    parser.add_argument('--y_scale', default='log', choices=['log', 'linear'],
                       help='superior limit to y-axis in the plot.')
    parser.add_argument('--remove_ylabel', action='store_true',
                        help='don include ylable')
    parser.add_argument('--remove_xlabel', action='store_true',
                        help='don include ylable')
    parser.add_argument('--remove_legend', action='store_true',
                        help='don include legend')
    parser.add_argument('--out_legend', action='store_true',
                        help='plot legend outside of the plot')
    parser.add_argument('--out_legend_bbox_y', default=1.1, type=float,
                        help='legend coordinate')
    parser.add_argument('--out_legend_bbox_x', default=0.98, type=float,
                        help='legend coordinate')
    parser.add_argument('--remove_bounds', action='store_true',
                        help='remove asymptotic bounds')
    parser.add_argument('--empirical_bounds', action='store_true',
                        help='use empirical bounds.')
    parser.add_argument('--second_marker_set', action='store_true',
                        help='don include ylabel')
    parser.add_argument('--experiment_plot', choices=['markers', 'error_bars', 'median_line', 'error_bars_connected'],
                        default='error_bars', help='don include ylabel')
    parser.add_argument('--fillbetween', choices=['from-lower-to-upper', 'closest-l2bound', 'only-show-ub'],
                        default='from-lower-to-upper',  help='don show fill between')
    parser.add_argument('--save', default='',
                        help='save plot in the given file (do not write extension). By default just show it.')
    args, unk = parser.parse_known_args()
    if args.plot_style:
        plt.style.use(args.plot_style)
    markers = ['*', 'o', 's', '<', '>', 'h', '*', 'o', 's', '<', '>', 'h']
    if args.second_marker_set:
        markers = ['<', '>', 'h', '*', 'o', 's', '*', 'o', 's', '<', '>', 'h']
    # Read files
    fig, ax = plt.subplots()
    for ii, file in enumerate(args.files):
        df = pd.read_csv(file + '.csv')
        with open(file + '.json') as f:
            config = json.load(f)
        plot_fn(ax, df, config, ii)
    if args.save:
        plt.savefig(args.save, transparent=True)
    else:
        plt.show()





