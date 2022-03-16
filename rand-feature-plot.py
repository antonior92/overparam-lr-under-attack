from interp_under_attack.random_features.activation_function_parameters import *
from interp_under_attack.random_features.analitic_functions_v import AnaliticalVFunctions
from interp_under_attack.random_features.asymptotics import compute_asymptotics, adversarial_bounds
from interp_under_attack.random_features.uniform_distribution_over_the_sphere import rand_matrix_asymptotic_l2_norm
from interp_under_attack.util import frac2int_vec, frac2int

markers = ['o', 's', '<', '>', 'h', '*']



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


def log_interp(x, num_points=1000):
    x_min = min(x)
    x_max = max(x)
    return np.logspace(np.log10(x_min), np.log10(x_max), num_points)


def plot_risk_per_ord(ax, p):
    i = 0
    risk = []
    for e in epsilon:
        r = df['advrisk-{:.1f}-{:.1f}'.format(p, e)]
        risk.append(r)
        # Plot empirical value
        new_xaxis, m, lerr, uerr = get_quantiles(proportion, r)
        ec = ax.errorbar(x=new_xaxis, y=m, yerr=[lerr, uerr], capsize=3.5, alpha=0.8,
                         marker='o', markersize=3.5, ls='', label=r'$\delta={:.1f}$'.format(e))
        l = ec.lines[0]
        # Plot upper bound
        lb, ub = adversarial_bounds(e, p, arisk, parameter_norm, mnorm, bottleneck, activation)
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
        #lb, ub = adversarial_bounds(e, p, arisk, parameter_norm, mnorm, bottleneck, activation)
        #if e == 0:
        #    ax.plot(proportions_for_bounds, ub, '-', color=l.get_color(), lw=2)
        #else:
        #    ax.fill_between(proportions_for_bounds, lb, ub, color=l.get_color(), alpha=0.2)
        #    ax.plot(proportions_for_bounds, ub, '-', color=l.get_color(), lw=1)
        #    ax.plot(proportions_for_bounds, lb, '-', color=l.get_color(), lw=1)

        # Increment
        i += 1


def plot_norm(ax):
    ax.plot(proportion, df['norm-2.0'], '*')
    ax.plot(proportions_for_bounds, np.array(parameter_norm))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Plot performance as a function of the proportion '
                                                 'n features / n samples rate.')
    parser.add_argument('--file', default='performance.csv',
                        help='input csv.')
    parser.add_argument('--plot_style', nargs='*', default=[],
                        help='plot styles to be used')
    parser.add_argument('-n', '--num_points', default=1000, type=int,
                        help='number of points in the asymptotics')
    parser.add_argument('--ord', type=float,
                        help='ord norm')
    parser.add_argument('--eps', type=float,
                        help='eps for the adver')
    parser.add_argument('--plot_type', choices=['risk_per_ord', 'risk_per_eps', 'norm'], default='risk_per_ord',
                        help='plot styles to be used')
    parser.add_argument('--y_min', default=None, type=float,
                        help='inferior limit to y-axis in the plot.')
    parser.add_argument('--y_max', default=None, type=float,
                        help='superior limit to y-axis in the plot.')
    parser.add_argument('--remove_ylabel', action='store_true',
                        help='don include ylable')
    parser.add_argument('--remove_legend', action='store_true',
                        help='don include legend')
    parser.add_argument('--save', default='',
                        help='save plot in the given file (do not write extension). By default just show it.')
    args, unk = parser.parse_known_args()
    if args.plot_style:
        plt.style.use(args.plot_style)

    df = pd.read_csv(args.file)
    ord, epsilon = zip(*[(float(k.split('-')[1]), float(k.split('-')[2])) for k in df.keys() if 'advrisk-' in k])
    epsilon = np.unique(epsilon)
    ord = np.unique(ord)

    inputdim_over_datasize = np.array(df['inputdim_over_datasize'])
    nfeatures_over_datasize = np.array(df['nfeatures_over_datasize'])
    seed = np.array(df['seed'])
    signal_amplitude = np.array(df['signal_amplitude'])[0]  # assuming all signal_amplitude are the same
    n_samples = np.array(df['num_train_samples'])[0]  # assuming a fixed n_train
    noise_std = np.array(df['noise_std'])[0]  # assuming all noise_std are the same
    activation = np.array(df['activation'])[0]  # assuming all features_kind are the same
    regularization = np.array(df['regularization'])[0]  # assuming all features_kind are the same
    fixed = np.array(df['fixed'])[0]  # assuming all features_kind are the same


    # Compute bounds
    activation_params = activation_function_parameters(activation)
    compute_vs = AnaliticalVFunctions()

    # compute assymptotics
    inputdim_over_datasize_for_bounds = log_interp(df['inputdim_over_datasize'], args.num_points)
    nfeatures_over_datasize_for_bounds = log_interp(df['nfeatures_over_datasize'], args.num_points)
    arisk = np.zeros(args.num_points)
    mnorm = np.zeros(args.num_points)
    parameter_norm = np.zeros(args.num_points)
    for i in tqdm(range(args.num_points)):
        arisk[i], parameter_norm[i] = compute_asymptotics(nfeatures_over_datasize_for_bounds[i] / inputdim_over_datasize_for_bounds[i],
                                                          1 / inputdim_over_datasize_for_bounds[i],
                                                          activation_params, regularization, signal_amplitude, noise_std, compute_vs)
        # we divide by np.sqrt(input_dim) because this factor appears in Mei and Montanri Eq. (1)
        mnorm[i] = rand_matrix_asymptotic_l2_norm(nfeatures_over_datasize_for_bounds[i] / inputdim_over_datasize_for_bounds[i])
    # compute bottleneck
    number_of_features = frac2int_vec(nfeatures_over_datasize_for_bounds, args.num_points)
    input_dimension = frac2int_vec(inputdim_over_datasize_for_bounds, args.num_points)
    bottleneck = np.minimum(number_of_features, input_dimension)

    # Plot arisk (one subplot per order)
    proportion = nfeatures_over_datasize
    proportions_for_bounds = nfeatures_over_datasize_for_bounds
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
        ax.set_ylim((10 ** args.y_min, 10 ** args.y_max))
    ax.set_xscale('log')
    ax.set_yscale('log')
    if not args.remove_legend:
        plt.legend()
    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()
