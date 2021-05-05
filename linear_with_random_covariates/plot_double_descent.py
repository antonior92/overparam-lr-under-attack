import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def asymptotic_risk(proportion, snr, features_kind, off_diag, noise_std=1.0):
    # This follows from Hastie Thm.1 (p.7) and is the same regardless of the covariance matrix
    underparametrized = proportion / (1 - proportion)
    if features_kind == 'isotropic':
        overparametrized = snr * (1 - 1 / proportion) + 1 / (proportion - 1)
    elif features_kind == 'equicorrelated':
        overparametrized = snr * (1 - off_diag) * (1 - 1 / proportion) + 1 / (proportion - 1)
    return noise_std ** 2 * ((proportion < 1) * underparametrized + (proportion > 1) * overparametrized)


def assymptotic_l2_norm_squared(proportion, snr, features_kind, off_diag, noise_std=1.0):
    # todo: get the close formula for the equicorrelated case.
    if features_kind == 'isotropic':
        underparametrized = snr + proportion / (1 - proportion)
        overparametrized = snr * 1 / proportion + 1 / (proportion - 1)
    elif features_kind == 'equicorrelated':
        underparametrized = snr + proportion / ((1 - proportion)*(1 - off_diag))
        overparametrized = snr * 1 / proportion + 1 / ((proportion - 1)*(1 - off_diag))
    return noise_std ** 2 * ((proportion < 1) * underparametrized + (proportion > 1) * overparametrized)


def adversarial_bounds(proportion, snr, noise_std, eps, ord, n_features, features_kind, off_diag):
    arisk = asymptotic_risk(proportion, snr, features_kind, off_diag)
    anorm = assymptotic_l2_norm_squared(proportion, snr, features_kind, off_diag)

    # Generalize to other norms,
    # using https://math.stackexchange.com/questions/218046/relations-between-p-norms
    if ord == np.inf:
        factor = n_features ** 1/2
    else:
        factor = n_features ** (1/2-1/ord)

    lower_eps = eps if ord >= 2 else eps * factor
    upper_eps = eps if ord <= 2 else eps * factor

    upper_bound = (np.sqrt(arisk) + upper_eps * np.sqrt(anorm))**2 + noise_std ** 2
    lower_bound = arisk + lower_eps**2 * anorm + noise_std ** 2
    return lower_bound, upper_bound


def assymptotic_lp_norm_squared(proportion, snr, features_kind, off_diag, ord, n_features, noise_std):
    anorm = assymptotic_l2_norm_squared(proportion, snr, features_kind, off_diag, noise_std)

    if ord == np.inf:
        factor = n_features ** 1/2
    else:
        factor = n_features ** (1/2-1/ord)

    lower_bound = anorm if ord >= 2 else anorm * factor**2
    upper_bound = anorm if ord <= 2 else anorm * factor**2
    return lower_bound, upper_bound


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot performance as a function of the proportion '
                                                 'n features / n samples rate.')
    parser.add_argument('--file', default='performance.csv',
                        help='input csv.')
    parser.add_argument('--plot_style', nargs='*', default=[],
                        help='plot styles to be used')
    parser.add_argument('-n', '--num_points', default=1000, type=int,
                        help='number of points')
    parser.add_argument('--y_min', default=None, type=float,
                        help='inferior limit to y-axis in the plot.')
    parser.add_argument('--y_max', default=None, type=float,
                        help='superior limit to y-axis in the plot.')
    parser.add_argument('--save', default='',
                        help='save plot in the given file (do not write extension). By default just show it.')
    args, unk = parser.parse_known_args()
    if args.plot_style:
        plt.style.use(args.plot_style)

    df = pd.read_csv(args.file)

    epsilon, risk = zip(*[(float(k.split('-')[1]), np.array(df[k])) for k in df.keys() if 'risk-' in k])
    proportion = np.array(df['proportion'])
    l2_parameter_norm = np.array(df['l2_param_norm'])
    lq_parameter_norm = np.array(df['lq_param_norm'])
    seed = np.array(df['seed'])
    snr = np.array(df['snr'])[0]  # assuming all snr are the same
    ord = np.array(df['ord'])[0]  # assuming all ord are the same
    n_train = np.array(df['n_train'])[0]  # assuming a fixed n_train
    noise_std = np.array(df['noise_std'])[0]  # assuming all noise_std are the same
    features_kind = np.array(df['features_kind'])[0]  # assuming all features_kind are the same
    # assuming all off_diag are the same
    off_diag = np.array(df['off_diag'])[0] if features_kind == 'equicorrelated' else None
    proportions_for_bounds = np.logspace(np.log10(min(proportion)), np.log10(max(proportion)), args.num_points)

    # Plot risk
    fig, ax = plt.subplots()
    markers = ['*', 'o', 's', '<', '>', 'h']
    i = 0
    for r, e in zip(risk, epsilon):
        # Plot empirical value
        l, = ax.plot(proportion, risk[i], markers[i], ms=4, label='$\\delta={}$'.format(e))
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Plot upper bound
        lb, ub = adversarial_bounds(proportions_for_bounds, snr, noise_std, e, ord,
                                    n_train * proportions_for_bounds, features_kind, off_diag)
        if e == 0:
            ax.plot(proportions_for_bounds, ub, '-', color=l.get_color(), lw=2)
        else:
            ax.fill_between(proportions_for_bounds, lb, ub, color=l.get_color(), alpha=0.2)
            ax.plot(proportions_for_bounds, ub, '-', color=l.get_color(), lw=1)
            ax.plot(proportions_for_bounds, lb, '-', color=l.get_color(), lw=1)

        # Labels
        ax.set_xlabel('$\\gamma$')
        ax.set_ylabel('Risk')

        # Plot vertical line at the interpolation threshold
        ax.axvline(1, ls='--')

        # Increment
        i += 1

    all_risk = np.stack(risk)
    y_min = 0.5 * np.min(all_risk) if args.y_min is None else args.y_min
    y_max = 2 * np.max(all_risk) if args.y_max is None else args.y_max
    ax.set_ylim((y_min, y_max))
    plt.legend()

    if args.save:
        plt.savefig(args.save + '.png')
    else:
        plt.show()

    # Get asymptotics for the parameter norm
    b = assymptotic_l2_norm_squared(proportions_for_bounds, snr, features_kind, off_diag, noise_std)
    # Plot l2 parameter norm
    fig, ax = plt.subplots()
    l, = ax.plot(proportion, l2_parameter_norm, '*', ms=4, label='$$l_2~~{\\rm norm}$$')
    ax.plot(proportions_for_bounds, np.sqrt(b), '-', color=l.get_color(), lw=2)
    # Plot lp parameter norm when available
    if ord != 2:
        l, = ax.plot(proportion, lq_parameter_norm, 'o', ms=4, label='$$l_q~~{\\rm norm}$$')
        lb, ub = assymptotic_lp_norm_squared(proportions_for_bounds, snr, features_kind, off_diag, ord,
                                             n_train * proportions_for_bounds, noise_std)
        ax.fill_between(proportions_for_bounds, np.sqrt(lb), np.sqrt(ub), color=l.get_color(), alpha=0.2)
        ax.plot(proportions_for_bounds, np.sqrt(ub), '-', color=l.get_color(), lw=1)
        ax.plot(proportions_for_bounds, np.sqrt(lb), '-', color=l.get_color(), lw=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$\\gamma$')
    ax.set_ylabel('Parameter Norm')
    plt.legend()
    if args.save:
        plt.savefig(args.save+'-norm.png')
    else:
        plt.show()





