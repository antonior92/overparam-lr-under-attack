import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def asymptotic_risk(proportion, snr, noise_std):
    underparametrized = proportion / (1 - proportion)
    overparametrized = snr * (1 - 1 / proportion) + 1 / (proportion - 1)
    return noise_std ** 2 * ((proportion < 1) * underparametrized + (proportion > 1) * overparametrized)


def assymptotic_l2_norm(proportion, snr, noise_std=1.0):
    underparametrized = snr + proportion / (1 - proportion)
    overparametrized = snr * 1 / proportion + 1 / (proportion - 1)
    return noise_std ** 2 * ((proportion < 1) * underparametrized + (proportion > 1) * overparametrized)


def adversarial_upper_bound(proportion, snr, noise_std, eps):
    arisk = asymptotic_risk(proportion, snr, 1.0)
    anorm = assymptotic_l2_norm(proportion, snr, 1.0)
    return (np.sqrt(arisk) + eps * np.sqrt(anorm))**2 + noise_std ** 2


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
                        help='input csv.')
    args, unk = parser.parse_known_args()
    if args.plot_style:
        plt.style.use(args.plot_style)

    df = pd.read_csv(args.file)

    epsilon, risk = zip(*[(float(k.split('-')[1]), np.array(df[k])) for k in df.keys() if 'risk-' in k])
    proportion = np.array(df['proportion'])
    seed = np.array(df['seed'])
    snr = np.array(df['snr'])[0]  # assuming all snr are the same
    noise_std = np.array(df['noise_std'])[0] # assuming all noise_std are the same

    underp = np.logspace(np.log10(min(proportion)), -0.000000001, args.num_points // 2)
    overp = np.logspace(0.0000000001, np.log10(max(proportion)), args.num_points - args.num_points // 2)
    proportions_for_ub = np.concatenate((underp, overp))

    fig, ax = plt.subplots()
    markers = ['*', 'o', 's', '<', '>', 'h']
    i = 0
    for r, e in zip(risk, epsilon):

        # Plot empirical value
        l, = ax.plot(proportion, risk[i], markers[i], ms=4, label='$\\delta={}$'.format(e))
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Plot upper bound
        ub = adversarial_upper_bound(proportions_for_ub, snr, noise_std, e)
        ax.plot(proportions_for_ub, ub, '-', color=l.get_color(), lw=2)

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
        plt.savefig(args.save)
    else:
        plt.show()




