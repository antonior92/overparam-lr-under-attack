import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


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

    epsilon, risk = zip(*[(float(k.split('-')[1]), np.array(df[k])) for k in df.keys() if 'arisk-' in k])
    proportion = np.array(df['proportion'])
    l2_parameter_norm = np.array(df['l2_param_norm'])
    lq_parameter_norm = np.array(df['lq_param_norm'])
    seed = np.array(df['seed'])
    feature_scaling_growth = np.array(df['feature_scaling_growth'])[0]  # assuming all snr are the same
    feature_std = np.array(df['feature_std'])[0]  # assuming all are the same
    ord = np.array(df['ord'])[0]  # assuming all are the same
    n_train = np.array(df['n_train'])[0]  # assuming all are the same
    n_train = np.array(df['n_train'])[0]  # assuming all are the same

    # Plot arisk
    fig, ax = plt.subplots()
    markers = ['*', 'o', 's', '<', '>', 'h']
    i = 0
    for r, e in zip(risk, epsilon):
        # Plot empirical value
        l, = ax.plot(proportion, risk[i], markers[i], ms=4, label='$\\delta={}$'.format(e))
        ax.set_xscale('log')
        ax.set_yscale('log')

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
    #ax.set_ylim((y_min, 1000))
    plt.legend()

    if args.save:
        plt.savefig(args.save + '.png')
    else:
        plt.show()


