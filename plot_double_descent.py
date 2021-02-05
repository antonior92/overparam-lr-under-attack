import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def asymptotic_risk(proportion, snr, noise_std):
    underparametrized = proportion / (1 - proportion)
    overparametrized = snr**2 * (1 - 1 / proportion) + 1 / (proportion - 1)
    return noise_std ** 2 * ((proportion < 1) * underparametrized + (proportion > 1) * overparametrized)


def assymptotic_l2_norm(proportion, snr, noise_std=1.0):
    underparametrized = snr**2 + proportion / (1 - proportion)
    overparametrized = snr**2 * 1 / proportion + 1 / (proportion - 1)
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
    args, unk = parser.parse_known_args()
    if args.plot_style:
        plt.style.use(args.plot_style)

    df = pd.read_csv(args.file)

    epsilon, risk = zip(*[(float(k.split('-')[1]), np.array(df[k])) for k in df.keys() if 'risk-' in k])
    proportion = np.array(df['proportion'])
    seed = np.array(df['seed'])
    snr = np.array(df['snr'])
    noise_std = np.array(df['noise_std'])

    i = 0
    fig, ax = plt.subplots()
    markers = ['*', 'o', 's', '<', '>', 'h']
    colors = ['r', 'b', 'g']
    for r, e in zip(risk, epsilon):

        # Plot empirical value
        l, = ax.plot(proportion, risk[i], markers[i], ms=4, label=e)
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Plot upper bound
        ub = adversarial_upper_bound(proportion, snr, noise_std, e)
        p_index = np.argsort(proportion)
        p = (proportion[p_index])[seed[p_index] == 0]
        ub = (ub[p_index])[seed[p_index] == 0]
        ax.plot(p, ub, '-', color=l.get_color())

        # Increment
        i += 1
    plt.legend()
    plt.show()



