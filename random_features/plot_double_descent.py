from activation_function_parameters import *
from analitic_functions_v import AnaliticalVFunctions
from uniform_distribution_over_the_sphere import rand_matrix_asymptotic_l2_norm


def compute_asymptotics(features_over_input_dim, samples_over_input_dim, activation_params,
                        regularization, snr, noise_std, compute_vs):
    # As defined in Eq (8) of Mei and Montanari
    mu_star = np.sqrt(activation_params['E{fn(G)**2}'] -
                      activation_params['E{fn(G)}']**2 - activation_params['E{G*fn(G)}']**2)

    zeta = activation_params['E{G*fn(G)}'] / mu_star
    corrected_regularizaton = regularization / mu_star ** 2
    xi_imag = np.sqrt(features_over_input_dim * samples_over_input_dim * corrected_regularizaton)
    rho = snr**2
    psi1 = features_over_input_dim  # as used in Mei and Montanari - to make equations bellow easier to read!
    psi2 = samples_over_input_dim  # as used in Mei and Montanari - to make equations bellow easier to read!

    vf, vs = compute_vs(psi1, psi2, zeta, 1j*xi_imag)

    # Implements Eq (16) of Mei and Montanari
    # I am assuming here that chi is a real number. And that, except for numerical errors
    # imag(vs * vf) was supposed to be zero
    chi = -np.imag(vf) * np.imag(vs)

    def m(p, q):
        """implement chi zeta monomial in compact format."""
        return chi ** p * zeta ** q

    # ---- Compute prediction risk ---- #
    # Implements eq (17) of Mei and Montanari
    E0 = - m(5, 6) + 3 * m(4, 4) + (psi1*psi2 - psi1 - psi2 + 1) * m(3, 6) - 2 * m(3, 4) - 3 * m(3, 2) + \
        (psi1 + psi2 - 3 * psi1 * psi2 + 1) * m(2, 4) + 2 * m(2, 2) + m(2, 0) + \
        3 * psi1 * psi2 * m(1, 2) - psi1 * psi2
    E1 = psi2 * m(3, 4) - psi2 * m(2, 2) + psi1 * psi2 * m(1, 2) - psi1 * psi2
    E2 = m(5, 6) - 3 * m(4, 4) + (psi1 - 1) * m(3, 6) + 2 * m(3, 4) + 3 * m(3, 2) + \
         (-psi1 - 1) * m(2, 4) - 2 * m(2, 2) - m(2, 0)

    B = E1 / E0  # Implements Eq (18) of Mei and Montanari
    V = E2 / E0  # Implements Eq (19) of Mei and Montanari
    R = rho / (1 + rho) * B + 1 / (1 + rho) * V  # Implements Eq (20) of Mei and Montanari
    predicted_risk = noise_std ** 2 * ((snr**2 + 1) * R + 1)  # Implements LHS of Eq (5) of Mei and Montanari

    # ---- Compute parameter norm ---- #
    # Implements Eq (48) of Mei and Montanari
    A1 = - rho / (1 + rho) * m(2, 0) * ((1 - psi2) * m(1, 4) - m(1, 2) + (1 + psi2) * m(0, 2) + 1) + \
         + 1 / (1 + rho) * m(2, 0) * (m(1, 2) - 1) * (m(2, 4) - 2 * m(1, 2) + m(0, 2) + 1)
    A0 = E0
    A = A1 / A0
    parameter_norm = noise_std * np.sqrt((snr**2 + 1) * A) / (mu_star)

    return predicted_risk, parameter_norm


def adversarial_bounds(eps, ord, predicted_risk, parameter_norm, mnorm):
    return predicted_risk + (eps * mnorm * parameter_norm)**2, \
           (np.sqrt(predicted_risk) + eps * mnorm * parameter_norm) ** 2


if __name__ == "__main__":
    # TODO: double check!
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

    proportion = np.array(df['proportion'])
    l2_parameter_norm = np.array(df['l2_param_norm'])
    seed = np.array(df['seed'])
    snr = np.array(df['snr'])[0]  # assuming all snr are the same
    n_samples = np.array(df['num_train_samples'])[0]  # assuming a fixed n_train
    n_features = np.array(df['n_features'])[0]  # assuming a fixed n_train
    input_dim = np.array(df['input_dim'])[0]  # assuming a fixed n_train
    noise_std = np.array(df['noise_std'])[0]  # assuming all noise_std are the same
    activation = np.array(df['activation'])[0]  # assuming all features_kind are the same
    regularization = np.array(df['regularization'])[0]  # assuming all features_kind are the same
    lower_proportion = np.log10(min(proportion))
    upper_proportion = np.log10(max(proportion))
    epsilon, adversarial_risk = zip(*[(float(k.split('-')[1]), np.array(df[k])) for k in df.keys() if 'risk-' in k])

    # Compute bounds
    activation_params = activation_function_parameters(activation)
    features_over_input_dim = n_features / input_dim
    samples_over_input_dim = n_samples / input_dim
    compute_vs = AnaliticalVFunctions()

    # compute assymptotics
    proportions_for_bounds = np.logspace(lower_proportion, upper_proportion, args.num_points)
    parameter_norm = np.zeros(len(proportions_for_bounds))
    risk = np.zeros(len(proportions_for_bounds))
    mnorm = np.zeros(len(proportions_for_bounds))
    for i, p2b in enumerate(tqdm(proportions_for_bounds)):
        n_features_i = max(int(p2b * n_samples), 1)
        risk[i], parameter_norm[i] = compute_asymptotics(n_features_i / input_dim, n_samples / input_dim,
                                                         activation_params, regularization, snr, noise_std, compute_vs)
        norm[i] = rand_matrix_asymptotic_l2_norm(n_features_i, input_dim) / np.sqrt(n_features_i)  # why???

    # Plot risk
    fig, ax = plt.subplots()
    markers = ['*', 'o', 's', '<', '>', 'h']
    i = 0
    for r, e in zip(adversarial_risk, epsilon):
        # Plot empirical value
        l, = ax.plot(proportion, r, markers[i], ms=4, label='$\\delta={}$'.format(e))
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Plot upper bound
        lb, ub = adversarial_bounds(e, 2.0, risk, parameter_norm, mnorm)
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
    #ax.set_ylim((y_min, y_max))
    plt.legend()
    ax.set_title('risk')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(df['proportion'], df['l2_param_norm'], '*')
    ax.plot(proportions_for_bounds, np.array(parameter_norm))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('parameter norm')
    plt.show()
