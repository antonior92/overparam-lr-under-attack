from activation_function_parameters import *
from analitic_functions_v import AnaliticalVFunctions
from uniform_distribution_over_the_sphere import rand_matrix_asymptotic_l2_norm
from random_feature_regression import frac2int, frac2int_vec

def compute_asymptotics(features_over_input_dim, samples_over_input_dim, activation_params,
                        regularization, signal_amplitude, noise_std, compute_vs):
    # As defined in Eq (8) of Mei and Montanari
    mu_star = np.sqrt(activation_params['E{fn(G)**2}'] -
                      activation_params['E{fn(G)}']**2 - activation_params['E{G*fn(G)}']**2)

    zeta = activation_params['E{G*fn(G)}'] / mu_star
    corrected_regularizaton = regularization / mu_star ** 2
    xi_imag = np.sqrt(features_over_input_dim * samples_over_input_dim * corrected_regularizaton)
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
    # Implements Eq (20) and (5) of Mei and Montanari
    predicted_risk = signal_amplitude ** 2 * B + noise_std ** 2 * V + noise_std ** 2

    # ---- Compute parameter norm ---- #
    # Implements Eq (48) of Mei and Montanari
    A1 = - signal_amplitude**2 * m(2, 0) * ((1 - psi2) * m(1, 4) - m(1, 2) + (1 + psi2) * m(0, 2) + 1) + \
         +  noise_std**2 * m(2, 0) * (m(1, 2) - 1) * (m(2, 4) - 2 * m(1, 2) + m(0, 2) + 1)
    A0 = E0
    A = A1 / A0
    parameter_norm = np.sqrt(A) / mu_star

    return predicted_risk, parameter_norm


def adversarial_bounds(eps, ord, predicted_risk, parameter_norm, mnorm, bottleneck, activation):
    l = lipshitz(activation)
    if ord == np.inf:
        factor = bottleneck ** 1/2
    else:
        factor = bottleneck ** (1/2-1/ord)

    upper_eps = eps if ord <= 2 else eps * factor

    return predicted_risk, (np.sqrt(predicted_risk) + upper_eps * l**2 * mnorm * parameter_norm) ** 2


def log_interp(x, num_points=1000):
    x_min = min(x)
    x_max = max(x)
    return np.logspace(np.log10(x_min), np.log10(x_max), num_points)


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

    inputdim_over_datasize = np.array(df['inputdim_over_datasize'])
    nfeatures_over_datasize = np.array(df['nfeatures_over_datasize'])
    l2_parameter_norm = np.array(df['l2_param_norm'])
    lq_parameter_norm = np.array(df['lq_param_norm'])
    seed = np.array(df['seed'])
    signal_amplitude = np.array(df['signal_amplitude'])[0]  # assuming all signal_amplitude are the same
    ord = np.array(df['ord'])[0]  # assuming all ord are the same
    n_samples = np.array(df['num_train_samples'])[0]  # assuming a fixed n_train
    noise_std = np.array(df['noise_std'])[0]  # assuming all noise_std are the same
    activation = np.array(df['activation'])[0]  # assuming all features_kind are the same
    regularization = np.array(df['regularization'])[0]  # assuming all features_kind are the same
    fixed = np.array(df['fixed'])[0]  # assuming all features_kind are the same
    epsilon, adversarial_risk = zip(*[(float(k.split('-')[1]), np.array(df[k])) for k in df.keys() if 'risk-' in k])


    # Compute bounds
    activation_params = activation_function_parameters(activation)
    compute_vs = AnaliticalVFunctions()

    # compute assymptotics
    inputdim_over_datasize_for_bounds = log_interp(df['inputdim_over_datasize'], args.num_points)
    nfeatures_over_datasize_for_bounds = log_interp(df['nfeatures_over_datasize'], args.num_points)
    risk = np.zeros(args.num_points)
    mnorm = np.zeros(args.num_points)
    parameter_norm = np.zeros(args.num_points)
    for i in tqdm(range(args.num_points)):
        risk[i], parameter_norm[i] = compute_asymptotics(nfeatures_over_datasize_for_bounds[i] / inputdim_over_datasize_for_bounds[i],
                                                         1 / inputdim_over_datasize_for_bounds[i],
                                                         activation_params, regularization, signal_amplitude, noise_std, compute_vs)
        # we divide by np.sqrt(input_dim) because this factor appears in Mei and Montanri Eq. (1)
        mnorm[i] = rand_matrix_asymptotic_l2_norm(nfeatures_over_datasize_for_bounds[i] / inputdim_over_datasize_for_bounds[i])
    # compute bottleneck
    number_of_features = frac2int_vec(nfeatures_over_datasize_for_bounds, args.num_points)
    input_dimension = frac2int_vec(inputdim_over_datasize_for_bounds, args.num_points)
    bottleneck = np.minimum(number_of_features, input_dimension)

    # Plot risk
    if fixed == 'inputdim_over_datasize':
        proportion = inputdim_over_datasize
        proportions_for_bounds = inputdim_over_datasize_for_bounds
    elif fixed == 'nfeatures_over_datasize':
        proportion = nfeatures_over_datasize
        proportions_for_bounds = nfeatures_over_datasize_for_bounds
    elif fixed == 'nfeatures_over_inputdim':
        proportion = nfeatures_over_datasize / inputdim_over_datasize
        proportions_for_bounds = nfeatures_over_datasize_for_bounds / inputdim_over_datasize_for_bounds
    else:
        raise ValueError('Invalid argument --fixed = {}.'.format(args.fixed))
    fig, ax = plt.subplots()
    markers = ['*', 'o', 's', '<', '>', 'h']
    i = 0
    for r, e in zip(adversarial_risk, epsilon):
        # Plot empirical value
        l, = ax.plot(proportion, r, markers[i], ms=4, label='$\\delta={}$'.format(e))
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Plot upper bound
        lb, ub = adversarial_bounds(e, ord, risk, parameter_norm, mnorm, bottleneck, activation)
        if e == 0:
            ax.plot(proportions_for_bounds, ub, '-', color=l.get_color(), lw=2)
        else:
            ax.fill_between(proportions_for_bounds, lb, ub, color=l.get_color(), alpha=0.2)
            ax.plot(proportions_for_bounds, ub, '-', color=l.get_color(), lw=1)

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
    plt.legend()
    ax.set_title('risk')
    if args.save:
        plt.savefig(args.save+'.png')
    else:
        plt.show()

    fig, ax = plt.subplots()
    ax.plot(proportion, l2_parameter_norm, '*')
    ax.plot(proportions_for_bounds, np.array(parameter_norm))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('l2 parameter norm')
    if args.save:
        plt.savefig(args.save+'-norm.png')
    else:
        plt.show()

    fig, ax = plt.subplots()
    ax.plot(proportions_for_bounds, np.array(mnorm))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('l2 parameter norm')
    if args.save:
        plt.savefig(args.save + '-matrix-norm.png')
    else:
        plt.show()
