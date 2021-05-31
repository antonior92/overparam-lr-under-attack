import numpy as np
from random_features.activation_function_parameters import lipshitz


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

    # ---- Compute prediction arisk ---- #
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
    A = A1 / E0
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