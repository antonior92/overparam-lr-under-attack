import sympy
import numpy as np


class AnalyticFunctionC0(object):
    def __init__(self):
        gamma, psi, c0 \
            = sympy.symbols('gamma psi c0')

        # Define 2nd order eq (35) from Hastie
        den1 = 1 + c0 * gamma
        den2 = 1 + c0 * (1 + (1 / psi)) * gamma
        eq_2nd_order = (1/gamma - 1) * den1 * den2 + (1 - psi) * den2 + psi * den1

        # Transform to polynomial coeficients
        coeffs = sympy.Poly(eq_2nd_order, c0).all_coeffs()
        print(coeffs)
        self.compute_coeffs = sympy.lambdify((gamma, psi), coeffs, 'numpy')

    def __call__(self, gamma, psi):
        coeffs = self.compute_coeffs(gamma, psi)
        a, b, c = coeffs
        print(coeffs)
        delta = b**2 - 4 * a * c
        print(delta)
        roots = np.roots(coeffs)
        print(gamma, psi, roots > 0)
        if (np.real(roots) > 0).any():
            return np.max(np.abs(roots))
        else:
            return np.NaN


compute_c0_scalar = AnalyticFunctionC0()


def compute_c0(proportions, proportion_latent):
    x = []
    for p in proportions.tolist():
        x += [compute_c0_scalar(p, proportion_latent/p)]
    return np.array(x)


def compute_normalized_bias_and_variance(proportions, proportion_latent):
    c0 = compute_c0(proportions, proportion_latent)
    psi = (proportion_latent / proportions)
    gamma = proportions
    # Implementing equations (33) and (34) from Hastie
    term1 = (1-psi) / (1 + c0 * gamma) ** 2
    aux = (1 + 1 / psi)
    den2 = (1 + c0 * aux * gamma) ** 2
    E1 = term1 + psi * aux**2 / den2
    E2 = term1 + (1 + psi) / den2

    # Implementing equations (31) and (32)
    v = c0 * gamma * E1/E2
    b = (1 + v) * aux / den2
    return b, v


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Check whether the property of non-negativeness actually holds
    proportions = np.logspace(-1, 2, 50)

    x = compute_c0(proportions, 0.0003)
    plt.plot(proportions, x)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    # TODO fix c0 computation
