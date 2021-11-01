import sympy
import numpy as np


class AnalyticFunctionC0(object):
    def __init__(self):
        gamma, psi, c0 \
            = sympy.symbols('gamma psi c0')

        # Define 2nd order eq (35) from Hastie
        den1 = gamma
        den2 = 1 + c0 * gamma
        den3 = 1 + c0 * (1 + 1 / psi) * gamma
        eq_2nd_order = den2 * den3 + (1 - psi) * den1 * den3 + psi * den1 * den2 - den1 * den2 * den3

        # Transform to polynomial coeficients
        coeffs = sympy.Poly(eq_2nd_order, c0).coeffs()
        self.compute_coeffs = sympy.lambdify((gamma, psi), coeffs, 'numpy')

    def __call__(self, gamma, psi):
        coeffs = self.compute_coeffs(gamma, psi)
        roots = np.roots(coeffs)
        print(roots)
        try:
            return roots[roots >= 0][0]
        except:
            return 0


compute_c0 = np.vectorize(AnalyticFunctionC0())


def compute_normalized_bias_and_variance(gamma, psi):
    c0 = compute_c0(gamma, psi)
    print('c0=', c0, 'gamma= ', gamma)
    # Implementing equations (33) and (34) from Hastie
    term1 = (1-psi) / (1 + c0 * gamma) ** 2
    aux = (1 + 1 / psi)
    den2 = (1 + c0 * aux * gamma) ** 2
    E1 = term1 + psi * aux**2 / den2
    E2 = term1 + (1 + psi) / den2

    # Implementing equations (31) and (32)
    v = c0 * gamma * E1/E2
    b = (1 + v) / ((1 + psi) * den2)
    return b, v


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Check whether the property of non-negativeness actually holds
    proportions = np.logspace(-2, 2, 50)

    x = compute_c0(proportions, 0.1/proportions)
    plt.semilogx(proportions, x)
    plt.show()

    # TODO fix c0 computation
