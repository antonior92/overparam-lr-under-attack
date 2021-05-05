import sympy
import numpy as np
from warnings import warn

class AnaliticalVFunctions(object):
    """Compute v1 and v2 defined as in equation (16)."""

    def __init__(self):
        # Compute solutions using sympy
        phi1, phi2, vf, vs, xi, zeta \
            = sympy.symbols('phi1 phi2 vf vs xi zeta')  # Here we denote v1 as vf and v2 as vs
        den = 1 - (zeta ** 2) * vs * vf
        equationf = sympy.expand(vf * (xi * den + vs * den + (zeta ** 2) * vs) + phi1 * den)  # Equation (15) from Mei and Montanari
        equations = sympy.expand(vs * (xi * den + vf * den + (zeta ** 2) * vf) + phi2 * den)  # Equation (15) from Mei and Montanari
        # equation_aux = equationf - equations
        # print sympy.factor(equation_aux) will show that (phi1 + vf*xi = vs*xi + phi2) or (vf*vs*zeta**2 - 1 = 0) which is
        # invalid and gives a zero denominator in Eq. (15)
        vs_new = sympy.expand(sympy.solve((phi1 + vf*xi) - (vs*xi + phi2), vs)[0])
        # Solve the equation obtained plugging it back in one of the original equation (it does not make any difference which)
        poly_eq = sympy.expand(equations.subs({vs: vs_new}))
        # hence it comes down to solve the polynomial
        coeffs = sympy.Poly(poly_eq, vf).coeffs()
        coeffs = [sympy.expand(c) for c in coeffs]
        self.compute_coeffs = sympy.lambdify((phi1, phi2, zeta, xi), sympy.Matrix(coeffs), 'numpy')


        #self.n_solutions = len(sol_vf)
        #self._vf_solutions = sympy.lambdify((phi1, phi2, zeta, xi), sympy.Matrix(sol_vf),  'numpy')
        self._vs_from_vf = sympy.lambdify((phi1, phi2, xi, vf), vs_new,  'numpy')
        self._equation_lhs = sympy.lambdify((phi1, phi2, zeta, xi, vf, vs),
                                             sympy.Matrix([equationf, equations]), 'numpy')
        self.poly_eq = sympy.lambdify((phi1, phi2, zeta, xi, vf), poly_eq,  'numpy')

    def equation_lhs(self, phi1_v, phi2_v, zeta_v, xi_v, v1, v2):
        """Check if solution when plugged back into the equations are close to zero."""
        return self._equation_lhs(phi1_v + 0j, phi2_v + 0j, zeta_v + 0j, xi_v + 0j, v1 + 0j, v2 + 0j),\
                self.poly_eq(phi1_v + 0j, phi2_v + 0j, zeta_v + 0j, xi_v + 0j, v1 + 0j), \
                self.subs_eq(phi1_v + 0j, phi2_v + 0j, zeta_v + 0j, xi_v + 0j)

    def solutions(self, phi1_v, phi2_v, zeta_v, xi_v):
        # Make sure the numbers feed to the function arre complex! Otherwise numpy just outputs NaNs...
        coeffs_symbolic = self.compute_coeffs(phi1_v + 0j, phi2_v + 0j, zeta_v + 0j, xi_v + 0j)
        vf = np.roots(coeffs_symbolic.flatten())
        vs = np.array([self._vs_from_vf(phi1_v + 0j, phi2_v + 0j, xi_v + 0j, vf_i) for vf_i in vf])
        return vf, vs

    def get_solution(self, vf, vs):
        valid_sol = (np.imag(vf) > 0) & (np.imag(vs) > 0)
        if sum(valid_sol) > 1:
            warn('multiple solutions are valid!')
        return vf[valid_sol][0], vs[valid_sol][0]

    def __call__(self, phi1_v, phi2_v, zeta_v, xi_v):
        """Compute v1 and v2"""
        vf, vs = self.solutions(phi1_v, phi2_v, zeta_v, xi_v)
        return self.get_solution(vf, vs)


if __name__ == "__main__":
    compute_vs = AnaliticalVFunctions()
    # Sanity check: do the solutions when plugged back into the equations are close to zero
    phi1_v, phi2_v, zeta_v = 0.1, 0.1, 0.1
    xi_v = np.sqrt(phi1_v * phi2_v * 1e-8) * 1j
    #v1, v2 = compute_vs.solutions(phi1_v, phi2_v, zeta_v, xi_v)

    p = compute_vs.compute_coeffs(phi1_v, phi2_v, zeta_v, xi_v).flatten()
    vf, vs = compute_vs.solutions(phi1_v, phi2_v, zeta_v, xi_v)
    print(np.polyval(p, np.roots(p)[0]))
    vf, vs = compute_vs(phi1_v, phi2_v, zeta_v, xi_v)
    print(vf, vs)
