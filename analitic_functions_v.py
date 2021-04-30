import sympy
import itertools
import numpy as np
import tqdm


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
        # invalid and gives a zero denominator
        vs_new = sympy.solve((phi1 + vf*xi) - (vs*xi + phi2), vs)[0]
        # Solve the equation obtained plugging it back in one of the original equation (it does not make any difference which)
        poly_eq= sympy.expand(equationf.subs({vs: vs_new})
        # hence it comes down to solve the polynomial
        sol_vf = sympy.solve(poly_eq, vf)
        self.n_solutions = len(sol)
        self._solutions = sympy.lambdify((phi1, phi2, zeta, xi), sympy.Matrix(sol), 'numpy')
        self._equation_lhs = sympy.lambdify((phi1, phi2, zeta, xi, vf, vs),
                                             sympy.Matrix([equationf, equations]), 'numpy')
        self.equation = sympy.lambdify((phi1, phi2, zeta, xi, vf, vs), equationf, 'numpy')
        self.equations = sympy.lambdify((phi1, phi2, zeta, xi, vf, vs), equations, 'numpy')

    def get_unique_solution(self, phi1_v, phi2_v, zeta_v):
        """Among all possible solutions get the only one that satisfy conditions

        It check conditions (i) from Mei and Montanari Def. 1 to guarantee the
        function has the right domain (Im(v1)> 0 and Im(v2)>0).

        It uses the property that for sufficiently large Xi it is the only
        solution that satisfy the conditions `v1_abs < phi1_v / xi_v`
        and `v2_abs < phi2_v / xi_v`. To try to deal with the case multiple
        solutions have the right domain.
        """
        valid_sol = list(range(self.n_solutions))
        condition_achieved = [False] * self.n_solutions
        for xi_imag in np.logspace(-8, 14, 10):
            if len(valid_sol) == 1:
                break
            # compute solutions
            s = self.solutions(phi1_v, phi2_v, zeta_v,  1j * xi_imag)
            for i in range(self.n_solutions):
                if i not in valid_sol:
                    break
                v1, v2 = s[i, :]
                # Check valid domain. If invalid domain remove solution
                if np.imag(v1) < 0 or np.imag(v2) < 0:
                    valid_sol.remove(i)
                else:
                    if abs(v1) < phi1_v / xi_imag and abs(v2) < phi2_v / xi_imag:
                        condition_achieved[i] = True
                    else:
                        condition_achieved[i] = False

        # If only one solution has the right domain (C_+) we use it
        if len(valid_sol) == 1:
            return valid_sol[0]
        # If multiple solution have the right domain we use the condition
        # on the magnitude of v1 and v2 as a tie breaker. If multiple
        # solution satisfy all the conditions we just return the first one.
        return [i for i in valid_sol if condition_achieved[i]][0]

    def equation_lhs(self, phi1_v, phi2_v, zeta_v, xi_v, v1, v2):
        """Check if solution when plugged back into the equations are close to zero."""
        return self._equation_lhs(phi1_v + 0j, phi2_v + 0j, zeta_v + 0j, xi_v + 0j, v1 + 0j, v2 + 0j)

    def solutions(self, phi1_v, phi2_v, zeta_v, xi_v):
        # Make sure the numbers feed to the function arre complex! Otherwise numpy just outputs NaNs...
        return self._solutions(phi1_v + 0j, phi2_v + 0j, zeta_v + 0j, xi_v + 0j)

    def __call__(self, phi1_v, phi2_v, zeta_v, xi_v):
        """Compute v1 and v2"""
        i = self.get_unique_solution(phi1_v, phi2_v, zeta_v)
        return self.solutions(phi1_v, phi2_v, zeta_v, xi_v)[i]


if __name__ == "__main__":
    compute_vs = AnaliticalVFunctions()
    # Sanity check: do the solutions when plugged back into the equations are close to zero
    phi1_v, phi2_v, zeta_v, xi_v = 0.1, 0.5, 1.6, 0.0000001 * 1j
    s = compute_vs.solutions(phi1_v, phi2_v, zeta_v, xi_v)
    for i in range(4):
        v1, v2 = s2[i, :]
        e = compute_vs.equation_lhs(phi1_v, phi2_v, zeta_v, xi_v, v1, v2)
        print(e)

    s2 = np.array([[complex(ss.evalf(subs={phi1:phi1_v, phi2:phi2_v, zeta:zeta_v, xi:xi_v})) for ss in s] for s in sol])
    # Get unique solution
    #i = compute_vs.get_unique_solution(phi1_v, phi2_v, zeta_v)
