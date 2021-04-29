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
        equationf = vf * (xi * den + vs * den + (zeta ** 2) * vs) + phi1 * den  # Equation (15) from Mei and Montanari
        equations = vs * (xi * den + vf * den + (zeta ** 2) * vf) + phi2 * den  # Equation (15) from Mei and Montanari
        sol = sympy.solve_poly_system([equationf, equations], vf, vs)
        self.sol = sol
        self.equationf = equationf
        self.equations = equations
        self.phi1 = phi1
        self.phi2 = phi2
        self.vf = vf
        self.vs = vs
        self.xi = xi
        self.zeta = zeta

    def _get_unique_solution(self, phi1_v, phi2_v, zeta_v):
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
        for xi_v in tqdm.tqdm(np.logspace(-8, 14, 10)):
            if len(valid_sol) == 1:
                break
            for i in range(self.n_solutions):
                if i not in valid_sol:
                    break
                v1, v2 = self.evaluate_ith_solution(i, phi1_v, phi2_v, zeta_v, xi_v)
                v1_imag, v2_imag = sympy.im(v1), sympy.im(v2)
                v1_abs, v2_abs = abs(v1), abs(v2)
                # Check valid domain. If invalid domain remove solution
                if v1_imag < 0 or v2_imag < 0:
                    valid_sol.remove(i)
                else:
                    if v1_abs < phi1_v / xi_v and v2_abs < phi2_v / xi_v:
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

    def _check_solution(self, v1, v2, phi1_v, phi2_v, zeta_v, xi_v, tol=1e-10):
        """Check if solution when plugged back into the equations are close to zero."""
        v1, v2 = self.evaluate_ith_solution(i, phi1_v, phi2_v, zeta_v, xi_v)
        subs = {self.vf: v1, self.vs: v2, self.phi1: phi1_v, self.phi2: phi2_v, self.xi: complex(0, xi_v), self.zeta: zeta_v}
        ef = abs(self.equationf).evalf(subs=subs)
        es = abs(self.equations).evalf(subs=subs)
        print(ef)
        print(es)
        assert ef < tol
        assert es < tol

    @property
    def n_solutions(self):
        """Return the number of possible solutions to the equations"""
        return len(self.sol)

    def evaluate_ith_solution(self, i, phi1_v, phi2_v, zeta_v, xi_v):
        subs = {self.phi1: phi1_v, self.phi2: phi2_v, self.xi: complex(0, xi_v), self.zeta: zeta_v}
        v = self.sol[i]
        v1 = v[0].evalf(subs=subs)
        v2 = v[1].evalf(subs=subs)
        return v1, v2

    def __call__(self, phi1_v, phi2_v, zeta_v, xi_v):
        """Compute v1 and v2"""
        i = self._get_unique_solution(phi1_v, phi2_v, zeta_v)
        return self.evaluate_ith_solution(i, phi1_v, phi2_v, zeta_v, xi_v)


if __name__ == "__main__":
    compute_vs = AnaliticalVFunctions()
    # Sanity check: do the solutions when plugged back into the equations are close to zero
    phi1_v, phi2_v, zeta_v, xi_v = 0.1, 0.5, 1.6, 0.1
    for i in range(4):
        v1, v2 = compute_vs.evaluate_ith_solution(i, phi1_v, phi2_v, zeta_v, xi_v)
        compute_vs._check_solution(v1, v2, phi1_v, phi2_v, zeta_v, xi_v)

    # Get unique solution
    i = compute_vs._get_unique_solution(phi1_v, phi2_v, zeta_v)
