
import sympy

if __name__ == "__main__":

    phi1, phi2, vf, vs, xi, zeta \
        = sympy.symbols('phi1 phi2 vf vs xi zeta')

    den = 1 - (zeta ** 2) * vs * vf
    equationf = sympy.Poly(vf * (xi * den + vs * den + (zeta ** 2) * vs) + phi1 * den, domain='CC')
    equations = sympy.Poly(vs * (xi * den + vf * den + (zeta ** 2) * vf) + phi2 * den, domain='CC')

    sol = sympy.solve_poly_system([equationf,  equations], vf, vs)

    sol[3][1].evalf(subs={phi1: 0.1, phi2: 0.1, xi:1000, zeta: 10})

    # TODO: check solutions and find the right one!