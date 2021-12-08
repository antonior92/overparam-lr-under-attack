# Import packages.
import cvxpy as cp
import numpy as np
import numpy.linalg as linalg
from interp_under_attack.adversarial_attack import compute_q


def adversarial_training(X, y, ord, eps, niter=10, verbose=True):
    """Compute parameter for linear model trained adversarially with unitary p-norm.

    :param X:
        A numpy array of shape = (n_points, input_dim) containing the inputs
    :param y:
        A numpy array of shape = (n_points,) containing true outcomes
    :param ord:
        The p-norm the adversarial attack is bounded. `ord` gives which p-norm is used
        ord = 2 is the euclidean norm. `ord` can a float value greater then or equal to 1 or np.inf,
        (for the infinity norm).
    :param eps:
        The magnitude of the attack during the trainign
    :return:
        An array containing `delta_x` of shape = (n_points, n_parameters)
        which should perturbate the input. The p-norm of each row is equal to 1.
        In order to obtain the adversarial attack bounded by `e` just multiply it
        `delta_x`.
    """
    m, n = X.shape

    def compute_cost(param_norm, error_l1, error_l2):
        return eps ** 2 * param_norm ** 2 + 2 / m * eps * error_l1 * param_norm + 1 / m * error_l2 ** 2

    q = compute_q(ord)
    param0 = 1 / np.sqrt(n) * np.random.randn(n)
    for i in range(niter):  # solve iterative procedure
        param = cp.Variable(n)
        error = X @ param - y
        error0 = X @ param0 - y
        if verbose:
            print("Cost ={}, Risk = {}, Parameter norm = {}".format(
                compute_cost(linalg.norm(param0, ord=ord),
                             linalg.norm(error0, ord=1),
                             linalg.norm(error0, ord=2)),
                linalg.norm(error0, ord=2),
                linalg.norm(param0, ord=ord)
            )
            )

        cost1 = compute_cost(cp.pnorm(param, p=ord), linalg.norm(error0, ord=1), cp.pnorm(error, p=2))
        cost2 = compute_cost(linalg.norm(param0, ord=q), cp.pnorm(error, p=1), cp.pnorm(error, p=2))

        try:
            prob = cp.Problem(cp.Minimize(cost1 + cost2))
            prob.solve()
            param0 = param.value
        except cp.error.SolverError:
            break

    return param0



# Define and solve the CVXPY problem.
if __name__ == '__main__':
    # Generate data.
    m = 20
    n = 23
    np.random.seed(1)
    X = np.random.randn(m, n)
    y = np.random.randn(m)

    param = adversarial_training(X, y, 2, 0.1)


    print(np.linalg.norm(X @ param - y))