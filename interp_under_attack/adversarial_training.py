# Import packages.
import cvxpy as cp
import numpy as np
import numpy.linalg as linalg
from interp_under_attack.adversarial_attack import compute_q


def adversarial_training(X, y, p, eps, niter=10, verbose=True):
    """Compute parameter for linear model trained adversarially with unitary p-norm.

    :param X:
        A numpy array of shape = (n_points, input_dim) containing the inputs
    :param y:
        A numpy array of shape = (n_points,) containing true outcomes
    :param p:
        The p-norm the adversarial attack is bounded. `p` gives which p-norm is used
        p = 2 is the euclidean norm. `p` can a float value greater then or equal to 1 or np.inf,
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

    q = compute_q(p)

    # Formulate problem
    param = cp.Variable(n)
    param_norm = cp.pnorm(param,  p=q)
    abs_error = cp.abs(X @ param - y)
    adv_loss = 1 / m * cp.sum((abs_error + eps * param_norm)**2)

    prob = cp.Problem(cp.Minimize(adv_loss))
    try:
        prob.solve()
        param0 = param.value
    except:
        param0 = np.zeros(n)

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
    print(np.linalg.norm(param))