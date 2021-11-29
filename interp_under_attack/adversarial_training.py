# Import packages.
import cvxpy as cp
import numpy as np

def adversarial_training(X, y, ord, adv, e):
    """Compute parameter for linear model trained adversarially with unitary p-norm.

    :param X:
        A numpy array of shape = (n_points, input_dim) containing the inputs
    :param y:
        A numpy array of shape = (n_points,) containing true outcomes
    :param ord:
        The p-norm the adversarial attack is bounded. `ord` gives which p-norm is used
        ord = 2 is the euclidean norm. `ord` can a float value greater then or equal to 1 or np.inf,
        (for the infinity norm).
    :param e:
        The magnitude of the attack during the trainign
    :return:
        An array containing `delta_x` of shape = (n_points, n_parameters)
        which should perturbate the input. The p-norm of each row is equal to 1.
        In order to obtain the adversarial attack bounded by `e` just multiply it
        `delta_x`.
    """
    pass


# Generate data.
m = 20
n = 15
np.random.seed(1)
X = np.random.randn(m, n)
y = np.random.randn(m)

eps = 0.1
q = 2

# Define and solve the CVXPY problem.
param0 = 1 / np.sqrt(n) * np.ones(n)

for i in range(10):  # solve iterative procedure
    param = cp.Variable(n)
    error = X @ param - y
    error_l1 =  np.linalg.norm(X @ param0 - y, ord=1)
    parameter_qnorm = np.linalg.norm(param0, ord=q)
    cost = eps ** 2 * cp.norm(param, p=q) ** 2 + 1 / m * cp.norm(error, p=2) ** 2 +\
           1 / m * eps * cp.norm(error, p=1) * parameter_qnorm + \
           1 / m * eps * error_l1 * cp.norm(param, p=q)

    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()
    param0 = param.value
    print("The optimal value is", prob.value)

