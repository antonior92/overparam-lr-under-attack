import numpy as np

implemented_activations = ['relu', 'tanh']


def get_activation(name):
    if name == 'relu':
        return lambda x: np.maximum(0, x)
    elif name == 'tanh':
        return lambda x: np.tanh(x)
    else:
        raise ValueError("Activation function not available!")


def get_activation_grad(name):
    if name == 'relu':
        return lambda x: np.where(x > 0, 1, 0)
    elif name == 'tanh':
        return lambda x:  1 - np.tanh(x)**2
    else:
        raise ValueError("Activation function not available!")


def activation_function_parameters(name):
    """Compute the parameters caracterizing an activation function.

    Given the `name` of an activation function returns three expectations
    for G ~ N(0, 1) a standard normal variable: E{fn(G)}, E{G * fn(G)}
    and E{fn(G) ** 2}.
    """
    parameters = {}
    if name == 'relu':  # It is straight forward to compute this numbers for the relu function
        # To compute the expectation of fn(G), just divide by two the mean of a
        # half-normal with variance 1 (https://en.wikipedia.org/wiki/Half-normal_distribution)
        parameters["E{fn(G)}"] = 1 / np.sqrt(2 * np.pi)
        # Both of this values are half of the variance of a normal distribution.
        # They are the same due to the definition of relu and they are easily obtained analytically
        # by writting down the integration and using the integrand is an even function.
        parameters["E{G*fn(G)}"] = 1 / 2
        parameters["E{fn(G)**2}"] = 1 / 2
    elif name == 'tanh':
        parameters["E{fn(G)}"] = 0  # from the symmetry
        # The next two results are harder to come by analitically. I just wrote down
        # the results I obtained for a large experiment
        parameters["E{G*fn(G)}"] = 0.60583
        parameters["E{fn(G)**2}"] = 0.39436
    else:
        raise ValueError("Activation function not available!")
    return parameters


def lipshitz(name):
    if name == 'relu' or name == 'tanh':
        return 1
    else:
        raise ValueError("Activation function not available!")


def minimum_variation(name, input_max=1):
    if name == 'relu':
        return 1
    elif name == 'tanh':
        return 2 * np.tanh(input_max)
    else:
        raise ValueError("Activation function not available!")


def estimate_function_parameters(fn, experiment_size=1000000, seed=0):
    rng = np.random.RandomState(seed)
    G = rng.randn(experiment_size)
    f = fn(G)
    return {"E{fn(G)}": np.mean(f), "E{G*fn(G)}": np.mean(G*f), "E{fn(G)**2}":np.mean(f * f)}


if __name__ == "__main__":
    print('Testing activation funciton parameters')
    for name in implemented_activations:

        print('--- {} ---'.format(name))
        fn = get_activation(name)
        parameters_exact = activation_function_parameters(name)
        parameters_estimated = estimate_function_parameters(fn)

        print('       |   E{fn(G)}   |    E{G*fn(G)}   |    E{fn(G)**2}   |')
        print('exact  |   {:5.3f}      |      {:4.3f}      |        {:4.3f}     |'
              .format(parameters_exact['E{fn(G)}'],
                      parameters_exact['E{G*fn(G)}'],
                      parameters_exact['E{fn(G)**2}']))
        print('estim  |   {:5.3f}      |      {:4.3f}      |        {:4.3f}     |'
              .format(parameters_estimated['E{fn(G)}'],
                      parameters_estimated['E{G*fn(G)}'],
                      parameters_estimated['E{fn(G)**2}']))

        print('   Checking derivative:'.format(name))
        x = np.random.randn(100)
        dx = 1e-8
        y = fn(x)
        dy = fn(x+dx) - y
        finite_diferences_approx = (fn(x+dx) - fn(x)) / dx
        gn = get_activation_grad(name)
        analytical_value = gn(x)
        print('       The maximum difference between analytical and finite differences approx. is {:e}.'
              .format(np.max(finite_diferences_approx - analytical_value)))





