import numpy as np


def uniform_distribution_over_the_sphere(n_samples: int, dimension: int, rng):
    """Generate i.i.d. samples. Each uniformly distributed over the sphere."""
    sphere_radius = np.sqrt(dimension)
    before_normalization = rng.randn(n_samples, dimension)
    X = sphere_radius * before_normalization / np.linalg.norm(before_normalization, ord=2, axis=-1, keepdims=True)
    return X


def rand_matrix_asymptotic_l2_norm(n_rows, n_cols):
    proportion = n_rows / n_cols
    return np.sqrt(n_cols) * (1 + np.sqrt(proportion))


# The matrix of uniform distribution over the sphere behaves exactly the same
# as a random matrix with gaussian entries for large enough values.
# The close formula for the l2 norm arise naturally from Marchenco-Pastur law.
# The example bellow illustrate the idea. We show that in the example bellow.
if __name__ == '__main__':
    from scipy.linalg import eigh, svd
    import matplotlib.pyplot as plt
    import seaborn as sns
    EPS = 1e-8
    seed = 0
    d = 3000
    proportion = 0.1

    rng = np.random.RandomState()
    n = int(d * proportion)
    X = rng.randn(n, d)
    X2 = uniform_distribution_over_the_sphere(n, d, rng)

    # compute eigenvalues of both matrices values
    eigvals, eigvecs = eigh(1 / d * X.T @ X)
    n_zero_eigvals = sum(eigvals < EPS)
    nonzero_eigvals = eigvals[eigvals >= EPS]

    eigvals2, eigvecs2 = eigh(1 / d * X2.T @ X2)
    n_zero_eigvals2 = sum(eigvals2 < EPS)
    nonzero_eigvals2 = eigvals2[eigvals2 >= EPS]

    # Get maximum and minimum
    M = (1 + np.sqrt(proportion))**2
    m = (1 - np.sqrt(proportion))**2
    v = np.linspace(m, M, 100)


    def marchenco_pastur(x):
        x = np.array(x)
        mp = 1/(2 * np.pi) * np.sqrt((M-x) * (x-m)) / (proportion * x)
        return np.where((m <= x) & (x <= M), mp, 0)


    sns.displot(data={'normal': nonzero_eigvals, 'unif sphere': nonzero_eigvals2}, kind='hist', stat='density',
                common_norm=False)
    plt.plot(v, marchenco_pastur(v), color='black')
    plt.show()

    # it is easy to verify the relation with of the eigenvalues and sigular values
    singular_values = svd(X, compute_uv=False)
    singular_values2 = svd(X2, compute_uv=False)
    assert (singular_values[::-1] ** 2 / d - nonzero_eigvals < EPS).all()
    assert (singular_values2[::-1] ** 2 / d - nonzero_eigvals2 < EPS).all()

    # It follows the value for the norm above
    print('assymptotic matrix norm = {}'.format(rand_matrix_asymptotic_l2_norm(n, d)))
    print('matrix norm, normal entries = {}'.format(np.linalg.norm(X, ord=2)))
    print('matrix norm, uniform over the sphere entries = {}'.format(np.linalg.norm(X2, ord=2)))




