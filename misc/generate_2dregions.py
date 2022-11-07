import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', default='2',
                        help='Plot region for the given norm')
    parser.add_argument('--save', default='',
                        help='save plot in the given file (do not write extension). By default just show it.')
    parser.add_argument('--plot_style', nargs='*', default=[],
                        help='plot styles to be used')
    args = parser.parse_args()

    if args.plot_style:
        plt.style.use(args.plot_style)
    n = 400
    m = 400
    norm_type = np.Inf if args.norm == 'inf' else float(args.norm)

    def f(x, y):
      return np.linalg.norm([x-2, y-2], norm_type)

    x = np.linspace(0, 4, m)
    y = np.linspace(0, 4, n)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(n):
      for j in range(m):
        Z[i, j] = f(X[i, j], Y[i, j])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, levels=[1], colors='black')
    ax.set_xlim([0, 4])
    ax.set_ylim([0, 4])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    if args.save:
        plt.savefig(args.save, transparent=True)
    else:
        plt.show()