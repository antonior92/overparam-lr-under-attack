import numpy as np
import scipy.linalg as linalg
import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import os
    plt.style.use(['./plot_style_files/mystyle.mplsty',
                   './plot_style_files/mycolors.mplsty',
                   './plot_style_files/mylegend2.mplsty',
                   ])
    save='out/figures'
    n_values = [800, 400, 200, 100]  # Use decreasing order so the figure looks nice
    n_test_vectors = 100
    n_features = np.array(np.ceil(np.logspace(1, 4)), dtype=int)

    def show(name):
        if save:
            plt.savefig(os.path.join(save, name) + '.pdf')
        else:
            plt.show()

    norms = {}
    for n in n_values:
        norms[n] = {'l2_m': [], 'l2_q1': [],'l2_q3': [],
                    'l1_m': [], 'l1_q1': [],'l1_q3': []}
        for m in tqdm.tqdm(n_features):
            X = np.random.randn(n, m)

            u, s, vh = linalg.svd(X, full_matrices=False, compute_uv=True)

            orth_proj = vh.T @ vh

            u = np.random.randn(m, n_test_vectors)
            proj_u = orth_proj @ u
            values = np.linalg.norm(proj_u, ord=2, axis=0) / np.linalg.norm(u, ord=2, axis=0)
            norms[n]['l2_m'].append(np.median(values))
            norms[n]['l2_q1'].append(np.quantile(values, q=0.25))
            norms[n]['l2_q3'].append(np.quantile(values, q=0.75))
            values = np.linalg.norm(proj_u, ord=1, axis=0) / np.linalg.norm(u, ord=2, axis=0)
            norms[n]['l1_m'].append(np.median(values))
            norms[n]['l1_q1'].append(np.quantile(values, q=0.25))
            norms[n]['l1_q3'].append(np.quantile(values, q=0.75))

    plt.style.use(['./plot_style_files/stacked.mplsty'])
    plt.figure()
    for n in n_values:
        m = np.array(norms[n]['l2_m'])
        q1, q3 = np.array(norms[n]['l2_q1']), np.array(norms[n]['l2_q3'])
        ec = plt.errorbar(x=n_features, y=m, yerr=[m-q1, q3-m], capsize=3, alpha=0.8,
                          marker='o', markersize=3,  ls='', label=r'n={}'.format(n))
        l = ec.lines[0]
        nf_positive = n_features[n_features >= n]
        plt.plot(nf_positive, 1/np.sqrt(nf_positive/n), color=l.get_color())
    plt.xscale('log')
    plt.xlabel('m')
    plt.ylabel(r'$$\|\Phi \beta\|_2$$')
    plt.legend(loc='lower left')
    show('projl2')

    plt.style.use(['./plot_style_files/stacked_bottom.mplsty'])
    plt.figure()
    for n in n_values:
        m = np.array(norms[n]['l1_m'])
        q1, q3 = np.array(norms[n]['l1_q1']), np.array(norms[n]['l1_q3'])
        ec = plt.errorbar(x=n_features, y=m, yerr=[m-q1, q3-m], capsize=3, alpha=0.8,
                          marker='o', markersize=3,  ls='', label=r'n={}'.format(n))
        l = ec.lines[0]
        nf_positive = n_features[n_features >= n]
        plt.plot(nf_positive, 0.8 * np.sqrt(n) * np.ones(len(nf_positive)), color=l.get_color())
    plt.xscale('log')
    plt.xlabel(r'$$m$$')
    plt.ylabel(r'$$\|\Phi \beta\|_1$$')
    plt.legend(loc='upper left')
    show('projl1')