import os
import numpy as np
import pandas as pd


def plot_error_bar(x, y, ax, lbl):
    new_x, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)

    y_values = np.zeros([len(new_x), max(counts)])
    secondindex = np.zeros(len(new_x), dtype=int)
    for n in range(len(x)):
        i = inverse[n]
        j = secondindex[i]
        y_values[i, j] = y[n]
        secondindex[i] += 1
    m = np.median(y_values, axis=1)
    lerr = m - np.quantile(y_values, 0.25, axis=1)
    uerr = np.quantile(y_values, 0.75, axis=1) - m
    l, = ax.plot(new_x, m)
    ax.fill_between(new_x, m - lerr, m + uerr, color=l.get_color(), alpha=0.4, label=lbl)


ls = ['-', ':', '--', '-.']
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import tqdm
    n_seeds = 5
    folders = ['out/results/magic_m8000_seed{}'.format(i) for i in range(n_seeds)]

    tps = ['mse_test', 'mse_train', 'sparsity']
    log_x = {t: True for t in ['mse_test', 'mse_train', 'sparsity']}
    log_y = {'mse_test': False, 'mse_train': True, 'sparsity': False}

    # Load dataset
    dsets = [np.load(os.path.join(f, 'dataset.npz')) for f in folders]

    # Load file
    list_df = []
    for f in folders:
        df_aux = pd.read_csv(os.path.join(f, 'experiments.csv'))
        df_aux['file'] = f + '/' + df_aux['file']
        list_df.append(df_aux)
    df = pd.concat(list_df, keys=range(n_seeds), names=['seed', 'n'])
    df.reset_index(0, inplace=True)
    df.reset_index(0, drop=True, inplace=True)

    # filter method
    methods = np.unique(df['method'])
    method_collected_data = {}
    for j, m in enumerate(methods):
        df_method = df[df['method'] == m]
        info = {'alpha': [],'mse_test': [], 'mse_train': [], 'sparsity': []}
        for i, l in tqdm.tqdm(df_method.iterrows()):
            dset = dsets[l.seed]
            X_train, X_test = dset['X_train'], dset['X_test']
            y_train, y_test = dset['y_train'], dset['y_test']
            fname = l['file']
            theta = np.load(fname + '.npy')
            y_pred = X_test @ theta
            mse = np.mean((y_test - y_pred) ** 2)
            info['mse_test'].append(mse)

            y_pred_train = X_train @ theta
            mse_train = np.mean((y_train - y_pred_train) ** 2)
            info['mse_train'].append(mse_train)

            sparsity = sum(np.abs(theta) > 1e-16) / len(theta)
            info['sparsity'].append(sparsity)
            info['alpha'].append(l['alpha'])
        method_collected_data[m] = info

    plt.style.use('./plot_style_files/mycolors.mplsty')
    offsets = {'ridge': 1, 'lasso': 1e-6, 'l2advtrain': 1e-5, 'linfadvtrain': 1e-6}
    fig, all_ax = plt.subplots(len(tps), sharex=True)
    for i, tp in enumerate(tps):
        ax = all_ax[i]
        for m, info in method_collected_data.items():
            x_axis = 1/np.array(info['alpha'])
            plot_error_bar(offsets[m]*(x_axis), np.array(info[tp]), ax, m)
        if log_x[tp]:
            ax.set_xscale('log')
        if log_y[tp]:
            ax.set_yscale('log')
        ax.grid()
    plt.legend()
    plt.show()