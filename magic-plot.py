import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def get_median_and_quantiles(x, y):
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
    return new_x, m, lerr, uerr


def plot_error_bar(new_x, m, lerr, uerr, ax, lbl, logy=False):
    if logy:
        l, = ax.plot(new_x, np.log10(m))
        ax.fill_between(new_x, np.log10(m - lerr), np.log10(m + uerr), color=l.get_color(), alpha=0.4, label=lbl)
    else:
        l, = ax.plot(new_x, m)
        ax.fill_between(new_x, m - lerr, m + uerr, color=l.get_color(), alpha=0.4, label=lbl)


if __name__ == '__main__':
    import pandas as pd
    import os
    df = pd.read_csv('out/results/magic_results.csv')
    plt.style.use(['./plot_style_files/mystyle.mplsty',
                   './plot_style_files/mycolors.mplsty',
                   './plot_style_files/mylegend2.mplsty',
                   ])
    save='out/figures'


    def show(name):
        if save:
            plt.savefig(os.path.join(save, name) + '.pdf')
        else:
            plt.show()

    methods_pretty_names = {'ridge': 'ridge',
                            'l2advtrain': 'adv. train $\ell_2$',
                            'lasso': 'lasso',
                            'linfadvtrain': 'adv. train $\ell_\infty$'}

    df_filtered = df[df['n_features'] == 16000]
    methods = ['ridge', 'l2advtrain','lasso', 'linfadvtrain']
    tps = ['mse_test', 'mse_train']
    log_x = {t: True for t in ['mse_test', 'mse_train']}
    log_y = {'mse_test': False, 'mse_train': True}
    offsets = {'ridge': 1, 'lasso': 1e-6, 'l2advtrain': 1e-5, 'linfadvtrain': 1e-6}

    # Plot test error
    plt.style.use(['./plot_style_files/stacked.mplsty'])
    fig, ax = plt.subplots()
    for m in methods:
        df_aux = df_filtered[df_filtered['method'] == m]
        x_axis = np.array(df_aux['alpha'])
        y_axis = np.array(df_aux['mse_test'])
        new_x, med, lerr, uerr = get_median_and_quantiles(x_axis, y_axis)
        plot_error_bar(1/new_x * offsets[m] , med, lerr, uerr, ax, methods_pretty_names[m])
        ax.set_xscale('log')
        ax.set_ylabel('MSE')
    plt.subplots_adjust(left=0.14)
    show('magic_test_vs_regul')

    # Plot train error
    plt.style.use(['./plot_style_files/stacked_bottom2.mplsty'])
    fig, ax = plt.subplots()
    for m in methods:
        df_aux = df_filtered[df_filtered['method'] == m]
        x_axis = np.array(df_aux['alpha'])
        y_axis = np.array(df_aux['mse_train'])
        new_x, med, lerr, uerr = get_median_and_quantiles(x_axis, y_axis)
        plot_error_bar(1/new_x * offsets[m], med, lerr, uerr, ax, methods_pretty_names[m])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('MSE')
        ax.set_xlabel('$$\delta$$')
    plt.subplots_adjust(left=0.14, bottom=0.4)
    plt.legend(bbox_to_anchor=(0.43, -0.75), loc='lower center', ncol=4)
    show('magic_train_vs_regul')

    # Filter by alpha
    all_df = []
    for m in methods:
        for n_features in np.unique(df['n_features']):
            df_filtered = df[df['n_features'] == n_features]
            df_aux = df_filtered[df_filtered['method'] == m]
            x_axis = np.array(df_aux['alpha'])
            y_axis = np.array(df_aux['mse_test'])
            new_x, med, lerr, uerr = get_median_and_quantiles(x_axis, y_axis)
            best_alpha = new_x[np.argmin(med)]
            all_df.append(df_aux[df_aux['alpha'] == best_alpha])
    df_concat = pd.concat(all_df)

    # Plot barplot
    plt.style.use(['./plot_style_files/one_half.mplsty'])
    fig, ax = plt.subplots()
    sns.boxplot(x="method", hue="n_features", y="mse_test",
               data=df_concat, ax=ax, palette="Set3")
    ax.set_ylim((0, 1.5))
    ax.set_ylabel('MSE')
    ax.set_xlabel('')
    ax.set_xticklabels(methods_pretty_names.values())
    plt.subplots_adjust(right=0.8)
    plt.legend(bbox_to_anchor =(1.15, 0.1), loc='lower center')
    show('magic_test_vs_size')