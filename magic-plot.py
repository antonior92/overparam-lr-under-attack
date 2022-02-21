import os
import numpy as np
import pandas as pd

ls = ['-', ':', '--', '-.']
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    folder = 'out/results/magic_2'

    # Load dataset
    dset = np.load(os.path.join(folder, 'dataset.npz'))
    X_train, X_test = dset['X_train'], dset['X_test']
    y_train, y_test = dset['y_train'], dset['y_test']

    # Load file
    df = pd.read_csv(os.path.join(folder, 'experiments.csv'))

    # filter method
    methods = np.unique(df['method'])
    for j, m in enumerate(methods):
        df_method = df[df['method'] == m]

        alphas = []
        mse_test_list = []
        mse_train_list = []
        for i, l in df_method.iterrows():
            fname = l['file']
            theta = np.load(os.path.join(folder, fname + '.npy'))
            y_pred = X_test @ theta
            mse = np.mean((y_test - y_pred) ** 2)
            mse_test_list.append(mse)

            y_pred_train = X_train @ theta
            mse_train = np.mean((y_train - y_pred_train) ** 2)
            mse_train_list.append(mse_train)

            alphas.append(l['alpha'])
        alphas = np.array(alphas)

        plt.semilogx(1 / alphas, mse_test_list, label=m, ls=ls[j])
    plt.legend()
    plt.show()