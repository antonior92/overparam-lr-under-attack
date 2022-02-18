import os
import numpy as np
import pandas as pd


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    folder = 'out/results/magic'
    method = 'lasso'

    # Load dataset
    dset = np.load(os.path.join(folder, 'dataset.npz'))
    X_train, X_test = dset['X_train'], dset['X_test']
    y_train, y_test = dset['y_train'], dset['y_test']

    # Load file
    df = pd.read_csv(os.path.join(folder, 'experiments.csv'))
    # filter method
    df = df[df['method'] == method]

    alphas = []
    mse_test_list = []
    mse_train_list = []
    for i, l in df.iterrows():
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

    plt.loglog(1 / alphas, mse_test_list)
    plt.show()