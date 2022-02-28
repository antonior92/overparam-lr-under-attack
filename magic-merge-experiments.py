import os
import numpy as np
import pandas as pd
import itertools

ls = ['-', ':', '--', '-.']
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import tqdm
    seeds = range(5)
    n_features = [1000, 2000, 4000, 8000, 16000]
    output_file = "out/results/magic_results.csv"
    folders = [(m, s, 'out/results/magic_m{}_seed{}'.format(m, s)) for s, m in itertools.product(seeds, n_features)]
    # Load file
    list_df = []
    for m, s, f in folders:
        df_aux = pd.read_csv(os.path.join(f, 'experiments.csv'))
        df_aux['file'] = f + '/' + df_aux['file']
        df_aux['dataset'] = os.path.join(f, 'dataset.npz')
        df_aux['seed'] = s
        df_aux['n_features'] = m
        if 'mse_test' not in df_aux.keys():
            df_aux['mse_test'] = -1  # Create columns that will be filled latter on
        if 'mse_train' not in df_aux.keys():
            df_aux['mse_train'] = -1
        list_df.append(df_aux)
    df = pd.concat(list_df, ignore_index=True)

    # filter method
    prev_dset_path = ''
    dset = 0
    for i, l in tqdm.tqdm(list(df.iterrows())):
        dset_path = l['dataset']
        if dset_path != prev_dset_path:
            dset = np.load(dset_path)
            prev_dset_path = dset_path
        X_train, X_test = dset['X_train'], dset['X_test']
        y_train, y_test = dset['y_train'], dset['y_test']
        fname = l['file']

        theta = np.load(fname + '.npy')
        y_pred = X_test @ theta
        mse = np.mean((y_test - y_pred) ** 2)
        df.loc[i, 'mse_test'] = mse

        y_pred_train = X_train @ theta
        mse_train = np.mean((y_train - y_pred_train) ** 2)
        df.loc[i, 'mse_train'] = mse_train
        df.to_csv(output_file, index=False)
