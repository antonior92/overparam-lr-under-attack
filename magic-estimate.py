# You can download and extract the MAGIC dataset by
# > wget http://mtweb.cs.ucl.ac.uk/mus/www/MAGICdiverse/MAGIC_diverse_FILES/BASIC_GWAS.tar.gz
# > tar -xvf BASIC_GWAS.tar.gz
import os
import pandas as pd
import sklearn.model_selection
import numpy as np
import tqdm
from interp_under_attack.adversarial_training import adversarial_training, lasso_cvx, ridge

l2advtrain = lambda xx, yy, ee: adversarial_training(xx, yy, 2, ee)
linfadvtrain = lambda xx, yy, ee: adversarial_training(xx, yy, np.Inf, ee)

if __name__ == '__main__':
    out_p = 'HET_2'
    pp = './WEBSITE/DATA'
    out_folder = 'out/results/magic'
    subsampl = 20  #use only every n-th input
    test_size = 250
    random_state = 0

    founder_names = ["Banco", "Bersee", "Brigadier", "Copain", "Cordiale", "Flamingo",
                     "Gladiator", "Holdfast", "Kloka", "MarisFundin", "Robigus", "Slejpner",
                     "Soissons", "Spark", "Steadfast", "Stetson"]


    # Genotype
    genotype = pd.read_csv(os.path.join(pp, 'MAGIC_IMPUTED_PRUNED/MAGIC_imputed.pruned.traw'), sep='\t')
    genotype.set_index('SNP', inplace=True)
    genotype = genotype.iloc[:,5:]
    colnames = genotype.keys()
    new_colnames = [c.split('_')[0] for c in colnames]
    genotype.rename(columns={c: new_c for c, new_c in zip(colnames, new_colnames)}, inplace=True)
    genotype = genotype.transpose()

    # Phenotype
    phenotype = pd.read_csv(os.path.join(pp, 'PHENOTYPES/NDM_phenotypes.tsv'), sep='\t')
    phenotype.set_index('line_name', inplace=True)
    phenotype.drop(founder_names, inplace=True)
    del phenotype['line_code']

    # Make genotype have the same index as phenotype
    genotype = genotype.reindex(phenotype.index,)

    # Replace NAs
    genotype = genotype.fillna(genotype.mean(axis=0))
    phenotype = phenotype.fillna(phenotype.mean(axis=0))

    # Formulate problem
    X = genotype.values
    y = phenotype[out_p].values

    # Reduce size (just for testing quickly)
    X = X[:, ::subsampl]

    # Train-val split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Rescale
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std


    np.savez(os.path.join(out_folder, 'dataset'),
             X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test)

    def compute_all_values(f, name, df, min_scale=-3, max_scale=10, n_alphas=20):
        # Compute ridge paths
        alphas = np.logspace(min_scale, max_scale, n_alphas)
        for a in tqdm.tqdm(alphas):
            theta = f(X_train, y_train, a)
            fname = name+'_{:0.8}'.format(a)
            np.save(os.path.join(out_folder, fname), theta)
            df = df.append({'method': name, 'alpha': a, 'file': fname}, ignore_index=True)
            df.to_csv(os.path.join(out_folder, 'experiments.csv'), index=False)
        return df

    df = pd.DataFrame(columns=['method', 'alpha', 'file'])
    df = compute_all_values(ridge, 'ridge', df)
    df = compute_all_values(lasso_cvx, 'lasso', df)
    df = compute_all_values(linfadvtrain, 'linfadvtrain', df)
    df = compute_all_values(l2advtrain, 'l2advtrain', df)
