import argparse
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np  # numpy > 1.10 so we can use np.linalg.norm(...,axis=axis, keepdims=keepdims)
import random
from interp_under_attack.linear_with_random_covariates.adversarial_risk import train_and_evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Double descent for l-2 adversarial attack')
    parser.add_argument('-o', '--output', default='./performance.csv',
                        help='output csv file.')
    parser.add_argument('--num_train_samples', type=int, default=100,
                       help='number of samples in the experiment')
    parser.add_argument('--num_test_samples', type=int, default=100,
                       help='number of samples in the experiment')
    parser.add_argument('-r', '--repetitions', type=int, default=4,
                        help='number of times each experiment is repeated')
    parser.add_argument('-p', '--ord',  default=[2.0], type=float, nargs='+',
                        help='ord is p norm of the adversarial attack.')
    parser.add_argument('-n', '--num_points', default=60, type=int,
                        help='number of points')
    parser.add_argument('-l', '--lower_proportion', default=-1, type=float,
                        help='the lowest value for the proportion (n features / n samples) is 10^l.')
    parser.add_argument('-u', '--upper_proportion', default=1, type=float,
                        help='the upper value for the proportion (n features / n samples) is 10^u.')
    parser.add_argument('-e', '--epsilon', default=[0, 0.1, 0.5, 1, 2], type=float, nargs='+',
                        help='the epsilon values used when computing the adversarial ttack')
    parser.add_argument('-s', '--noise_std', type=float, default=1.0,
                        help='standard deviation of the additive noise added.')
    parser.add_argument('-f', '--features_kind', choices=['isotropic', 'equicorrelated'], default='isotropic',
                        help='how the features are generated')
    parser.add_argument('--datagen_parameter', choices=['gaussian_prior', 'constant'], default='gaussian_prior',
                        help='how the features are generated')
    parser.add_argument('--off_diag', default=0.5, type=float,
                        help='value of diagonal values. Default is 0.5. Only take effect when '
                             'features_kind = equicorrelated.')
    parser.add_argument('--signal_amplitude', type=float, default=1.0,
                         help='signal amplitude. I.e. \|beta*\|_2')
    args, unk = parser.parse_known_args()
    print(args)

    tqdm.write("Estimating performance as a function of proportion...")
    list_dict = []
    proportions = np.logspace(args.lower_proportion, args.upper_proportion, args.num_points)
    run_instances = list(itertools.product(range(args.repetitions), proportions))

    # Some of the executions are computationally heavy and others are not. We shuffle the configurations
    # so the progress bar can give a more accurate notion of the time to completion
    random.shuffle(run_instances)
    prev_mdl = None  # used only if reuse_weights is True
    df = pd.DataFrame(columns=['proportion', 'seed'] + ['norm-{:.1f}'.format(p) for p in args.ord] +
                              ['advrisk-{:.1f}-{:.1f}'.format(p, e) for p, e in itertools.product(args.ord, args.epsilon)])
    for seed, proportion in tqdm(run_instances, smoothing=0.03):
        n_features = max(int(proportion * args.num_train_samples), 1)
        risk, pnorms, l2distance = train_and_evaluate(
            args.num_train_samples, n_features, args.noise_std, args.signal_amplitude,
            args.epsilon, args.ord, args.num_test_samples, args.features_kind,
            args.off_diag, args.datagen_parameter, seed)
        dict1 = {'proportion': proportion, 'n_features': n_features, 'n_train':args.num_train_samples,
                 'n_test': args.num_test_samples, 'ord': args.ord, 'features_kind': args.features_kind, 'seed': seed,
                 'signal_amplitude': args.signal_amplitude, 'noise_std': args.noise_std,
                 'datagen_parameter': args.datagen_parameter, 'l2distance': l2distance}
        if args.features_kind =='equicorrelated':
            dict1['off_diag'] = args.off_diag
        df = df.append({**risk, **pnorms, **dict1}, ignore_index=True)
        df.to_csv(args.output, index=False)
    tqdm.write("Done")