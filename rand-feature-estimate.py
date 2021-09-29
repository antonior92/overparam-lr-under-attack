if __name__ == '__main__':
    from interp_under_attack.random_features.random_feature_regression import train_and_evaluate
    from interp_under_attack.random_features.activation_function_parameters import implemented_activations
    from interp_under_attack.util import frac2int
    import itertools
    from tqdm import tqdm
    import pandas as pd
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description='Double descent for l-2 adversarial attack')
    parser.add_argument('-o', '--output', default='./performance.csv',
                        help='output csv file.')
    parser.add_argument('--num_train_samples', type=int, default=100,
                        help='number of samples in the experiment.')
    parser.add_argument('--num_test_samples', type=int, default=100,
                        help='number of samples in the experiment.')
    parser.add_argument('-r', '--repetitions', type=int, default=1,
                        help='number of times each experiment is repeated')
    parser.add_argument('-p', '--ord', type=float, default=[2.0], nargs='+',
                        help='ord is p norm of the adversarial attack.')
    parser.add_argument('-n', '--num_points', default=60, type=int,
                        help='number of points')
    parser.add_argument('--n_adv_steps', default=100, type=int,
                        help='number of steps used in the adversarial attack')
    parser.add_argument('-l', '--lower_proportion', default=-1, type=float,
                        help='the lowest value for the proportion (n features / n samples) is 10^l.')
    parser.add_argument('-u', '--upper_proportion', default=1, type=float,
                        help='the upper value for the proportion (n features / n samples) is 10^u.')
    parser.add_argument('--fixed_proportion', default=0.5, type=float,
                        help='the value of the proportion that is fixed')
    parser.add_argument('--fixed', choices=['inputdim_over_datasize', 'nfeatures_over_datasize',
                                            'nfeatures_over_inputdim'], default='nfeatures_over_datasize',
                        help='what is fixed in the problem.')
    parser.add_argument('--datagen_parameter', choices=['gaussian_prior', 'constant'], default='gaussian_prior',
                        help='how the features are generated')
    parser.add_argument('-s', '--noise_std', type=float, default=1,
                        help='standard deviation of the additive noise added.')
    parser.add_argument('--regularization', type=float, default=1e-6,
                        help='type of ridge regularization.')
    parser.add_argument('--activation', choices=implemented_activations, default='relu',
                        help='activations function')
    parser.add_argument('-e', '--epsilon', default=[0, 0.1, 0.5, 1.0, 2.0], type=float, nargs='+',
                        help='the epsilon values used when computing the adversarial attack')
    parser.add_argument('--signal_amplitude', type=float, default=1,
                        help='signal amplitude. I.e. \|beta*\|_2')
    args, unk = parser.parse_known_args()
    print(args)

    # Compute performance for varying number of features
    tqdm.write("Estimating performance as a function of proportion...")
    df = pd.DataFrame(columns=['inputdim_over_datasize', 'nfeatures_over_datasize', 'seed', 'datasize'])
    proportions = np.logspace(args.lower_proportion, args.upper_proportion, args.num_points)
    run_instances = list(itertools.product(range(args.repetitions), proportions))
    for seed, proportion in tqdm(run_instances, smoothing=0.03):
        # Get problem type
        if args.fixed == 'inputdim_over_datasize':
             inputdim_over_datasize = args.fixed_proportion
             nfeatures_over_datasize = proportion
        elif args.fixed == 'nfeatures_over_datasize':
            inputdim_over_datasize = args.fixed_proportion
            nfeatures_over_datasize = proportion
        elif args.fixed == 'nfeatures_over_inputdim':
            inputdim_over_datasize = args.fixed_proportion * proportion
            nfeatures_over_datasize = proportion
        else:
            raise ValueError('Invalid argument --fixed = {}.'.format(args.fixed))
        n_features = frac2int(nfeatures_over_datasize, args.num_train_samples)
        input_dim = frac2int(inputdim_over_datasize, args.num_train_samples)
        risk, pnorms = \
            train_and_evaluate(args.num_train_samples, n_features, input_dim, args.noise_std, args.signal_amplitude,
                               args.num_test_samples, args.activation, args.regularization, args.ord,
                               args.epsilon, args.n_adv_steps, args.datagen_parameter, seed)
        dict1 = {'inputdim_over_datasize': inputdim_over_datasize, 'nfeatures_over_datasize': nfeatures_over_datasize,
                 'seed': seed, 'datasize': args.num_train_samples}
        df = df.append({**risk, **pnorms, **dict1, **vars(args)}, ignore_index=True)
        df.to_csv(args.output, index=False)
    tqdm.write("Done")