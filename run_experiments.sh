mkdir results  # if it does not exist already

python estimate_advrisk_linear.py --num_test_samples 100 --num_train_samples 100 -o results/equicorrelated-constant.csv \
    --features_kind equicorrelated --ord 2 inf --datagen_param constant -e 0.1 -u 2
python estimate_advrisk_linear.py --num_test_samples 100 --num_train_samples 100 -o results/isotropic-gaussian-prior.csv \
    --ord 1.5 2 20
python estimate_advrisk_rf.py -n 60 -r 4 -u 1 -l -1 --epsilon 0 0.1 1.0 --ord 2 --noise_std 0 \
     -o results/l2-random-feature.csv --fixed nfeatures_over_inputdim --datagen_parameter constant
