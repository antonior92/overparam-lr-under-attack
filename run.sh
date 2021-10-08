# Run all experiments and generate all figures
mkdir out
mkdir out/results  # if it does not exist already
mkdir out/figures  # if it does not exist already
RESULTS=out/results
FIGURES=out/figures
STYLE="plot_style_files/mystyle.mplsty"

# Generate Figure 1
# TODO: In this case the l2 norm start increase again after somepoint!!! This definetly needs some attention.
python linear-estimate.py --num_test_samples 100 --num_train_samples 100 -o $RESULTS/equicorrelated-constant \
    --features_kind equicorrelated --ord 2 inf --datagen_param constant -e 0 0.1 0.5 1.0 -u 3
python linear-plot.py --file out/results/equicorrelated-constant  --plot_style $STYLE plot_style_files/one_half.mplsty \
  plot_style_files/mycolors.mplsty   --plot_type risk_per_eps  --remove_bounds --eps 0 \
  --save $FIGURES/equicorrelated-constant.pdf


# Generate Figure 2
python linear-estimate.py --num_test_samples 300 --num_train_samples 300 -o out/results/equicorrelated-gaussian-prior\
    --ord 1.5 2 20 --features_kind equicorrelated --off_diag 0.9
python linear-plot.py --file out/results/equicorrelated-gaussian-prior \
    --plot_style $STYLE plot_style_files/one_half.mplsty --ord 2  --save out/figures/equicorrelated-gaussian-prior-l2.pdf


# Generate Figure 3
python linear-estimate.py --num_test_samples 500 --num_train_samples 500 -o out/results/isotropic-gaussian-prior \
    --ord 1.5 2 20
python linear-plot.py --file out/results/isotropic-gaussian-prior \
  --plot_style $STYLE plot_style_files/one_half.mplsty  \
  plot_style_files/mycolors.mplsty  --plot_type risk_per_eps --second_marker_set --eps 2.0 \
  --save out/figures/isotropic-gaussian-prior-variouslp.pdf


# Generate Figure 4 (still not there)
python linear-estimate.py  --num_test_samples 100 --num_train_samples 100 -o out/results/isotropic-constant\
    --features_kind  isotropic --ord 2 inf --datagen_param constant -e 0.1 -u 3 --signal_amplitude 2
python linear-plot.py --file out/results/isotropic-constant  --plot_style $STYLE plot_style_files/one_half.mplsty \
  plot_style_files/mycolors.mplsty   --plot_type risk_per_eps  --plot_type risk_per_eps

# Generate Figure 5 (still not there)
python linear-estimate.py  --num_test_samples 100 --num_train_samples 200 -o test   \
  --features_kind latent --ord 2 inf --datagen_param gaussian_prior --latent 1 -e 0 0.1 0.5 \
  --signal_amplitude 1 --noise_std 0 -u 2  --num_latent 20
python linear-plot.py --file test  --plot_style  --plot_type risk_per_eps  --remove_bounds
python linear-plot.py --file test  --plot_style  --plot_type norm --remove_bounds

# From previous run

python estimate_advrisk_linear.py --num_test_samples 100 --num_train_samples 100 -o results/equicorrelated0.9-gaussian-prior.csv \
  --features_kind equicorrelated --ord 2 --off_diag 0.9
python estimate_advrisk_rand_features.py -n 60 -r 4 -u 1 -l -1 --epsilon 0 0.1 1.0 --ord 2 --noise_std 0 \
     -o results/l2-random-feature.csv --fixed nfeatures_over_inputdim --datagen_parameter constant


echo "Generate Figures 1..."

echo "Generate Figures 2..."
python plot_linear.py --file results/equicorrelated0.9-gaussian-prior.csv --save figures/equicorrelated-gaussian-prior-l2.pdf --plot_style $STYLE plot_style_files/one_half.mplsty  --ord 2 --y_min -0.2 --y_max 4.6

echo "Generate Figures 3..."
python plot_linear.py --file results/isotropic-gaussian-prior.csv --save figures/isotropic-gaussian-prior-l1.5.pdf --plot_style $STYLE plot_style_files/one_third_with_ylabel.mplsty --ord 1.5 --y_min -0.2 --y_max 4.6
python plot_linear.py --file results/isotropic-gaussian-prior.csv --save figures/isotropic-gaussian-prior-l2.pdf --plot_style $STYLE plot_style_files/one_third_without_ylabel.mplsty --remove_ylabel --remove_legend --ord 2 --y_min -0.2 --y_max 4.6
python plot_linear.py --file results/isotropic-gaussian-prior.csv --save figures/isotropic-gaussian-prior-l20.pdf --plot_style $STYLE plot_style_files/one_third_without_ylabel.mplsty  --remove_ylabel --remove_legend --ord 20 --y_min -0.2 --y_max 4.6
