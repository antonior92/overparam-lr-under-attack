# Run all experiments and generate all figures
mkdir out
mkdir out/results  # if it does not exist already
mkdir out/figures  # if it does not exist already
RESULTS=out/results
FIGURES=out/figures
STYLE="plot_style_files/mystyle.mplsty"

# Generate Figure 1 - Intro figure
# TODO: In this case the l2 norm start increase again after somepoint!!! This definetly needs some attention.
python linear-estimate.py --num_test_samples 100 --num_train_samples 100 -o $RESULTS/equicorrelated-constant \
    --features_kind equicorrelated --ord 2 inf --datagen_param constant -e 0 0.1 0.5 1.0 -u 3 --noise_std 0
python linear-plot.py --file out/results/equicorrelated-constant  --plot_style $STYLE plot_style_files/one_half.mplsty \
  plot_style_files/mycolors.mplsty   --plot_type norm --eps 0 \
  --save $FIGURES/equicorrelated-constant.pdf --remove_bounds

# Generate Figure 2
python linear-estimate.py --num_test_samples 500 --num_train_samples 500 -o out/results/isotropic-gaussian-prior\
    --ord 2 --features_kind isotropic --signal_amplitude 4
python linear-plot.py --file out/results/isotropic-gaussian-prior \
    --plot_style $STYLE plot_style_files/one_half.mplsty --ord 2 \
      --save out/figures/isotropic-gaussian-prior.pdf

# Generate Figure S1
for RR in 0.5 1 2;
do
  python linear-estimate.py --num_test_samples 500 --num_train_samples 500 -o out/results/isotropic-gaussian-prior-r"$RR"\
      --ord 2 --features_kind isotropic --signal_amplitude $RR
  python linear-plot.py --file out/results/isotropic-gaussian-prior-r"$RR" \
      --plot_style $STYLE plot_style_files/one_third_with_ylabel.mplsty --ord 2 \
      --save out/figures/isotropic-gaussian-prior-r"$RR".pdf
done;


# Generate Figure S2 and S3
for SCALING in sqrt sqrtlog;
do for RR in 0.5 1 2;
  do
    python linear-estimate.py --num_test_samples 500 --num_train_samples 500 -o out/results/isotropic-gaussian-prior-r"$RR"-"$SCALING"\
        --ord 2 --features_kind isotropic --signal_amplitude $RR --scaling $SCALING
    python linear-plot.py --file out/results/isotropic-gaussian-prior-r"$RR"-"$SCALING" \
        --plot_style $STYLE plot_style_files/one_third_with_ylabel.mplsty --ord 2 \
        --save out/figures/isotropic-gaussian-prior-r"$RR"-"$SCALING".pdf
  done;
done;

# TODO: implement the figure that will become figure 3. Linear estimate must now allow the
# number of features to be constant while the number of datapoints vary.
# We must also allow different values of signal amplitude in the same plot
python linear-estimate.py --num_test_samples 1000 --num_train_samples 1000 -o out/results/isotropic-sqrt-swepover_train\
    --ord 2 --features_kind isotropic --signal_amplitude 4 --scaling sqrt --swep_over num_train_samples -l 0.2 -u 1.5 \
     --eps 1
python linear-plot.py --file out/results/isotropic-sqrt-swepover_train \
       --plot_style $STYLE plot_style_files/one_third_with_ylabel.mplsty --ord 2 --xaxis n

# Generate Figure 3 -> Make it figure 4
python linear-estimate.py --num_test_samples 500 --num_train_samples 500 -o out/results/isotropic-gaussian-prior\
    --ord 2 --features_kind isotropic --signal_amplitude 4
python linear-plot.py --file out/results/isotropic-gaussian-prior \
    --plot_style $STYLE plot_style_files/one_half.mplsty --ord 2 \
      --save out/figures/isotropic-gaussian-prior.pdf


# Generate Figure 4
rm out/results/isotropic-gaussian-prior.csv out/results/isotropic-gaussian-prior.json
python linear-estimate.py --num_test_samples 200 --num_train_samples 200 -o out/results/isotropic-gaussian-prior \
    --ord 1.5 2 20 inf -u 2
python linear-plot.py --file out/results/isotropic-gaussian-prior \
  --plot_style $STYLE plot_style_files/one_half.mplsty  \
  plot_style_files/mycolors.mplsty  --plot_type risk_per_eps --second_marker_set --eps 0  \
  --save out/figures/isotropic-gaussian-prior-variouslp.pdf
python linear-plot.py --file out/results/isotropic-gaussian-prior \
  --plot_style $STYLE plot_style_files/one_half.mplsty  \
  plot_style_files/mycolors.mplsty  --plot_type norm --second_marker_set


# Generate Figure 4
python linear-estimate.py  --num_test_samples 100 --num_train_samples 100 -o out/results/isotropic-constant\
    --features_kind  isotropic --ord 2 inf --datagen_param constant -e 0.1 -u 3 --signal_amplitude 2
python linear-plot.py --file out/results/isotropic-constant  --plot_style $STYLE plot_style_files/one_half.mplsty \
  plot_style_files/mycolors.mplsty   --plot_type risk_per_eps  --plot_type risk_per_eps \
  --save $FIGURES/isotropic-constant.pdf

# Generate Figure 5 (still not there)
python linear-estimate.py  --num_test_samples 100 --num_train_samples 200 -o test   \
  --features_kind latent --ord 2 inf --datagen_param constant --latent 1 -e 0 0.1 0.5 \
  --signal_amplitude 1 --noise_std 0 -u 2  --num_latent 20
python linear-plot.py --file test  --plot_style  --plot_type risk_per_eps  --remove_bounds
python linear-plot.py --file test  --plot_style  --plot_type norm --remove_bounds



# Generate Figure 2*
python linear-estimate.py --num_test_samples 300 --num_train_samples 300 -o out/results/equicorrelated-gaussian-prior\
    --ord 1.5 2 20 --features_kind equicorrelated --off_diag 0.9
python linear-plot.py --file out/results/equicorrelated-gaussian-prior \
    --plot_style $STYLE plot_style_files/one_half.mplsty --ord 2  --save out/figures/equicorrelated-gaussian-prior-l2.pdf


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
