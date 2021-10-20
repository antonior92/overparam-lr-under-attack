# Run all experiments and generate all figures
mkdir out
mkdir out/results  # if it does not exist already
mkdir out/figures  # if it does not exist already
RESULTS=out/results
FIGURES=out/figures
STYLE="plot_style_files/mystyle.mplsty"

##############################
## ISOTROPIC FEATURES MODEL ##
################################
# Generate Figure 2
python linear-estimate.py --num_test_samples 500 --num_train_samples 500 -o out/results/isotropic\
    --ord 2 --features_kind isotropic --signal_amplitude 4
python linear-plot.py --file out/results/isotropic-gaussian-prior \
    --plot_style $STYLE plot_style_files/one_half.mplsty --ord 2 \
      --save out/figures/isotropic.pdf

# Generate Figure S1
for RR in 0.5 1 2 4;
do
  python linear-estimate.py --num_test_samples 500 --num_train_samples 500 -o out/results/isotropic-r"$RR"\
      --ord 2 --features_kind isotropic --signal_amplitude $RR
  python linear-plot.py --file out/results/isotropic-r"$RR" \
      --plot_style $STYLE plot_style_files/one_third_with_ylabel.mplsty --ord 2 \
      --save out/figures/isotropic-r"$RR".pdf
done;


# Generate Figure S2 and S3
for SCALING in sqrt sqrtlog;
do for RR in 0.5 1 2 4;
  do
    python linear-estimate.py --num_test_samples 500 --num_train_samples 500 -o out/results/isotropic-r"$RR"-"$SCALING"\
        --ord 2 --features_kind isotropic --signal_amplitude $RR --scaling $SCALING
    python linear-plot.py --file out/results/isotropic-r"$RR"-"$SCALING" \
        --plot_style $STYLE plot_style_files/one_third_with_ylabel.mplsty --ord 2 \
        --save out/figures/isotropic-r"$RR"-"$SCALING".pdf
  done;
done;

# Generate Figure S4 (a)
python linear-plot.py out/results/isotropic-r{0.5,1,2,4} --plot_type advrisk \
  --second_marker_set --labels '$r^2=0.5$' '$r^2=1$' '$r^2=2$' '$r^2=4$' --eps 0 0 0 0 \
  --plot_style $STYLE plot_style_files/one_half.mplsty  plot_style_files/mycolors.mplsty \
  --save out/figures/isotropic-predrisk.pdf


# Generate Figure S4 (b)
python linear-plot.py out/results/isotropic-r4-{sqrt,sqrtlog} out/results/isotropic-r4 --plot_type norm \
  --second_marker_set --labels '$\eta(m) = \sqrt{m}$' '$\eta(m) = \sqrt{\log{m}}$' '$\eta(m) = 1$'  \
  --plot_style $STYLE plot_style_files/one_half.mplsty  plot_style_files/mycolors.mplsty \
  --save out/figures/isotropic-norm.pdf

# TODO: implement the figure. Linear estimate must now allow the
# number of features to be constant while the number of datapoints vary.
# We must also allow different values of signal amplitude in the same plot
python linear-estimate.py --num_test_samples 1000 --num_train_samples 1000 -o out/results/isotropic-sqrt-swepover_train\
    --ord 2 --features_kind isotropic --signal_amplitude 4 --scaling sqrt --swep_over num_train_samples -l 0.2 -u 1.5 \
     --eps 1
python linear-plot.py --file out/results/isotropic-sqrt-swepover_train \
       --plot_style $STYLE plot_style_files/one_third_with_ylabel.mplsty --ord 2 --xaxis n


# Generate Figure 3
python linear-estimate.py --num_test_samples 500 --num_train_samples 500 -o out/results/isotropic-gaussian-prior \
    --ord 1.5 2 20 --signal_amplitude 1
python linear-plot.py --file out/results/isotropic-gaussian-prior \
  --plot_style $STYLE plot_style_files/one_half.mplsty  \
  plot_style_files/mycolors.mplsty  --plot_type risk_per_eps --second_marker_set --eps 2.0 \
   --fillbetween closest-l2bound \
  --save out/figures/isotropic-gaussian-prior-variouslp.pdf

# Generate Figure 4
for SCALING in sqrt sqrtlog;
do
  python linear-estimate.py --num_test_samples 500 --num_train_samples 500 -o out/results/isotropic-gaussian-prior-linf-"$SCALING" \
      --ord inf --eps 1.0 -u 2 --scaling $SCALING
done;
python linear-plot.py out/results/isotropic-gaussian-prior-linf-sqrt \
  out/results/isotropic-gaussian-prior-linf-sqrtlog --plot_type advrisk \
  --fillbetween only-show-ub --plot_style $STYLE plot_style_files/one_half.mplsty \
  --second_marker_set --save out/figures/linf-isotropic.pdf --labels '$\eta(m) = \sqrt{m}$' \
  '$\eta(m) = \sqrt{\log(m)}$'

###############################
## EQUICORRELATED ATTACKS ##
###############################

# Generate Figure S5 (a)
python linear-estimate.py --num_test_samples 500 --num_train_samples 500 -o out/results/equicorrelated \
      --ord 1.5 2 20 --features_kind equicorrelated --signal_amplitude 4
python linear-plot.py --file out/results/equicorrelated\
        --plot_style $STYLE plot_style_files/one_half.mplsty --ord 2 \
        --save out/figures/equicorrelated.pdf
# Generate Figure S5 (b) and (c)
for SCALING in sqrt sqrtlog;
do
  python linear-estimate.py --num_test_samples 500 --num_train_samples 500 -o out/results/equicorrelated-"$SCALING"\
      --ord 1.5 2 20  --features_kind equicorrelated --signal_amplitude 4 --scaling $SCALING
  python linear-plot.py --file out/results/equicorrelated-"$SCALING" \
      --plot_style $STYLE plot_style_files/one_half.mplsty --ord 2 \
        --save out/figures/equicorrelated-"$SCALING".pdf
done


# Generate Figure S6 (a)
python linear-plot.py --file out/results/equicorrelated \
  --plot_style $STYLE plot_style_files/one_half.mplsty  \
  plot_style_files/mycolors.mplsty  --plot_type risk_per_eps --second_marker_set --eps 2.0\
   --fillbetween closest-l2bound \
   --save out/figures/equicorrelated-variouslp.pdf
# Generate Figure S6 (b) and (c)
for SCALING in sqrt sqrtlog;
do
  python linear-plot.py --file out/results/equicorrelated-"$SCALING"  \
  --plot_style $STYLE plot_style_files/one_half.mplsty  \
  plot_style_files/mycolors.mplsty  --plot_type risk_per_eps --second_marker_set --eps 2.0\
   --fillbetween closest-l2bound \
   --save out/figures/equicorrelated-variouslp-"$SCALING".pdf
done

# Generate Figure S7

for SCALING in sqrt sqrtlog;
do
  python linear-estimate.py --num_test_samples 500 --num_train_samples 500 -o out/results/equicorrelated-linf-"$SCALING"-0.5 \
      --ord inf --eps 1.0 -u 2 --scaling $SCALING --features_kind equicorrelated
done;
python linear-plot.py out/results/equicorrelated-linf-{sqrt,sqrtlog}-0.5 --plot_type advrisk \
  --fillbetween only-show-ub --plot_style $STYLE plot_style_files/one_half.mplsty \
  --second_marker_set --labels '$\eta(m) = \sqrt{m}$' '$\eta(m) = \sqrt{\log(m)}$' \
   --save out/figures/equicorrelated-linf.pdf

##################
## LATENT MODEL ##
##################

# Generate Figure 1
python linear-estimate.py  --num_test_samples 500 --num_train_samples 500 -o out/results/latent-sqrt  \
  --features_kind latent --ord 2 1.5 20 inf -e 0 0.1 \
  --signal_amplitude 1 --noise_std 0.1 -u 2  --num_latent 20 --scaling sqrt
python linear-plot.py out/results/latent-sqrt out/results/latent-sqrt  out/results/latent-sqrt  \
  --plot_type advrisk --eps 0 0.1 0.1 --ord inf 2 inf\
  --remove_bounds --second_marker_set --labels "no adv." '$\ell_2$ adv.' '$\ell_\infty$ adv.' \
  --plot_style $STYLE plot_style_files/one_half.mplsty  plot_style_files/mycolors.mplsty   \
  --save out/figures/latent.pdf
# Generate Figure 4(a)
python linear-estimate.py  --num_test_samples 500 --num_train_samples 500 -o  out/results/latent-logsqrt  \
  --features_kind latent --ord  2 1.5 20 inf -e 0 0.1 \
  --signal_amplitude 1 --noise_std 0.1 -u 2  --num_latent 20 --scaling sqrtlog
python linear-plot.py  out/results/latent-sqrt out/results/latent-logsqrt --plot_type advrisk --ord 2 2 \
  --remove_bounds --second_marker_set --labels '$\eta(m) = \sqrt{m}$' '$\eta(m) = \sqrt{\log(m)}$' \
  --plot_style $STYLE plot_style_files/stacked.mplsty  plot_style_files/mycolors.mplsty   \
  --save out/figures/latent-l2.pdf --remove_xlabel
# Generate Figure 4(b)
python linear-plot.py  out/results/latent-sqrt out/results/latent-logsqrt --plot_type advrisk --ord inf inf \
  --remove_bounds --second_marker_set --labels '$\eta(m) = \sqrt{m}$' '$\eta(m) = \sqrt{\log(m)}$' \
  --plot_style $STYLE plot_style_files/stacked_bottom.mplsty  plot_style_files/mycolors.mplsty \
  --save out/figures/latent-linf.pdf --remove_legend


# Generate Figure S8
python linear-plot.py  out/results/latent-sqrt out/results/latent-logsqrt --plot_type advrisk --ord 2 2 \
  --remove_bounds --second_marker_set --labels '$\eta(m) = \sqrt{m}$' '$\eta(m) = \sqrt{\log(m)}$' \
  --plot_style $STYLE plot_style_files/one_half.mplsty   plot_style_files/mycolors.mplsty   \
  --save out/figures/latent-l1.5.pdf --remove_xlabel
python linear-plot.py  out/results/latent-sqrt out/results/latent-logsqrt --plot_type advrisk --ord 20 20 \
  --remove_bounds --second_marker_set --labels '$\eta(m) = \sqrt{m}$' '$\eta(m) = \sqrt{\log(m)}$' \
  --plot_style $STYLE plot_style_files/one_half.mplsty  plot_style_files/mycolors.mplsty \
  --save out/figures/latent-l20.pdf --remove_legend


# Generate Figure S9
for SCALING in sqrt sqrtlog;
  do for NOISE in 1 0.5 0.1 0;
  do
    python linear-estimate.py  --num_test_samples 200 --num_train_samples 200 -o out/results/latent-"$SCALING"-"$NOISE"  \
      --features_kind latent --ord 2 1.5 20 inf -e 0 0.1 0.5 2  \
      --signal_amplitude 1 --noise_std $NOISE -u 2  --num_latent 20 --scaling "$SCALING"
  done;
done


for SCALING in sqrt sqrtlog;
  do for ORD in 2 inf;
  do
    python linear-plot.py out/results/latent-"$SCALING"-{1,0.5,0.1,0} --ord $ORD $ORD $ORD $ORD --eps 0.1 0.1 0.1 0.1 --plot_type advrisk \
      --plot_style $STYLE plot_style_files/stacked.mplsty --remove_bounds --second_marker_set \
      --labels '$\sigma_\xi = 1$' '$\sigma_\xi = 0.5$' '$\sigma_\xi = 0.1$' '$\sigma_\xi = 0$' \
       --save latent-noise-l"$ORD"-"$SCALING" --remove_legend
  done;
done
# Repeat the last one with the legend so it overide the plot (Not very clen :) )
python linear-plot.py out/results/latent-sqrt-{1,0.5,0.1,0} --ord inf inf inf inf  --eps 0.1 0.1 0.1 0.1 --plot_type advrisk \
      --plot_style $STYLE plot_style_files/stacked.mplsty --remove_bounds --second_marker_set \
      --labels '$\sigma_\xi = 1$' '$\sigma_\xi = 0.5$' '$\sigma_\xi = 0.1$' '$\sigma_\xi = 0$' \
       --save latent-noise-linf-sqrt


# Generate Figure S10

for SCALING in sqrt sqrtlog;
  do for ORD in 2 inf;
  do
  python linear-plot.py out/results/latent-"$SCALING"-0.1 \
      --plot_style $STYLE plot_style_files/stacked.mplsty  --ord $ORD  --remove_bounds \
       --save latent-eps-l"$ORD"-"$SCALING" --remove_legend
  done;
done;
python linear-plot.py out/results/latent-sqrt-0.1 \
      --plot_style $STYLE plot_style_files/stacked.mplsty  --ord inf  --remove_bounds \
       --save latent-eps-linf-sqrt

#####################
## NONLINEAR MODEL ##
#####################


python rand-feature-plot.py --file out/results/l2-random-feature.csv --plot_style \
   $STYLE plot_style_files/one_half.mplsty --save figures/l2-random-feature.pdf
