# Run all experiments and generate all figures
mkdir out
mkdir out/results  # if it does not exist already
mkdir out/figures  # if it does not exist already
RESULTS=out/results
FIGURES=out/figures
STYLE="plot_style_files/mystyle.mplsty"

rep(){
	for i in $(seq "$2");
	do echo -n "$1"
	 echo -n " "
	done
}

# TODO: Change all the plots to error bars

##############################
## ISOTROPIC FEATURES MODEL ##
################################
# Generate Figure 2
python linear-estimate.py --num_test_samples 500 --num_train_samples 500 -o out/results/isotropic \
    --ord 2 --features_kind isotropic --signal_amplitude 4
python linear-plot.py --file out/results/isotropic \
    --plot_style $STYLE plot_style_files/one_half.mplsty --ord 2 \
      --save out/figures/isotropic.pdf

# Generate Figure S1
for RR in 0.5 1 2 4;
do
  python linear-estimate.py --num_test_samples 500 --num_train_samples 500 -o out/results/isotropic-r"$RR" \
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
  --plot_type advrisk --eps 0 0.1 0.1 --ord inf 2 inf --experiment_plot error_bars \
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


######################
## DIABETES EXAMPLE ##
######################
# Generate Figure 7
python diabetes_example.py --plot_style $STYLE plot_style_files/stacked_bottom.mplsty --save out/figures



######################################################
## Adversarial training overparametrized: isotropic ##
######################################################
# Generate Figure 8
FEATURE_KIND=isotropic
SCALING=none
RESULTS=out/results/advtrain_new
for REGUL_TYPE in advtrain-l2 advtrain-linf ridge lasso;
    do
  for REG in 1 0.5 0.1 0.05 0.01;
        do
    python linear-estimate.py --num_test_samples 200 --num_train_samples 200  --scaling $SCALING \
      --features_kind $FEATURE_KIND --signal_amplitude 4 --ord 2 1.5 20 inf \
      --training $REGUL_TYPE --regularization $REG -ord 2 1.5 20 inf \
      -o "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-"$REG"
  done;
done;

for REGUL_TYPE in advtrain-l2 lasso ridge;
    do
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.5,0.1,0.05,0.01} \
    --labels {0.5,0.1,0.05,0.01} --experiment_plot error_bars \
    --plot_type train_mse --remove_bounds --remove_legend \ %  --save "$FIGURES"/train_mse-"$REGUL_TYPE".pdf \
    --plot_style $STYLE plot_style_files/stacked.mplsty
done;
REGUL_TYPE=advtrain-linf
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.5,0.1,0.05,0.01} \
    --labels "$\delta = "{0.5,0.1,0.05,0.01}"$" --experiment_plot error_bars --y_min -12 --y_max 2 \
    --plot_type train_mse --remove_bounds --save "$FIGURES"/train_mse-"$REGUL_TYPE".pdf \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty


# Figure 9: TODO fix to isotropic / Figure out why runs from hyperion seem more noisy
RESULTS=out/results/advtrain
FEATURE_KIND=isotropic
SCALING=sqrt
for REGUL_TYPE in advtrain-l2 advtrain-linf ridge lasso;
    do
  for REG in 0.5 0.4 0.3 0.2 0.1 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.02 0.01 0.009 0.008 0.007 0.006 0.005
          do
    python linear-estimate.py --num_test_samples 200 --num_train_samples 200  --scaling $SCALING \
          --features_kind $FEATURE_KIND --signal_amplitude 4 --ord 2 1.5 20 inf --training $REGUL_TYPE --regularization $REG \
            --ord 2 1.5 20 inf -l -0.5  -o "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-"$REG"
  done;
done;


REGUL_TYPE=advtrain-l2
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.09,0.08,0.07,0.06,0.05,0.03,0.01} \
    --ord  $(rep inf 7) --experiment_plot error_bars \
    --plot_type norm --remove_bounds --remove_legend --y_scale linear \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty plot_style_files/mycolors.mplsty \
    --save "$FIGURES"/norm-"$REGUL_TYPE".pdf

REGUL_TYPE=ridge
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.5,0.3,0.1,0.05,0.03,0.01,0.008} \
    --ord  $(rep inf 7) --experiment_plot error_bars \
    --plot_type norm --remove_bounds --remove_legend  --y_scale linear \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty plot_style_files/mycolors.mplsty \
    --save "$FIGURES"/norm-"$REGUL_TYPE".pdf

REGUL_TYPE=advtrain-linf
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.03,0.02,0.01,0.008,0.007,0.006,0.005} \
    --ord  $(rep inf 7) --experiment_plot error_bars \
    --plot_type norm --remove_bounds --remove_legend --y_scale linear \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty plot_style_files/mycolors.mplsty \
    --save "$FIGURES"/norm-"$REGUL_TYPE".pdf

REGUL_TYPE=lasso
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.04,0.03,0.02,0.01,0.008,0.007,0.005} \
    --ord  $(rep inf 7) --experiment_plot error_bars  \
    --plot_type norm --remove_bounds --remove_legend  --y_scale linear \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty plot_style_files/mycolors.mplsty \
    --save "$FIGURES"/norm-"$REGUL_TYPE".pdf



# Figure 10
# This is now running on hyperion:experiment_adversarial_training2
RESULTS=out/results/advtrain_new
FEATURE_KIND=isotropic
SCALING=sqrt
REGUL_TYPE=(advtrain-l2 ridge advtrain-linf lasso min-l2norm)
REG=(0.01 0.01 0.01 0.01 0.0)
for i in 0 1 2 3 4;
    do
    python linear-estimate.py --num_test_samples 100 --num_train_samples 100  --scaling $SCALING \
          --features_kind $FEATURE_KIND --signal_amplitude 4 -l -1 -u 2  \
          --ord 2 1.5 20 inf --eps 0.5 0.1 0.05 0.01 0.005 0.001 0.0 \
          --training ${REGUL_TYPE[$i]} \
          --regularization ${REG[$i]} \
          --ord 2 1.5 20 inf   -o "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"${REGUL_TYPE[$i]}"-long
done;


python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-{advtrain-l2,ridge,advtrain-linf,lasso,min-l2norm}-long \
    --plot_type advrisk --eps $(rep 0.01 5)  --ord $(rep inf 5) --experiment_plot error_bars \
    --remove_bounds --second_marker_set --labels "adv. train $\ell_2$" "ridge"  "adv. train $\ell_{\infty}$" lasso "min. norm" \
    --plot_style $STYLE plot_style_files/one_half.mplsty plot_style_files/mycolors.mplsty \
    --save "$FIGURES"/"$FEATURE_KIND"-advrisk-regularized.pdf


###################################################
## Adversarial training overparametrized: latent ##
###################################################

# ---- Figure "9" for Latent ----
RESULTS=out/results/advtrain_new
FEATURE_KIND=latent
SCALING=sqrt
for REGUL_TYPE in advtrain-l2 advtrain-linf ridge lasso;
do
  for REG in 0.5 0.4 0.3 0.2 0.1 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.02 0.01 0.009 0.008 0.007 0.006 0.005
  do
    python linear-estimate.py --num_test_samples 200 --num_train_samples 200  --scaling $SCALING \
          --features_kind $FEATURE_KIND --signal_amplitude 1  --noise_std 0.1 -l -0.5 --eps 0.5 0.1 0.05 0.01 0.005 0.001 0.0 \
           --ord 2 1.5 20 inf --training $REGUL_TYPE \
           --regularization $REG --ord 2 1.5 20 inf \
           -o "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-"$REG"
  done;
done;

REGUL_TYPE=advtrain-l2
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.4,0.3,0.2,0.1,0.05,0.01} \
    --ord  $(rep inf 7) --experiment_plot error_bars \
    --plot_type norm --remove_bounds --remove_legend --y_scale linear \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty plot_style_files/mycolors.mplsty


REGUL_TYPE=ridge
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{1.0,0.5,0.3,0.1,0.05,0.03,0.01} \
    --ord  $(rep inf 7) --experiment_plot error_bars \
    --plot_type norm --remove_bounds --remove_legend  --y_scale linear \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty plot_style_files/mycolors.mplsty

REGUL_TYPE=advtrain-linf
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.06,0.05,0.04,0.03,0.02,0.01,0.005} \
    --ord  $(rep inf 7) --experiment_plot error_bars \
    --plot_type norm --remove_bounds --remove_legend --y_scale linear \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty plot_style_files/mycolors.mplsty

REGUL_TYPE=lasso
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.06,0.05,0.04,0.03,0.02,0.01,0.005} \
    --ord  $(rep inf 7) --experiment_plot error_bars  \
    --plot_type norm --remove_bounds --remove_legend  --y_scale linear \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty plot_style_files/mycolors.mplsty


#  ---- Figure "10" for Latent ----
# NOW RUNNING on hyperion:410452.advtrain
RESULTS=out/results/advtrain_new
FEATURE_KIND=latent
SCALING=sqrt
REGUL_TYPE=(advtrain-l2 ridge advtrain-linf lasso)
REG=(0.01 0.01 0.03 0.03)
for i in 0 1 2 3 4;
    do
    python linear-estimate.py --num_test_samples 200 --num_train_samples 200  --scaling $SCALING \
          --features_kind $FEATURE_KIND --signal_amplitude 1  --noise_std 0.1 -l -1 -u 2  \
          --ord 2 1.5 20 inf --eps 0.5 0.1 0.05 0.01 0.005 0.001 0.0 \
          --training ${REGUL_TYPE[$i]} \
          --regularization ${REG[$i]} \
          -o "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"${REGUL_TYPE[$i]}"-long
done;

python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-advtrain-l2-0.03 \
    "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-ridge-0.03  \
    "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-lasso-0.03 \
    "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-advtrain-linf-0.03  \
    --plot_type advrisk --eps $(rep 0.01 4)  --ord $(rep inf 4) --experiment_plot error_bars \
    --remove_bounds --second_marker_set --y_scale linear

python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-advtrain-l2-0.03 \
    "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-ridge-0.03  \
    "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-lasso-0.03 \
    "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-advtrain-linf-0.03  \
    --plot_type advrisk --eps $(rep 0.01 4)  --ord $(rep inf 4) --experiment_plot error_bars \
    --remove_bounds --second_marker_set --y_scale linear



###################
## MAGIC example ##
###################

wget http://mtweb.cs.ucl.ac.uk/mus/www/MAGICdiverse/MAGIC_diverse_FILES/BASIC_GWAS.tar.gz
tar -xvf BASIC_GWAS.tar.gz

# Testing whether the optimal point changes with the number of features.
#  Now running on hyperion:410452.advtrain
for S in 40 35 30 25 20 15;
do
  python magic-estimate.py -o ./out/results/magic_s"$S" -s $S
done;

#####################
## NONLINEAR MODEL ##
#####################


python rand-feature-plot.py --file out/results/l2-random-feature.csv --plot_style \
   $STYLE plot_style_files/one_half.mplsty --save figures/l2-random-feature.pdf



## SOME ADDITIONAL TESTS!!


## TEST
python linear-estimate.py --num_test_samples 200 --num_train_samples 200 -o test  \
  --features_kind latent --ord 2 1.5 20 inf -e 0 0.1 -r 10 \
  --signal_amplitude 0.1 --noise_std 10 -u 2  --num_latent 20 --scaling none
python linear-plot.py performance --plot_type advrisk --eps 0  --ord inf
python linear-plot.py performance  --plot_type norm



## Presentation

python linear-plot.py out/results/latent-sqrt \
  --plot_type advrisk --eps 0 --ord inf \
  --second_marker_set --remove_legend \
  --plot_style $STYLE plot_style_files/one_half.mplsty  plot_style_files/mycolors.mplsty   \
  --save latent_space_risk.pdf


python linear-plot.py out/results/latent-sqrt \
  --plot_type advrisk --eps 0.1 --ord inf \
  --second_marker_set --remove_legend --fillbetween only-show-ub --remove_bounds \
  --plot_style $STYLE plot_style_files/one_half.mplsty  plot_style_files/mycolors.mplsty   \
  --save latent_space_linfrisk.pdf
python linear-plot.py out/results/latent-sqrt \
  --plot_type advrisk --eps 0.1 --ord 2 \
  --second_marker_set --remove_legendm --fillbetween only-show-ub --remove_bounds \
  --plot_style $STYLE plot_style_files/one_half.mplsty  plot_style_files/mycolors.mplsty   \
  --save latent_space_l2risk.pdf

## Test mispecif (TODO: fix.)
python linear-estimate.py --features_kind mispecif --num_train_samples 100 --num_test_samples 100 -u 1.5 --mispec_factor 3 --signal_amplitude 2 -e 0 --dont_shuffle
python linear-plot.py performance --plot_style $STYLE plot_style_files/one_half.mplsty plot_style_files/mycolors.mplsty

## Test Regularized
REGUL_TYPE=advtrain-l2
for REG in 1 0.05 0.1 0.005 0.01;
  do python linear-estimate.py --num_test_samples 200 --num_train_samples 200 \
      --features_kind isotropic --signal_amplitude 4 --training $REGUL_TYPE --regularization $REG \
      -o out/results/"$REGUL_TYPE"-"$REG"
done;
python linear-plot.py out/results/"$REGUL_TYPE"-{1,0.5,0.1,0.05,0.01} --ord 2 2 2 2 2 --eps 0.1 0.1 0.1 0.1 0.1 \
  --plot_type advrisk --remove_bounds --labels {1,0.5,0.1,0.05,0.01}
python linear-plot.py out/results/"$REGUL_TYPE"-{1,0.5,0.1,0.05,0.01}  --ord 2 2 2 2 2 --eps 0.5 0.5 0.5 0.5 0.5 \
  --plot_type train_mse --remove_bounds --labels {1,0.5,0.1,0.05,0.01}
python linear-plot.py out/results/"$REGUL_TYPE"-{1,0.5,0.1,0.05,0.01}  --ord 2 2 2 2 2 --eps 0.5 0.5 0.5 0.5 0.5 \
  --plot_type norm --remove_bounds --labels {1,0.5,0.1,0.05,0.01}


python linear-estimate.py --num_test_samples 40 --num_train_samples 40 \
      --features_kind isotropic --signal_amplitude 4 --training lasso --regularization 0.000005 \
      -n 50 -r 4
python linear-plot.py performance --plot_type train_mse --remove_bounds --eps 0.5
python linear-plot.py performance --remove_bounds