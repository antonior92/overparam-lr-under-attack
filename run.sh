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
python linear-estimate.py --num_test_samples 300 --num_train_samples 300 -o out/results/isotropic \
    --ord 2 --features_kind isotropic --signal_amplitude 2 -r 10
python linear-plot.py --file out/results/isotropic \
    --plot_style $STYLE plot_style_files/one_half3.mplsty plot_style_files/mylegend.mplsty --ord 2 \
    --save out/figures/isotropic.pdf


# Generate Figure S2
for SCALING in sqrt sqrtlog none;
do for RR in 0.5 1 2 4;
  do
    python linear-estimate.py --num_test_samples 300 --num_train_samples 300 -o out/results/isotropic-r"$RR"-"$SCALING"\
        --ord 2 --features_kind isotropic --signal_amplitude $RR --scaling $SCALING -r 10
  done;
done;

for SCALING in sqrt sqrtlog none;
  do for RR in 0.5 1 2;
    do python linear-plot.py --file out/results/isotropic-r"$RR"-"$SCALING" \
        --plot_style $STYLE plot_style_files/one_third_with_ylabel.mplsty  plot_style_files/mylegend.mplsty --remove_legend --ord 2 \
        --save out/figures/isotropic-r"$RR"-"$SCALING".pdf
    done;
done;
RRR=0.5
SCALING=none
python linear-plot.py --file out/results/isotropic-r"$RRR"-"$SCALING" \
    --plot_style $STYLE plot_style_files/one_third_with_ylabel.mplsty plot_style_files/mylegend.mplsty  --ord 2 \
    --save out/figures/isotropic-r"$RRR"-"$SCALING".pdf


# Generate Figure S1 (a)
python linear-plot.py out/results/isotropic-r{0.5,1,2,4} --plot_type advrisk \
  --second_marker_set --labels '$r^2=0.5$' '$r^2=1$' '$r^2=2$' '$r^2=4$' --eps 0 0 0 0 \
  --plot_style $STYLE   plot_style_files/one_third_with_ylabel.mplsty   plot_style_files/mycolors.mplsty plot_style_files/mylegend.mplsty  \
  --save out/figures/isotropic-predrisk.pdf


# Generate Figure S1 (b)
python linear-plot.py out/results/isotropic-r2-{sqrt,sqrtlog} out/results/isotropic-r2 --plot_type norm \
  --second_marker_set --labels '$\eta(m) = \sqrt{m}$' '$\eta(m) = \sqrt{\log{m}}$' '$\eta(m) = 1$'  \
  --plot_style $STYLE  plot_style_files/one_third_with_ylabel.mplsty plot_style_files/mycolors.mplsty plot_style_files/mylegend.mplsty \
  --y_min -0.2 --y_max 3.5 \
  --save out/figures/isotropic-norm.pdf

# Generate Figure 3
python linear-estimate.py --num_test_samples 100 --num_train_samples 100 -o out/results/isotropic-gaussian-prior \
    --ord 1 2 inf --signal_amplitude 1 -r 10
python linear-plot.py --file out/results/isotropic-gaussian-prior out/results/isotropic-gaussian-prior out/results/isotropic-gaussian-prior \
   --labels  "$\ell_\infty$ adv." "$\ell_2$ adv."  "$\ell_1$ adv." \
  --plot_style $STYLE plot_style_files/one_half.mplsty plot_style_files/mylegend.mplsty \
  plot_style_files/mycolors.mplsty plot_style_files/stacked.mplsty  --plot_type advrisk --second_marker_set --ord inf 2 1 \
    --eps 2.0 2.0 2.0 --fillbetween closest-l2bound --remove_xlabel --out_legend \
   --save out/figures/isotropic-variouslp.pdf
python linear-plot.py --file  out/results/isotropic-gaussian-prior out/results/isotropic-gaussian-prior out/results/isotropic-gaussian-prior \
  --plot_style $STYLE plot_style_files/one_half.mplsty plot_style_files/mylegend.mplsty \
  plot_style_files/mycolors.mplsty plot_style_files/stacked_bottom.mplsty \
   --labels "$\|\hat{\beta}\|_1$" "$\|\hat{\beta}\|_2$" "$\|\hat{\beta}\|_\infty$"  \
   --plot_type norm --second_marker_set --ord inf 2 1  \
   --fillbetween closest-l2bound --out_legend \
   --save out/figures/isotropic-variouslp-norm.pdf



# Generate Figure 5
for SCALING in sqrt sqrtlog;
do
  python linear-estimate.py --num_test_samples 100 --num_train_samples 100 -o out/results/isotropic-linf-"$SCALING" \
      --ord 2 inf --eps 0 1.0 -u 2 --scaling $SCALING -r 10
done;
python linear-plot.py out/results/isotropic-linf-sqrt out/results/isotropic-linf-sqrtlog --plot_type advrisk --ord inf inf \
  --fillbetween only-show-ub --plot_style $STYLE plot_style_files/one_half3.mplsty plot_style_files/mylegend.mplsty \
  --second_marker_set --labels '$\eta(m) = \sqrt{m}$' \
  '$\eta(m) = \sqrt{\log(m)}$' \
  --save out/figures/linf-isotropic.pdf



###############################
## EQUICORRELATED ATTACKS ##
###############################

# Generate Figure S3
for SCALING in none sqrt sqrtlog;
do
  python linear-estimate.py --num_test_samples 300 --num_train_samples 300 -o out/results/equicorrelated-"$SCALING"\
      --ord 1.5 2 20  --features_kind equicorrelated --signal_amplitude 4 --scaling $SCALING -r 10
  python linear-plot.py --file out/results/equicorrelated-"$SCALING" \
      --plot_style $STYLE plot_style_files/one_third_with_ylabel.mplsty  plot_style_files/mylegend.mplsty --ord 2 --remove_legend \
      --save out/figures/equicorrelated-"$SCALING".pdf
done
SCALING=sqrt
python linear-plot.py --file out/results/equicorrelated-"$SCALING" \
    --plot_style $STYLE plot_style_files/one_third_with_external_legend.mplsty  plot_style_files/mylegend.mplsty --ord 2 \
     --out_legend --out_legend_bbox_y 1 \
    --save out/figures/equicorrelated-"$SCALING".pdf


# Generate Figure S4 (a)
python linear-plot.py --file out/results/equicorrelated \
   --plot_type risk_per_eps --second_marker_set --eps 2.0 \
   --fillbetween closest-l2bound \
   --plot_style $STYLE plot_style_files/one_third_with_ylabel.mplsty  plot_style_files/mylegend.mplsty  plot_style_files/mycolors.mplsty \
   --save out/figures/equicorrelated-variouslp.pdf
# Generate Figure S4 (b) and (c)
SCALING=none
python linear-plot.py --file out/results/equicorrelated-"$SCALING" --scaling \
  --plot_type risk_per_eps --second_marker_set --eps 2.0 --y_min 0.5 --y_max 6.3 \
  --fillbetween closest-l2bound \
  --plot_style $STYLE plot_style_files/one_third_with_ylabel.mplsty  plot_style_files/mylegend.mplsty  plot_style_files/mycolors.mplsty \
  --save out/figures/equicorrelated-variouslp-"$SCALING".pdf
for SCALING in sqrt sqrtlog;
do
   python linear-plot.py --file out/results/equicorrelated-"$SCALING" --scaling \
   --plot_type risk_per_eps --second_marker_set --eps 2.0 \
   --fillbetween closest-l2bound \
   --plot_style $STYLE plot_style_files/one_third_with_ylabel.mplsty  plot_style_files/mylegend.mplsty  plot_style_files/mycolors.mplsty \
   --save out/figures/equicorrelated-variouslp-"$SCALING".pdf
done

# Generate Figure S5

for SCALING in sqrt sqrtlog;
do
  python linear-estimate.py --num_test_samples 100 --num_train_samples 100 -o out/results/equicorrelated-linf-"$SCALING"-0.5 \
      --ord inf --eps 0.1 -u 2 --scaling $SCALING --features_kind equicorrelated -r 10
done;
python linear-plot.py out/results/equicorrelated-linf-{sqrt,sqrtlog}-0.5 --plot_type advrisk --ord inf inf \
  --fillbetween only-show-ub --plot_style $STYLE plot_style_files/one_third_with_ylabel.mplsty plot_style_files/mylegend.mplsty \
  --second_marker_set --labels '$\eta(m) = \sqrt{m}$' \
  '$\eta(m) = \sqrt{\log(m)}$' \
  --save out/figures/linf-equicorrelated.pdf

python linear-plot.py out/results/equicorrelated-linf-{sqrt,sqrtlog}-0.5 --ord inf inf \
  --fillbetween only-show-ub --plot_style $STYLE plot_style_files/one_third_with_ylabel.mplsty plot_style_files/mylegend.mplsty \
  --second_marker_set --labels '$\eta(m) = \sqrt{m}$' \
  '$\eta(m) = \sqrt{\log(m)}$' --plot_type norm \
  --save out/figures/linf-equicorrelated-norm.pdf

##################
## LATENT MODEL ##
##################

# Generate Figure 1
python linear-estimate.py  --num_test_samples 200 --num_train_samples 200 -o out/results/latent-sqrt  \
  --features_kind latent --ord 2 1.5 20 inf -e 0 0.1 \
  --signal_amplitude 1 --noise_std 0.1 -u 2  --num_latent 20 --scaling sqrt -r 10
python linear-plot.py out/results/latent-sqrt out/results/latent-sqrt  out/results/latent-sqrt  \
  --plot_type advrisk --eps 0.1 0.1 0 --ord   inf 2 inf --experiment_plot error_bars \
  --remove_bounds --second_marker_set --labels  '$\ell_\infty$ adv.' '$\ell_2$ adv.' "no adv." \
  --plot_style $STYLE plot_style_files/one_half2.mplsty  plot_style_files/mycolors.mplsty  plot_style_files/mylegend.mplsty \
  --save out/figures/latent.pdf
# Generate Figure 6(a)
python linear-estimate.py  --num_test_samples 200 --num_train_samples 200 -o  out/results/latent-logsqrt  \
  --features_kind latent --ord  2 1.5 20 inf -e 0 0.1 \
  --signal_amplitude 1 --noise_std 0.1 -u 2  --num_latent 20 --scaling sqrtlog  -r 10
python linear-plot.py  out/results/latent-sqrt out/results/latent-logsqrt --plot_type advrisk --ord 2 2 --eps 0.1 0.1 \
  --remove_bounds --second_marker_set --labels '$\eta(m) = \sqrt{m}$' '$\eta(m) = \sqrt{\log(m)}$' \
  --plot_style $STYLE plot_style_files/stacked.mplsty  plot_style_files/mycolors.mplsty  plot_style_files/mylegend.mplsty \
  --save out/figures/latent-l2.pdf --remove_xlabel --empirical_bounds
# Generate Figure 6(b)
python linear-plot.py  out/results/latent-sqrt out/results/latent-logsqrt --plot_type advrisk --ord inf inf --eps 0.1 0.1 \
  --remove_bounds --second_marker_set --labels '$\eta(m) = \sqrt{m}$' '$\eta(m) = \sqrt{\log(m)}$' \
  --plot_style $STYLE plot_style_files/stacked_bottom.mplsty  plot_style_files/mycolors.mplsty plot_style_files/mylegend.mplsty \
  --save out/figures/latent-linf.pdf --remove_legend --empirical_bounds


# Generate Figure S6
python linear-plot.py  out/results/latent-sqrt out/results/latent-logsqrt --plot_type advrisk --ord 2 2 --eps 0.1 0.1 \
  --second_marker_set --labels '$\eta(m) = \sqrt{m}$' '$\eta(m) = \sqrt{\log(m)}$' \
  --plot_style $STYLE plot_style_files/one_third_with_ylabel2.mplsty  plot_style_files/mycolors.mplsty  plot_style_files/mylegend.mplsty \
  --save out/figures/latent-l2-asympt.pdf
python linear-plot.py  out/results/latent-sqrt out/results/latent-logsqrt --plot_type advrisk --ord inf inf --eps 0.1 0.1 \
 --second_marker_set --labels '$\eta(m) = \sqrt{m}$' '$\eta(m) = \sqrt{\log(m)}$' --fillbetween only-show-ub \
  --plot_style $STYLE plot_style_files/one_third_with_ylabel.mplsty  plot_style_files/mycolors.mplsty plot_style_files/mylegend.mplsty \
  --remove_legend --save out/figures/latent-linf-asympt.pdf


# Generate Figure S7
for SCALING in sqrt sqrtlog;
  do for NOISE in 1 0.5 0.1 0;
  do
    python linear-estimate.py  --num_test_samples 200 --num_train_samples 200 -o out/results/latent-"$SCALING"-"$NOISE"  \
      --features_kind latent --ord 2 1.5 20 inf -e 0 0.1 0.5 2  \
      --signal_amplitude 1 --noise_std $NOISE -u 2  --num_latent 20 --scaling "$SCALING" -r 10
  done;
done
python linear-plot.py  out/results/latent-sqrt-{1,0.5,0.1,0}  --plot_type advrisk --ord 2 2 2 2 --eps 0.1 0.1 0.1 0.1 \
    --second_marker_set  --remove_legend   \
    --plot_style $STYLE plot_style_files/one_third_with_ylabel2.mplsty  plot_style_files/mycolors.mplsty  plot_style_files/mylegend.mplsty \
    --save out/figures/latent-noise-l2-sqrt.pdf
python linear-plot.py  out/results/latent-sqrt-{1,0.5,0.1,0}  --plot_type advrisk --ord inf inf inf inf --eps 0.1 0.1 0.1 0.1 \
    --second_marker_set --remove_legend --fillbetween only-show-ub  \
    --plot_style $STYLE plot_style_files/one_third_with_ylabel2.mplsty  plot_style_files/mycolors.mplsty  plot_style_files/mylegend.mplsty \
    --save out/figures/latent-noise-linf-sqrt.pdf
python linear-plot.py  out/results/latent-sqrtlog-{1,0.5,0.1,0}  --plot_type advrisk --ord 2 2 2 2 --eps 0.1 0.1 0.1 0.1 \
    --second_marker_set --remove_legend  \
    --plot_style $STYLE plot_style_files/one_third_with_ylabel2.mplsty  plot_style_files/mycolors.mplsty  plot_style_files/mylegend.mplsty \
    --save out/figures/latent-noise-l2-sqrtlog.pdf
python linear-plot.py  out/results/latent-sqrtlog-{1,0.5,0.1,0}  --plot_type advrisk --ord inf inf inf inf --eps 0.1 0.1 0.1 0.1 \
    --second_marker_set --labels '$\sigma_\xi = 1$' '$\sigma_\xi = 0.5$' '$\sigma_\xi = 0.1$' '$\sigma_\xi = 0$' --fillbetween only-show-ub  \
    --plot_style $STYLE plot_style_files/one_third_with_ylabel2.mplsty  plot_style_files/mycolors.mplsty  plot_style_files/mylegend.mplsty \
    --save out/figures/latent-noise-linf-sqrtlog.pdf



# Generate Figure S8
python linear-plot.py  $(rep out/results/latent-sqrt-0.1 4)  --plot_type advrisk --ord 2 2 2 2 --eps 0 0.1 0.5 2.0 \
    --second_marker_set  --remove_legend  \
    --plot_style $STYLE plot_style_files/one_third_with_ylabel2.mplsty  plot_style_files/mycolors.mplsty  plot_style_files/mylegend.mplsty \
    --save out/figures/latent-eps-l2-sqrt.pdf
python linear-plot.py  $(rep out/results/latent-sqrt-0.1 4)  --plot_type advrisk --ord inf inf inf inf --eps 0 0.1 0.5 2.0  \
    --second_marker_set --remove_legend --fillbetween only-show-ub  \
    --plot_style $STYLE plot_style_files/one_third_with_ylabel2.mplsty  plot_style_files/mycolors.mplsty  plot_style_files/mylegend.mplsty \
    --save out/figures/latent-eps-linf-sqrt.pdf
python linear-plot.py  $(rep out/results/latent-sqrtlog-0.1 4)  --plot_type advrisk --ord 2 2 2 2 --eps 0 0.1 0.5 2.0  \
    --second_marker_set --remove_legend  \
    --plot_style $STYLE plot_style_files/one_third_with_ylabel2.mplsty  plot_style_files/mycolors.mplsty  plot_style_files/mylegend.mplsty \
    --save out/figures/latent-eps-l2-sqrtlog.pdf
python linear-plot.py  $(rep out/results/latent-sqrtlog-0.1 4)   --plot_type advrisk --ord inf inf inf inf --eps 0 0.1 0.5 2.0  \
    --second_marker_set --labels '$\delta = $'{2.0,0.1,0.1,0} --fillbetween only-show-ub  \
    --plot_style $STYLE plot_style_files/one_third_with_ylabel2.mplsty  plot_style_files/mycolors.mplsty  plot_style_files/mylegend.mplsty \
    --save out/figures/latent-eps-linf-sqrtlog.pdf


########################################
## Discussion about random projection ##
########################################
# Generate Figure 4
python norm_projections.py

######################
## DIABETES EXAMPLE ##
######################
# Generate Figure XX
python diabetes_example.py --plot_style $STYLE plot_style_files/stacked_bottom.mplsty --save out/figures


# TODO: fix to isotropic / Figure out why runs from hyperion seem more noisy
######################################################
## Adversarial training overparametrized: isotropic ##
######################################################
# Generate Figure X(Will be moved to follow up paper)
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

REGUL_TYPE=ridge
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.5,0.1,0.05,0.01} \
    --labels "$\delta = "{0.5,0.1,0.05,0.01}"$" --experiment_plot error_bars --y_min -12 --y_max 2 \
    --plot_type train_mse --remove_bounds --remove_xlabel \
    --plot_style $STYLE plot_style_files/stacked.mplsty plot_style_files/mylegend.mplsty \
    --save "$FIGURES"/train_mse-"$REGUL_TYPE".pdf

for REGUL_TYPE in advtrain-l2 lasso;
    do
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.5,0.1,0.05,0.01} \
    --labels {0.5,0.1,0.05,0.01} --experiment_plot error_bars \
    --plot_type train_mse --remove_bounds --remove_legend  --remove_xlabel\
    --plot_style $STYLE plot_style_files/stacked.mplsty \
    --save "$FIGURES"/train_mse-"$REGUL_TYPE".pdf
done;
REGUL_TYPE=advtrain-linf
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.5,0.1,0.05,0.01} \
    --labels "$\delta = "{0.5,0.1,0.05,0.01}"$" --experiment_plot error_bars --y_min -12 --y_max 2 \
    --plot_type train_mse --remove_bounds --remove_legend\
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty \
    --save "$FIGURES"/train_mse-"$REGUL_TYPE".pdf


# Figure S9
RESULTS=out/results/advtrain_new
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
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.01,0.03,0.05,0.06,0.07,0.08,0.09}\
    --ord  $(rep inf 7) --experiment_plot error_bars \
    --plot_type norm --remove_bounds --y_scale linear \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty plot_style_files/mycolors.mplsty plot_style_files/mylegend.mplsty  \
    --out_legend --labels {0.01,0.03,0.05,0.06,0.07,0.08,0.09} \
    --save "$FIGURES"/norm-"$REGUL_TYPE".pdf

REGUL_TYPE=ridge
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.008,0.01,0.03,0.05,0.1,0.3,0.5} \
    --ord  $(rep inf 7) --experiment_plot error_bars \
    --plot_type norm --remove_bounds --y_scale linear \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty plot_style_files/mycolors.mplsty plot_style_files/mylegend.mplsty  \
    --out_legend --labels {0.008,0.01,0.03,0.05,0.1,0.3,0.5} \
    --save "$FIGURES"/norm-"$REGUL_TYPE".pdf

REGUL_TYPE=advtrain-linf
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.005,0.006,0.007,0.008,0.01,0.02,0.03} \
    --ord  $(rep inf 7) --experiment_plot error_bars \
    --plot_type norm --remove_bounds --out_legend --y_scale linear \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty plot_style_files/mycolors.mplsty plot_style_files/mylegend.mplsty \
    --labels {0.005,0.006,0.007,0.008,0.01,0.02,0.03} \
    --save "$FIGURES"/norm-"$REGUL_TYPE".pdf

REGUL_TYPE=lasso
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.005,0.007,0.008,0.01,0.02,0.03,0.04} \
    --ord  $(rep inf 7) --experiment_plot error_bars  \
    --plot_type norm --remove_bounds --out_legend  --y_scale linear \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty plot_style_files/mycolors.mplsty plot_style_files/mylegend.mplsty \
    --labels {0.005,0.007,0.008,0.01,0.02,0.03,0.04} \
    --save "$FIGURES"/norm-"$REGUL_TYPE".pdf



# Figure S10
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


python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-{min-l2norm,advtrain-l2,ridge,advtrain-linf,lasso}-long \
    --plot_type advrisk --eps $(rep 0.01 5)  --ord $(rep inf 5) --experiment_plot error_bars \
    --remove_bounds --second_marker_set --labels "min. norm"  "adv. train $\ell_2$" "ridge"  "adv. train $\ell_{\infty}$" lasso \
    --plot_style $STYLE plot_style_files/one_half4.mplsty plot_style_files/mycolors2.mplsty plot_style_files/mylegend.mplsty \
    --save "$FIGURES"/"$FEATURE_KIND"-advrisk-regularized.pdf

python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-{min-l2norm,advtrain-l2,ridge,advtrain-linf,lasso}-long \
    --plot_type advrisk --eps $(rep 0.0 5)  --ord $(rep inf 5) --experiment_plot error_bars \
    --remove_bounds --second_marker_set --remove_legend \
    --plot_style $STYLE plot_style_files/one_half4.mplsty plot_style_files/mycolors2.mplsty  plot_style_files/mylegend.mplsty \
    --save "$FIGURES"/"$FEATURE_KIND"-stdrisk-regularized.pdf

###################################################
## Adversarial training overparametrized: latent ##
###################################################
# ---- Figure XX ---- (It will go to the next paper)
FEATURE_KIND=latent
SCALING=none
RESULTS=out/results/advtrain_new
for REGUL_TYPE in advtrain-l2 advtrain-linf ridge lasso;
    do
  for REG in 1 0.5 0.1 0.05 0.01;
        do
    python linear-estimate.py --num_test_samples 200 --num_train_samples 200  --scaling $SCALING \
      --features_kind $FEATURE_KIND --signal_amplitude 1  --noise_std 0.1 -l -0.5 --eps 0.5 0.1 0.05 0.01 0.005 0.001 0.0 \
      --ord 2 1.5 20 inf --training $REGUL_TYPE \
      --regularization $REG -ord 2 1.5 20 inf \
      -o "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-"$REG"
  done;
done;

REGUL_TYPE=ridge
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.5,0.1,0.05,0.01} \
    --labels "$\delta = "{0.5,0.1,0.05,0.01}"$" --experiment_plot error_bars \
    --plot_type train_mse --remove_bounds \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty  plot_style_files/mylegend.mplsty plot_style_files/mycolors.mplsty  \
    --save "$FIGURES"/train_mse-"$REGUL_TYPE"-"$FEATURE_KIND".pdf
REGUL_TYPE=lasso
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.5,0.1,0.05,0.01} \
    --experiment_plot error_bars \
    --plot_type train_mse --remove_bounds --remove_legend \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty  plot_style_files/mylegend.mplsty plot_style_files/mycolors.mplsty  \
    --save "$FIGURES"/train_mse-"$REGUL_TYPE"-"$FEATURE_KIND".pdf
REGUL_TYPE=advtrain-linf
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.5,0.1,0.05,0.01} \
    --labels "$\delta = "{0.5,0.1,0.05,0.01}"$" --experiment_plot error_bars --y_min -12 --y_max 2  \
    --plot_type train_mse --remove_bounds \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty  plot_style_files/mylegend.mplsty plot_style_files/mycolors.mplsty   \
    --save "$FIGURES"/train_mse-"$REGUL_TYPE"-"$FEATURE_KIND".pdf
REGUL_TYPE=advtrain-l2
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.5,0.1,0.05,0.01} \
    --labels "$\delta = "{0.5,0.1,0.05,0.01}"$" --experiment_plot error_bars  \
    --plot_type train_mse --remove_bounds \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty  plot_style_files/mylegend.mplsty plot_style_files/mycolors.mplsty  \
    --save "$FIGURES"/train_mse-"$REGUL_TYPE"-"$FEATURE_KIND".pdf


# ---- Figure 7 ----
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
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.01,0.05,0.08,0.1,0.2,0.3,0.4} \
    --ord  $(rep inf 7) --experiment_plot error_bars \
    --plot_type norm --remove_bounds --y_scale linear \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty plot_style_files/mycolors.mplsty plot_style_files/mylegend.mplsty \
    --out_legend --labels "$\delta = "{0.01,0.05,0.08,0.1,0.2,0.3,0.4}"$" --out_legend_bbox_x 0.96  \
    --save "$FIGURES"/norm-"$REGUL_TYPE"-"$FEATURE_KIND".pdf

REGUL_TYPE=ridge
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.01,0.03,0.05,0.1,0.3,0.5,1.0} \
    --ord  $(rep inf 7) --experiment_plot error_bars \
    --plot_type norm --remove_bounds  --y_scale linear \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty plot_style_files/mycolors.mplsty plot_style_files/mylegend.mplsty  \
    --out_legend --labels "$\delta = "{0.01,0.03,0.05,0.1,0.3,0.5,1.0}"$" --out_legend_bbox_x 0.96 \
    --save "$FIGURES"/norm-"$REGUL_TYPE"-"$FEATURE_KIND".pdf

REGUL_TYPE=advtrain-linf
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.02,0.03,0.04,0.05,0.06,0.07,0.08} \
    --ord  $(rep inf 7) --experiment_plot error_bars \
    --plot_type norm --remove_bounds --y_scale linear \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty plot_style_files/mycolors.mplsty plot_style_files/mylegend.mplsty  \
    --out_legend --labels "$\delta = "{0.01,0.03,0.05,0.1,0.3,0.5,1.0}"$" --out_legend_bbox_x 0.96 \
    --save "$FIGURES"/norm-"$REGUL_TYPE"-"$FEATURE_KIND".pdf

REGUL_TYPE=lasso
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-{0.02,0.03,0.04,0.05,0.06,0.07,0.08} \
    --ord  $(rep inf 7) --experiment_plot error_bars  \
    --plot_type norm --remove_bounds  --y_scale linear \
    --plot_style $STYLE plot_style_files/stacked_bottom.mplsty plot_style_files/mycolors.mplsty plot_style_files/mylegend.mplsty  \
    --out_legend --labels "$\delta = "{0.01,0.03,0.05,0.1,0.3,0.5,1.0}"$" --out_legend_bbox_x 0.96 \
    --save "$FIGURES"/norm-"$REGUL_TYPE"-"$FEATURE_KIND".pdf


#  ---- Figure 8----
RESULTS=out/results/advtrain_new
FEATURE_KIND=latent
SCALING=sqrt
REGUL_TYPE=(advtrain-l2 ridge advtrain-linf lasso min-l2norm)
REG=(0.01 0.01 0.03 0.03 0.00)
for i in 0 1 2 3 4;
    do
    python linear-estimate.py --num_test_samples 200 --num_train_samples 200  --scaling $SCALING \
          --features_kind $FEATURE_KIND --signal_amplitude 1  --noise_std 0.1 -l -1 -u 2  \
          --ord 2 1.5 20 inf --eps 0.5 0.1 0.05 0.01 0.005 0.001 0.0 \
          --training ${REGUL_TYPE[$i]} \
          --regularization ${REG[$i]} \
          -o "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"${REGUL_TYPE[$i]}"-long
done;

python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-{min-l2norm,advtrain-l2,ridge,lasso,advtrain-linf}-long \
    --plot_type advrisk --eps $(rep 0.01 5)  --ord $(rep inf 5) --experiment_plot error_bars \
    --remove_bounds --second_marker_set --y_scale log --labels "min. norm" "adv. train $\ell_2$" "ridge"  "adv. train $\ell_{\infty}$" lasso  \
    --plot_style $STYLE plot_style_files/one_half4.mplsty plot_style_files/mycolors2.mplsty plot_style_files/mylegend.mplsty \
    --save "$FIGURES"/"$FEATURE_KIND"-advrisk-regularized.pdf


# Not show anywhere... Equivalent to supplement material Fig S10(b)
python linear-plot.py "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-{advtrain-l2,ridge,lasso,advtrain-linf,min-l2norm}-long \
    --plot_type advrisk --eps $(rep 0.0 5)  --ord $(rep inf 5) --experiment_plot error_bars \
    --remove_bounds --second_marker_set --y_scale log --labels {advtrain-l2,ridge,lasso,advtrain-linf} --labels "adv. train $\ell_2$" "ridge"  "adv. train $\ell_{\infty}$" lasso "min. norm" \
    --plot_style $STYLE plot_style_files/one_half4.mplsty plot_style_files/mylegend.mplsty \
    --save "$FIGURES"/"$FEATURE_KIND"-stdrisk-regularized.pdf --y_max 1.3 --y_min -3.5




###################
## MAGIC example ##
###################

wget http://mtweb.cs.ucl.ac.uk/mus/www/MAGICdiverse/MAGIC_diverse_FILES/BASIC_GWAS.tar.gz
tar -xvf BASIC_GWAS.tar.gz



# NOTE: Comparing with the original paper:
# running the script WEBSITE/genomic.prediction.r
# We can extract the metric (1-R^2).
#    = mean((test.pheno$HET_2- pred.rr[,1])^2) / mean((test.pheno$HET_2- mean(test.pheno$HET_2))^2)
# Which we can compute there and
# Notice that this is the metric we have been plotting in magic-merge-experiments.py
# The value obtained there is:
# - for 454/50 train/test split is:
#           ridge = 0.73784, lasso = 0.519179, elnet = 0.5166093
# - for 254/250 train/test split is:
#           ridge = 0.0.7933152, lasso = 0.5268, elnet = 0.5308153
# OBS: this may be different for different seeds...
# Since the cross validation and test set depend on the seed
# Generate multiple runs
for M in 1000 2000 4000 8000 16000;
do
  for SEED in 0 1 2 3 4;
  do
    python magic-estimate.py -o ./out/results/magic_m"$M"_seed"$SEED" -m $M -r $SEED --grid 100
  done;
done;

python magic-merge-experiments.py

#  Generate Fig. XX
python magic-plot.py


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
