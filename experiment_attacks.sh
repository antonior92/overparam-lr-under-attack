# Run all experiments and generate all figures
mkdir out
mkdir out/results  # if it does not exist already
mkdir out/results/advtrain
mkdir out/figures  # if it does not exist already
mkdir out/figures/advtrain
RESULTS=out/results/advtrain
FIGURES=out/figures/advtrain

# This is now running on hyperion:636772.experiment_adversarial_training
N=100
SIGNAL_AMPLITUDE=4
for FEATURE_KIND in latent isotropic;
do
  for SCALING in sqrt sqrtlog none;
  do
    python linear-estimate.py --num_test_samples $N --num_train_samples $N  --scaling $SCALING \
      --features_kind $FEATURE_KIND --signal_amplitude $SIGNAL_AMPLITUDE --ord 2 1.5 20 inf -l -0.5 \
      -o "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-noreg
    for REGUL_TYPE in advtrain-l2 advtrain-linf ridge;
    do
      for REG in 1 0.5 0.1 0.05 0.01;
      do python linear-estimate.py --num_test_samples $N --num_train_samples $N --scaling $SCALING  \
          --features_kind $FEATURE_KIND --signal_amplitude $SIGNAL_AMPLITUDE --training $REGUL_TYPE --regularization $REG \
          --ord 2 1.5 20 inf -l -0.5  -o "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-"$REG"
      done;
    done;
  done;
done;

# Prepare for plotting
STYLE="plot_style_files/mystyle.mplsty"

rep(){
	for i in $(seq "$2");
	do echo -n "$1"
	 echo -n " "
	done
}

# TODO: update plot
for FEATURE_KIND in latent isotropic;
do
  for SCALING in sqrt sqrtlog none;
  do
    NN="$RESULTS"/"$SCALING"-"$FEATURE_KIND"
    for REGUL_TYPE in advtrain-l2 advtrain-linf ridge;
    do

      for ORD in 2 inf;
      do
        PLT_TYPE=advrisk
        for EPS in 0 0.1 0.5 1.0 2.0;
        do
            python linear-plot.py "$NN"-"$REGUL_TYPE"-{0.1,0.5,0.01,0.05}  "$NN"-noreg --ord $(rep $ORD 6) \
           --eps  $(rep $EPS 6)  --labels {1,0.5,0.1,0.05}  noreg --experiment_plot median_line \
           --plot_type $PLT_TYPE --remove_bounds --save $FIGURES/risk-e"$EPS"o"$ORD"-"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-.pdf
        done;
        PLT_TYPE='norm'
        python linear-plot.py "$NN"-"$REGUL_TYPE"-{0.1,0.5,0.01,0.05} "$NN"-noreg --ord $(rep $ORD 6) \
            --labels {1,0.5,0.1,0.05}  noreg --experiment_plot median_line \
            --plot_type $PLT_TYPE --remove_bounds  --save $FIGURES/norm"$ORD"-"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE".pdf
      done
      PLT_TYPE="train_mse"
      python linear-plot.py "$NN"-"$REGUL_TYPE"-{0.1,0.5,0.01,0.05} "$NN"-noreg --ord $(rep $ORD 6) \
          --eps  $(rep $EPS 6)  --labels {0.1,0.5,0.01,0.05}  noreg --experiment_plot median_line \
          --plot_type $PLT_TYPE --remove_bounds --save $FIGURES/train_mse-"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE".pdf
    done;
  done;
done;

