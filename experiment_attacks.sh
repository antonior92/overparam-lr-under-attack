# Run all experiments and generate all figures
mkdir out
mkdir out/results  # if it does not exist already
mkdir out/results/advtrain
mkdir out/figures  # if it does not exist already
mkdir out/figures/advtrain
RESULTS=out/results
FIGURES=out/figures/advtrain
STYLE="plot_style_files/mystyle.mplsty"

rep(){
	for i in $(seq "$2");
	do echo -n "$1"
	 echo -n " "
	done
}

# This is now running on hyperion:
N=100
SIGNAL_AMPLITUDE=4
for FEATURE_KIND in latent isotropic;
do
  for SCALING in sqrt sqrtlog none;
  do
    python linear-estimate.py --num_test_samples $N --num_train_samples $N \
      --features_kind $FEATURE_KIND --signal_amplitude $SIGNAL_AMPLITUDE --ord 2 1.5 20 inf \
      -o "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-noreg -l -0.5
    for REGUL_TYPE in advtrain-l2 advtrain-linf ridge;
    do
      for REG in 1 0.5 0.1 0.05 0.01;
      do python linear-estimate.py --num_test_samples $N --num_train_samples $N \
          --features_kind $FEATURE_KIND --signal_amplitude $SIGNAL_AMPLITUDE --training $REGUL_TYPE --regularization $REG \
          -o "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-"$REG" -l -0.5
      done;
    done;
  done;
done;

# TODO: update plot
for REGUL_TYPE in advtrain-l2 ridge;
do
  PLT_TYPE=advrisk
  for EPS in 0 0.1 0.5 1.0 2.0;
  do python linear-plot.py out/results/"$REGUL_TYPE"-{1,0.1,0.5,0.01,0.05} out/results/control-noreg --ord $(rep 2 6) \
      --eps  $(rep $EPS 6)  --labels {1,0.5,0.1,0.05,0.01}  noreg --experiment_plot median_line\
      --plot_type $PLT_TYPE --remove_bounds --save $FIGURES/"$PLT_TYPE"-"$REGUL_TYPE"-e"$EPS".pdf
  done;
  PLT_TYPE="train_mse"
  python linear-plot.py out/results/"$REGUL_TYPE"-{1,0.1,0.5,0.01,0.05} out/results/control-noreg --ord $(rep 2 6) \
      --eps  $(rep $EPS 6)  --labels {1,0.5,0.1,0.05,0.01}  noreg --experiment_plot median_line\
      --plot_type $PLT_TYPE --remove_bounds --save $FIGURES/"$PLT_TYPE"-"$REGUL_TYPE".pdf
done;

REGUL_TYPE=advtrain-linf
PLT_TYPE=advrisk
for EPS in 0 0.1 0.5 1.0 2.0;
do python linear-plot.py out/results/"$REGUL_TYPE"-{10,1,0.1,0.01,0.001} out/results/control-noreg --ord $(rep 2 6) \
    --eps  $(rep $EPS 6)  --labels {10,1,0.1,0.01,0.001} noreg --experiment_plot median_line\
    --plot_type $PLT_TYPE --remove_bounds --save $FIGURES/"$PLT_TYPE"-"$REGUL_TYPE"-e"$EPS".pdf
done;
PLT_TYPE="train_mse"
python linear-plot.py out/results/"$REGUL_TYPE"-{10,1,0.1,0.01,0.001} out/results/control-noreg --ord $(rep 2 6) \
    --eps  $(rep $EPS 6)  --labels {10,1,0.1,0.01,0.001}  noreg --experiment_plot median_line\
    --plot_type $PLT_TYPE --remove_bounds --save $FIGURES/"$PLT_TYPE"-"$REGUL_TYPE".pdf