# Run all experiments and generate all figures
mkdir out
mkdir out/results  # if it does not exist already
mkdir out/results/advtrain_new
RESULTS=out/results/advtrain_new

# This is now running on hyperion:410452.advtrain
COMMON_PARAMS="--ord 2 1.5 20 inf --eps 0.5 0.1 0.05 0.01 0.005 0.001 0.0  --num_test_samples 200 --num_train_samples 200"
for FEATURE_KIND in latent ; # isotropic
do
  if [ $FEATURE_KIND == latent ]; # Set parameter
  then
     PP="--signal_amplitude 1 --noise_std 0.1 -l -0.5"
  else
     PP="--signal_amplitude 4 --noise_std 1 "
  fi;
  for SCALING in sqrt; # sqrtlog # none
  do
    python linear-estimate.py $PP $COMMON_PARAMS --scaling $SCALING --features_kind $FEATURE_KIND \
      -o "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-noreg
    for REGUL_TYPE in advtrain-l2 advtrain-linf ridge lasso;
    do

      case $REGUL_TYPE in
        advtrain-l2)
          ALL_REGS="1.0 0.6 0.5 0.4 0.3 0.1" #"0.09 0.08 0.07 0.06 0.05 0.03 0.01"
          ;;
        ridge)
          ALL_REGS="1.0 0.7 0.5" #"0.5 0.3 0.1 0.05 0.03 0.01 0.008"
          ;;
        advtrain-linf)
          ALL_REGS="0.5 0.1 0.09 0.08 0.07 0.06 0.05 0.04" #"0.03 0.02 0.01 0.008 0.007 0.006 0.005"
          ;;
        lasso)
          ALL_REGS="0.5 0.03 0.1 0.09 0.08 0.07 0.06 0.05" #"0.04 0.03 0.02 0.01 0.008 0.007 0.005"
          ;;
        *)
          echo -n "unknown"
          ;;
      esac

      for REG in $ALL_REGS;
      do python linear-estimate.py  $PP $COMMON_PARAMS  --scaling $SCALING  --features_kind $FEATURE_KIND \
       --training $REGUL_TYPE --regularization $REG \
        -o "$RESULTS"/"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE"-"$REG"
      done;
    done;
  done;
done;


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
    for REGUL_TYPE in advtrain-l2 advtrain-linf ridge lasso;
    do
      for ORD in 2 inf;
      do
        PLT_TYPE=advrisk
        for EPS in 0 0.1 0.5 1.0 2.0;
        do
            python linear-plot.py "$NN"-"$REGUL_TYPE"-{0.1,0.5,0.01,0.05}  "$NN"-noreg --ord $(rep $ORD 6) \
           --eps  $(rep $EPS 6)  --labels {1,0.5,0.1,0.05}  noreg --experiment_plot median_line \
           --plot_type $PLT_TYPE --remove_bounds --save $FIGURES/risk-e"$EPS"o"$ORD"-"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE".pdf
        done;
        PLT_TYPE='norm'
        python linear-plot.py "$NN"-"$REGUL_TYPE"-{0.1,0.5,0.01,0.05} "$NN"-noreg --ord $(rep $ORD 6) \
            --labels {1,0.5,0.1,0.05}  noreg --experiment_plot median_line \
            --plot_type $PLT_TYPE --remove_bounds  --save $FIGURES/norm-o"$ORD"-"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE".pdf
      done
      PLT_TYPE="train_mse"
      python linear-plot.py "$NN"-"$REGUL_TYPE"-{0.1,0.5,0.01,0.05} "$NN"-noreg \
          --eps  $(rep $EPS 6)  --labels {0.1,0.5,0.01,0.05}  noreg --experiment_plot median_line \
          --plot_type $PLT_TYPE --remove_bounds --save $FIGURES/train_mse-"$SCALING"-"$FEATURE_KIND"-"$REGUL_TYPE".pdf
  done;
done;
done;
