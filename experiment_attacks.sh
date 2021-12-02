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

N=100
python linear-estimate.py --num_test_samples $N --num_train_samples $N \
      --features_kind isotropic --signal_amplitude 4 \
      -o out/results/control-noreg

N=100
for REGUL_TYPE in advtrain-l2 advtrain-linf ridge;
do
  for REG in 0.5 0.05; #10 1  0.1 0.05 0.01 0.001;
  do python linear-estimate.py --num_test_samples $N --num_train_samples $N \
      --features_kind isotropic --signal_amplitude 4 --training $REGUL_TYPE --regularization $REG \
      -o out/results/"$REGUL_TYPE"-"$REG"
  done;
done;


python linear-plot.py out/results/control-noreg out/results/advtrain-l2-{1,0.1,0.5,0.01,0.05} --ord $(rep 2 6) \
    --eps  $(rep 2.0 6)  --labels noreg {1,0.5,0.1,0.05,0.01} --experiment_plot error_bars \
    --plot_type advrisk --remove_bounds
python linear-plot.py out/results/control-noreg out/results/advtrain-l2-{1,0.1,0.5,0.01,0.05} --ord $(rep 2 6) \
    --eps  $(rep 2.0 6)  --labels noreg {1,0.5,0.1,0.05,0.01} --experiment_plot error_bars \
     --plot_type train_mse

python linear-plot.py out/results/control-noreg out/results/ridge-{1,0.1,0.5,0.01,0.05} --ord $(rep 2 6) \
    --eps  $(rep 2.0 6)  --labels noreg {1,0.5,0.1,0.05,0.01} --experiment_plot error_bars \
    --plot_type advrisk --remove_bounds
python linear-plot.py out/results/ridge-{1,0.1,0.5,0.01,0.05} --ord $(rep 2 5) \
    --eps  $(rep 2.0 5)  --labels {1,0.5,0.1,0.05,0.01} --experiment_plot error_bars \
     --plot_type train_mse

python linear-plot.py out/results/control-noreg out/results/advtrain-linf-{10,1,0.1,0.01,0.001} --ord $(rep 2 6) \
    --eps  $(rep 2.0 6)  --labels noreg {10,1,0.1,0.01,0.001} --experiment_plot error_bars \
    --plot_type advrisk --remove_bounds
python linear-plot.py out/results/control-noreg out/results/advtrain-linf-{10,1,0.1,0.01,0.001} --ord $(rep 2 6) \
    --eps  $(rep 2.0 6)  --labels noreg {10,1,0.1,0.01,0.001} --experiment_plot error_bars \
     --plot_type train_mse