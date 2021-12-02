# Run all experiments and generate all figures
mkdir out
mkdir out/results  # if it does not exist already
mkdir out/figures  # if it does not exist already
RESULTS=out/results
FIGURES=out/figures
STYLE="plot_style_files/mystyle.mplsty"

N=100
for REGUL_TYPE in advtrain-l2 advtrain-linf ridge;
do
  for REG in 10 1 0.1 0.01 0.001;
  do python linear-estimate.py --num_test_samples $N --num_train_samples $N \
      --features_kind isotropic --signal_amplitude 4 --training $REGUL_TYPE --regularization $REG \
      -o out/results/"$REGUL_TYPE"-"$REG"
  done;
done;


#python linear-plot.py out/results/"$REGUL_TYPE"-{10,1,0.1,0.01,0.001} --ord 2 2 2 2 2 --eps 0.5 0.5 0.5 0.5 0.5 \
#  --plot_type advrisk --remove_bounds --labels {10,1,0.1,0.01,0.001}
#python linear-plot.py out/results/"$REGUL_TYPE"-{10,1,0.1,0.01,0.001} --ord 2 2 2 2 2 --eps 0.5 0.5 0.5 0.5 0.5 \
#  --plot_type train_mse --remove_bounds --labels {10,1,0.1,0.01,0.001}