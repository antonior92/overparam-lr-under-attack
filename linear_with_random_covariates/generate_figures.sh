echo "Create directories..."
mkdir results  # if it does not exist already
mkdir figures  # if it does not exist already
STYLE="ggplot ../mystyle.mplsty"
export PYTHONPATH="${PYTHONPATH}::../"

echo "Generate data for Figures 1 and 2..."
for ORD in 2 1.5 20 inf;
do for TP in isotropic equicorrelated;
  do
    python adversarial_risk.py --num_test_samples 100 --num_train_samples 100 -o results/l"$ORD"-"$TP".csv --features_kind $TP --ord $ORD
    python plot_double_descent.py --file results/l"$ORD"-"$TP".csv --plot_style $STYLE --y_max 1e5 --save figures/l"$ORD"-"$TP"
  done;
done;
