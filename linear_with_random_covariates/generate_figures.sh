echo "Create directories..."
mkdir results  # if it does not exist already
mkdir figures  # if it does not exist already
STYLE="ggplot ../mystyle.mplsty"
export PYTHONPATH="${PYTHONPATH}::../"


echo "Generate data for Figures 1..."
for ORD in 2 inf;
do
  python adversarial_risk.py --num_test_samples 100 --num_train_samples 100 -o results/l"$ORD"-worst-case.csv \
    --features_kind equicorrelated --ord $ORD --datagen_param constant -e 0.1
  python plot_double_descent.py --file results/l"$ORD"-worst-case.csv
done;



echo "Generate data for Figures 2..."
python adversarial_risk.py --num_test_samples 100 --num_train_samples 100 -o results/isotropic-gaussian-prior.csv --ord 1.5 2 20
python plot_double_descent.py --file results/isotropic-gaussian-prior.csv --save figures/isotropic-gaussian-prior

