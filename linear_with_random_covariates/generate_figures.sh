echo "Create directories..."
mkdir results  # if it does not exist already
mkdir figures  # if it does not exist already
STYLE="../mystyle.mplsty"
export PYTHONPATH="${PYTHONPATH}::../"


echo "Generate data for Figures 1..."
python adversarial_risk.py --num_test_samples 100 --num_train_samples 100 -o results/equicorrelated-constant.csv \
    --features_kind equicorrelated --ord 2 inf --datagen_param constant -e 0.1 -u 1.5
python plot_double_descent.py --file results/equicorrelated-constant.csv --save figures/equicorrelated-constant --plot_style $STYLE --plot_type risk_per_eps


echo "Generate data for Figures 2..."
python adversarial_risk.py --num_test_samples 100 --num_train_samples 100 -o results/isotropic-gaussian-prior.csv --ord 1.5 2 20
python plot_double_descent.py --file results/isotropic-gaussian-prior.csv --save figures/isotropic-gaussian-prior --plot_style $STYLE

