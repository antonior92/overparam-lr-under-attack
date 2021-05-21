#!/bin/bash

echo "Create directories..."
mkdir results  # if it does not exist already
mkdir figures  # if it does not exist already
STYLE="ggplot"
export PYTHONPATH="${PYTHONPATH}::../"



for REGUL in 1e-5 1e-6 1e-7 1e-8;
do
  echo $REGUL
  python random_feature_regression.py -e 0.0 0.1 0.5 1.0 2.0 -o results/regul_$REGUL.csv -n 10 -u 0.5 --regularization $REGUL --fixed_proportion 2.0
  python plot_double_descent.py --file results/regul_$REGUL.csv --plot_style $STYLE --save figures/regul_$REGUL
done;