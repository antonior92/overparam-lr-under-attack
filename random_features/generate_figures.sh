#!/bin/bash

echo "Create directories..."
mkdir results  # if it does not exist already
mkdir figures  # if it does not exist already
STYLE="ggplot ../mystyle.mplsty"
export PYTHONPATH="${PYTHONPATH}::../"


echo "Generating Figure..."
python random_feature_regression.py -e 0.0 0.1 0.5 1.0 2.0 -o results/test.csv -n 10
python plot_double_descent.py --file results/test.csv --plot_style $STYLE --save figures/test
