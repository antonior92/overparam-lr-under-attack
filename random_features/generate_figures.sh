#!/bin/bash

echo "Create directories..."
mkdir results  # if it does not exist already
mkdir figures  # if it does not exist already
STYLE="../mystyle.mplsty"
export PYTHONPATH="${PYTHONPATH}::../"


echo "Generating Figure 3..."
python random_feature_regression.py -n 60 -r 4 -u 1 -l -1 --epsilon 0 0.1 1.0 --ord 2 --noise_std 0 -o results/l2-random-feature.csv --fixed nfeatures_over_inputdim --datagen_parameter constant
python plot_double_descent.py --file results/l2-random-feature.csv --plot_style $STYLE ../one_half.mplsty --save figures/l2-random-feature.pdf

