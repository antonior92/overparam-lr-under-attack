#!/bin/bash

echo "Create directories..."
mkdir results  # if it does not exist already
mkdir figures  # if it does not exist already
STYLE="ggplot ../mystyle.mplsty"
export PYTHONPATH="${PYTHONPATH}::../"


for $ORD in 2;
do
echo "Generating Figure..."
python random_feature_regression.py -n 20 -u 1 -l -1 --epsilon 0.1 --ord $ORD --noise_std 0 -o results/l"$ORD"-attack.csv
python plot_double_descent.py --file results/l"$ORD"-attack.csv --plot_style $STYLE --save figures/l"$ORD"-attack
done;