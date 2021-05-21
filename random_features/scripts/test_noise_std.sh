#!/bin/bash

echo "Create directories..."
mkdir results  # if it does not exist already
mkdir figures  # if it does not exist already
STYLE="ggplot"
export PYTHONPATH="${PYTHONPATH}::../"



for NOISE_STD in 0.1 0.01 0.001 0.0001;
do
  echo $NOISE_STD
  python random_feature_regression.py -e 0.0 0.1 0.5 1.0 2.0 -o results/noise_std_$NOISE_STD.csv -n 10 -u 0.5 --snr 100 --noise_std $NOISE_STD
  python plot_double_descent.py --file results/noise_std_$NOISE_STD.csv --plot_style $STYLE --save figures/noise_std_$NOISE_STD
done;