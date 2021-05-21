#!/bin/bash

echo "Create directories..."
mkdir results  # if it does not exist already
mkdir figures  # if it does not exist already
STYLE="ggplot"
export PYTHONPATH="${PYTHONPATH}::../"



for SNR in 0.1 1 10 100 1000 10000 100000;
do
  echo $SNR
  python random_feature_regression.py -e 0.0 0.1 0.5 1.0 2.0 -o results/snr_$SNR.csv -n 10 -u 0.5 --snr $SNR
  python plot_double_descent.py --file results/snr_$SNR.csv --plot_style $STYLE --save figures/snr_$SNR
done;