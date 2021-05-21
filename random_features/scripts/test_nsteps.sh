#!/bin/bash

echo "Create directories..."
mkdir results  # if it does not exist already
mkdir figures  # if it does not exist already
STYLE="ggplot"
export PYTHONPATH="${PYTHONPATH}::../"



for NSTEPS in 110 200 400;
do
  echo $NSTEPS
  python random_feature_regression.py -e 0.0 0.1 0.5 1.0 2.0 -o results/n_steps_$NSTEPS.csv -n 10 -u 0.5 --n_adv_steps $NSTEPS
  python plot_double_descent.py --file results/n_steps_$NSTEPS.csv --plot_style $STYLE --save figures/n_steps_$NSTEPS
done;