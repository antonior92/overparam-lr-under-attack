#!/bin/bash

echo "Create directories..."
mkdir results  # if it does not exist already
mkdir figures  # if it does not exist already
STYLE="ggplot ../mystyle.mplsty"
export PYTHONPATH="${PYTHONPATH}::../"


for $ORD in 2 inf;
do
echo "Generating Figure..."
python empirical_experiment.py --ord 2
python plot_double_descent.py
done;