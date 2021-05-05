#!/bin/bash


files=$@
for f in $files
do
    echo "file =  $f"
    python plot_double_descent.py --file $f --save $f.png --y_min 0.5 --y_max 1e6
done