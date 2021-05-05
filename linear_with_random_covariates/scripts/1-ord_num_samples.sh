#!/bin/bash


N_PTS=60
DIR=results-1
mkdir $DIR

for P in 1 2 5 10 inf;
do
    for N in 100 200 300;
    do
        python adversarial_risk.py --ord $P -n $N_PTS --num_train_samples $N -o $DIR/perfomance_${P}_${N}.csv
    done;
done;