
for NORM in 1 1.5 2 20 inf;
do
python generate_2dregions.py --norm $NORM --plot_style style.mplsyt --save l"$NORM".png
done;
