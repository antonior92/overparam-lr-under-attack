# interpolators-under-attack

```bash
for N in 100 300 500 700;
do
  python adversarial_risk.py --ord 2 --num_train_samples $N --num_test_samples 700
  python plot_double_descent.py --plot_style ggplot mystyle.mplsty --save plot_"$N".png --y_min 1e-1 --y_max 1e3 
done;
```