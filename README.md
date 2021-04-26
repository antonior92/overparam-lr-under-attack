# interpolators-under-attack


Figure 1 (a)
```bash
python adversarial_risk.py --num_test_samples 100 --num_train_samples 100 -o results/l2-isotropic.csv
python plot_double_descent.py --file results/l2-isotropic.csv --plot_style ggplot mystyle.mplsty --y_max 1e5 --save figures/l2-isotropic.png
```

Figure 1 (b)
```bash
python adversarial_risk.py --num_test_samples 100 --num_train_samples 100 -f equicorrelated --off_diag 0.5 -o results/l2-equicorrelated.csv
python plot_double_descent.py --file results/l2-equicorrelated.csv --plot_style ggplot mystyle.mplsty --y_max 1e5 --save figures/l2-equicorrelated.png
```

Figure 2 (a)
```bash
python adversarial_risk.py --num_test_samples 100 --num_train_samples 100 -o results/l1.5-isotropic.csv --ord 1.5
python plot_double_descent.py --file results/l1.5-isotropic.csv --plot_style ggplot mystyle.mplsty --y_max 1e5 --save figures/l1.5-isotropic.png --save_norm_plot figures/l1.5-isotropic-norm.png
```

Figure 2 (c)
```bash
python adversarial_risk.py --num_test_samples 100 --num_train_samples 100 -o results/l20-isotropic.csv --ord 20
python plot_double_descent.py --file results/l20-isotropic.csv --plot_style ggplot mystyle.mplsty --y_max 1e5 --save figures/l20-isotropic.png --save_norm_plot figures/l20-isotropic-norm.png
```

Figure 3(a)
```
python adversarial_risk.py --num_test_samples 100 --num_train_samples 100 -o results/linf-equicorrelated.csv --ord inf 
python plot_double_descent.py --file results/linf-equicorrelated.csv --plot_style ggplot mystyle.mplsty --save figures/linf-isotropic.png --save_norm_plot figures/linf-isotropic-norm.png
```
