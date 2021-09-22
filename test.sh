rm testresult.csv
python estimate_advrisk_linear.py --num_test_samples 100 --num_train_samples 100 -o testresult.csv \
    --features_kind isotropic --ord 2 --datagen_param gaussian_prior -e 0 -u 1.3 -r 1 -n 15 --signal_amplitude 10 --off_diag 0.5

python plot_linear.py --file testresult.csv  --plot_style $STYLE \
  plot_style_files/one_half.mplsty plot_style_files/mycolors.mplsty   --plot_type risk_per_eps

python plot_linear.py --file testresult.csv  --plot_style $STYLE \
  plot_style_files/one_half.mplsty plot_style_files/mycolors.mplsty   --plot_type norm

python plot_linear.py --file testresult.csv  --plot_style $STYLE \
  plot_style_files/one_half.mplsty plot_style_files/mycolors.mplsty   --plot_type distance