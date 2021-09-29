mkdir figures  # if it does not exist already
STYLE="plot_style_files/mystyle.mplsty"

echo "Generate Figures 1..."
python plot_linear.py --file results/equicorrelated-constant.csv  --plot_style $STYLE plot_style_files/one_half.mplsty plot_style_files/mycolors.mplsty   --plot_type risk_per_eps  --plot_type risk_per_eps --remove_bounds --second_marker_set --save figures/equicorrelated-constant.pdf

echo "Generate Figures 2..."
python plot_linear.py --file results/equicorrelated0.9-gaussian-prior.csv --save figures/equicorrelated-gaussian-prior-l2.pdf --plot_style $STYLE plot_style_files/one_half.mplsty  --ord 2 --y_min -0.2 --y_max 4.6

echo "Generate Figures 3..."
python plot_linear.py --file results/isotropic-gaussian-prior.csv --save figures/isotropic-gaussian-prior-l1.5.pdf --plot_style $STYLE plot_style_files/one_third_with_ylabel.mplsty --ord 1.5 --y_min -0.2 --y_max 4.6
python plot_linear.py --file results/isotropic-gaussian-prior.csv --save figures/isotropic-gaussian-prior-l2.pdf --plot_style $STYLE plot_style_files/one_third_without_ylabel.mplsty --remove_ylabel --remove_legend --ord 2 --y_min -0.2 --y_max 4.6
python plot_linear.py --file results/isotropic-gaussian-prior.csv --save figures/isotropic-gaussian-prior-l20.pdf --plot_style $STYLE plot_style_files/one_third_without_ylabel.mplsty  --remove_ylabel --remove_legend --ord 20 --y_min -0.2 --y_max 4.6

echo "Generating Figure *..."
python plot_rand_features.py --file results/l2-random-feature.csv --plot_style $STYLE plot_style_files/one_half.mplsty --save figures/l2-random-feature.pdf
