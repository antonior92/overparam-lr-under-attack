# Overparametrized Linear Regresiong under Adversarial Attacks

Companion code to the paper:
```
“Overparameterized Linear Regression under Adversarial Attacks,” A. H. Ribeiro and T. B. Schön. IEEE Transactions on Signal Processing, 2023, doi: 10.1109/TSP.2023.3246228.
```
Links: [arXiv](https://arxiv.org/abs/2204.06274) - [IEEE](https://ieeexplore.ieee.org/document/10048547)


## Running instructions
- To run the experiments described in the paper run:
```bash
sh run_experiments.sh
```
The folder `results/` will be created and the results of the experiments
will be saved in `.csv` format.

- To generate the figures
```bash
sh generate_figures.sh
```
It read the results of the empirical experiments from `results/` and use them to generate
the figures. The folder `figures/` will be created and the results of the experiments
will be saved in `.pdf` format.

## Requirements
This code was tested on python 3.7. It uses numpy, sympy, pandas
tqdm, scipy, matplotlib. Numpy should have a version equal to or 
greater then 1.10.0.

We use tex to generate the plots. If you want to turn it down, just
comment out the lines:
```python
text.usetex: True
text.latex.preamble: \usepackage{newtxmath}
```
in `plot_style_file/mystyle.mplsty`.




