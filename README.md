# Overparametrized Linear Regresiong under Adversarial Attacks

Companion code to the paper:
```
“Overparameterized Linear Regression under Adversarial Attacks,” A. H. Ribeiro and T. B. Schön. IEEE Transactions on Signal Processing, 2023, doi: 10.1109/TSP.2023.3246228.
```
Links: [arXiv](https://arxiv.org/abs/2204.06274) - [IEEE](https://ieeexplore.ieee.org/document/10048547)


## Running instructions
The bash script `run.sh` runs all the experiments described in the paper. It includes comments indicating the command used to generate each figure. I would suggest running the ones you are interested in (and not the full script, since it can be quite long).

The folders `results/` and `figures/` will be created. Experiments are typically run in two stages. First, a compute-intensive step is executed, and the resulting experiments are saved in `.csv` format inside the `results/` folder.
Later, a plotting function reads these files and generates the corresponding figures (in `.pdf` format), which are saved in the `figures/` folder.

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




