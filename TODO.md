# Implementation
- [x] Add p = inf to compute adv attack
- [x] Add p = 1 to compute adv attack
- [x] Allow number of test samples to be different from the number of training samples

-----

# Experiments
- [ ] For different p-norms see what is the effect of changing the number of samples. By the derived upper bound,
      for p != 2 the risk might depend not only on the ratio (num_parameters / num_samples) but also on the 
      (num_parameters). With the risk increasing with it for p > 2 and decreasing wiht it for p < 2.