# Implementation
- [x] Add p = inf to compute adv attack
- [x] Add p = 1 to compute adv attack
- [x] Allow number of test samples to be different from the number of training samples
- [x] Double check the boundaries for the case there is no attack
- [x] Add asymptotics for other p-norms
- [ ] Plot parameter norm
- [ ] Double check the bounds. Sometimes the upper bound does not seems like 
      an upper bound. Increase the sample size even further to be able to draw
      conclusions.
- [ ] The assumption in plot_double_descent.py that ord, snr... 
      are constant is not very elegant... Fix it latter...

-----

# Experiments
- [x] For different p-norms see what is the effect of changing the number of samples. By the derived upper bound,
      for p != 2 the risk might depend not only on the ratio (num_parameters / num_samples) but also on the 
      (num_parameters). With the risk increasing with it for p > 2 and decreasing wiht it for p < 2.
  

----

# Now running
- [x] Different orders and different training sizes (hydra3:10014.interpolators-under-attack)