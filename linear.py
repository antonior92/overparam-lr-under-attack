import numpy as np
from interp_under_attack.latent_space import compute_normalized_bias_and_variance


###########################
# Asymptotic computations #
###########################
def asymptotic_risk(proportion, signal_amplitude, noise_std, features_kind, off_diag, proportion_latent):
    # This follows from Hastie Thm.1 (p.7) and is the same regardless of the covariance matrix
    # Does not account for the noise
    if features_kind == 'latent':
        psi = (proportion_latent / proportion)
        b, v = compute_normalized_bias_and_variance(proportion, 1/psi)
        noise_std = np.sqrt(noise_std ** 2 + signal_amplitude**2 * psi / (1+psi))  # redefine noise std
    else:
        # The variance term
        v_underparametrized = proportion / (1 - proportion)
        v_overparametrized = 1 / (proportion - 1)
        v = (proportion < 1) * v_underparametrized + (proportion > 1) * v_overparametrized

        # The bias term
        b_underparametrized = 0
        if features_kind == 'isotropic':
            b_overparametrized = (1 - 1 / proportion)
        elif features_kind == 'equicorrelated':
            b_overparametrized = (1 - off_diag) * (1 - 1 / proportion)
        else:
            raise ValueError
        b = (proportion < 1) * b_underparametrized + (proportion > 1) * b_overparametrized

    return noise_std ** 2 * v + signal_amplitude ** 2 * b + noise_std ** 2


def assymptotic_l2_norm(proportion, signal_amplitude, noise_std, features_kind, off_diag, proportion_latent):
    if features_kind == 'latent':
        return np.ones_like(proportion)

    if features_kind == 'isotropic':
        v_underparametrized = proportion / (1 - proportion)
        v_overparametrized = 1 / (proportion - 1)
    elif features_kind == 'equicorrelated':
        v_underparametrized = proportion / ((1 - proportion) * (1 - off_diag))
        v_overparametrized = 1 / ((proportion - 1) * (1 - off_diag))
    else:
        raise ValueError
    v = (proportion < 1) * v_underparametrized + (proportion > 1) * v_overparametrized

    b_underparametrized = 1
    b_overparametrized = 1 / proportion
    b = (proportion < 1) * b_underparametrized + (proportion > 1) * b_overparametrized

    return np.sqrt(noise_std ** 2 * v + signal_amplitude ** 2 * b)


def assymptotic_l2_distance(proportion, signal_amplitude, noise_std, features_kind, off_diag, proportion_latent):
    if features_kind == 'latent':
        return np.ones_like(proportion)

    if features_kind == 'isotropic':
        v_underparametrized = proportion / (1 - proportion)
        v_overparametrized = 1 / (proportion - 1)
    elif features_kind == 'equicorrelated':
        v_underparametrized = proportion / ((1 - proportion) * (1 - off_diag))
        v_overparametrized = 1 / ((proportion - 1)* (1 - off_diag))
    else:
        raise ValueError
    v = (proportion < 1) * v_underparametrized + (proportion > 1) * v_overparametrized

    b_underparametrized = 0
    b_overparametrized = 1 - 1 / proportion
    b = (proportion < 1) * b_underparametrized + (proportion > 1) * b_overparametrized

    return np.sqrt(noise_std ** 2 * v + signal_amplitude ** 2 * b)


def lp_norm_bounds(ord, sz):
    # Generalize to other norms,
    # using https://math.stackexchange.com/questions/218046/relations-between-p-norms
    if ord == np.inf:
        factor = sz ** (1.0/2.0)
    else:
        factor = sz ** (1.0/2.0-1.0/ord)

    lfactor = 1 if ord >= 2 else factor
    ufactor = 1 if ord <= 2 else factor

    return lfactor, ufactor


def adversarial_bounds(arisk, anorm, eps, ord, n_features):
    lb, ub = lp_norm_bounds(ord, n_features)

    upper_bound = (np.sqrt(arisk) + eps * ub * anorm)**2
    lower_bound = arisk + (eps * lb * anorm) ** 2

    return lower_bound, upper_bound

