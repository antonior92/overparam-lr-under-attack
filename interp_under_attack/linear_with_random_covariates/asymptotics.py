import numpy as np


def asymptotic_risk(proportion, signal_amplitude, noise_std, features_kind, off_diag):
    # This follows from Hastie Thm.1 (p.7) and is the same regardless of the covariance matrix

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
    b = (proportion < 1) * b_underparametrized + (proportion > 1) * b_overparametrized

    return noise_std ** 2 * v + signal_amplitude ** 2 * b


def assymptotic_l2_norm_squared(proportion, signal_amplitude, noise_std, features_kind, off_diag):

    if features_kind == 'isotropic':
        v_underparametrized = proportion / (1 - proportion)
        v_overparametrized = 1 / (proportion - 1)
    elif features_kind == 'equicorrelated':
        v_underparametrized = proportion / ((1 - proportion) * (1 - off_diag))
        v_overparametrized = 1 / ((proportion - 1)* (1 - off_diag))
    v = (proportion < 1) * v_underparametrized + (proportion > 1) * v_overparametrized

    b_underparametrized = 1
    b_overparametrized = 1 / proportion
    b = (proportion < 1) * b_underparametrized + (proportion > 1) * b_overparametrized

    return noise_std ** 2 * v + signal_amplitude ** 2 * b


def l2_distance_between_parameters(proportion, signal_amplitude, noise_std, features_kind, off_diag):
    if features_kind == 'isotropic':
        v_underparametrized = proportion / (1 - proportion)
        v_overparametrized = 1 / (proportion - 1)
    elif features_kind == 'equicorrelated':
        v_underparametrized = proportion / ((1 - proportion) * (1 - off_diag))
        v_overparametrized = 1 / ((proportion - 1)* (1 - off_diag))
    v = (proportion < 1) * v_underparametrized + (proportion > 1) * v_overparametrized

    b_underparametrized = 0
    b_overparametrized = 1 - 1 / proportion
    b = (proportion < 1) * b_underparametrized + (proportion > 1) * b_overparametrized

    return noise_std ** 2 * v + signal_amplitude ** 2 * b


def lp_norm_squared(l2_norm_squared, ord, sz):
    # Generalize to other norms,
    # using https://math.stackexchange.com/questions/218046/relations-between-p-norms
    if ord == np.inf:
        factor = sz ** 1/2
    else:
        factor = sz ** (1/2-1/ord)

    lfactor = 1 if ord >= 2 else factor
    ufactor = 1 if ord <= 2 else factor

    lower_bound = l2_norm_squared * lfactor ** 2
    upper_bound = l2_norm_squared * ufactor ** 2

    return lower_bound, upper_bound


def adversarial_bounds(arisk, anorm, noise_std, eps, ord, n_features, datagen_parameter):
    lqnorm_lb, lqnorm_ub = lp_norm_squared(anorm, ord, n_features)

    upper_bound = (np.sqrt(arisk) + eps * np.sqrt(lqnorm_ub))**2 + noise_std ** 2
    lower_bound = arisk + eps**2 * lqnorm_lb + noise_std ** 2

    return lower_bound, upper_bound