#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 10:06:55 2021

"""

import numpy as np
import scipy
import scipy.stats


def compute_moments(
    log_likelihood_ratio_func, dens_func, max_order, left=-np.inf, right=np.inf
):
    """
    Compute the moments of the log likelihood ratio up to order n
    """

    moments = [0] * max_order
    errs = [0] * max_order
    for ii in range(max_order):
        order = ii + 1

        def integrand(x):
            return log_likelihood_ratio_func(x) ** order * dens_func(x)

        moments[ii], errs[ii] = scipy.integrate.quad(integrand, left, right, 
                                 epsabs = 1e-15, epsrel = 1e-15, limit = 100)
    return moments, errs


def compute_cumulants(moments, order = 2):
    """
    Compute the cumulants from the moments up to order 4
    """
    assert len(moments) >= 4, "You must have moments at least up to order 4."
    kappas = [0] * 4
    if order == 3:
        kappas = [0] * 5
    
    kappas[0] = moments[0]
    kappas[1] = moments[1] - moments[0] ** 2
    kappas[2] = moments[2] - 3 * moments[1] * moments[0] + 2 * moments[0] ** 3
    kappas[3] = (
        moments[3]
        - 4 * moments[2] * moments[0]
        - 3 * moments[1] ** 2
        + 12 * moments[1] * moments[0] ** 2
        - 6 * moments[0] ** 4
    )
    if order == 3:
        kappas[4] = (
            moments[4]
            - 5 * moments[3] * moments[0]
            - 10 * moments[2] * moments[1]
            + 20 * moments[2] * moments[0] ** 2
            + 30 * moments[1] ** 2 * moments[0]
            - 60 * moments[1] * moments[0] ** 3
            + 24 * moments[0] ** 5
        )   
    return kappas


def _approx_Fn_edgeworth(x, mu, sigma_square, kappa_3, kappa_4, kappa_5 = None):
    """
    Compute the approximated value of Fn(x), with 2nd order Egdeworth
    Input:
        x - The data point where you want to evaluate delta_Fn.
        sigma_square - An array. It consists of variances of n variables.
        kappa_3 - An array. It consists of 3rd order cumulants of n variables.
        kappa_4 - An array. It consists of 4th order cumulants of n variables.
        kappa_5 - An array. It consists of 4th order cumulants of n variables.
                  Default is None, if not None, use the 3rd order expansion.
    """
    m = np.sum(mu)
    inv_sigma_n = 1.0 / np.sqrt(np.sum(sigma_square))
    kap_3 = np.sum(kappa_3)
    kap_4 = np.sum(kappa_4)
    x = (x - m) * inv_sigma_n
    expansion = scipy.stats.norm.cdf(x) + (
        -1.0 / 6.0 * inv_sigma_n ** 3 * kap_3 * (x ** 2 - 1.0)
        - 1.0 / 24.0 * inv_sigma_n ** 4 * kap_4 * (x ** 3 - 3 * x)
        - 1.0 / 72.0 * inv_sigma_n ** 6 * kap_3 ** 2 * (x ** 5 - 10 * x ** 3 + 15 * x)
    ) * scipy.stats.norm.pdf(x)
    if kappa_5:
        expansion -= (
        + 1.0 / 120.0 * inv_sigma_n ** 5 * np.sum(kappa_5) * (x ** 4 - 6 * x ** 2 + 3)
        + 1.0 / 144.0 * inv_sigma_n ** 7 * kap_3 * kap_4 * (x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15)
        + 1.0 / 1296.0 * inv_sigma_n ** 9 * kap_3 ** 3 * (x ** 8 - 28 * x ** 6 + 210 * x ** 4 - 420 * x ** 2 + 105)
        ) * scipy.stats.norm.pdf(x)
    return expansion


def _approx_fn_edgeworth(x, mu, sigma_square, kappa_3, kappa_4, kappa_5 = None):
    """
    Compute the approximated value of fn(x) with 2nd order Edgeworth expansion.
    Input:
        x - The data point where you want to evaluate delta_Fn.
        sigma_square - An array. It consists of variances of n variables.
        kappa_3 - An array. It consists of 3rd order cumulants of n variables.
        kappa_4 - An array. It consists of 4th order cumulants of n variables.
        kappa_5 - An array. It consists of 4th order cumulants of n variables.
                  Default is None, if not None, use the 3rd order expansion.
    """
    m = np.sum(mu)
    inv_sigma_n = 1.0 / np.sqrt(np.sum(sigma_square))
    kap_3 = np.sum(kappa_3)
    kap_4 = np.sum(kappa_4)
    x = (x - m) * inv_sigma_n
    expansion =  (1.0
        + 1.0 / 6.0 * inv_sigma_n ** 3 * kap_3 * (x ** 3 - 3 * x)
        + 1.0 / 24.0 * inv_sigma_n ** 4 * kap_4 * (x ** 4 - 6 * x ** 2 + 3)
        + 1.0 / 72.0 * inv_sigma_n ** 6 * kap_3 ** 2 * (x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15)
    ) * scipy.stats.norm.pdf(x)
    if kappa_5:
        expansion += (
        + 1.0 / 120.0 * inv_sigma_n ** 5 * np.sum(kappa_5) * (x ** 5 - 10 * x ** 3 + 15 * x)
        + 1.0 / 144.0 * inv_sigma_n ** 7 * kap_3 * kap_4 * (x ** 7 - 21 * x ** 5 + 105 * x ** 3 - 105 * x)
        + 1.0 / 1296.0 * inv_sigma_n ** 9 * kap_3 ** 3 * (x ** 9 - 36 * x ** 7 + 378 * x ** 5 - 1260 * x ** 3 + 945 * x)
        ) * scipy.stats.norm.pdf(x)
    return expansion



def approx_eps_delta_edgeworth(
        c, mu_X, mu_Y, sigma_square_X, sigma_square_Y, 
        kappa_3_X, kappa_4_X, kappa_5_X, kappa_3_Y, kappa_4_Y, kappa_5_Y
        ):
    """
    Compute the approximated value of eps and delta function at c, i.e. 
    eps(c), delta(c) using the Edgeworth expansion.
    Input:
        c - The curoff point where you want to evaluate Likelihood ratio test.
        sigma_square_X - An array of the variance of n variables of X
        sigma_square_Y - An array of the variance of n variables of Y
        kappa_3_X  - An array of the 3rd order cumulants of n variables of X.
        kappa_3_Y  - An array of the 3rd order cumulants of n variables of Y.
        kappa_4_X  - An array of the 4th order cumulants of n variables of X.
        kappa_4_Y  - An array of the 4th order cumulants of n variables of Y.
        kappa_5_X  - An array of the 5th order cumulants of n variables of X.
                     Default is None. If specified, 3rd oreder Edgeworth
        kappa_5_Y  - An array of the 5th order cumulants of n variables of Y.
                     Default is None. If specified, 3rd oreder Edgeworth
        mu - (Eq[Tn] - Ep[Tn]) /  sqrt(Var_P(Tn))
        use_cornish_fisher - A boolean. If true, use the cornish-fisher expansion to
            compute the 1 - alpha quantile of Fn.
    Output:
        eps(c), delta(c)
    """
    
    fx = _approx_fn_edgeworth(c, mu_X, sigma_square_X, kappa_3_X, kappa_4_X, kappa_5_X)
    fy = _approx_fn_edgeworth(c, mu_Y, sigma_square_Y, kappa_3_Y, kappa_4_Y, kappa_5_Y)
    Fx = _approx_Fn_edgeworth(c, mu_X, sigma_square_X, kappa_3_X, kappa_4_X, kappa_5_X)
    Fy = _approx_Fn_edgeworth(c, mu_Y, sigma_square_Y, kappa_3_Y, kappa_4_Y, kappa_5_Y)

    return np.log(fy/fx), 1 - Fy - fy / fx * (1 - Fx)

def compute_eps_delta_repeat_GDP_function(log_likelihood_ratio_func, dens_func_X, 
                                 dens_func_Y, order = 2):
    """
    Compute the approximated value of eps(), delta()
    Inputs:
        log_likelihood_ratio_func - the PRV form
        dens_func_X & Y - the density function of X (null) and (alternative)
        order - either 2 or 3, 
    Output:
        A function that could evaluate at any point c, num_composition
    """
    assert order in [2, 3], "Only support Edgeworth for order 2 or 3"
    max_order = 4 if (order == 2) else 5
    moments_X, errs_X = compute_moments(log_likelihood_ratio_func, dens_func_X, max_order, left=-np.inf, right=np.inf)
    moments_Y, errs_Y = compute_moments(log_likelihood_ratio_func, dens_func_Y, max_order, left=-np.inf, right=np.inf)
    # Calculate cumulants
    kappasX = compute_cumulants(moments_X, order = order)
    kappasY = compute_cumulants(moments_Y, order = order)
    def f(c, num_composition):
        mu_X = kappasX[0] * num_composition
        sigma_square_X = [kappasX[1]] * num_composition
        kappa_3_X = [kappasX[2]] * num_composition
        kappa_4_X = [kappasX[3]] * num_composition
        kappa_5_X = None if order == 2 else [kappasX[4]] * num_composition
        
        mu_Y = kappasY[0] * num_composition
        sigma_square_Y = [kappasY[1]] * num_composition
        kappa_3_Y = [kappasY[2]] * num_composition
        kappa_4_Y = [kappasY[3]] * num_composition
        kappa_5_Y = None if order == 2 else [kappasY[4]] * num_composition

        
        return approx_eps_delta_edgeworth(c, mu_X, mu_Y, sigma_square_X, sigma_square_Y, kappa_3_X, kappa_4_X, 
                                          kappa_5_X, kappa_3_Y, kappa_4_Y, kappa_5_Y)
    return f, kappasX[0], kappasY[0]



def DP_SGD_privacy_eps_delta_function(sigma, p, order = 2):
    """
    A wrapper for Gaussian Mechanism with sigma and p. Return a function.
    """
    mu = 1 / sigma
    def log_likelihood_ratio_func(x):
        if x > 0:
            return mu * x + np.log((1 - p) * np.exp(-mu * x) + p * np.exp(- mu * mu / 2))
        return np.log(1 - p + p * np.exp(mu * x - mu * mu / 2))
    def dens_func_X(x):
        return scipy.stats.norm.pdf(x)
    def dens_func_Y(x):
        return (1 - p) * scipy.stats.norm.pdf(x) + p * scipy.stats.norm.pdf(x, loc = mu)
    
    return compute_eps_delta_repeat_GDP_function(log_likelihood_ratio_func, dens_func_X, 
                                 dens_func_Y, order = order)


def DP_SGD_eps_list_at_delta(sigma, num_composition_list, p, delta, order = 2, test = False):
    """
    The function that takes in a list of NUM_COMPOSITION's, and evaluate 
    eps at the given DELTA.

    Parameters
    ----------
    sigma : noise rate in the Gaussian Mechanism.
    num_composition_list : the list of num_composition's to evaluate
    p : sampling probability.
    delta : the given delta level.
    order : Edgeworth expansion order. The default is 2.

    Returns
    -------
    eps_list : the desired eps list.

    """
    eps_list = []
    if test:
        c_list = []
    f, mx, my = DP_SGD_privacy_eps_delta_function(sigma, p, order = order)
    for num_composition in num_composition_list:
        g = lambda c: f(c, num_composition)
        target = lambda c: g(c)[1] - delta
        c_star = scipy.optimize.fsolve(target, x0 = (mx + my) / 2 * num_composition, xtol = 1e-12)
        if test:
            c_list.append(c_star)
        eps_list.append(g(c_star)[0])
    if test:
        print(f"c_list is {c_list}")
        print(f"eps_list is {eps_list}")
    return eps_list
        

def DP_SGD_eps_delta_list_at_step(sigma, num_composition, p, c_list, order = 2):
    eps_list = []
    delta_list = []
    f, _, _ = DP_SGD_privacy_eps_delta_function(sigma, p, order = order)
    for c in c_list:
        eps, delta = f(c, num_composition)
        eps_list.append(eps)
        delta_list.append(delta)
    return eps_list, delta_list


#
#def PRV_DP_SGD_eps_list_at_delta(sigma, num_composition_list, p, delta, order = 2):
#    """
#    Use the PRV with Edgeworth expansion.
#    """
#    mu = 1 / sigma
#    def log_likelihood_ratio_func(x):
#        if x > 0:
#            return mu * x + np.log((1 - p) * np.exp(-mu * x) + p * np.exp(- mu * mu / 2))
#        return np.log(1 - p + p * np.exp(mu * x - mu * mu / 2))
#    def dens_func_X(x):
#        return scipy.stats.norm.pdf(x)
#    def dens_func_Y(x):
#        return (1 - p) * scipy.stats.norm.pdf(x) + p * scipy.stats.norm.pdf(x, loc = mu)
#    
#        
#    assert order in [2, 3], "Only support Edgeworth for order 2 or 3"
#    max_order = 4 if (order == 2) else 5
#    moments_X, errs_X = compute_moments(log_likelihood_ratio_func, dens_func_X, max_order, left=-np.inf, right=np.inf)
#    moments_Y, errs_Y = compute_moments(log_likelihood_ratio_func, dens_func_Y, max_order, left=-np.inf, right=np.inf)
#    # Calculate cumulants
#    kappasX = compute_cumulants(moments_X, order = order)
#    kappasY = compute_cumulants(moments_Y, order = order)
#    
#    eps_list = []
#    def f(eps, num_composition):
#        mu_X = [kappasX[0]] * num_composition
#        sigma_square_X = [kappasX[1]] * num_composition
#        kappa_3_X = [kappasX[2]] * num_composition
#        kappa_4_X = [kappasX[3]] * num_composition
#        
#        mu_Y = [kappasY[0]] * num_composition
#        sigma_square_Y = [kappasY[1]] * num_composition
#        kappa_3_Y = [kappasY[2]] * num_composition
#        kappa_4_Y = [kappasY[3]] * num_composition
#        
#        Fx = _approx_Fn_edgeworth(eps, mu_X, sigma_square_X, kappa_3_X, kappa_4_X)
#        Fy = _approx_Fn_edgeworth(eps, mu_Y, sigma_square_Y, kappa_3_Y, kappa_4_Y)
#        return 1 - Fy - np.exp(eps) * (1 - Fx)
#    for num_composition in num_composition_list:
#        g = lambda eps: f(eps, num_composition) - delta
#        eps_star = scipy.optimize.fsolve(g, x0 = 0.2)
#        eps_list.append(eps_star)
#    return eps_list
        
        