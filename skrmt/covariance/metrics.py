"""Metrics Module

This module contains the implementation of several metrics to
measure estimators quality.
"""

import numpy as np


def loss_mv(sigma_tilde, sigma):
    """Computes minimum variance loss function.

    Computes minimum variance loss function between estimated covariance
    matrix (Sigma tilde) and population covariance matrix (Sigma).

    Args:
        sigma_tilde (numpy array): estimated covariance matrix.
        sigma (numpy array): population covariance matrix.

    Returns:
        loss (double): calculated minimum variance loss.

    References:
        Ledoit, O. and Wolf, M.
            "Analytical nonlinear shrinkage of large-dimensional covariance matrices".
            Annals of Statistics. 48.5 (2020): 3043-3065

    """
    p_size = len(sigma)

    # pylint: disable=raise-missing-from
    try:
        sigma_tilde_inv = np.linalg.inv(sigma_tilde)
    except np.linalg.LinAlgError:
        raise ValueError("Unable to invert estimated covariance matrix")

    try:
        sigma_inv = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        raise ValueError("Unable to invert population covariance matrix")


    mtx = np.matmul(np.matmul(sigma_tilde_inv, sigma), sigma_tilde_inv)
    loss = np.trace(mtx)/p_size/(np.trace(sigma_tilde_inv)/p_size)**2 - p_size/np.trace(sigma_inv)
    return loss


def loss_frobenius(sigma_tilde, sigma):
    """Computes Fröbenius loss function.

    Computes Fröbenius loss function between estimated covariance
    matrix (Sigma tilde) and population covariance matrix (Sigma).

    Args:
        sigma_tilde (numpy array): estimated covariance matrix.
        sigma (numpy array): population covariance matrix.

    Returns:
        loss (double): calculated Fröbenius loss.

    References:
        Ledoit, O. and Wolf, M.
            "Analytical nonlinear shrinkage of large-dimensional covariance matrices".
            Annals of Statistics. 48.5 (2020): 3043-3065

    """
    p_size = len(sigma)
    loss = np.trace(sigma_tilde - sigma)**2/p_size
    return loss


def prial_mv(exp_sample, exp_sigma_tilde, exp_fsopt):
    """Computes percentage relative improvement in average loss.

    Computes percentage relative improvement in average loss using
    minimum variance losses. The given expectations must have been
    calculated using Monte Carlo simulations.

    Args:
        exp_sample (double): expected MV loss between sample covariance
            matrix and population matrix.
        exp_sigma_tilde (double): expected MV loss between estimated covariance
            matrix and population matrix.
        exp_fsopt (double): expected MV loss between S^* covariance
            matrix (i.e., finite-sample optimal covariance matrix)
            and population matrix.

    Returns:
        prial (double): calculated percentage relative improvement in average loss.

    References:
        Ledoit, O. and Wolf, M.
            "Analytical nonlinear shrinkage of large-dimensional covariance matrices".
            Annals of Statistics. 48.5 (2020): 3043-3065

    """
    return (exp_sample - exp_sigma_tilde)/(exp_sample - exp_fsopt)
