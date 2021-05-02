"""Tridiagonal utils module

This module contains utilities for tridiagolization or to handle
tridiagonal matrices.

"""

import numpy as np


def tridiag_eigval_neg(tridiag_mtx):
    """Calculates number of negative eigenvalues.

    Given a tridiagonal matrix, this function calculates the
    number of negative eigenvalues using Sturm sequences.

    Args:
        tridiag_mtx (numpy array): tridiagonal matrix

    Returns:
        integer representing the number of negative eigenvalues.
    """
    n_size = len(tridiag_mtx) # number of rows = number of columns
    sturm_seq = np.zeros(n_size)

    sturm_seq[0] = tridiag_mtx[n_size-1][n_size-1] # R[0] = a_1 by default
    for i in range(1, n_size): # O(n)
        sturm_seq[i] = tridiag_mtx[n_size-i-1][n_size-i-1] - \
                        ((tridiag_mtx[n_size-i-1][n_size-i]**2)/sturm_seq[i-1])

    # number of negative values in R sequence
    return (sturm_seq<0).sum()


def tridiag_eigval_hist(tridiag_mtx, bins=100, interval=(-2,2), density=False):
    """Computes efficiently eigenvalue histogram.

    Computes the eigenvalue histogram of the given matrix, using the
    specified bins between the introduced interval. The given matrix has
    to be tridiagonal, so this function builds the histogram efficiently
    using Sturm sequences, avoiding to calculate eigenvalues.

    Args:
        tridiag_mtx (numpy array): tridiagonal matrix
        bins (int or sequence, default=100): If bins is an integer, it defines the number of
            equal-width bins in the range. If bins is a sequence, it defines the
            bin edges, including the left edge of the first bin and the right
            edge of the last bin; in this case, bins may be unequally spaced.
        interval (tuple, default=(-2,2)): Delimiters (xmin, xmax) of the histogram.
            The lower and upper range of the bins. Lower and upper outliers are ignored.
        density (bool, default=False): If True, draw and return a probability
            density: each bin will display the bin's raw count divided by the total
            number of counts and the bin width, so that the area under the histogram
            integrates to 1. If set to False, the absolute frequencies of the eigenvalues
            are returned.

    Returns:
        (tuple) tuple containing:
            observed (array): List of eigenvalues frequencies per bin. If density is
            True these values are the relative frequencies in order to get an area under
            the histogram equal to 1. Otherwise, this list contains the absolute
            frequencies of the eigenvalues.
            bins (array): The edges of the bins. Length nbins + 1 (nbins left edges and
            right edge of last bin)
    """
    if not isinstance(interval, tuple):
        raise ValueError("interval argument must be a tuple")

    if isinstance(bins, int):
        # calculating bin delimiters
        bin_delimiters = np.linspace(interval[0], interval[1], num=bins+1) # O(m)
    elif isinstance(bins, (tuple, list)):
        # bin delimiters directly introduced
        bin_delimiters = np.array(bins)
    else:
        raise ValueError("bins argument must be a tuple, list or integer")

    n_size = int(len(tridiag_mtx)) # number of rows = number of columns
    # + 2 because of (-inf, interval[0]) and (interval[1], inf) intervals
    histogram = np.zeros(len(bin_delimiters))

    # Building matrix A (aux_mtx) in linear time
    diag = np.arange(n_size)
    diag_1 = diag[:-1] # = np.arange(n-1)
    aux_mtx = np.zeros((n_size,n_size))
    aux_mtx[diag_1, diag_1+1] = tridiag_mtx[diag_1, diag_1+1]
    aux_mtx[diag_1+1, diag_1] = tridiag_mtx[diag_1+1, diag_1]

    prev = 0
    for i in range(bins+1): # O(m)
        aux_mtx[diag, diag] = tridiag_mtx[diag, diag] - bin_delimiters[i]
        current = tridiag_eigval_neg(aux_mtx) # O(n)
        histogram[i] = current - prev
        prev = current

    if density:
        # dont need (-inf, interval[0]) interval
        histogram[1:] = histogram[1:] / (sum(histogram[1:]) * np.diff(bin_delimiters))

    # dont want (-inf, interval[0]) interval
    return histogram[1:], bin_delimiters


def householder_reduction(mtx, ret_iterations=False):
    """Householder reduction method for tridiagonalization.

    Computes Householder reduction method for tridiagonalization. It
    transforms a given symmetric matrix in its tridiagonal form, keeping the
    same eigenvalues as the original.

    Args:
        mtx (numpy array): symmetric matrix to be tridiagonalized.
        ret_iterations (bool, default=False): If set to True, it returns
            rotation matrices used to perform the tridiagonalization.

    Returns:
        (tuple) tuple containing:
            mtx (nparray): tridiagonalized matrix.
            mtx_list (nparray): list of matrices representing the evolution of the given matrix
            after each rotation. Only returned if ret_iterations is set to True.
            rot_list (nparray): list of applied rotation matrices. Only returned if
            ret_iterations is set to True.

    References:
        R. Hildebrand. “Householder numerically with mathematica.” 2007.
            http://buzzard.ups.edu/courses/2007spring/projects/hildebrand-paper-revised.pdf
    """
    n_size = len(mtx)

    mtx_list = [mtx]
    rot_list = []

    for j in range(n_size-2):
        if mtx[j+1, j] >= 0:
            alpha = -np.sqrt((mtx[j+1:n_size,j]**2).sum())
        else:
            alpha = np.sqrt((mtx[j+1:n_size,j]**2).sum())

        r_var = np.sqrt(alpha**2/2 - alpha/2 * mtx[j+1,j])

        x_vec = np.zeros((n_size,1))
        x_vec[j+1] = (mtx[j+1,j] - alpha)/(2*r_var)
        for k in range(j+2, n_size):
            x_vec[k] = mtx[k,j]/(2*r_var)

        rot = np.identity(n_size) - 2*np.matmul(x_vec, x_vec.transpose())
        mtx = np.matmul(rot, np.matmul(mtx, rot))

        mtx_list.append(mtx)
        rot_list.append(rot)

    if ret_iterations:
        return mtx, mtx_list, rot_list
    return mtx
