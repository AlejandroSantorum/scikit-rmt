"""Tridiagonal utils module

This module contains utilities for tridiagolization or to handle
tridiagonal matrices.

"""

from typing import Tuple, Union, Sequence
import numpy as np


def tridiag_eigval_neg(diag: np.ndarray, offdiag: np.ndarray) -> int:
    """Calculates number of negative eigenvalues.

    Given a tridiagonal matrix, this function calculates the
    number of negative eigenvalues using Sturm sequences.

    Args:
        diag (numpy array): Main diagonal of length n.
        offdiag (numpy array): First upper diagonal of length n-1.

    Returns:
        integer representing the number of negative eigenvalues.
    
    References:
        - Albrecht, J. and Chan, C.P. and Edelman, A.
            "Sturm sequences and random eigenvalue distributions".
            Foundations of Computational Mathematics. 9.4 (2008): 461-483.
        - Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
            Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

    """
    n_size = len(diag)  # number of rows = number of columns
    if n_size == 0:
        return 0

    # start at the bottom-right corner
    # R[0] = a_1 by default
    pivot = diag[-1]
    n_negative = int(pivot < 0)

    # apply the Sturm recurrence in reverse order.
    for i in range(n_size - 2, -1, -1):  # O(n)
        pivot = diag[i] - offdiag[i] ** 2 / pivot
        n_negative += int(pivot < 0)

    # number of negative values in R sequence
    return n_negative


def tridiag_eigval_hist(
    tridiag_mtx: np.ndarray,
    interval: Tuple,
    bins: Union[int, Sequence] = 100,
    density: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes efficiently eigenvalue histogram.

    Calculates the eigenvalue histogram of the given matrix, using the
    specified bins between the introduced interval. The given matrix has
    to be tridiagonal, so this function builds the histogram efficiently
    using Sturm sequences, avoiding to calculate eigenvalues.

    Args:
        tridiag_mtx (numpy array): tridiagonal matrix
        interval (tuple): Delimiters (xmin, xmax) of the histogram. The lower and upper
            range of the bins. Lower and upper outliers are ignored.
        bins (int or sequence, default=100): If bins is an integer, it defines the number of
            equal-width bins in the range. If bins is a sequence, it defines the
            bin edges, including the left edge of the first bin and the right
            edge of the last bin; in this case, bins may be unequally spaced.
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
            bin_delimiters (array): The edges of the bins. Length nbins + 1 (nbins left edges
            and right edge of last bin).
    
    References:
        - Albrecht, J. and Chan, C.P. and Edelman, A.
            "Sturm sequences and random eigenvalue distributions".
            Foundations of Computational Mathematics. 9.4 (2008): 461-483.
        - Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
            Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

    """
    if not isinstance(interval, tuple):
        raise ValueError("interval argument must be a tuple")

    if isinstance(bins, int):
        # calculating bin delimiters
        bin_delimiters = np.linspace(interval[0], interval[1], num=bins+1) # O(m)
    elif isinstance(bins, (tuple, list)):
        # bin delimiters directly introduced
        bin_delimiters = np.asarray(bins)
    else:
        raise ValueError("bins argument must be a tuple, list or integer")

    # extract the diagonals once: O(n) storage and work.
    diag = np.diag(tridiag_mtx).copy()
    offdiag = np.diag(tridiag_mtx, k=1).copy()

    n_size = int(len(tridiag_mtx)) # number of rows = number of columns
    # + 2 because of (-inf, interval[0]) and (interval[1], inf) intervals
    histogram = np.zeros(len(bin_delimiters))

    # reuse this buffer instead of allocating a new shifted
    # diagonal for every bin edge.
    shifted_diag = np.empty(
        n_size,
        dtype=np.result_type(diag, bin_delimiters, np.float64),
    )

    prev = 0
    for i, edge in enumerate(bin_delimiters):
        # compute diag - edge without allocating another array
        np.subtract(diag, edge, out=shifted_diag)
        # compute the number of negative eigenvalues of the shifted matrix
        current = tridiag_eigval_neg(shifted_diag, offdiag)  # O(n)

        histogram[i] = current - prev
        prev = current

    # exclude eigenvalues below the first bin edge
    histogram = histogram[1:]

    if density:
        # dont need (-inf, interval[0]) interval
        histogram = histogram / (
            np.sum(histogram) * np.diff(bin_delimiters)
        )

    return histogram, bin_delimiters


def householder_reduction(
    mtx: np.ndarray,
    ret_iterations: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        - R. Hildebrand. “Householder numerically with mathematica.” 2007.
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
