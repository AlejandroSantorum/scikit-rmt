"""Utils functions

This sub-module contains several useful functions to run and manage various simulations.
"""

import numpy as np
from typing import Union, Sequence
import matplotlib.pyplot as plt

from .base_ensemble import _Ensemble
from .gaussian_ensemble import GaussianEnsemble
from .tracy_widom_approximator import TW_Approximator
from .misc import get_bins_centers_and_contour


def plot_spectral_hist_and_law(
    ensemble: _Ensemble,
    bins: Union[int, Sequence] = 100,
    savefig_path: str = None,
):
    """TODO
    """
    # plotting ensemble spectral histogram
    observed, bin_edges = ensemble._plot_eigval_hist(bins=bins, density=True, normalize=True)
    centers = np.asarray(get_bins_centers_and_contour(bin_edges))

    # getting the class that implements the ensemble spectral law
    law_class = ensemble._law_class
    # computing PDF on the bin centers
    pdf = law_class.pdf(centers)
    # plotting PDF
    plt.plot(centers, pdf, color='red', linewidth=2)

    # Saving plot or showing it
    if savefig_path:
        plt.savefig(savefig_path, dpi=1200)
    else:
        plt.show()


def plot_max_eigvals_tracy_widom(
    ensemble: _Ensemble,
    n_eigvals: int = 1,
    bins: Union[int, Sequence] = 100,
    random_state: int = None,
    savefig_path: str = None,
):
    """Plots the histogram of the maximum eigenvalues of a random ensemble with the
    Tracy-Widom PDF.

    It computes samples of normalized maximum eigenvalues of the specified random matrix
    ensemble and plots their histogram alongside the Tracy-Widom PDF.

    Args:
        ensemble (_Ensemble): a random matrix ensemble instance.
        n_eigvals (int, default=1): number of maximum eigenvalues to compute. This is the number
            of times the random matrix is re-sampled in order to get several samples of the maximum
            eigenvalue.
        bins (int or sequence, default=100): If bins is an integer, it defines the number of
            equal-width bins in the range. If bins is a sequence, it defines the
            bin edges, including the left edge of the first bin and the right
            edge of the last bin; in this case, bins may be unequally spaced.
        random_state (int, default=None): random seed to initialize the pseudo-random
            number generator of numpy. This has to be any integer between 0 and 2**32 - 1
            (inclusive), or None (default). If None, the seed is obtained from the clock.
        savefig_path (string, default=None): path to save the created figure. If it is not
            provided, the plot is shown at the end of the routine.
    
    """
    max_eigvals = rand_mtx_max_eigvals(
        ensemble=ensemble,
        normalize=True,
        n_eigvals=n_eigvals,
        random_state=random_state,
    )

    interval = (max_eigvals.min(), max_eigvals.max())

    observed, bin_edges = np.histogram(max_eigvals, bins=bins, range=interval, density=True)
    width = bin_edges[1]-bin_edges[0]
    plt.bar(bin_edges[:-1], observed, width=width, align='edge')

    centers = get_bins_centers_and_contour(bin_edges)

    tw_approx = TW_Approximator(beta=ensemble.beta)
    tw_pdf = tw_approx.pdf(centers)

    plt.plot(centers, tw_pdf, color='red', linewidth=2)

    plt.title("Comparing maximum eigenvalues histogram with Tracy-Widom law", fontweight="bold")
    plt.xlabel("x")
    plt.ylabel("probability density")

    # Saving plot or showing it
    if savefig_path:
        plt.savefig(savefig_path, dpi=1200)
    else:
        plt.show()


def rand_mtx_max_eigvals(
    ensemble: _Ensemble,
    normalize: bool = False,
    n_eigvals: int = 1,
    random_state: int = None,
):
    """Computes several times the maximum eigenvalue of different random matrix samples.

    It generates several samples of the maximum eigenvalue of the specified random matrix
    by sampling several times the randon matrix of the corresponding ensemble and computing
    the largest eigenvalue. The eigenvalues can be then normalized using the scaling and
    normalization constants of the Tracy-Widom distribution.

    Args:
        ensemble (_Ensemble): a random matrix ensemble instance.
        normalize (bool, default=False): whether to normalize the computed maximum eigenvalues
            using the scaling and normalization constants of the Tracy-Widom distribution. This
            is useful if the goal is to compare the distribution of the maximum eigenvalues with
            the Tracy-Widom distribution.
        n_eigvals (int, default=1): number of maximum eigenvalues to compute. This is the number
            of times the random matrix is re-sampled in order to get several samples of the maximum
            eigenvalue.
        random_state (int, default=None): random seed to initialize the pseudo-random
            number generator of numpy. This has to be any integer between 0 and 2**32 - 1
            (inclusive), or None (default). If None, the seed is obtained from the clock.
    
    Returns:
        numpy array (ndarray) containing the sampled maximum eigenvalues of the given ensemble. 
    
    """
    if random_state is not None:
        np.random.seed(random_state)

    max_eigvals = []
    for _ in range(n_eigvals):
        ensemble.sample(random_state=None)
        max_eigval = ensemble.eigvals(normalize=False).max()
        max_eigvals.append(max_eigval)
    
    max_eigvals = np.asarray(max_eigvals)
    
    if normalize:
        # `n` is the matrix sample size. Usually it's a square matrix, so shape[0] = shape[1]
        n_size = ensemble.matrix.shape[1]

        # Tracy-Widom eigenvalue normalization constants
        eigval_scale = 1.0/np.sqrt(ensemble.beta)
        size_scale = 1.0
        if ensemble.beta == 4:
            size_scale = 1/np.sqrt(2)

        max_eigvals = size_scale*(n_size**(1/6))*(eigval_scale*max_eigvals - (2.0 * np.sqrt(n_size)))

    return max_eigvals
