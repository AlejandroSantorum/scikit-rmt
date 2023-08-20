"""Utils functions

This sub-module contains several useful functions to run and manage various simulations.
"""

import numpy as np
from typing import Union, Sequence
import matplotlib.pyplot as plt

from .base_ensemble import _Ensemble
from .gaussian_ensemble import GaussianEnsemble
from .wishart_ensemble import WishartEnsemble
from .misc import get_bins_centers_and_contour


def plot_spectral_hist_and_law(
    ensemble: _Ensemble,
    bins: Union[int, Sequence] = 100,
    savefig_path: str = None,
):
    """Plots the spectrum histogram of a random matrix ensemble alongside the
    PDF of the corresponding spectral law.

    It illustrates the histogram of the spectrum of a given random matrix ensemble
    alongside the PDF of the corresponding spectral law, i.e.:
    - If `ensemble` is `GaussianEnsemble` then is plotted Wigner's Semicircle law PDF.
    - If `ensemble` is `WishartEnsemble` then is plotted Marchenko-Pastur law PDF.
    - If `ensemble` is `ManovaEnsemble` then is plotted the PDF of the Manova ensemble
    formulated by Wachter.

    Args:
        ensemble (_Ensemble): a random matrix ensemble instance. The only supported types
            are `GaussianEnsemble`, `WishartEnsemble` and `ManovaEnsemble`.
        bins (int or sequence, default=100): If bins is an integer, it defines the number of
            equal-width bins in the range. If bins is a sequence, it defines the
            bin edges, including the left edge of the first bin and the right
            edge of the last bin; in this case, bins may be unequally spaced.
        savefig_path (string, default=None): path to save the created figure. If it is not
            provided, the plot is shown at the end of the routine.
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

    plt.xlabel("x")
    plt.ylabel("density")
    plt.title("Ensemble spectral histogram vs law PDF", fontweight="bold")

    # Saving plot or showing it
    if savefig_path:
        plt.savefig(savefig_path, dpi=1000)
    else:
        plt.show()


def standard_vs_tridiag_hist(
    ensemble: Union[GaussianEnsemble, WishartEnsemble],
    bins: Union[int, Sequence] = 100,
    savefig_path: str = None,
    random_state: int = None,
):
    """Plots and compares the spectral histogram of a random matrix using its
    standard form vs using the corresponding tridiagonal form.

    This function simulates the histogramming of a given random matrix ensemble
    in its standard form and it compares the former with the equivalent histogram
    computed using the tridiagonal matrix form. This is useful to illustrate both
    types of matrix have the same spectral distribution.

    Args:
        ensemble (_Ensemble): a random matrix ensemble instance. The only supported types
            are `GaussianEnsemble` and `WishartEnsemble`, since spectral optimizations
            based on their tridiagonal forms are known.
        bins (int or sequence, default=100): If bins is an integer, it defines the number of
            equal-width bins in the range. If bins is a sequence, it defines the
            bin edges, including the left edge of the first bin and the right
            edge of the last bin; in this case, bins may be unequally spaced.
        savefig_path (string, default=None): path to save the created figure. If it is not
            provided, the plot is shown at the end of the routine.
        random_state (int, default=None): random seed to initialize the pseudo-random
                number generator of numpy. This has to be any integer between 0 and 2**32 - 1
                (inclusive), or None (default). If None, the seed is obtained from the clock.
    
    References:
        - Albrecht, J. and Chan, C.P. and Edelman, A.
            "Sturm sequences and random eigenvalue distributions".
            Foundations of Computational Mathematics. 9.4 (2008): 461-483.
        - Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
            Journal of Mathematical Physics. 43.11 (2002): 5830-5847.
    
    """
    plt.rcParams['figure.dpi'] = 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ### Standard form
    ensemble.resample(tridiagonal_form=False, random_state=random_state)
    obs_std, bin_edges_std = ensemble.eigval_hist(bins=bins, density=True, normalize=True)

    width = bin_edges_std[1] - bin_edges_std[0]
    ax1.bar(bin_edges_std[:-1], obs_std, width=width, align='edge')

    ### Tridiagonal form
    ensemble.resample(tridiagonal_form=True, random_state=random_state)
    obs_tridiag, bin_edges_tridiag = ensemble.eigval_hist(bins=bins, density=True, normalize=True)

    width = bin_edges_tridiag[1] - bin_edges_tridiag[0]
    ax2.bar(bin_edges_tridiag[:-1], obs_tridiag, width=width, align='edge')

    # Plot info
    ax1.set_xlabel("x")
    ax2.set_xlabel("x")
    ax1.set_ylabel("density")
    ax2.set_ylabel("density")

    ax1.set_title("Standard matrix form")
    ax2.set_title("Tridiagonal matrix form")

    fig.suptitle("Spectral histogram of random matrices in standard form vs tridiagonal form", fontweight="bold")
    
    # Saving plot or showing it
    if savefig_path:
        plt.savefig(savefig_path, dpi=1000)
    else:
        plt.show()
