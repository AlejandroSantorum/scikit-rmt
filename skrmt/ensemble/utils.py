"""Utils functions

This sub-module contains several useful functions to run and manage various simulations.
"""

import numpy as np
from typing import Union, Sequence
import matplotlib.pyplot as plt

from .base_ensemble import _Ensemble
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
        plt.savefig(savefig_path, dpi=1200)
    else:
        plt.show()
