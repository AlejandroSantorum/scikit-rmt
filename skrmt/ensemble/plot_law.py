"""Plot Law module

This module contains several functions that simulate various
random matrix laws, including Wigner's Semicircle Law,
Marchenko-Pastur Law and Tracy-Widom Law.

"""

import numpy as np
import matplotlib.pyplot as plt

from .gaussian_ensemble import GaussianEnsemble
from .wishart_ensemble import WishartEnsemble

def wigner_semicircular_law(ensemble='goe', n_size=1000, bins=100, interval=None,
                            density=False, savefig_path=None):
    """Calculates and plots Wigner's Semicircle Law using Gaussian Ensemble.

    Calculates and plots Wigner's Semicircle Law using Gaussian Ensemble random matrices.
    Gaussian (Hermite) ensemble has improved routines (using tridiagonal forms and Sturm
    sequences) to avoid calculating the eigenvalues, so the histogram
    is built using certain techniques to boost efficiency.

    Args:
        ensemble ('goe', 'gue' or 'gse', default='goe'): ensemble to draw the
            random matrices to study Wigner's Law.
        n_size (int, default=1000): random matrix size n times n.
        bins (int or sequence, default=100): If bins is an integer, it defines the number
            of equal-width bins in the range. If bins is a sequence, it defines the
            bin edges, including the left edge of the first bin and the right
            edge of the last bin; in this case, bins may be unequally spaced.
        interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram.
            The lower and upper range of the bins. Lower and upper outliers are ignored.
        density (bool, default=False): If True, draw and return a probability
            density: each bin will display the bin's raw count divided by the total
            number of counts and the bin width, so that the area under the histogram
            integrates to 1. If set to False, the absolute frequencies of the eigenvalues
            are returned.
        fig_path (string, default=None): path to save the created figure. If it is not
            provided, the plot is shown are the end of the routine.

    References:
        Albrecht, J. and Chan, C.P. and Edelman, A.
            "Sturm sequences and random eigenvalue distributions".
            Foundations of Computational Mathematics. 9.4 (2008): 461-483.
        Dumitriu, I. and Edelman, A.
            "Matrix Models for Beta Ensembles".
            Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

    """
    # pylint: disable=too-many-arguments
    if n_size<1:
        raise ValueError("matrix size must be positive")

    if ensemble == 'goe':
        ens = GaussianEnsemble(beta=1, n=n_size, use_tridiagonal=True)
        if interval is None:
            interval = (-2,2)
    elif ensemble == 'gue':
        ens = GaussianEnsemble(beta=2, n=n_size, use_tridiagonal=True)
        if interval is None:
            interval = (-3,3)
    elif ensemble == 'gse':
        ens = GaussianEnsemble(beta=4, n=n_size, use_tridiagonal=True)
        if interval is None:
            interval = (-4,4)
    else:
        raise ValueError("ensemble not supported")

    # Wigner eigenvalue normalization constant
    norm_const = 1/np.sqrt(n_size/2)

    observed, bins = ens.eigval_hist(bins=bins, interval=interval,
                                     density=density, norm_const=norm_const)
    width = bins[1]-bins[0]
    plt.bar(bins[:-1], observed, width=width, align='edge')

    plt.title("Eigenvalue density histogram")
    plt.xlabel("x")
    plt.ylabel("density")

    # Saving plot or showing it
    if savefig_path:
        plt.savefig(savefig_path, dpi=1200)
    else:
        plt.show()


def marchenko_pastur_law(ensemble='wre', p_size=3000, n_size=10000, bins=100, interval=None,
                         density=False, savefig_path=None):
    """Calculates and plots Wigner's Semicircle Law using Gaussian Ensemble.

    Calculates and plots Marchenko-Pastur Law using Wishart Ensemble random matrices.
    Wishart (Laguerre) ensemble has improved routines (using tridiagonal forms and Sturm
    sequences) to avoid calculating the eigenvalues, so the histogram
    is built using certain techniques to boost efficiency.

    Args:
        ensemble ('wre', 'wce' or 'wqe', default='wre'): ensemble to draw the
            random matrices to study Marchenko-Pastur Law.
        p_size (int, default=3000): number of rows of the guassian matrix that generates
            the matrix of the corresponding ensemble.
        n_size (int, default=10000): number of columns of the guassian matrix that generates
            the matrix of the corresponding ensemble.
        bins (int or sequence, default=100): If bins is an integer, it defines the number
            of equal-width bins in the range. If bins is a sequence, it defines the
            bin edges, including the left edge of the first bin and the right
            edge of the last bin; in this case, bins may be unequally spaced.
        interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram.
            The lower and upper range of the bins. Lower and upper outliers are ignored.
        density (bool, default=False): If True, draw and return a probability
            density: each bin will display the bin's raw count divided by the total
            number of counts and the bin width, so that the area under the histogram
            integrates to 1. If set to False, the absolute frequencies of the eigenvalues
            are returned.
        fig_path (string, default=None): path to save the created figure. If it is not
            provided, the plot is shown are the end of the routine.

    References:
        Albrecht, J. and Chan, C.P. and Edelman, A.
            "Sturm sequences and random eigenvalue distributions".
            Foundations of Computational Mathematics. 9.4 (2008): 461-483.
        Dumitriu, I. and Edelman, A.
            "Matrix Models for Beta Ensembles".
            Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

    """
    # pylint: disable=too-many-arguments
    if n_size<1:
        raise ValueError("matrix size must be positive")

    if ensemble == 'wre':
        ens = WishartEnsemble(beta=1, p=p_size, n=n_size, use_tridiagonal=True)
        if interval is None:
            ratio = p_size/n_size
            lambda_plus = (1 + np.sqrt(ratio))**2
            lambda_minus = (1 - np.sqrt(ratio))**2
            interval = (lambda_minus, lambda_plus)
    elif ensemble == 'wce':
        ens = WishartEnsemble(beta=2, p=p_size, n=n_size, use_tridiagonal=True)
        if interval is None:
            interval = (0.2, 5)
    elif ensemble == 'wqe':
        ens = WishartEnsemble(beta=4, p=p_size, n=n_size, use_tridiagonal=True)
        if interval is None:
            interval = (0.5, 10)
    else:
        raise ValueError("ensemble not supported")

    # Wigner eigenvalue normalization constant
    norm_const = 1/n_size

    observed, bins = ens.eigval_hist(bins=bins, interval=interval,
                                     density=density, norm_const=norm_const)
    width = bins[1]-bins[0]
    plt.bar(bins[:-1], observed, width=width, align='edge')

    plt.title("Eigenvalue density histogram")
    plt.xlabel("x")
    plt.ylabel("density")

    # Saving plot or showing it
    if savefig_path:
        plt.savefig(savefig_path, dpi=1200)
    else:
        plt.show()


def tracy_widom_law(ensemble='goe', n_size=100, times=1000, bins=100, interval=None,
                    density=False, savefig_path=None):
    """Calculates and plots Tracy-Widom Law using Gaussian Ensemble.

    Calculates and plots Tracy-Widom Law using Gaussian Ensemble random matrices.
    Because we need to obtain the largest eigenvalue of each sampled random matrix,
    we need to sample a certain amount them. For each random matrix sammpled, its
    largest eigenvalue is calcualted in order to simulate its density.

    Args:
        ensemble ('goe', 'gue' or 'gse', default='goe'): ensemble to draw the
            random matrices to study Wigner's Law.
        n_size (int, default=100): random matrix size n times n.
        times (int, default=1000): number of times to sample a random matrix.
        bins (int or sequence, default=100): If bins is an integer, it defines the number
            of equal-width bins in the range. If bins is a sequence, it defines the
            bin edges, including the left edge of the first bin and the right
            edge of the last bin; in this case, bins may be unequally spaced.
        interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram.
            The lower and upper range of the bins. Lower and upper outliers are ignored.
        density (bool, default=False): If True, draw and return a probability
            density: each bin will display the bin's raw count divided by the total
            number of counts and the bin width, so that the area under the histogram
            integrates to 1. If set to False, the absolute frequencies of the eigenvalues
            are returned.
        fig_path (string, default=None): path to save the created figure. If it is not
            provided, the plot is shown are the end of the routine.

    References:
        Albrecht, J. and Chan, C.P. and Edelman, A.
            "Sturm sequences and random eigenvalue distributions".
            Foundations of Computational Mathematics. 9.4 (2008): 461-483.
        Dumitriu, I. and Edelman, A.
            "Matrix Models for Beta Ensembles".
            Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

    """
    # pylint: disable=too-many-arguments
    if n_size<1 or times<1:
        raise ValueError("matrix size or number of repetitions must be positive")

    if ensemble == 'goe':
        ens = GaussianEnsemble(beta=1, n=n_size, use_tridiagonal=False)
    elif ensemble == 'gue':
        ens = GaussianEnsemble(beta=2, n=n_size, use_tridiagonal=False)
    elif ensemble == 'gse':
        ens = GaussianEnsemble(beta=4, n=n_size, use_tridiagonal=False)
    else:
        raise ValueError("ensemble not supported")

    eigvals = np.asarray([])
    for _ in range(times):
        vals = ens.eigvals()
        eigvals = np.append(eigvals, vals.max())
        ens.sample()

    # Wigner/Tracy-Widom eigenvalue normalization constant
    eigvals = eigvals/np.sqrt(n_size/2)

    if interval is None:
        xmin=eigvals.min()
        xmax=eigvals.max()
        interval=(xmin, xmax)

    # using numpy to obtain histogram in the given interval and no. of bins
    observed, bins = np.histogram(eigvals, bins=bins, range=interval, density=density)
    width = bins[1]-bins[0]
    plt.bar(bins[:-1], observed, width=width, align='edge')

    plt.title("Eigenvalue density histogram")
    plt.xlabel("x")
    plt.ylabel("density")

    # Saving plot or showing it
    if savefig_path:
        plt.savefig(savefig_path, dpi=1200)
    else:
        plt.show()
