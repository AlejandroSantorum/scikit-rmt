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
        plt.savefig(savefig_path)
    else:
        plt.show()


def marchenko_pastur_law(ensemble='wre', p_size=3000, n_size=10000, bins=100, interval=None,
                         density=False, savefig_path=None):
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
        plt.savefig(savefig_path)
    else:
        plt.show()


def tracy_widom_law(ensemble='goe', n_size=100, times=1000, bins=100, interval=None,
                    density=False, savefig_path=None):
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
        plt.savefig(savefig_path)
    else:
        plt.show()
