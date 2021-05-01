import numpy as np
import matplotlib.pyplot as plt

from .gaussian_ensemble import GaussianEnsemble
from .wishart_ensemble import WishartEnsemble

def wigner_semicircular_law(ensemble='goe', n=1000, bins=100, interval=None, density=False, savefig_path=None):
    if n<1:
        raise ValueError("matrix size must be positive")

    if ensemble == 'goe':
        ens = GaussianEnsemble(beta=1, n=n, use_tridiagonal=True)
        if interval is None:
            interval = (-2,2)
    elif ensemble == 'gue':
        ens = GaussianEnsemble(beta=2, n=n, use_tridiagonal=True)
        if interval is None:
            interval = (-3,3)
    elif ensemble == 'gse':
        ens = GaussianEnsemble(beta=4, n=n, use_tridiagonal=True)
        if interval is None:
            interval = (-4,4)
    else:
        raise ValueError("ensemble not supported")
    
    # Wigner eigenvalue normalization constant
    norm_const = 1/np.sqrt(n/2)

    observed, bins = ens.eigval_hist(bins=bins, interval=interval, density=density, norm_const=norm_const)
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


def marchenko_pastur_law(ensemble='wre', p=3000, n=10000, bins=100, interval=None, density=False, savefig_path=None):
    if n<1:
        raise ValueError("matrix size must be positive")

    if ensemble == 'wre':
        ens = WishartEnsemble(beta=1, p=p, n=n, use_tridiagonal=True)
        if interval is None:
            ratio = p/n
            lambda_plus = (1 + np.sqrt(ratio))**2
            lambda_minus = (1 - np.sqrt(ratio))**2
            xmin = lambda_minus
            xmax = lambda_plus
            interval = (xmin, xmax)
    elif ensemble == 'wce':
        ens = WishartEnsemble(beta=2, p=p, n=n, use_tridiagonal=True)
        if interval is None:
            interval = (0.2, 5)
    elif ensemble == 'wqe':
        ens = WishartEnsemble(beta=4, p=p, n=n, use_tridiagonal=True)
        if interval is None:
            interval = (0.5, 10)
    else:
        raise ValueError("ensemble not supported")
    
    # Wigner eigenvalue normalization constant
    norm_const = 1/n

    observed, bins = ens.eigval_hist(bins=bins, interval=interval, density=density, norm_const=norm_const)
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


def tracy_widom_law(ensemble='goe', n=100, t=1000, bins=100, interval=None, density=False, savefig_path=None):
    if n<1 or t<1:
        raise ValueError("matrix size or number of repetitions must be positive")

    if ensemble == 'goe':
        ens = GaussianEnsemble(beta=1, n=n, use_tridiagonal=False)
    elif ensemble == 'gue':
        ens = GaussianEnsemble(beta=2, n=n, use_tridiagonal=False)
    elif ensemble == 'gse':
        ens = GaussianEnsemble(beta=4, n=n, use_tridiagonal=False)
    else:
        raise ValueError("ensemble not supported")

    eigvals = np.asarray([])
    for i in range(t):
        vals = ens.eigvals()
        new_val = vals.max()
        eigvals = np.append(eigvals, new_val)
        ens.sample()
    
    # Wigner/Tracy-Widom eigenvalue normalization constant
    eigvals = eigvals/np.sqrt(n/2)

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





