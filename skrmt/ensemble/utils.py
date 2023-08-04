"""Utils functions

This sub-module contains several useful functions to run and manage various simulations.
"""

import numpy as np
from typing import Union, Sequence
import matplotlib.pyplot as plt

from ._base_ensemble import _Ensemble
from .tracy_widom_approximator import TW_Approximator


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


def plot_func(interval, func, num_x_vals=1000, plot_title=None, plot_ylabel=None, savefig_path=None):
    """Plots a given 1D function (callable) within the provided interval.

    It plots a given 1-dimensional function (python Callable) within the provided interval.
    The x values are computed by generating `n_

    Args:
        interval (tuple): Delimiters (xmin, xmax) of the histogram.
        func (callable): Function to be evaluated. The image of the function builds
            the y-axis values that are plotted.
        num_x_vals (int, default=100): It defines the number of evenly spaced x values
            within the given interval or range in which the function (callable) is evaluated.
        plot_title (string, default=None): Title of the plot.
        plot_ylabel (string, default=None): Label of the y-axis.
        savefig_path (string, default=None): path to save the created figure. If it is not
            provided, the plot is shown at the end of the routine.
    
    """
    if not isinstance(interval, tuple):
        raise ValueError("interval argument must be a tuple")
    
    (xmin, xmax) = interval

    xx = np.linspace(xmin, xmax, num=num_x_vals)
    yy = func(xx)

    plt.plot(xx, yy)
    plt.xlabel("x")
    if plot_ylabel:
        plt.ylabel(plot_ylabel)
    
    if plot_title:
        plt.title(plot_title)

    if savefig_path:
        plt.savefig(savefig_path, dpi=800)
    else:
        plt.show()


def relu(x):
    """Element-wise maximum between the value and zero.

    Args:
        x (ndarray): list of numbers to compute its element-wise maximum.
    
    Returns:
        array_like consisting in the element-wise maximum vector of the given values.
    """
    return np.maximum(x, np.zeros_like(x))


def indicator(x, start=None, stop=None, inclusive="both"):
    r"""Element-wise indicator function within a real interval.
    The interval can be left-closed, right-closed, closed or open.
    Visit https://en.wikipedia.org/wiki/Indicator_function for more information.

    Args:
        x (ndarray): list of numbers to compute its element-wise indicator image.
        start (double, default=None): left value of the interval. If not provided,
            the left value is equivalent to :math:`- \infty`.
        stop (double, default=None): right value of the interval. If not provided,
            the right value is equivalent to :math:`+ \infty`.
        inclusive (string, default="both"): type of interval. For left-closed interval
            use "left", for right-closed interval use "right", for closed interval use
            "both" and for open interval use "neither".

    Returns:
        array_like consisting in the element-wise indicator function image of the given values.
    """
    if start is None and stop is None:
        raise ValueError("Error: provide start and/or stop for indicator function.")

    INCLUSIVE_OPTIONS = ["both", "left", "right", "neither"]
    if inclusive not in INCLUSIVE_OPTIONS:
        raise ValueError(f"Error: invalid interval inclusive parameter: {inclusive}\n"
                         "\t inclusive has to be one of the following: {INCLUSIVE_OPTIONS}.")

    if start is not None:
        if inclusive == "both" or inclusive == "left":
            condition = (start <= x)
        elif inclusive == "neither" or inclusive == "right":
            condition = (start < x)
    
    if (start is not None) and (stop is not None):
        if inclusive == "both" or inclusive == "right":
            condition = np.logical_and(condition, (x <= stop))
        elif inclusive == "neither" or inclusive == "left":
            condition = np.logical_and(condition, (x < stop))
    elif stop:
        if inclusive == "both" or inclusive == "right":
            condition = (x <= stop)
        elif inclusive == "neither" or inclusive == "left":
            condition = (x < stop)

    return np.where(condition, 1.0, 0.0)


def get_bins_centers_and_contour(bin_edges):
    """Calculates the centers and contour of the given the bins edges.

    Computes the centers of the given the bins edges. Also, the smallest and the largest
    bin delimitiers are included to define the countour of the representation interval.

    Args:
        bin_edges (list): list of numbers (floats) that specify each bin delimiter.

    Returns:
        list of numbers (floats) consisting in the list of bin centers and contour.
    
    """
    centers = [bin_edges[0]] # Adding initial contour
    l = len(bin_edges)
    for i in range(l-1):
        centers.append((bin_edges[i]+bin_edges[i+1])/2) # Adding centers
    centers.append(bin_edges[-1]) # Adding final contour
    return centers
