"""Utils functions

This sub-module contains several useful functions to run and manage various simulations.
"""

import numpy as np
import matplotlib.pyplot as plt

from ._base_ensemble import _Ensemble


def rand_mtx_max_eigvals(
    ensemble: _Ensemble,
    normalize: bool = False,
    size: int = 1,
    random_state: int = None,
):
    if random_state is not None:
        np.random.seed(random_state)

    max_eigvals = []
    for _ in range(size):
        ensemble.sample(random_state=None)
        max_eigval = ensemble.eigvals(normalize=normalize).max()
        max_eigvals.append(max_eigval)

    return np.asarray(max_eigvals)


def plot_func(interval, func, bins=1000, plot_title=None, plot_ylabel=None, savefig_path=None):
    """Plots a given function (callable) within the provided interval.

    Args:
        interval (tuple): Delimiters (xmin, xmax) of the histogram.
        func (callable): Function to be evaluated. The image of the function builds
            the y-axis values that are plotted.
        bins (int, default=100): It defines the number of equal-width bins within the
            provided interval or range.
        plot_title (string, default=None): Title of the plot.
        plot_ylabel (string, default=None): Label of the y-axis.
        savefig_path (string, default=None): path to save the created figure. If it is not
            provided, the plot is shown at the end of the routine.
    
    """
    if not isinstance(interval, tuple):
        raise ValueError("interval argument must be a tuple")
    
    (xmin, xmax) = interval

    xx = np.linspace(xmin, xmax, num=bins)
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


def get_bins_centers_and_contour(bins):
    """Calculates the centers and contour of the given bins.

    Computes the centers of the given bins. Also, the smallest and the largest bin
    delimitiers are included to define the countour of the representation interval.

    Args:
        bins (list): list of numbers (floats) that specify each bin delimiter.

    Returns:
        list of numbers (floats) consisting in the list of bin centers and contour.
    
    """
    centers = [bins[0]] # Adding initial contour
    l = len(bins)
    for i in range(l-1):
        centers.append((bins[i]+bins[i+1])/2) # Adding centers
    centers.append(bins[-1]) # Adding final contour
    return centers