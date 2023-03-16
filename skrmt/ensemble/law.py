import numpy as np


def _relu(x):
    """Element-wise maximum between the value and zero.

    Args:
        x (ndarray): list of numbers to compute its element-wise maximum.
    
    Returns:
        array_like consisting in the element-wise maximum vector of the given values.
    """
    return np.maximum(x, np.zeros_like(x))


def _indicator(x, start=None, stop=None, inclusive="both"):
    """Element-wise indicator function within a real interval.
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
    if not start and not stop:
        raise ValueError("Error: provide start and/or stop for indicator function.")

    INCLUSIVE_OPTIONS = ["both", "left", "right", "neither"]
    if inclusive not in INCLUSIVE_OPTIONS:
        raise ValueError(f"Error: invalid interval inclusive parameter: {inclusive}\n"
                         "\t inclusive has to be one of the following: {INCLUSIVE_OPTIONS}.")

    if start:
        if inclusive == "both" or inclusive == "left":
            condition = (start <= x)
        elif inclusive == "neither" or inclusive == "right":
            condition = (start < x)
    
    if start and stop:
        if inclusive == "both" or inclusive == "right":
            condition = np.logical_and(condition, (x <= stop))
        elif inclusive == "neither" or inclusive == "left":
            condition = np.logical_and(condition, (x < stop))
    elif stop:
        if inclusive == "both" or inclusive == "right":
            condition = (x <= stop)
        elif inclusive == "neither" or inclusive == "left":
            condition = (x < stop)
    
    return np.where(condition, 1, 0)



class WignerSemicircleDistribution:

    def __init__(self, ensemble="goe", sigma=1.0):
        try:
            self.beta = ["goe", "gue", None, "gse"].index(ensemble) + 1
        except ValueError:
            raise ValueError(f"Ensemble '{ensemble}' not supported."
                            " Check that ensemble is one of the following: 'goe', 'gue' or 'gse'.")
    
        self.ensemble = ensemble
        self.sigma = sigma
        self.radius = 2.0 * np.sqrt(self.beta) * sigma

    def pdf(self, x):
        return _relu(2.0 * np.sqrt(self.radius**2 - x**2) / (np.pi * self.radius**2))
    
    def cdf(self, x):
        return 0.5 + (x * np.sqrt(self.radius**2 - x**2))/(np.pi * self.radius**2) + \
            (np.arcsin(x/self.radius)) / np.pi
