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
    
    return np.where(condition, 1.0, 0.0)



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


class MarchenkoPasturDistribution:

    ARCTAN_OF_INFTY = np.pi/2

    def __init__(self, ratio, ensemble="wre", sigma=1.0):
        if ratio < 0:
            raise ValueError(f"Error: invalid ratio. It has to be positive. Provided ratio = {ratio}.")
        try:
            self.beta = ["wre", "wce", None, "wqe"].index(ensemble) + 1
        except ValueError:
            raise ValueError(f"Error: Ensemble '{ensemble}' not supported."
                            " Check that ensemble is one of the following: 'wre', 'wce' or 'wqe'.")

        self.ratio = ratio
        self.ensemble = ensemble
        self.sigma = sigma
        self.lambda_minus = self.beta * self.sigma**2 * (1 - np.sqrt(self.ratio))**2
        self.lambda_plus = self.beta * self.sigma**2 * (1 + np.sqrt(self.ratio))**2
        self._var = self.beta * self.sigma**2

    def pdf(self, x):
        return np.sqrt(_relu(self.lambda_plus - x) * _relu(x - self.lambda_minus)) \
            / (2.0 * np.pi * self.ratio * self._var * x)

    def cdf(self, x):
        np.seterr(divide='ignore')
        np.seterr(invalid='ignore')

        acum = _indicator(x, start=self.lambda_plus, inclusive="left")
        acum += np.where(_indicator(x, start=self.lambda_minus, stop=self.lambda_plus, inclusive="left"),
                         self._cdf_aux_f(x), 0.0)

        if self.ratio <= 1:
            if acum.shape == (): return float(acum)
            return acum
        
        acum += np.where(_indicator(x, start=self.lambda_minus, stop=self.lambda_plus, inclusive="left"),
                         (self.ratio-1)/(2*self.ratio), 0.0)
        acum += np.where(_indicator(x, start=0, stop=self.lambda_minus, inclusive="left"),
                         (self.ratio-1)/self.ratio, 0.0)

        if acum.shape == (): return float(acum)
        return acum

    def _cdf_aux_f(self, x):
        first_arctan_term = np.where(x == self.lambda_minus,
                                    MarchenkoPasturDistribution.ARCTAN_OF_INFTY,
                                    np.arctan((self._cdf_aux_r(x)**2 - 1)/(2 * self._cdf_aux_r(x)))
                                )

        second_arctan_term = np.where(x == self.lambda_minus,
                                      MarchenkoPasturDistribution.ARCTAN_OF_INFTY,
                                      np.arctan((self.lambda_minus*self._cdf_aux_r(x)**2 - self.lambda_plus) \
                                                / (2*self._var*(1-self.ratio)*self._cdf_aux_r(x)))
                                )
        return 1/(2*np.pi*self.ratio) * (np.pi*self.ratio \
                                         + (1/self._var)*np.sqrt(_relu(self.lambda_plus-x)*_relu(x-self.lambda_minus)) \
                                         - (1+self.ratio)*first_arctan_term + (1-self.ratio)*second_arctan_term)

    def _cdf_aux_r(self, x):
        return np.sqrt((self.lambda_plus-x)/(x - self.lambda_minus))