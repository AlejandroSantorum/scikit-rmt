import numpy as np
import matplotlib.pyplot as plt

from .gaussian_ensemble import GaussianEnsemble
from .wishart_ensemble import WishartEnsemble
from .manova_ensemble import ManovaEnsemble
from .tracy_widom_approximator import TW_Approximator


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

    def __init__(self, beta=1, sigma=1.0):
        if beta not in [1,2,4]:
            raise ValueError(f"Error: invalid beta. It has to be 1,2 or 4. Provided beta = {beta}.")

        self.beta = beta
        self.sigma = sigma
        self.radius = 2.0 * np.sqrt(self.beta) * sigma
        self._gaussian_ens = None
    
    def rvs(self, size):
        if size <= 0:
            raise ValueError(f"Error: invalid sample size. It has to be positive. Provided size = {size}.")
        
        if not self._gaussian_ens:
            self._gaussian_ens = GaussianEnsemble(beta=self.beta, n=size, use_tridiagonal=False, sigma=self.sigma)
        else:
            self._gaussian_ens.set_size(size, resample_mtx=True)
        
        _eigval_norm_const = 1/np.sqrt(size)
        if self.beta == 4:
            return _eigval_norm_const * self._gaussian_ens.eigvals()[::2]
        return _eigval_norm_const * self._gaussian_ens.eigvals()

    def pdf(self, x):
        return 2.0 * np.sqrt(_relu(self.radius**2 - x**2)) / (np.pi * self.radius**2)
    
    def cdf(self, x):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.select(
                condlist=[x >= self.radius, x <= -self.radius],
                choicelist=[1.0, 0.0],
                default=(0.5 + (x * np.sqrt(self.radius**2 - x**2))/(np.pi * self.radius**2) + \
                         (np.arcsin(x/self.radius)) / np.pi)
            )
    
    def plot_pdf(self, interval=None, bins=1000, savefig_path=None):
        if not interval:
            interval = (-self.radius, self.radius)
        
        xx = np.linspace(interval[0], interval[1], num=bins)
        yy = self.pdf(xx)

        plt.plot(xx, yy)
        plt.xlabel("x")
        plt.ylabel("probability density")

        if savefig_path:
            plt.savefig(savefig_path, dpi=800)
        else:
            plt.show()
    
    def plot_cdf(self, interval=None, bins=1000, savefig_path=None):
        if not interval:
            interval = (-self.radius - 0.1, self.radius + 0.1)
        
        xx = np.linspace(interval[0], interval[1], num=bins)
        yy = self.cdf(xx)

        plt.plot(xx, yy)
        plt.xlabel("x")
        plt.ylabel("cumulative distribution")

        if savefig_path:
            plt.savefig(savefig_path, dpi=800)
        else:
            plt.show()



class MarchenkoPasturDistribution:

    ARCTAN_OF_INFTY = np.pi/2

    def __init__(self, ratio, beta=1, sigma=1.0):
        if beta not in [1,2,4]:
            raise ValueError(f"Error: invalid beta. It has to be 1,2 or 4. Provided beta = {beta}.")
        if ratio <= 0:
            raise ValueError(f"Error: invalid ratio. It has to be positive. Provided ratio = {ratio}.")

        self.ratio = ratio
        self.beta = beta
        self.sigma = sigma
        self.lambda_minus = self.beta * self.sigma**2 * (1 - np.sqrt(self.ratio))**2
        self.lambda_plus = self.beta * self.sigma**2 * (1 + np.sqrt(self.ratio))**2
        self._var = self.beta * self.sigma**2
        self._wishart_ens = None
    
    def rvs(self, size):
        if size <= 0:
            raise ValueError(f"Error: invalid sample size. It has to be positive. Provided size = {size}.")
        
        _n = int(np.round(size / self.ratio))

        if not self._wishart_ens:
            self._wishart_ens = WishartEnsemble(beta=self.beta, p=size, n=_n, use_tridiagonal=False, sigma=self.sigma)
        else:
            self._wishart_ens.set_size(p=size, n=_n, resample_mtx=True)
        
        _eigval_norm_const = 1/_n 
        if self.beta == 4:
            return _eigval_norm_const * self._wishart_ens.eigvals()[::2]
        return _eigval_norm_const * self._wishart_ens.eigvals()

    def pdf(self, x):
        return np.sqrt(_relu(self.lambda_plus - x) * _relu(x - self.lambda_minus)) \
            / (2.0 * np.pi * self.ratio * self._var * x)

    def cdf(self, x):
        with np.errstate(divide='ignore', invalid='ignore'):
            acum = _indicator(x, start=self.lambda_plus, inclusive="left")
            acum += np.where(_indicator(x, start=self.lambda_minus, stop=self.lambda_plus, inclusive="left"),
                            self._cdf_aux_f(x), 0.0)

            if self.ratio <= 1:
                return acum
            
            acum += np.where(_indicator(x, start=self.lambda_minus, stop=self.lambda_plus, inclusive="left"),
                            (self.ratio-1)/(2*self.ratio), 0.0)
            acum += np.where(_indicator(x, start=0, stop=self.lambda_minus, inclusive="left"),
                            (self.ratio-1)/self.ratio, 0.0)

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

    def plot_pdf(self, interval=None, bins=1000, savefig_path=None):
        if not interval:
            interval = (self.lambda_minus, self.lambda_plus)
        
        xx = np.linspace(interval[0], interval[1], num=bins)
        yy = self.pdf(xx)

        plt.plot(xx, yy)
        plt.xlabel("x")
        plt.ylabel("probability density")

        if savefig_path:
            plt.savefig(savefig_path, dpi=800)
        else:
            plt.show()
    
    def plot_cdf(self, interval=None, bins=1000, savefig_path=None):
        if not interval:
            interval = (self.lambda_minus, self.lambda_plus)
        
        xx = np.linspace(interval[0], interval[1], num=bins)
        yy = self.cdf(xx)

        plt.plot(xx, yy)
        plt.xlabel("x")
        plt.ylabel("cumulative distribution")

        if savefig_path:
            plt.savefig(savefig_path, dpi=800)
        else:
            plt.show()


class TracyWidomDistribution:

    def __init__(self, beta=1):
        if beta not in [1,2,4]:
            raise ValueError(f"Error: invalid beta. It has to be 1,2 or 4. Provided beta = {beta}.")

        self.beta = beta
        self.tw_approx = TW_Approximator(beta=self.beta)

    def pdf(self, x):
        return self.tw_approx.pdf(x)

    def cdf(self, x):
        return self.tw_approx.cdf(x)

    def plot_pdf(self, interval=None, bins=1000, savefig_path=None):
        if not interval:
            interval = (-5, 4-self.beta)
        
        xx = np.linspace(interval[0], interval[1], num=bins)
        yy = self.pdf(xx)

        plt.plot(xx, yy)
        plt.xlabel("x")
        plt.ylabel("probability density")

        if savefig_path:
            plt.savefig(savefig_path, dpi=800)
        else:
            plt.show()
    
    def plot_cdf(self, interval=None, bins=1000, savefig_path=None):
        if not interval:
            interval = (-5, 4-self.beta)
        
        xx = np.linspace(interval[0], interval[1], num=bins)
        yy = self.cdf(xx)

        plt.plot(xx, yy)
        plt.xlabel("x")
        plt.ylabel("cumulative distribution")

        if savefig_path:
            plt.savefig(savefig_path, dpi=800)
        else:
            plt.show()


class ManovaSpectrumDistribution:

    def __init__(self, a, b, beta=1):
        if beta not in [1,2,4]:
            raise ValueError(f"Error: invalid beta. It has to be 1,2 or 4. Provided beta = {beta}.")
        if a <= 0 or b <= 0:
            raise ValueError("Error: invalid matrix parameters. They have to be both positive.\n"
                             f"\tProvided a = {a} and b = {b}.")

        if a < 1 or b < 1:
            print(f"Warning: Setting a < 1 (a = {a}) or b < 1 (b = {b}) may cause numerical instability.")

        self.a = a
        self.b = b
        self.beta = beta
        self.lambda_term1 = np.sqrt((a/(a+b)) * (1 - (1/(a+b))))
        self.lambda_term2 = np.sqrt((1/(a+b)) * (1 - (a/(a+b))))
        self.lambda_minus = (self.lambda_term1 - self.lambda_term2)**2
        self.lambda_plus = (self.lambda_term1 + self.lambda_term2)**2
        self._manova_ens = None
    
    def rvs(self, size):
        if size <= 0:
            raise ValueError(f"Error: invalid sample size. It has to be positive. Provided size = {size}.")
        
        _n1 = int(np.round(size * self.a))
        _n2 = int(np.round(size * self.b))

        if not self._manova_ens:
            self._manova_ens = ManovaEnsemble(beta=self.beta, m=size, n1=_n1, n2=_n2)
        else:
            self._manova_ens.set_size(m=size, n1=_n1, n2=_n2, resample_mtx=True)
        
        _eigval_norm_const = 1.0
        if self.beta == 4:
            return _eigval_norm_const * self._manova_ens.eigvals()[::2].real
        return _eigval_norm_const * self._manova_ens.eigvals().real
    
    def pdf(self, x):
        with np.errstate(divide='ignore', invalid='ignore'): 
            return np.where(np.logical_and(x > self.lambda_minus, x < self.lambda_plus),
                            (self.a + self.b) * np.sqrt((self.lambda_plus - x) * (x - self.lambda_minus)) \
                                / (2.0 * np.pi * x * (1-x)),
                            0.0)

    def plot_pdf(self, interval=None, bins=1000, savefig_path=None):
        if not interval:
            interval = (self.lambda_minus, self.lambda_plus)
        
        xx = np.linspace(interval[0], interval[1], num=bins)
        yy = self.pdf(xx)

        plt.plot(xx, yy)
        plt.xlabel("x")
        plt.ylabel("probability density")

        if savefig_path:
            plt.savefig(savefig_path, dpi=800)
        else:
            plt.show()
