"""Law module

This module contains classes that implement the main spectral distributions.
When the limiting behaviour of the spectrum of a random matrix ensemble is
well-known, it is often described with a mathematical proven law.
Probability density functions and cumulative distribution functions are provided
for the Wigner Semicircle Law, Marchenko-Pastur Law, Tracy-Widom Law and for the
spectrum of the Manova Ensemble.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import collections.abc

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


def _plot_func(interval, func, bins=1000, plot_title=None, plot_ylabel=None, savefig_path=None):
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


def _get_bins_centers_and_contour(bins):
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


class WignerSemicircleDistribution:
    """Wigner Semicircle Distribution class.

    The Wigner Semicircle Law describes the spectrum of the Wigner random matrices.
    In particular, random matrices of the Gaussian Ensemble are Wigner matrices,
    and therefore the spectrum of the random matrices of this ensemble converge
    to the Wigner Semicircle Law when the matrix size goes to infinity. This
    class provides methods to sample eigenvalues following Wigner Semicircle
    distribution, computing the PDF, computing the CDF and simple methods to
    plot the former two.

    Attributes:
        beta (int): descriptive integer of the Gaussian ensemble type.
            For GOE beta=1, for GUE beta=2, for GSE beta=4.
        center (float): center of the distribution. Since the distribution
            has the shape of a semicircle, the center corresponds to its center.
        sigma (float): scale of the distribution. This value also corresponds
            to the standard deviation of the random entries of the sampled matrix.
        radius (float): radius of the semicircle of the Wigner law. This depends on
            the scale (sigma) and on beta.
    
    References:
        - Wigner, E.
            "Characteristic Vectors of Bordered Matrices With Infinite Dimensions".
            Annals of Mathematics. 62.3. (1955).
        - Wigner, E.
            "On the Distribution of the Roots of Certain Symmetric Matrices".
            Annals of Mathematics. 67.2. (1958).
    
    """

    def __init__(self, beta=1, center=0.0, sigma=1.0):
        """Constructor for WignerSemicircleDistribution class.

        Initializes an instance of this class with the given parameters.

        Args:
            beta (int, default=1): descriptive integer of the Gaussian ensemble type.
                For GOE beta=1, for GUE beta=2, for GSE beta=4.
            center (float, default=0.0): center of the distribution. Since the distribution
                has the shape of a semicircle, the center corresponds to its center.
            sigma (float, default=1.0): scale of the distribution. This value also corresponds
                to the standard deviation of the random entries of the sampled matrix.
        
        """
        if beta not in [1,2,4]:
            raise ValueError(f"Error: invalid beta. It has to be 1,2 or 4. Provided beta = {beta}.")

        self.beta = beta
        self.center = center
        self.sigma = sigma
        self.radius = 2.0 * np.sqrt(self.beta) * self.sigma
        self._gaussian_ens = None
    
    def rvs(self, size):
        """Samples ranfom variates following this distribution.

        Args:
            size (int): sample size.
        
        Returns:
            numpy array with the generated samples.
        """
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
        """Computes PDF of the Wigner Semicircle Law.

        Args:
            x (float or ndarray): value or (numpy) array of values in which compute the PDF.
        
        Returns:
            float or numpy array with the computed PDF in the given value(s).
        
        """
        return 2.0 * np.sqrt(_relu(self.radius**2 - (x-self.center)**2)) / (np.pi * self.radius**2)
    
    def cdf(self, x):
        """Computes CDF of the Wigner Semicircle Law.

        Args:
            x (float or ndarray): value or (numpy) array of values in which compute the CDF.
        
        Returns:
            float or numpy array with the computed CDF in the given value(s).
        
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.select(
                condlist=[x >= (self.center + self.radius), x <= (self.center - self.radius)],
                choicelist=[1.0, 0.0],
                default=(0.5 + ((x-self.center) * np.sqrt(self.radius**2 - (x-self.center)**2))/(np.pi * self.radius**2) \
                         + (np.arcsin((x-self.center)/self.radius)) / np.pi)
            )
    
    def plot_pdf(self, interval=None, bins=1000, savefig_path=None):
        """Plots the PDF of the Wigner Semicircle Law.

        Args:
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram. If not
                provided, the used interval is calculated depending on beta, center, radius and scale.
            bins (int, default=100): It defines the number of equal-width bins within the
                provided interval or range.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown at the end of the routine.
        
        """
        if interval is None:
            interval = (self.center - self.radius - 0.1, self.center + self.radius + 0.1)
        
        _plot_func(
            interval, func=self.pdf, bins=bins, 
            plot_ylabel="probability density", savefig_path=savefig_path
        )
    
    def plot_cdf(self, interval=None, bins=1000, savefig_path=None):
        """Plots the CDF of the Wigner Semicircle Law.

        Args:
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram. If not
                provided, the used interval is calculated depending on beta, center, radius and scale.
            bins (int, default=100): It defines the number of equal-width bins within the
                provided interval or range.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown at the end of the routine.
        
        """
        if interval is None:
            interval = (self.center - self.radius - 0.1, self.center + self.radius + 0.1)
        
        _plot_func(
            interval, func=self.cdf, bins=bins, 
            plot_ylabel="cumulative distribution", savefig_path=savefig_path
        )
    

    def plot_empirical_pdf(self, n_size=1000, bins=100, interval=None, density=False,
                           plot_law_pdf=False, savefig_path=None):
        """Computes and plots Wigner's semicircle empirical law using Gaussian Ensemble.

        Calculates and plots Wigner's semicircle empirical law using Gaussian Ensemble
        random matrices. Gaussian (Hermite) ensemble has improved routines (using
        tridiagonal forms and Sturm sequences) to avoid calculating the eigenvalues,
        so the histogram is built using certain techniques to boost efficiency.

        Args:
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
            plot_law_pdf (bool, default=False): If True, the theoretical law is plotted.
                If set to False, just the empirical histogram is shown. This parameter is only
                considered when the argument 'density' is set also to True.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown are the end of the routine.

        References:
            - Albrecht, J. and Chan, C.P. and Edelman, A.
                "Sturm sequences and random eigenvalue distributions".
                Foundations of Computational Mathematics. 9.4 (2008): 461-483.
            - Dumitriu, I. and Edelman, A.
                "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        """
        # pylint: disable=too-many-arguments
        if n_size<1:
            raise ValueError("matrix size must be positive")

        if self.center != 0.0:
            print(f"Warning: The given center is not 0.0 (center = {self.center}).\n"
                  "\t It is only available the centered empirical Wigner's semicircle law.")
        
        use_tridiag = (self.sigma == 1.0)
        if not use_tridiag:
            print(f"Warning: The given scale is not the standard (sigma = {self.sigma}).\n"
                "\t Tridiagonal histogramming is deactivated.\n"
                "\t It is adviced to set sigma=1.0 to optimize and boost histogramming.")

        ens = GaussianEnsemble(beta=self.beta, n=n_size, sigma=self.sigma, use_tridiagonal=use_tridiag)

        # default plotting interval in case it's not provided
        if interval is None:
            interval = (-self.radius, self.radius)

        # Wigner eigenvalue normalization constant
        if use_tridiag:
            norm_const = 1/np.sqrt(n_size) if self.beta==4 else 1/np.sqrt(n_size/2)
        else:
            norm_const = 1/np.sqrt(n_size)

        observed, bins = ens.eigval_hist(bins=bins, interval=interval,
                                        density=density, norm_const=norm_const)
        width = bins[1]-bins[0]
        plt.bar(bins[:-1], observed, width=width, align='edge')

        # Plotting Wigner Semicircle Law pdf
        if plot_law_pdf and density:
            centers = np.asarray(_get_bins_centers_and_contour(bins))
            pdf = self.pdf(centers)
            plt.plot(centers, pdf, color='red', linewidth=2)
        elif plot_law_pdf and not density:
            print("Warning: Wigner's Semicircle Law PDF is only plotted when density is True.")

        plt.title("Wigner Semicircle Law - Empirical density histogram", fontweight="bold")
        plt.xlabel("x")
        plt.ylabel("probability density")

        # Saving plot or showing it
        if savefig_path:
            plt.savefig(savefig_path, dpi=1200)
        else:
            plt.show()



class MarchenkoPasturDistribution:
    """Marchenko-Pastur Distribution class.

    The Marchenko-Pastur Law describes the spectrum of the Wishart random matrices.
    Therefore the spectrum of the random matrices of this ensemble converge
    to the Marchenko-Pastur Law when the matrix size goes to infinity. This
    class provides methods to sample eigenvalues following Marchenko-Pastur
    distribution, computing the PDF, computing the CDF and simple methods to
    plot the former two.

    Attributes:
        ratio (float): random matrix size ratio. This is the ratio between the
            number of degrees of freedom 'p' and the sample size 'n'. The value
            of ratio = p/n.
        beta (int): descriptive integer of the Wishart ensemble type.
            For WRE beta=1, for WCEE beta=2, for WQE beta=4.
        sigma (float): scale of the distribution. This value also corresponds
            to the standard deviation of the random entries of the sampled matrix.
        lambda_minus (float): lower bound of the support of the Marchenko-Pastur Law.
            It depends on beta, on the scale (sigma) and on the ratio.
        lambda_plus (float): upper bound of the support of the Marchenko-Pastur Law.
            It depends on beta, on the scale (sigma) and on the ratio.
    
    References:
        - Bar, Z.D. and Silverstain, J.W.
            "Spectral Analysis of Large Dimensional Random Matrices".
            2nd edition. Springer. (2010).
    
    """

    ARCTAN_OF_INFTY = np.pi/2

    def __init__(self, ratio, beta=1, sigma=1.0):
        """Constructor for MarchenkoPasturDistribution class.

        Initializes an instance of this class with the given parameters.

        Args:
            ratio (float): random matrix size ratio. This is the ratio between the
                number of degrees of freedom 'p' and the sample size 'n'. The value
                of ratio = p/n.
            beta (int, default=1): descriptive integer of the Wishart ensemble type.
                For WRE beta=1, for WCE beta=2, for WQE beta=4.
            sigma (float, default=1.0): scale of the distribution. This value also corresponds
                to the standard deviation of the random entries of the sampled matrix.
        
        """
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
        """Samples ranfom variates following this distribution.

        Args:
            size (int): sample size.
        
        Returns:
            numpy array with the generated samples.
        """
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
        """Computes PDF of the Marchenko-Pastur Law.

        Args:
            x (float or ndarray): value or (numpy) array of values in which compute the PDF.
        
        Returns:
            float or numpy array with the computed PDF in the given value(s).
        
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.sqrt(_relu(self.lambda_plus - x) * _relu(x - self.lambda_minus)) \
                / (2.0 * np.pi * self.ratio * self._var * x)

    def cdf(self, x):
        """Computes CDF of the Marchenko-Pastur Law.

        Args:
            x (float or ndarray): value or (numpy) array of values in which compute the CDF.
        
        Returns:
            float or numpy array with the computed CDF in the given value(s).
        
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            acum = _indicator(x, start=self.lambda_plus, inclusive="left")
            acum += np.where(_indicator(x, start=self.lambda_minus, stop=self.lambda_plus, inclusive="left"),
                            self._cdf_aux_f(x), 0.0)

            if self.ratio <= 1:
                return acum
            
            acum += np.where(_indicator(x, start=self.lambda_minus, stop=self.lambda_plus, inclusive="left"),
                            (self.ratio-1)/(2*self.ratio), 0.0)

            ### This would need to be added if the extra density point at zero is measured
            # https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution
            # acum += np.where(_indicator(x, start=0, stop=self.lambda_minus, inclusive="left"),
            #                 (self.ratio-1)/self.ratio, 0.0)
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
        """Plots the PDF of the Marchenko-Pastur Law.

        Args:
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram. If not
                provided, the used interval is calculated depending on beta, ratio, and scale.
            bins (int, default=100): It defines the number of equal-width bins within the
                provided interval or range.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown at the end of the routine.
        
        """
        if interval is None:
            interval = (self.lambda_minus, self.lambda_plus)
        
        _plot_func(
            interval, func=self.pdf, bins=bins, 
            plot_ylabel="probability density", savefig_path=savefig_path
        )
    
    def plot_cdf(self, interval=None, bins=1000, savefig_path=None):
        """Plots the CDF of the Marchenko-Pastur Law.

        Args:
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram. If not
                provided, the used interval is calculated depending on beta, ratio, and scale.
            bins (int, default=100): It defines the number of equal-width bins within the
                provided interval or range.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown at the end of the routine.
        
        """
        if interval is None:
            interval = (self.lambda_minus, self.lambda_plus)
        
        _plot_func(
            interval, func=self.cdf, bins=bins, 
            plot_ylabel="cumulative distribution", savefig_path=savefig_path
        )

    def plot_empirical_pdf(self, p_size=1000, n_size=3000, bins=100, interval=None,
                           density=False, plot_law_pdf=False, savefig_path=None):
        """Computes and plots Marchenko-Pastur empirical law using Wishart Ensemble random matrices.

        Calculates and plots Marchenko-Pastur empirical law using Wishart Ensemble random matrices.
        Wishart (Laguerre) ensemble has improved routines (using tridiagonal forms and Sturm
        sequences) to avoid calculating the eigenvalues, so the histogram is built using certain
        techniques to boost efficiency. This optimization is only used when the ratio p_size/n_size
        is less or equal than 1.

        Args:
            p_size (int, default=1000): number of rows of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            n_size (int, default=3000): number of columns of the guassian matrix that generates
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
            plot_law_pdf (bool, default=False): If True, the limiting theoretical law is plotted.
                If set to False, just the empirical histogram is shown. This parameter is only
                considered when the argument 'density' is set also to True.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown are the end of the routine.

        References:
            - Albrecht, J. and Chan, C.P. and Edelman, A.
                "Sturm sequences and random eigenvalue distributions".
                Foundations of Computational Mathematics. 9.4 (2008): 461-483.
            - Dumitriu, I. and Edelman, A.
                "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        """
        # pylint: disable=too-many-arguments
        if n_size<1 or p_size<1:
            raise ValueError("matrix size must be positive")
        
        if interval and interval[0] == 0:
            print("Warning: setting the beginning of the interval to zero may generate numerical errors.")
            print(f"Setting interval to (-0.01, {interval[1]})")
            interval = (-0.01, interval[1])

        # calculating constants depending on matrix sizes
        ratio = p_size/n_size
        lambda_plus = self.beta * self.sigma**2 * (1 + np.sqrt(ratio))**2
        lambda_minus = self.beta * self.sigma**2 * (1 - np.sqrt(ratio))**2
        use_tridiag_ratio = (ratio <= 1)
        if not use_tridiag_ratio:
            print("Warning: Cannot use tridiagonal histogramming if 'p' (degrees of freedom) is "
                " greater than 'n' (sample size).\n"
                f"\t Provided n={n_size} and p={p_size}. Tridiagonal histogramming is therefore deactivated.\n"
                "\t It is adviced to increase sample size (`n`) to optimize and boost histogramming.")

        # computing interval according to the matrix size ratio and support
        if interval is None:
            if ratio <= 1:
                interval = (lambda_minus, lambda_plus)
            else:
                interval = (min(-0.05, lambda_minus), lambda_plus)
        
        use_tridiag_sigma = (self.sigma == 1.0)
        if not use_tridiag_sigma:
            print(f"Warning: The given scale is not the standard (sigma = {self.sigma}).\n"
                "\t Tridiagonal histogramming is deactivated.\n"
                "\t It is adviced to set sigma=1.0 to optimize and boost histogramming.")

        ens = WishartEnsemble(beta=self.beta, p=p_size, n=n_size, sigma=self.sigma,
                            use_tridiagonal=(use_tridiag_ratio and use_tridiag_sigma))

        # Wigner eigenvalue normalization constant
        norm_const = 1/n_size 

        observed, bins = ens.eigval_hist(bins=bins, interval=interval,
                                        density=density, norm_const=norm_const)
        width = bins[1]-bins[0]
        plt.bar(bins[:-1], observed, width=width, align='edge')

        # Plotting theoretical graphic
        if plot_law_pdf and density:
            centers = np.array(_get_bins_centers_and_contour(bins))
            # creating new instance with the updated ratio depending on the given matrix sizes
            mpd = MarchenkoPasturDistribution(beta=self.beta, ratio=ratio, sigma=self.sigma)
            pdf = mpd.pdf(centers)
            plt.plot(centers, pdf, color='red', linewidth=2)

        plt.title("Marchenko-Pastur Law - Empirical density histogram", fontweight="bold")
        plt.xlabel("x")
        plt.ylabel("probability density")
        if ratio > 1:
            if plot_law_pdf and density:
                ylim_vals = pdf
            else:
                ylim_vals = observed
            try:
                plt.ylim(0, np.max(ylim_vals)+0.25*np.max(ylim_vals))
            except ValueError:
                second_highest_val = np.partition(ylim_vals.flatten(), -2)[-2]
                plt.ylim(0, second_highest_val+0.25*second_highest_val)

        # Saving plot or showing it
        if savefig_path:
            plt.savefig(savefig_path, dpi=1200)
        else:
            plt.show()



class TracyWidomDistribution:
    """Tracy-Widom Distribution class.

    The Tracy-Widom Law describes the behaviour of the largest eigenvalue of the
    Wigner random matrices. In particular, random matrices of the Gaussian Ensemble
    are Wigner matrices, and therefore the largest eigenvalues of the spectrum of
    the random matrices of this ensemble converge to the Tracy-Widom Law when the
    matrix size and the sample size (number of times a Wigner matrix is generated to
    compute the largest eigenvalue) goes to infinity. This class provides methods to
    sample eigenvalues following Tracy-Widom distribution, computing the PDF,
    computing the CDF and simple methods to plot the former two.

    Attributes:
        beta (int): descriptive integer of the Gaussian ensemble type.
            For GOE beta=1, for GUE beta=2, for GSE beta=4.
    
    References:
        - Bauman, S.
            "The Tracy-Widom Distribution and its Application to Statistical Physics".
            http://web.mit.edu/8.334/www/grades/projects/projects17/SamBauman.pdf
            MIT Department of Physics. (2017).
        - Tracy, C.A. and Widom, H.
            "On orthogonal and symplectic matrix ensembles".
            Communications in Mathematical Physics. 177.3. (1996).
        - Bejan, A.
            "Largest eigenvalues and sample covariance matrices".
            Tracy-Widom and Painleve II: Computational aspects and
            realization in S-Plus with applications, M.Sc. dissertation,
            Department of Statistics, The University of Warwick. (2005).
        - Borot, G. and Nadal, C.
            "Right tail expansion of Tracy-Widom beta laws".
            Random Matrices: Theory and Applications. 01.03. (2012).
    
    """

    def __init__(self, beta=1):
        """Constructor for TracyWidomDistribution class.

        Initializes an instance of this class with the given parameters.

        Args:
            beta (int, default=1): descriptive integer of the Gaussian ensemble type.
                For GOE beta=1, for GUE beta=2, for GSE beta=4.
        
        """
        if beta not in [1,2,4]:
            raise ValueError(f"Error: invalid beta. It has to be 1,2 or 4. Provided beta = {beta}.")

        self.beta = beta
        self.tw_approx = TW_Approximator(beta=self.beta)

    def rvs(self, size, mtx_size=100):
        """Samples ranfom variates following this distribution.

        Args:
            size (int): sample size.
            mtx_size (int, default=100): matrix size. Remember the Tracy-Widom Law describes the
                limiting behaviour of the largest eigenvalue of a Wigner matrix. Therefore,
                a matrix has to be generated to get each sample. This argument specifies the size
                of the matrix.
        
        Returns:
            numpy array with the generated samples.
        """
        if size <= 0:
            raise ValueError(f"Error: invalid sample size. It has to be positive. Provided size = {size}.")
        if mtx_size <= 0:
            raise ValueError(f"Error: invalid matrix size. It has to be positive. Provided matrix size = {mtx_size}.")

        self._gaussian_ens = GaussianEnsemble(beta=self.beta, n=mtx_size, use_tridiagonal=False)

        max_eigvals = []
        for _ in range(size):
            max_eigvals.append(self._gaussian_ens.eigvals().max())
            self._gaussian_ens.sample()
        max_eigvals = np.asarray(max_eigvals)

        # Tracy-Widom eigenvalue distr. normalization constants
        eigval_scale = 1.0
        size_scale = 1.0
        if self.beta == 2:
            eigval_scale = 1/np.sqrt(2)
        if self.beta == 4:
            eigval_scale = size_scale = 1/np.sqrt(2)
            mtx_size *= 2

        max_eigvals = size_scale*(mtx_size**(1/6))*(eigval_scale*max_eigvals - (2.0*np.sqrt(mtx_size)))
        return max_eigvals

    def pdf(self, x):
        """Computes PDF of the Tracy-Widom Law.

        Args:
            x (float or ndarray): value or (numpy) array of values in which compute the PDF.
        
        Returns:
            float or numpy array with the computed PDF in the given value(s).
        
        """
        return self.tw_approx.pdf(x)

    def cdf(self, x):
        """Computes CDF of the Tracy-Widom Law.

        Args:
            x (float or ndarray): value or (numpy) array of values in which compute the CDF.
        
        Returns:
            float or numpy array with the computed CDF in the given value(s).
        
        """
        return self.tw_approx.cdf(x)

    def plot_pdf(self, interval=None, bins=1000, savefig_path=None):
        """Plots the PDF of the Tracy-Widom Law.

        Args:
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram. If not
                provided, the used interval is calculated depending on beta.
            bins (int, default=100): It defines the number of equal-width bins within the
                provided interval or range.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown at the end of the routine.
        
        """
        if interval is None:
            interval = (-5, 4-self.beta)
        
        _plot_func(
            interval, func=self.pdf, bins=bins, 
            plot_ylabel="probability density", savefig_path=savefig_path
        )
    
    def plot_cdf(self, interval=None, bins=1000, savefig_path=None):
        """Plots the PDF of the Tracy-Widom Law.

        Args:
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram. If not
                provided, the used interval is calculated depending on beta.
            bins (int, default=100): It defines the number of equal-width bins within the
                provided interval or range.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown at the end of the routine.
        
        """
        if interval is None:
            interval = (-5, 4-self.beta)
        
        _plot_func(
            interval, func=self.cdf, bins=bins, 
            plot_ylabel="cumulative distribution", savefig_path=savefig_path
        )

    def plot_empirical_pdf(self, n_size=100, times=1000, bins=100, interval=None,
                           density=False, plot_law_pdf=False, savefig_path=None):
        """Computes and plots Tracy-Widom empirical law using Gaussian Ensemble.

        Calculates and plots Tracy-Widom empirical law using Gaussian Ensemble random matrices.
        Because we need to obtain the largest eigenvalue of each sampled random matrix,
        we need to sample a certain amount them. For each random matrix sammpled, its
        largest eigenvalue is calcualted in order to simulate its density.

        Args:
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
            plot_law_pdf (bool, default=False): If True, the limiting theoretical law is plotted.
                If set to False, just the empirical histogram is shown. This parameter is only
                considered when the argument 'density' is set also to True.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown are the end of the routine.

        References:
            - Albrecht, J. and Chan, C.P. and Edelman, A.
                "Sturm sequences and random eigenvalue distributions".
                Foundations of Computational Mathematics. 9.4 (2008): 461-483.
            - Dumitriu, I. and Edelman, A.
                "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        """
        # pylint: disable=too-many-arguments
        if n_size<1 or times<1:
            raise ValueError("matrix size or number of repetitions must be positive")

        ens = GaussianEnsemble(beta=self.beta, n=n_size, use_tridiagonal=False)

        eigvals = []
        for _ in range(times):
            eigvals.append(ens.eigvals().max())
            ens.sample()
        eigvals = np.asarray(eigvals)

        # Tracy-Widom eigenvalue distr. normalization constants
        eigval_scale = 1.0
        size_scale = 1.0
        if self.beta == 2:
            eigval_scale = 1/np.sqrt(2)
        elif self.beta == 4:
            eigval_scale = size_scale = 1/np.sqrt(2)
            n_size *= 2
        eigvals = size_scale*(n_size**(1/6))*(eigval_scale*eigvals - (2.0*np.sqrt(n_size)))

        if interval is None:
            xmin=eigvals.min()
            xmax=eigvals.max()
            interval=(xmin, xmax)

        # using numpy to obtain histogram in the given interval and no. of bins
        observed, bins = np.histogram(eigvals, bins=bins, range=interval, density=density)
        width = bins[1]-bins[0]
        plt.bar(bins[:-1], observed, width=width, align='edge')

        # Plotting theoretical graphic
        if plot_law_pdf and density:
            centers = _get_bins_centers_and_contour(bins)
            pdf = self.pdf(centers)
            plt.plot(centers, pdf, color='red', linewidth=2)

        plt.title("Tracy-Widom Law - Empirical density histogram", fontweight="bold")
        plt.xlabel("x")
        plt.ylabel("probability density")

        # Saving plot or showing it
        if savefig_path:
            plt.savefig(savefig_path, dpi=1200)
        else:
            plt.show()



class ManovaSpectrumDistribution:
    """Manova Spectrum Distribution class.

    The spectrum of the random matrices of the Manova Ensemble converge to a
    well-defined function implemented in this class. The class provides methods
    to sample eigenvalues of the Manova Ensemble, computing the PDF, computing
    the CDF and simple methods to plot the former two.

    Attributes:
        a (float): first random matrix size ratio. This is the ratio between the
            number of degrees of freedom 'p' and the first sample size 'n1'. The value
            of a = p/n1. Remember a Manova randon matrix is considered a double-Wishart
            matrix, that's why there are two sample sizes 'n1' and 'n2' (see below).
        b (float): second random matrix size ratio. This is the ratio between the
            number of degrees of freedom 'p' and the second sample size 'n2'. The value
            of b = p/n2. Remember a Manova randon matrix is considered a double-Wishart
            matrix, that's why there are two sample sizes 'n1' and 'n2'.
        beta (int, default=1): descriptive integer of the Manova ensemble type.
            For MRE beta=1, for WME beta=2, for MQE beta=4.
        sigma (float): scale of the distribution. This value also corresponds
            to the standard deviation of the random entries of the sampled matrix.
        lambda_minus (float): lower bound of the support of the Manova spectrum distribution.
            It depends on beta, on the scale (sigma) and on the ratio.
        lambda_plus (float): upper bound of the support of the Manova spectrum distribution.
            It depends on beta, on the scale (sigma) and on the ratio.
    
    References:
        - Erdos, L. and Farrell, B.
            "Local Eigenvalue Density for General MANOVA Matrices".
            Journal of Statistical Physics. 152.6 (2013): 1003-1032.
    
    """

    def __init__(self, a, b, beta=1):
        """Constructor for ManovaSpectrumDistribution class.

        Initializes an instance of this class with the given parameters.

        Args:
            a (float): first random matrix size ratio. This is the ratio between the
                number of degrees of freedom 'p' and the first sample size 'n1'. The value
                of a = p/n1. Remember a Manova randon matrix is considered a double-Wishart
                matrix, that's why there are two sample sizes 'n1' and 'n2' (see below).
            b (float): second random matrix size ratio. This is the ratio between the
                number of degrees of freedom 'p' and the second sample size 'n2'. The value
                of b = p/n2. Remember a Manova randon matrix is considered a double-Wishart
                matrix, that's why there are two sample sizes 'n1' and 'n2'.
            beta (int, default=1): descriptive integer of the Manova ensemble type.
                For MRE beta=1, for WME beta=2, for MQE beta=4.
        
        """
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
        """Samples random variates following this distribution.

        Args:
            size (int): sample size.
        
        Returns:
            numpy array with the generated samples.
        """
        if size <= 0:
            raise ValueError(f"Error: invalid sample size. It has to be positive. Provided size = {size}.")
        
        _n1 = int(np.round(size * self.a))
        _n2 = int(np.round(size * self.b))

        if not self._manova_ens:
            self._manova_ens = ManovaEnsemble(beta=self.beta, m=size, n1=_n1, n2=_n2)
        else:
            self._manova_ens.set_size(m=size, n1=_n1, n2=_n2, resample_mtx=True)
        
        # Here, _eigval_norm_const = 1.0
        if self.beta == 4:
            return self._manova_ens.eigvals()[::2].real
        return self._manova_ens.eigvals().real
    
    def pdf(self, x):
        """Computes PDF of the Manova Spectrum distribution.

        Args:
            x (float or ndarray): value or (numpy) array of values in which compute the PDF.
        
        Returns:
            float or numpy array with the computed PDF in the given value(s).
        
        """
        with np.errstate(divide='ignore', invalid='ignore'): 
            return np.where(np.logical_and(x > self.lambda_minus, x < self.lambda_plus),
                            (self.a + self.b) * np.sqrt((self.lambda_plus - x) * (x - self.lambda_minus)) \
                                / (2.0 * np.pi * x * (1-x)),
                            0.0)
    
    def __cdf(self, x):
        if x <= self.lambda_minus:
            return 0.0

        if x >= self.lambda_plus:
            return 1.0

        return quad(self.pdf, self.lambda_minus, x)[0]

    def cdf(self, x):
        """Computes CDF of the Manova Spectrum distribution.

        Args:
            x (float or ndarray): value or (numpy) array of values in which compute the CDF.
        
        Returns:
            float or numpy array with the computed CDF in the given value(s).
        
        """
        # if x is array-like
        if isinstance(x, (collections.abc.Sequence, np.ndarray)):
            y_ret = []
            for val in x:
                y_ret.append(self.__cdf(val))
            return np.asarray(y_ret)
        
        # if x is a number (int or float)
        return self.__cdf(x)

    def plot_pdf(self, interval=None, bins=1000, savefig_path=None):
        """Plots the PDF of the Manova Spectrum distribution.

        Args:
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram. If not
                provided, the used interval is calculated depending on beta, a, and b.
            bins (int, default=100): It defines the number of equal-width bins within the
                provided interval or range.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown at the end of the routine.
        
        """
        if interval is None:
            interval = (self.lambda_minus, self.lambda_plus)
        
        _plot_func(
            interval, func=self.pdf, bins=bins, 
            plot_ylabel="probability density", savefig_path=savefig_path
        )

    def plot_cdf(self, interval=None, bins=1000, savefig_path=None):
        """Plots the CDF of the Manova Spectrum distribution.

        Args:
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram. If not
                provided, the used interval is calculated depending on beta, a, and b.
            bins (int, default=100): It defines the number of equal-width bins within the
                provided interval or range.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown at the end of the routine.
        
        """
        if interval is None:
            interval = (self.lambda_minus, self.lambda_plus)

        _plot_func(
            interval, func=self.cdf, bins=bins, 
            plot_ylabel="cumulative distribution", savefig_path=savefig_path
        )

    def plot_empirical_pdf(self, m_size=1000, n1_size=3000, n2_size=3000, bins=100, interval=None,
                           density=False, plot_law_pdf=False, savefig_path=None):
        """Computes and plots Manova spectrum empirical pdf and analytical distribution.

        Calculates and plots Manova spectrum empirical pdf using Manova Ensemble random matrices.

        Args:
            m_size (int, default=1000): number of rows of the two Wishart Ensemble matrices that
                generates the matrix of the corresponding ensemble.
            n1_size (int, default=3000): number of columns of the first Wishart Ensemble matrix
                that generates the matrix of the corresponding ensemble.
            n2_size (int, default=3000): number of columns of the second Wishart Ensemble matrix
                that generates the matrix of the corresponding ensemble.    
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
            plot_law_pdf (bool, default=False): If True, the limiting theoretical law is plotted.
                If set to False, just the empirical histogram is shown. This parameter is only
                considered when the argument 'density' is set also to True.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown are the end of the routine.

        References:
            - Laszlo, L. and Farrel, B.
                "Local Eigenvalue Density for General MANOVA Matrices".
                Journal of Statistical Physics. 152.6 (2013): 1003-1032.
            - Albrecht, J. and Chan, C.P. and Edelman, A.
                "Sturm sequences and random eigenvalue distributions".
                Foundations of Computational Mathematics. 9.4 (2008): 461-483.
            - Dumitriu, I. and Edelman, A.
                "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        """
        if m_size<1 or n1_size<1 or n2_size<1:
            raise ValueError("matrix size must be positive")
        
        ens = ManovaEnsemble(beta=self.beta, m=m_size, n1=n1_size, n2=n2_size)

        a = n1_size/m_size
        b = n2_size/m_size
        if a <= 1 or b <= 1:
            print("Warning: sample size ('n1_size' or 'n2_size') too small compared \
                to degrees of freedom ('m_size'). It may cause numerical instability.")
        lambda_term1 = np.sqrt((a/(a+b)) * (1 - (1/(a+b))))
        lambda_term2 = np.sqrt((1/(a+b)) * (1 - (a/(a+b))))
        lambda_minus = (lambda_term1 - lambda_term2)**2
        lambda_plus = (lambda_term1 + lambda_term2)**2

        if interval is None:
            interval = [lambda_minus, lambda_plus]
            if a <= 1:
                interval[0] = min(-0.05, lambda_minus)
            if b <= 1:
                interval[1] = max(lambda_plus, 1.05)
            interval = tuple(interval)

        observed, bins = ens.eigval_hist(bins=bins, interval=interval, density=density, avoid_img=True)

        width = bins[1]-bins[0]
        plt.bar(bins[:-1], observed, width=width, align='edge')

        # Plotting theoretical graphic
        if plot_law_pdf and density:
            centers = np.array(_get_bins_centers_and_contour(bins))
            # creating new instance with the updated ratios depending on the given matrix sizes
            msd = ManovaSpectrumDistribution(beta=self.beta, a=a, b=b)
            pdf = msd.pdf(centers)
            plt.plot(centers, pdf, color='red', linewidth=2)

        plt.title("Manova Spectrum - Empirical density histogram", fontweight="bold")
        plt.xlabel("x")
        plt.ylabel("probability density")
        if a <= 1 or b <= 1:
            if plot_law_pdf and density:
                ylim_vals = pdf
            else:
                ylim_vals = observed
            try:
                plt.ylim(0, np.max(ylim_vals) + 0.25*np.max(ylim_vals))
            except ValueError:
                second_highest_val = np.partition(ylim_vals.flatten(), -2)[-2]
                plt.ylim(0, second_highest_val + 0.25*second_highest_val)

        # Saving plot or showing it
        if savefig_path:
            plt.savefig(savefig_path, dpi=1200)
        else:
            plt.show()
