"""Spectral Law module

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
from scipy.stats import rv_continuous
from scipy import interpolate
import collections.abc

from .tracy_widom_approximator import TW_Approximator
from .misc import relu, indicator, plot_func, get_bins_centers_and_contour


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
        self.default_interval = (self.center - self.radius, self.center + self.radius)

    def rvs(self, size, random_state: int = None):
        """Samples ranfom variates following this distribution.
        This uses the relationship between Wigner Semicircle law and Beta distribution.

        Args:
            size (int): sample size.
            random_state (int, default=None): random seed to initialize the pseudo-random
                number generator of numpy. This has to be any integer between 0 and 2**32 - 1
                (inclusive), or None (default). If None, the seed is obtained from the clock.
        
        Returns:
            numpy array with the generated samples.
        """
        if size <= 0:
            raise ValueError(f"Error: invalid sample size. It has to be positive. Provided size = {size}.")
        
        if random_state is not None:
            np.random.seed(random_state)

        # Use relationship with beta distribution
        beta_samples = np.random.beta(1.5, 1.5, size=size)
        return self.center + 2*self.radius*beta_samples - self.radius

    def pdf(self, x):
        """Computes PDF of the Wigner Semicircle Law.

        Args:
            x (float or ndarray): value or (numpy) array of values in which compute the PDF.
        
        Returns:
            float or numpy array with the computed PDF in the given value(s).
        
        """
        return 2.0 * np.sqrt(relu(self.radius**2 - (x-self.center)**2)) / (np.pi * self.radius**2)
    
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
    
    def plot_pdf(self, interval=None, num_x_vals=1000, savefig_path=None):
        """Plots the PDF of the Wigner Semicircle Law.

        Args:
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram. If not
                provided, the used interval is calculated depending on beta, center, radius and scale.
            num_x_vals (int, default=100): It defines the number of evenly spaced x values
                within the given interval or range in which the function (callable) is evaluated.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown at the end of the routine.
        
        """
        if interval is None:
            interval = (self.center - self.radius - 0.1, self.center + self.radius + 0.1)
        
        plot_func(
            interval, func=self.pdf, num_x_vals=num_x_vals, plot_title="Wigner Semicircle law PDF", 
            plot_ylabel="probability density", savefig_path=savefig_path
        )
    
    def plot_cdf(self, interval=None, num_x_vals=1000, savefig_path=None):
        """Plots the CDF of the Wigner Semicircle Law.

        Args:
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram. If not
                provided, the used interval is calculated depending on beta, center, radius and scale.
            num_x_vals (int, default=100): It defines the number of evenly spaced x values
                within the given interval or range in which the function (callable) is evaluated.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown at the end of the routine.
        
        """
        if interval is None:
            interval = (self.center - self.radius - 0.1, self.center + self.radius + 0.1)
        
        plot_func(
            interval, func=self.cdf, num_x_vals=num_x_vals, plot_title="Wigner Semicircle law CDF",
            plot_ylabel="cumulative distribution", savefig_path=savefig_path
        )

    def plot_empirical_pdf(self, sample_size=10000, bins=100, interval=None, density=False,
                           plot_law_pdf=False, savefig_path=None, random_state=None):
        """Computes and plots Wigner's semicircle empirical law.

        Calculates and plots Wigner's semicircle empirical law using random samples generated
        using the relationship between the Wigner Semicircle law and the Beta distribution:
        the Wigner's Semicircle distribution it is a scaled Beta distribution with parameters
        :math:`\alpha = \beta = 3/2`.

        Args:
            sample_size (int, default=1000): number of random samples that can be interpreted as
                random eigenvalues of a Wigner matrix. This is the sample size.
            bins (int or sequence, default=100): If bins is an integer, it defines the number
                of equal-width bins in the range. If bins is a sequence, it defines the
                bin edges, including the left edge of the first bin and the right
                edge of the last bin; in this case, bins may be unequally spaced.
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram.
                The lower and upper range of the bins. Lower and upper outliers are ignored.
                If one of the bounds of the specified interval is outside of the minimum default
                interval, this will be adjusted to show the distribution bulk properly.
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
            random_state (int, default=None): random seed to initialize the pseudo-random
                number generator of numpy. This has to be any integer between 0 and 2**32 - 1
                (inclusive), or None (default). If None, the seed is obtained from the clock.

        References:
            - Albrecht, J. and Chan, C.P. and Edelman, A.
                "Sturm sequences and random eigenvalue distributions".
                Foundations of Computational Mathematics. 9.4 (2008): 461-483.
            - Dumitriu, I. and Edelman, A.
                "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        """
        # pylint: disable=too-many-arguments
        if sample_size<1:
            raise ValueError("Negative eigenvalue sample size (given = {sample_size})."
                             " Please, provide a sample size greater or equal than 1.")
        
        if interval is None:
            range_interval = self.default_interval
        else:
            xmin = min(interval[0], self.default_interval[0])
            xmax = max(interval[1], self.default_interval[1])
            if interval[0] > self.default_interval[0]:
                print(f"Lower bound of interval too large. Setting lower bound to {xmin}.")
            if interval[1] < self.default_interval[1]:
                print(f"Upper bound of interval too large. Setting upper bound to {xmax}.")
            range_interval = (xmin, xmax)
            print(f"Setting plot interval to {range_interval}.")

        random_samples = self.rvs(size=sample_size, random_state=random_state)
        observed, bin_edges = np.histogram(random_samples, bins=bins, range=range_interval, density=density)

        width = bin_edges[1]-bin_edges[0]
        plt.bar(bin_edges[:-1], observed, width=width, align='edge')

        # Plotting Wigner Semicircle Law pdf
        if plot_law_pdf and density:
            centers = np.asarray(get_bins_centers_and_contour(bin_edges))
            pdf = self.pdf(centers)
            plt.plot(centers, pdf, color='red', linewidth=2)
        elif plot_law_pdf and not density:
            print("Warning: Wigner's Semicircle Law PDF is only plotted when density is True.")

        plt.title("Wigner Semicircle Law - Eigenvalue histogram", fontweight="bold")
        plt.xlabel("x")
        plt.ylabel("probability density")

        # Saving plot or showing it
        if savefig_path:
            plt.savefig(savefig_path, dpi=1200)
        else:
            plt.show()



class MarchenkoPasturDistribution(rv_continuous):
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
            ratio (float): random matrix size ratio (:math:`\lambda`). This is the ratio
                between the number of degrees of freedom :math:`p` and the sample size :math:`n`.
                The value of ratio is computed as :math:`\lambda = p/n`.
            beta (int, default=1): descriptive integer of the Wishart ensemble type (:math:`\beta`).
                For WRE beta=1, for WCE beta=2, for WQE beta=4.
            sigma (float, default=1.0): scale of the distribution (:math:`\sigma`). This value also
                corresponds to the standard deviation of the random entries of the sampled matrix.
        
        """
        super().__init__()

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
        self._set_default_interval()
        # when the support is finite (lambda_minus, lambda_plus) it is better
        # to explicity approximate the inverse of the CDF to implement rvs
        self._approximate_inv_cdf()
    
    def _set_default_interval(self):
        # computing interval according to the matrix size ratio and support
        if self.ratio <= 1:
            self.default_interval = (self.lambda_minus, self.lambda_plus)
        else:
            self.default_interval = (min(-0.05, self.lambda_minus), self.lambda_plus)
    
    def _approximate_inv_cdf(self):
        # https://gist.github.com/amarvutha/c2a3ea9d42d238551c694480019a6ce1
        x_vals = np.linspace(self.lambda_minus, self.lambda_plus, 1000)
        _pdf = self._pdf(x_vals)
        _cdf = np.cumsum(_pdf)      # approximating CDF
        cdf_y = _cdf/_cdf.max()     # normalizing approximated CDF to 1.0
        self._inv_cdf = interpolate.interp1d(cdf_y, x_vals)

    def _rvs(self, size, random_state, _random_state=None):
        if _random_state is not None:
            np.random.seed(_random_state)

        uniform_samples = np.random.random(size=size)
        return self._inv_cdf(uniform_samples)

    def _pdf(self, x):
        """Computes PDF of the Marchenko-Pastur Law.

        Args:
            x (float or ndarray): value or (numpy) array of values in which compute the PDF.
        
        Returns:
            float or numpy array with the computed PDF in the given value(s).
        
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.sqrt(relu(self.lambda_plus - x) * relu(x - self.lambda_minus)) \
                / (2.0 * np.pi * self.ratio * self._var * x)

    def _cdf(self, x):
        """Computes CDF of the Marchenko-Pastur Law.

        Args:
            x (float or ndarray): value or (numpy) array of values in which compute the CDF.
        
        Returns:
            float or numpy array with the computed CDF in the given value(s).
        
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            acum = indicator(x, start=self.lambda_plus, inclusive="left")
            acum += np.where(indicator(x, start=self.lambda_minus, stop=self.lambda_plus, inclusive="left"),
                            self._cdf_aux_f(x), 0.0)

            if self.ratio <= 1:
                return acum
            
            acum += np.where(indicator(x, start=self.lambda_minus, stop=self.lambda_plus, inclusive="left"),
                            (self.ratio-1)/(2*self.ratio), 0.0)

            ### This would need to be added if the extra density point at zero is measured
            # https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution
            # acum += np.where(indicator(x, start=0, stop=self.lambda_minus, inclusive="left"),
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
                                         + (1/self._var)*np.sqrt(relu(self.lambda_plus-x)*relu(x-self.lambda_minus)) \
                                         - (1+self.ratio)*first_arctan_term + (1-self.ratio)*second_arctan_term)

    def _cdf_aux_r(self, x):
        return np.sqrt((self.lambda_plus-x)/(x - self.lambda_minus))

    def plot_pdf(self, interval=None, num_x_vals=1000, savefig_path=None):
        """Plots the PDF of the Marchenko-Pastur Law.

        Args:
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram. If not
                provided, the used interval is calculated depending on beta, ratio, and scale.
            num_x_vals (int, default=100): It defines the number of evenly spaced x values
                within the given interval or range in which the function (callable) is evaluated.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown at the end of the routine.
        
        """
        if interval is None:
            interval = (self.lambda_minus, self.lambda_plus)
        
        plot_func(
            interval, func=self._pdf, num_x_vals=num_x_vals, plot_title="Marchenko-Pastur law PDF",
            plot_ylabel="probability density", savefig_path=savefig_path
        )
    
    def plot_cdf(self, interval=None, num_x_vals=1000, savefig_path=None):
        """Plots the CDF of the Marchenko-Pastur Law.

        Args:
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram. If not
                provided, the used interval is calculated depending on beta, ratio, and scale.
            num_x_vals (int, default=100): It defines the number of evenly spaced x values
                within the given interval or range in which the function (callable) is evaluated.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown at the end of the routine.
        
        """
        if interval is None:
            interval = (self.lambda_minus, self.lambda_plus)
        
        plot_func(
            interval, func=self._cdf, num_x_vals=num_x_vals, plot_title="Marchenko-Pastur law CDF",
            plot_ylabel="cumulative distribution", savefig_path=savefig_path
        )

    def plot_empirical_pdf(self, sample_size=1000, bins=100, interval=None, density=False,
                           plot_law_pdf=False, savefig_path=None, random_state=None):
        """Computes and plots Marchenko-Pastur empirical law using Wishart Ensemble random matrices.

        Calculates and plots Marchenko-Pastur empirical law using Wishart Ensemble random matrices.
        The size of the sampled matrix will depend on the `n_size` (:math:`n`) parameter and on the
        ratio :math:`\lambda` given when instantiating this class, unless the parameter `p_size` (:math:`p`)
        is also given. In this last case, the ratio :math:`\lambda` for the empirical pdf plotting is
        computed as :math:`\lambda = p/n`. If only the sample size :math:`n` is provided, the number
        of degrees of freedom :math:`p` is computed as :math:`[\lambda * n]`.
        Wishart (Laguerre) ensemble has improved routines (using tridiagonal forms and Sturm
        sequences) to avoid calculating the eigenvalues, so the histogram is built using certain
        techniques to boost efficiency. This optimization is only used when the ratio p_size/n_size
        is less or equal than 1.

        Args:
            n_size (int, default=3000): number of columns of the guassian matrix that generates
                the matrix of the corresponding ensemble. This is the sample size. The number of
                degrees of freedom is computed depending on this argument and on the given ratio,
                unless the argument `p_size` is also provided, which in this case the ratio is
                re-computed as ratio=p_size/n_size.
            p_size (int, default=None): number of rows of the guassian matrix that generates
                the matrix of the corresponding ensemble. If provided, the current ratio is ignored
                (but not replaced) and the new ratio=p_size/n_size is used instead.
            bins (int or sequence, default=100): If bins is an integer, it defines the number
                of equal-width bins in the range. If bins is a sequence, it defines the
                bin edges, including the left edge of the first bin and the right
                edge of the last bin; in this case, bins may be unequally spaced.
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram.
                The lower and upper range of the bins. Lower and upper outliers are ignored.
                If one of the bounds of the specified interval is outside of the minimum default
                interval, this will be adjusted to show the distribution bulk properly.
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
            random_state (int, default=None): random seed to initialize the pseudo-random
                number generator of numpy. This has to be any integer between 0 and 2**32 - 1
                (inclusive), or None (default). If None, the seed is obtained from the clock.

        References:
            - Albrecht, J. and Chan, C.P. and Edelman, A.
                "Sturm sequences and random eigenvalue distributions".
                Foundations of Computational Mathematics. 9.4 (2008): 461-483.
            - Dumitriu, I. and Edelman, A.
                "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        """
        # pylint: disable=too-many-arguments
        if sample_size<1:
            raise ValueError("Negative eigenvalue sample size (given = {sample_size})."
                             " Please, provide a sample size greater or equal than 1.")

        if interval is None:
            range_interval = self.default_interval
        else:
            xmin = min(interval[0], self.default_interval[0])
            xmax = max(interval[1], self.default_interval[1])
            if interval[0] > self.default_interval[0]:
                print(f"Lower bound of interval too large. Setting lower bound to {xmin}.")
            if interval[1] < self.default_interval[1]:
                print(f"Upper bound of interval too large. Setting upper bound to {xmax}.")
            range_interval = (xmin, xmax)
            print(f"Setting plot interval to {range_interval}.")
        
        random_samples = self._rvs(size=sample_size, random_state=random_state, _random_state=random_state)
        observed, bin_edges = np.histogram(random_samples, bins=bins, range=range_interval, density=density)

        width = bin_edges[1]-bin_edges[0]
        plt.bar(bin_edges[:-1], observed, width=width, align='edge')

        # Plotting theoretical graphic
        if plot_law_pdf and density:
            centers = np.asarray(get_bins_centers_and_contour(bin_edges))
            # creating new instance with the approximated ratio depending on the given matrix sizes
            pdf = self._pdf(centers)
            plt.plot(centers, pdf, color='red', linewidth=2)

        plt.title("Marchenko-Pastur Law - Eigenvalue histogram", fontweight="bold")
        plt.xlabel("x")
        plt.ylabel("probability density")
        if self.ratio > 1:
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



class TracyWidomDistribution(rv_continuous):
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
        super().__init__()

        if beta not in [1,2,4]:
            raise ValueError(f"Error: invalid beta. It has to be 1,2 or 4. Provided beta = {beta}.")

        self.beta = beta
        self.tw_approx = TW_Approximator(beta=self.beta)
        self.default_interval = (-5, 4-self.beta)

    def _pdf(self, x):
        """Computes PDF of the Tracy-Widom Law.

        Args:
            x (float or ndarray): value or (numpy) array of values in which compute the PDF.
        
        Returns:
            float or numpy array with the computed PDF in the given value(s).
        
        """
        return self.tw_approx.pdf(x)

    def _cdf(self, x):
        """Computes CDF of the Tracy-Widom Law.

        Args:
            x (float or ndarray): value or (numpy) array of values in which compute the CDF.
        
        Returns:
            float or numpy array with the computed CDF in the given value(s).
        
        """
        return self.tw_approx.cdf(x)

    def plot_pdf(self, interval=None, num_x_vals=1000, savefig_path=None):
        """Plots the PDF of the Tracy-Widom Law.

        Args:
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram. If not
                provided, the used interval is calculated depending on beta.
            num_x_vals (int, default=100): It defines the number of evenly spaced x values
                within the given interval or range in which the function (callable) is evaluated.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown at the end of the routine.
        
        """
        if interval is None:
            interval = self.default_interval
        
        plot_func(
            interval, func=self._pdf, num_x_vals=num_x_vals, plot_title="Tracy-Widom law PDF",
            plot_ylabel="probability density", savefig_path=savefig_path
        )
    
    def plot_cdf(self, interval=None, num_x_vals=1000, savefig_path=None):
        """Plots the PDF of the Tracy-Widom Law.

        Args:
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram. If not
                provided, the used interval is calculated depending on beta.
            num_x_vals (int, default=100): It defines the number of evenly spaced x values
                within the given interval or range in which the function (callable) is evaluated.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown at the end of the routine.
        
        """
        if interval is None:
            interval = self.default_interval
        
        plot_func(
            interval, func=self._cdf, num_x_vals=num_x_vals, plot_title="Tracy-Widom law CDF",
            plot_ylabel="cumulative distribution", savefig_path=savefig_path
        )

    def plot_empirical_pdf(self, sample_size=1000, bins=100, interval=None, density=False,
                           plot_law_pdf=False, savefig_path=None, random_state=None):
        """Computes and plots Tracy-Widom empirical law using Gaussian Ensemble.

        Calculates and plots Tracy-Widom empirical law using Gaussian Ensemble random matrices.
        Because we need to obtain the largest eigenvalue of each sampled random matrix,
        we need to sample a certain amount them. For each random matrix sammpled, its
        largest eigenvalue is calcualted in order to simulate its density.

        Args:
            n_size (int, default=100): random matrix size n times n. This is the sample size.
            times (int, default=1000): number of times to sample a random matrix.
            bins (int or sequence, default=100): If bins is an integer, it defines the number
                of equal-width bins in the range. If bins is a sequence, it defines the
                bin edges, including the left edge of the first bin and the right
                edge of the last bin; in this case, bins may be unequally spaced.
            density (bool, default=False): If True, draw and return a probability
                density: each bin will display the bin's raw count divided by the total
                number of counts and the bin width, so that the area under the histogram
                integrates to 1. If set to False, the absolute frequencies of the eigenvalues
                are returned.
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram.
                The lower and upper range of the bins. Lower and upper outliers are ignored.
                If one of the bounds of the specified interval is outside of the minimum default
                interval, this will be adjusted to show the distribution bulk properly.
            plot_law_pdf (bool, default=False): If True, the limiting theoretical law is plotted.
                If set to False, just the empirical histogram is shown. This parameter is only
                considered when the argument 'density' is set also to True.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown are the end of the routine.
            random_state (int, default=None): random seed to initialize the pseudo-random
                number generator of numpy. This has to be any integer between 0 and 2**32 - 1
                (inclusive), or None (default). If None, the seed is obtained from the clock.

        References:
            - Albrecht, J. and Chan, C.P. and Edelman, A.
                "Sturm sequences and random eigenvalue distributions".
                Foundations of Computational Mathematics. 9.4 (2008): 461-483.
            - Dumitriu, I. and Edelman, A.
                "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        """
        # pylint: disable=too-many-arguments
        if sample_size<1:
            raise ValueError("Negative eigenvalue sample size (given = {sample_size})."
                             " Please, provide a sample size greater or equal than 1.")

        if interval is None:
            range_interval = self.default_interval
        else:
            xmin = min(interval[0], self.default_interval[0])
            xmax = max(interval[1], self.default_interval[1])
            if interval[0] > self.default_interval[0]:
                print(f"Lower bound of interval too large. Setting lower bound to {xmin}.")
            if interval[1] < self.default_interval[1]:
                print(f"Upper bound of interval too large. Setting upper bound to {xmax}.")
            range_interval = (xmin, xmax)
            print(f"Setting plot interval to {range_interval}.")

        random_samples = self.rvs(size=sample_size, random_state=random_state)
        observed, bin_edges = np.histogram(random_samples, bins=bins, range=range_interval, density=density)

        width = bin_edges[1]-bin_edges[0]
        plt.bar(bin_edges[:-1], observed, width=width, align='edge')

        # Plotting theoretical graphic
        if plot_law_pdf and density:
            centers = get_bins_centers_and_contour(bin_edges)
            pdf = self._pdf(centers)
            plt.plot(centers, pdf, color='red', linewidth=2)

        plt.title("Tracy-Widom Law - Eigenvalue histogram", fontweight="bold")
        plt.xlabel("x")
        plt.ylabel("probability density")

        # Saving plot or showing it
        if savefig_path:
            plt.savefig(savefig_path, dpi=1200)
        else:
            plt.show()



class ManovaSpectrumDistribution(rv_continuous):
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

    def __init__(self, ratio_a, ratio_b, beta=1):
        """Constructor for ManovaSpectrumDistribution class.

        Initializes an instance of this class with the given parameters.

        Args:
            ratio_a (float): first random matrix size ratio. This is the ratio between the
                number of degrees of freedom 'p' and the first sample size 'n1'. The value
                of a = p/n1. Remember a Manova randon matrix is considered a double-Wishart
                matrix, that's why there are two sample sizes 'n1' and 'n2' (see below).
            ratio_b (float): second random matrix size ratio. This is the ratio between the
                number of degrees of freedom 'p' and the second sample size 'n2'. The value
                of b = p/n2. Remember a Manova randon matrix is considered a double-Wishart
                matrix, that's why there are two sample sizes 'n1' and 'n2'.
            beta (int, default=1): descriptive integer of the Manova ensemble type.
                For MRE beta=1, for WME beta=2, for MQE beta=4.
        
        """
        super().__init__()

        if beta not in [1,2,4]:
            raise ValueError(f"Error: invalid beta. It has to be 1,2 or 4. Provided beta = {beta}.")
        if ratio_a <= 0 or ratio_b <= 0:
            raise ValueError("Error: invalid matrix parameters. They have to be both positive.\n"
                             f"\tProvided a = {ratio_a} and b = {ratio_b}.")

        if ratio_a < 1 or ratio_b < 1:
            print(f"Warning: Setting a < 1 (a = {ratio_a}) or b < 1 (b = {ratio_b}) may cause numerical instability.")

        self.ratio_a = ratio_a
        self.ratio_b = ratio_b
        self.beta = beta
        self.lambda_term1 = np.sqrt((ratio_a/(ratio_a + ratio_b)) * (1 - (1/(ratio_a + ratio_b))))
        self.lambda_term2 = np.sqrt((1/(ratio_a + ratio_b)) * (1 - (ratio_a/(ratio_a + ratio_b))))
        self.lambda_minus = (self.lambda_term1 - self.lambda_term2)**2
        self.lambda_plus = (self.lambda_term1 + self.lambda_term2)**2
        self._set_default_interval()
        # when the support is finite (lambda_minus, lambda_plus) it is better
        # to explicity approximate the inverse of the CDF to implement rvs
        self._approximate_inv_cdf()
    
    def _set_default_interval(self):
        interval = [self.lambda_minus, self.lambda_plus]
        if self.ratio_a <= 1:
            interval[0] = min(-0.05, self.lambda_minus)
        if self.ratio_b <= 1:
            interval[1] = max(self.lambda_plus, 1.05)
        self.default_interval = tuple(interval)

    def _approximate_inv_cdf(self):
        # https://gist.github.com/amarvutha/c2a3ea9d42d238551c694480019a6ce1
        x_vals = np.linspace(self.lambda_minus, self.lambda_plus, 1000)
        _pdf = self._pdf(x_vals)
        _cdf = np.cumsum(_pdf)      # approximating CDF
        cdf_y = _cdf/_cdf.max()     # normalizing approximated CDF to 1.0
        self._inv_cdf = interpolate.interp1d(cdf_y, x_vals)

    def _rvs(self, size, random_state, _random_state=None):
        if _random_state is not None:
            np.random.seed(_random_state)

        uniform_samples = np.random.random(size=size)
        return self._inv_cdf(uniform_samples)
    
    def __pdf_float(self, x):
        if x <= self.lambda_minus:
            return 0.0

        if x >= self.lambda_plus:
            return 0.0

        return (self.ratio_a + self.ratio_b) * np.sqrt((self.lambda_plus - x) * (x - self.lambda_minus)) \
                                / (2.0 * np.pi * x * (1-x))

    def _pdf(self, x):
        """Computes PDF of the Manova Spectrum distribution.

        Args:
            x (float or ndarray): value or (numpy) array of values in which compute the PDF.
        
        Returns:
            float or numpy array with the computed PDF in the given value(s).
        
        """
        # if x is array-like
        if isinstance(x, (collections.abc.Sequence, np.ndarray)):
            # TODO: Vectorize this loop in case x is array-like
            y_ret = []
            for val in x:
                y_ret.append(self.__pdf_float(val))
            return np.asarray(y_ret)
        # if x is a number (int or float)
        return self.__pdf_float(x)
    
    def __cdf_float(self, x):
        if x <= self.lambda_minus:
            return 0.0

        if x >= self.lambda_plus:
            return 1.0

        return quad(self.pdf, self.lambda_minus, x)[0]

    def _cdf(self, x):
        """Computes CDF of the Manova Spectrum distribution.

        Args:
            x (float or ndarray): value or (numpy) array of values in which compute the CDF.
        
        Returns:
            float or numpy array with the computed CDF in the given value(s).
        
        """
        # if x is array-like
        if isinstance(x, (collections.abc.Sequence, np.ndarray)):
            # TODO: Vectorize this loop in case x is array-like
            y_ret = []
            for val in x:
                y_ret.append(self.__cdf_float(val))
            return np.asarray(y_ret)
        
        # if x is a number (int or float)
        return self.__cdf_float(x)

    def plot_pdf(self, interval=None, num_x_vals=1000, savefig_path=None):
        """Plots the PDF of the Manova Spectrum distribution.

        Args:
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram. If not
                provided, the used interval is calculated depending on beta, a, and b.
            num_x_vals (int, default=100): It defines the number of evenly spaced x values
                within the given interval or range in which the function (callable) is evaluated.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown at the end of the routine.
        
        """
        if interval is None:
            interval = (self.lambda_minus, self.lambda_plus)
        
        plot_func(
            interval, func=self._pdf, num_x_vals=num_x_vals, plot_title="Manova spectrum PDF",
            plot_ylabel="probability density", savefig_path=savefig_path
        )

    def plot_cdf(self, interval=None, num_x_vals=1000, savefig_path=None):
        """Plots the CDF of the Manova Spectrum distribution.

        Args:
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram. If not
                provided, the used interval is calculated depending on beta, a, and b.
            num_x_vals (int, default=100): It defines the number of evenly spaced x values
                within the given interval or range in which the function (callable) is evaluated.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown at the end of the routine.
        
        """
        if interval is None:
            interval = (self.lambda_minus, self.lambda_plus)

        plot_func(
            interval, func=self._cdf, num_x_vals=num_x_vals, plot_title="Manova spectrum CDF",
            plot_ylabel="cumulative distribution", savefig_path=savefig_path
        )

    def plot_empirical_pdf(self, sample_size=1000, bins=100, interval=None, density=False,
                           plot_law_pdf=False, savefig_path=None, random_state=None):
        """Computes and plots Manova spectrum empirical pdf and analytical distribution.

        Calculates and plots Manova spectrum empirical pdf using Manova Ensemble random matrices.
        The size of the sampeld matrices will depend on the `m_size` (:math:`m`) parameter (no. of
        degrees of freedom) and on the ratios :math:`a` and :math:`b` given when instantiating this
        class, unless the parameters `n1_size` (:math:`n_1`) and/or `n2_size` (:math:`n_2`) are also
        given. In this last case, the new ratio :math:`a` for the empirical pdf plotting is computed
        as :math:`a = n_1/m`, and the new ratio :math:`b` is calculated as :math:`b = n_2/m`. If only
        the number of degrees of freedom :math:`m` is provided, the sample sizes :math:`n_1` and :math:`n_2`
        are computed as :math:`n_1 = [a * m]` and :math:`n_2 = [b * m]` respectively.

        Args:
            m_size (int, default=1000): number of rows of the two Wishart Ensemble matrices that
                generates the matrix of the corresponding ensemble. This is the number of degrees
                of freedom (:math:`m`). The sample sizes (:math:`n_1` and :math:`n_2`) of the two
                matrices are computed from this value and the ratios :math:`a` and :math:`b` given
                to instantiate this class.
            n1_size (int, default=None): number of columns of the first Wishart Ensemble matrix
                that generates the matrix of the corresponding ensemble (:math:`n_1`). If provided,
                the ratio :math:`a` is ignored (but not replaced) and the new ratio :math:`a = n_1/m`
                is used instead to plot the empirical pdf.
            n2_size (int, default=None): number of columns of the second Wishart Ensemble matrix
                that generates the matrix of the corresponding ensemble (:math:`n_2`). If provided,
                the ratio :math:`b` is ignored (but not replaced) and the new ratio :math:`b = n_2/m`
                is used instead to plot the empirical pdf.
            bins (int or sequence, default=100): If bins is an integer, it defines the number
                of equal-width bins in the range. If bins is a sequence, it defines the
                bin edges, including the left edge of the first bin and the right
                edge of the last bin; in this case, bins may be unequally spaced.
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram.
                The lower and upper range of the bins. Lower and upper outliers are ignored.
                If one of the bounds of the specified interval is outside of the minimum default
                interval, this will be adjusted to show the distribution bulk properly.
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
            random_state (int, default=None): random seed to initialize the pseudo-random
                number generator of numpy. This has to be any integer between 0 and 2**32 - 1
                (inclusive), or None (default). If None, the seed is obtained from the clock.

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
        # pylint: disable=too-many-arguments
        if sample_size<1:
            raise ValueError("Negative eigenvalue sample size (given = {sample_size})."
                             " Please, provide a sample size greater or equal than 1.")

        if interval is None:
            range_interval = self.default_interval
        else:
            xmin = min(interval[0], self.default_interval[0])
            xmax = max(interval[1], self.default_interval[1])
            if interval[0] > self.default_interval[0]:
                print(f"Lower bound of interval too large. Setting lower bound to {xmin}.")
            if interval[1] < self.default_interval[1]:
                print(f"Upper bound of interval too large. Setting upper bound to {xmax}.")
            range_interval = (xmin, xmax)
            print(f"Setting plot interval to {range_interval}.")

        random_samples = self._rvs(size=sample_size, random_state=random_state, _random_state=random_state)
        observed, bin_edges = np.histogram(random_samples, bins=bins, range=range_interval, density=density)

        width = bin_edges[1]-bin_edges[0]
        plt.bar(bin_edges[:-1], observed, width=width, align='edge')

        # Plotting theoretical graphic
        if plot_law_pdf and density:
            centers = np.asarray(get_bins_centers_and_contour(bin_edges))
            # creating new instance with the approximated ratios depending on the given matrix sizes
            pdf = self._pdf(centers)
            plt.plot(centers, pdf, color='red', linewidth=2)

        plt.title("Manova Spectrum - Eigenvalue histogram", fontweight="bold")
        plt.xlabel("x")
        plt.ylabel("probability density")
        if self.ratio_a <= 1 or self.ratio_b <= 1:
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
