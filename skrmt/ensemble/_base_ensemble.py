"""Base Ensemble Module

This module contains the general implementation of the matrix ensembles.
This file contains the common attributes and methods for all the
random matrix ensembles. It also defines the basic interface to be
supported by inherited classes.

"""

from abc import ABCMeta, abstractmethod

import numpy as np
import matplotlib.pyplot as plt


#########################################################################
### ABSTRACT CLASS: ENSEMBLE

class _Ensemble(metaclass=ABCMeta):
    """General abstract ensemble class.

    This class contains common attributes and methods for all the
    ensembles. It also defines the basic interface to be
    supported by inherited classes.

    Attributes:
        matrix (numpy array): instance of the random matrix ensemble
            of size n times n.

    """

    @abstractmethod
    def __init__(self):
        self.matrix = None

    @abstractmethod
    def sample(self):
        """Samples new random matrix.

        The sampling algorithm depends on the inherited classes, so it should be
        specified by them.

        Returns:
            numpy array containing new matrix sampled.
        """
        # pylint: disable=unnecessary-pass
        pass

    @abstractmethod
    def set_size(self):
        # pylint: disable=unnecessary-pass
        # pylint: disable=missing-function-docstring
        # this will be commented at inherited classes
        pass

    @abstractmethod
    def eigvals(self):
        # pylint: disable=unnecessary-pass
        # pylint: disable=missing-function-docstring
        # this will be commented at inherited classes
        pass

    @abstractmethod
    def eigval_pdf(self):
        # pylint: disable=unnecessary-pass
        # pylint: disable=missing-function-docstring
        # this will be commented at inherited classes
        pass

    def eigval_hist(self, bins, interval=None, density=False, norm_const=None):
        """Calculates the histogram of the matrix eigenvalues

        Calculates the histogram of the current sampled matrix eigenvalues. Some ensembles
        like Gaussian (Hermite) ensemble or Wishart (Laguerre) ensemble might have
        improved routines to avoid calculating the eigenvalues, so instead the histogram
        is built using certain techniques to boost efficiency.
        It is important to underline that this function works with real eigenvalues,
        if the matrix eigenvalues are complex, they are casted to its real part.

        Args:
            bins (int or sequence): If bins is an integer, it defines the number of
                equal-width bins in the range. If bins is a sequence, it defines the
                bin edges, including the left edge of the first bin and the right
                edge of the last bin; in this case, bins may be unequally spaced.
            interval (tuple): Delimiters (xmin, xmax) of the histogram.
                The lower and upper range of the bins. Lower and upper outliers are ignored.
            density (bool, default=False): If True, draw and return a probability
                density: each bin will display the bin's raw count divided by the total
                number of counts and the bin width, so that the area under the histogram
                integrates to 1. If set to False, the absolute frequencies of the eigenvalues
                are returned.
            norm_const (float, default=None): Eigenvalue normalization constant. By default,
                it is set to None, so eigenvalues are not normalized. However, it is advisable
                to specify a normalization constant to observe eigenvalue spectrum, e.g.
                1/sqrt(n/2) if you want to analyze Wigner's Semicircular Law.

        Returns:
            (tuple) tuple containing:
                observed (array): List of eigenvalues frequencies per bin. If density is
                True these values are the relative frequencies in order to get an area under
                the histogram equal to 1. Otherwise, this list contains the absolute
                frequencies of the eigenvalues.
                bins (array): The edges of the bins. Length nbins + 1 (nbins left edges and
                right edge of last bin)

        Raises:
            ValueError if interval is not a tuple.

        References:
            Albrecht, J. and Chan, C.P. and Edelman, A.
                "Sturm sequences and random eigenvalue distributions".
                Foundations of Computational Mathematics. 9.4 (2008): 461-483.
            Dumitriu, I. and Edelman, A.
                "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        """
        if interval is not None:
            if not isinstance(interval, tuple):
                raise ValueError("interval argument must be a tuple")

        # calculating eigenvalues using standard algorithm
        eigvals = self.eigvals()

        if norm_const:
            eigvals = norm_const*eigvals

        # using numpy to obtain histogram in the given interval and no. of bins
        observed, bins = np.histogram(eigvals, bins=bins, range=interval, density=density)
        return observed, bins


    def plot_eigval_hist(self, bins, interval=None, density=False, norm_const=None, fig_path=None):
        """Calculates and plots the histogram of the matrix eigenvalues

        Calculates and plots the histogram of the current sampled matrix eigenvalues.
        Some ensembles like Gaussian (Hermite) ensemble or Wishart (Laguerre) ensemble
        have improved routines to avoid calculating the eigenvalues, so the histogram
        is built using certain techniques to boost efficiency. It is important to
        underline that this function works with real and complex eigenvalues: if the
        matrix eigenvalues are complex, they are plotted in the complex plane next to a
        heap map to study eigenvalue density.

        Args:
            bins (int or sequence): If bins is an integer, it defines the number of
                equal-width bins in the range. If bins is a sequence, it defines the
                bin edges, including the left edge of the first bin and the right
                edge of the last bin; in this case, bins may be unequally spaced.
            interval (tuple): Delimiters (xmin, xmax) of the histogram.
                The lower and upper range of the bins. Lower and upper outliers are ignored.
            density (bool, default=False): If True, draw and return a probability
                density: each bin will display the bin's raw count divided by the total
                number of counts and the bin width, so that the area under the histogram
                integrates to 1. If set to False, the absolute frequencies of the eigenvalues
                are returned.
            norm_const (float, default=None): Eigenvalue normalization constant. By default,
                it is set to None, so eigenvalues are not normalized. However, it is advisable
                to specify a normalization constant to observe eigenvalue spectrum, e.g.
                1/sqrt(n/2) if you want to analyze Wigner's Semicircular Law.
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
        if not isinstance(interval, tuple):
            raise ValueError("interval argument must be a tuple")

        observed, bins = self.eigval_hist(bins=bins, interval=interval,
                                          density=density, norm_const=norm_const)
        width = bins[1]-bins[0]
        plt.bar(bins[:-1], observed, width=width, align='edge')

        plt.title("Eigenvalue density histogram (matrix size: "+\
                      str(len(self.matrix))+"x"+str(len(self.matrix))+")", fontweight="bold")
        plt.xlabel("x")
        plt.ylabel("density")

        # Saving plot or showing it
        if fig_path:
            plt.savefig(fig_path, dpi=1200)
        else:
            plt.show()
