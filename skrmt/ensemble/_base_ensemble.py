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
        pass

    @abstractmethod
    def set_size(self):
        pass

    @abstractmethod
    def eigval_pdf(self):
        pass

    def eigval_hist(self, bins, interval=None, normed_hist=True):
        """Calculates the histogram of the matrix eigenvalues

        Calculates the histogram of the current sampled matrix eigenvalues. Some ensembles
        like Gaussian (Hermite) ensemble or Wishart (Laguerre) ensemble might have
        improved routines to avoid calculating the eigenvalues, so instead the histogram
        is built using certain techniques to boost efficiency.

        Args:
            bins (int or sequence): If bins is an integer, it defines the number of
                equal-width bins in the range. If bins is a sequence, it defines the
                bin edges, including the left edge of the first bin and the right
                edge of the last bin; in this case, bins may be unequally spaced.
            interval (tuple): Delimiters (xmin, xmax) of the histogram.
                The lower and upper range of the bins. Lower and upper outliers are ignored.
            normed_hist (bool, default=True): If True, draw and return a probability
                density: each bin will display the bin's raw count divided by the total
                number of counts and the bin width, so that the area under the histogram
                integrates to 1. If set to False, the absolute frequencies of the eigenvalues
                are returned.
        
        Returns:
            (tuple) tuple containing:
                observed (array): List of eigenvalues frequencies per bin. If normed_hist=True
                these values are the relative frequencies in order to get an area under the histogram
                equal to 1. Otherwise, this list contains the absolute frequencies of the eigenvalues.
                bins (array): The edges of the bins. Length nbins + 1 (nbins left edges and right edge of last bin)
        
        Raises:
            ValueError if interval is not a tuple.

        """
        if not isinstance(interval, tuple):
            raise ValueError("interval argument must be a tuple")

        # calculating eigenvalues using standard algorithm
        eigvals = np.linalg.eigvals(self.matrix)

        # using matplotlib to obtain histogram in the given interval and with the specified no. of bins
        observed, bins = np.histogram(eigvals, bins=bins, range=interval, density=normed_hist)
        #plt.clf()
        return observed, bins
