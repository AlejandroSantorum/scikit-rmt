"""Base Ensemble Module

This module contains the general implementation of the matrix ensembles.
This file contains the common attributes and methods for all the
random matrix ensembles. It also defines the basic interface to be
supported by inherited classes.

"""

from typing import Union, Sequence, Tuple
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
        _eigvals (numpy array): array of computed eigenvalues. This array
            is None until the method `eigvals` is called. The computed
            eigenvalues are cached in the attribute _eigvals to avoid
            re-computing them. The eigenvalues are re-calculated again
            if the matrix sample changes.
        eigval_norm_cost (float): constant that is used to normalize the
            eigenvalues so their support is always the same independently
            of the sample size. This is really useful for plotting, since
            the representation interval will always be the same no matter
            how many eigenvalues are sampled.
    """

    @abstractmethod
    def __init__(self) -> None:
        self.matrix = None
        self._eigvals = None
        # default eigenvalue normalization constant
        self.eigval_norm_const = 1.0

    @abstractmethod
    def sample(self, random_state: int = None) -> np.ndarray:
        """Samples new random matrix.

        The sampling algorithm depends on the inherited classes, so it should be
        specified by them.

        Args:
            random_state (int, default=None): random seed to initialize the pseudo-random
                number generator of numpy. This has to be any integer between 0 and 2**32 - 1
                (inclusive), or None (default). If None, the seed is obtained from the clock.

        Returns:
            numpy array containing new matrix sampled.
        """
        # pylint: disable=unnecessary-pass
        pass

    @abstractmethod
    def resample(self, random_state: int = None) -> np.ndarray:
        """This is an alias for the method ``sample``. It samples new random matrix.

        The sampling algorithm depends on the inherited classes, so it should be
        specified by them.

        Args:
            random_state (int, default=None): random seed to initialize the pseudo-random
                number generator of numpy. This has to be any integer between 0 and 2**32 - 1
                (inclusive), or None (default). If None, the seed is obtained from the clock.

        Returns:
            numpy array containing new matrix sampled.
        """
        # pylint: disable=unnecessary-pass
        pass

    def _set_eigval_norm_const(self, eigval_norm_const: float) -> None:
        """Sets a custom eigenvalue normalization constant.

        This updates the normalization constant applied to the computed eigenvalues.
        Eigenvalue normalization is useful because normalized eigenvalues always have
        the same support independently of the sample size.

        Args:
            eigval_norm_const (float): new eigenvalue normalization constant.
        """
        # pylint: disable=unnecessary-pass
        self.eigval_norm_const = eigval_norm_const

    @abstractmethod
    def eigvals(self, normalize: bool = False) -> np.ndarray:
        # pylint: disable=unnecessary-pass
        # pylint: disable=missing-function-docstring
        # this will be commented at inherited classes
        pass

    @abstractmethod
    def joint_eigval_pdf(self, eigvals: np.ndarray = None) -> float:
        # pylint: disable=unnecessary-pass
        # pylint: disable=missing-function-docstring
        # this will be commented at inherited classes
        pass

    def eigval_hist(
        self,
        bins: Union[int, Sequence] = 100,
        interval: Tuple = None,
        density: bool = False,
        normalize: bool = False,
        avoid_img: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the histogram of the matrix eigenvalues.

        Calculates the histogram of the current sampled matrix eigenvalues. Some ensembles
        like Gaussian (Hermite) ensemble or Wishart (Laguerre) ensemble might have
        improved routines to avoid calculating the eigenvalues, so instead the histogram
        is built using certain techniques to boost efficiency.
        It is important to underline that this function works with real eigenvalues,
        if the matrix eigenvalues are complex, they are casted to its real part.

        Args:
            bins (int or sequence, default=100): If bins is an integer, it defines the number of
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
            normalize (bool, default=False): Whether to normalize the computed eigenvalues
                by the default normalization constant (see references). Defaults to False,
                i.e., the eigenvalues are normalized. Normalization makes the eigenvalues
                to be in the same support independently of the sample size.
            avoid_img (bool, default=False): If True, eigenvalue imaginary part is ignored.
                This should be used when the eigenvalue compatation is expected to generate
                complex eigenvalues with really small imaginary part because of computing
                rounding errors. E.g.: MANOVA Ensemble eigenvalues.

        Returns:
            (tuple) tuple containing:
                observed (array): List of eigenvalues frequencies per bin. If density is
                True these values are the relative frequencies in order to get an area under
                the histogram equal to 1. Otherwise, this list contains the absolute
                frequencies of the eigenvalues.
                bin_edges (array): The edges of the bins. Length nbins + 1 (nbins left edges
                and right edge of last bin)

        Raises:
            ValueError if interval is not a tuple.

        References:
            - Albrecht, J. and Chan, C.P. and Edelman, A.
                "Sturm sequences and random eigenvalue distributions".
                Foundations of Computational Mathematics. 9.4 (2008): 461-483.
            - Dumitriu, I. and Edelman, A.
                "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        """
        if interval is not None:
            if not isinstance(interval, tuple):
                raise ValueError("interval argument must be a tuple")

        # calculating eigenvalues using standard algorithm
        eigvals = self.eigvals(normalize=normalize)
        # ignoring imaginary part because of computing rounding errors
        if avoid_img:
            eigvals = eigvals.real

        # using numpy to obtain histogram in the given interval and bins
        observed, bin_edges = np.histogram(eigvals, bins=bins, range=interval, density=density)
        return observed, bin_edges

    def _plot_eigval_hist(
        self,
        bins,
        interval=None,
        density=False,
        normalize=False,
        avoid_img=False,
    ):
        # computing eigenvalue histogram
        observed, bin_edges = self.eigval_hist(bins=bins, interval=interval, density=density,
                                               normalize=normalize, avoid_img=avoid_img)
        # plotting eigenvalue histogram
        width = bin_edges[1]-bin_edges[0]
        plt.bar(bin_edges[:-1], observed, width=width, align='edge')
        return observed, bin_edges

    def plot_eigval_hist(
        self,
        bins: Union[int, Sequence],
        interval: Tuple = None,
        density: bool = False,
        normalize: bool = False,
        savefig_path: str = None,
        avoid_img: bool = False,
    ) -> None:
        """Computes and plots the histogram of the matrix eigenvalues.

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
            normalize (bool, default=False): Whether to normalize the computed eigenvalues
                by the default normalization constant (see references). Defaults to False,
                i.e., the eigenvalues are normalized. Normalization makes the eigenvalues
                to be in the same support independently of the sample size.
            avoid_img (bool, default=False): If True, eigenvalue imaginary part is ignored.
                This should be used when the eigenvalue compatation is expected to generate
                complex eigenvalues with really small imaginary part because of computing
                rounding errors. E.g.: MANOVA Ensemble eigenvalues.
            savefig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown at the end of the routine.

        References:
            - Albrecht, J. and Chan, C.P. and Edelman, A.
                "Sturm sequences and random eigenvalue distributions".
                Foundations of Computational Mathematics. 9.4 (2008): 461-483.
            - Dumitriu, I. and Edelman, A.
                "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        """
        # pylint: disable=too-many-arguments
        if interval is not None:
            if not isinstance(interval, tuple):
                raise ValueError("interval argument must be a tuple")

        # histogramming and plotting
        self._plot_eigval_hist(
            bins=bins, interval=interval, density=density, normalize=normalize, avoid_img=avoid_img
        )

        # plot labels, title and illustration
        plt.title("Eigenvalue histogram", fontweight="bold")
        plt.xlabel("x")
        plt.ylabel("density") if density else plt.ylabel("frequency")

        # Saving plot or showing it
        if savefig_path:
            plt.savefig(savefig_path, dpi=1000)
        else:
            plt.show()
