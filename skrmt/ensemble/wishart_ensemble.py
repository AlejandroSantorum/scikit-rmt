"""Wishart Ensemble Module

This module contains the implementation of the Wishart Ensemble, also
known as Laguerre Ensemble. This ensemble of random matrices contains
mainly three sub-ensembles: Wishart Real Ensemble, Wishart Complex Ensemble
and Wishart Quaternion Ensemble.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy import sparse

from ._base_ensemble import _Ensemble
from .tridiagonal_utils import tridiag_eigval_hist


#########################################################################
### Wishart Ensemble = Laguerre Ensemble

class WishartEnsemble(_Ensemble):
    """General Wishart Ensemble class.

    This class contains common attributes and methods for all the
    Wishart ensembles. Wishart Ensembles are divided in:
    - Wishart Real Ensemble (WRE, beta=1): the random matrices of
    this ensemble are formed by multiplying a random real standard
    gaussian matrix of size p times n by its transpose.
    - Wishart Complex Ensemble (WCE, beta=2): the random matrices
    of this ensemble are formed by multiplying a random complex
    standard gaussian matrix of size p times n by its transpose.
    - Wishart Quaternion Ensemble (WQE, beta=4): the random matrices
    of this ensemble are formed by: sampling two random complex
    standard guassian matrices (X and Y), stacking them to create
    matrix A = [X  Y; -conj(Y)  conj(X)]. Finally matrix A is
    multiplied by its transpose in order to generate a matrix of
    the Wishart Quaternion Ensemble.

    Attributes:
        matrix (numpy array): instance of the WishartReal, WishartComplex
            or WishartQuaternion random ensembles. If it is an instance
            of WishartReal or WishartComplex, the random matrix is of
            size n times n. If it is a WishartQuaternion, the random matrix
            is of size 2n times 2n.
        beta (int): descriptive integer of the Wishart ensemble type.
            For Real beta=1, for Complex beta=2, for Quaternion beta=4.
        p (int): number of rows of the guassian matrix that generates
            the matrix of the corresponding ensemble.
        n (int): number of columns of the guassian matrix that generates
            the matrix of the corresponding ensemble.
        use_tridiagonal (bool): if set to True, Gaussian Ensemble
            matrices are sampled in its tridiagonal form, which has the same
            eigenvalues than its standard form. Otherwise, it is sampled using
            its standard form.

    References:
        Albrecht, J. and Chan, C.P. and Edelman, A.
            "Sturm sequences and random eigenvalue distributions".
            Foundations of Computational Mathematics. 9.4 (2008): 461-483.
        Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
            Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

    """

    def __init__(self, beta, p, n, use_tridiagonal=False):
        """Constructor for WishartEnsemble class.

        Initializes an instance of this class with the given parameters.

        Args:
            beta (int): descriptive integer of the Wishart ensemble type.
                For Real beta=1, for Complex beta=2, for Quaternion beta=4
            p (int): number of rows of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            n (int): number of columns of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            use_tridiagonal (bool, default=False): if set to True, Wishart Ensemble
            matrices are sampled in its tridiagonal form, which has the same
            eigenvalues than its standard form.

        """
        super().__init__()
        # pylint: disable=invalid-name
        self.p = p
        self.n = n
        self.beta = beta
        self.use_tridiagonal = use_tridiagonal
        self.matrix = self.sample()

    def set_size(self, p, n, resample_mtx=False):
        # pylint: disable=arguments-differ
        """Setter of matrix size.

        Sets the matrix size. Useful if it has been initialized with a different value.

        Args:
            p (int): number of rows of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            n (int): number of columns of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            resample_mtx (bool, default=False): If set to True, the ensemble matrix is
                resampled with the new dimensions.

        """
        self.p = p
        self.n = n
        if resample_mtx:
            self.matrix = self.sample()

    # pylint: disable=inconsistent-return-statements
    def sample(self):
        """Samples new Wishart Ensemble random matrix.

        The sampling algorithm depends on the specification of
        use_tridiagonal parameter. If use_tridiagonal is set to True,
        a Wishart Ensemble random matrix in its tridiagonal form
        is sampled. Otherwise, it is sampled using the standard
        form.

        Returns:
            numpy array containing new matrix sampled.

        References:
            Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.
        """
        if self.use_tridiagonal:
            return self.sample_tridiagonal()

        if self.beta == 1:
            return self._sample_wre()
        if self.beta == 2:
            return self._sample_wce()
        if self.beta == 4:
            return self._sample_wqe()

    def _sample_wre(self):
        p_size = self.p
        n_size = self.n
        # p by n matrix of random Gaussians
        mtx = np.random.randn(p_size,n_size)
        # symmetrize matrix
        self.matrix = np.matmul(mtx, mtx.transpose())
        return self.matrix

    def _sample_wce(self):
        p_size = self.p
        n_size = self.n
        # p by n random complex matrix of random Gaussians
        mtx = np.random.randn(p_size,n_size) + (0+1j)*np.random.randn(p_size,n_size)
        # hermitian matrix
        self.matrix = np.matmul(mtx, mtx.transpose())
        return self.matrix

    def _sample_wqe(self):
        p_size = self.p
        n_size = self.n
        # p by n random complex matrix of random Gaussians
        x_mtx = np.random.randn(p_size,n_size) + (0+1j)*np.random.randn(p_size,n_size)
        # p by n random complex matrix of random Gaussians
        y_mtx = np.random.randn(p_size,n_size) + (0+1j)*np.random.randn(p_size,n_size)
        # [X Y; -conj(Y) conj(X)]
        mtx = np.block([
                        [x_mtx              , y_mtx],
                        [-np.conjugate(y_mtx), np.conjugate(x_mtx)]
                    ])
        # hermitian matrix
        self.matrix = np.matmul(mtx, mtx.transpose())
        return self.matrix

    def sample_tridiagonal(self):
        '''Samples a Wishart Ensemble random matrix in its tridiagonal form.

        Samples a random matrix of the specified Wishart Ensemble (remember,
        beta=1 is Real, beta=2 is Complex and beta=4 is Quaternion) in its
        tridiagonal form.

        Returns:
            numpy array containing new matrix sampled.

        References:
            Albrecht, J. and Chan, C.P. and Edelman, A.
                "Sturm sequences and random eigenvalue distributions".
                Foundations of Computational Mathematics. 9.4 (2008): 461-483.
            Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        '''
        a_val = self.n*self.beta/ 2
        # sampling chi-squares
        dfs = np.arange(self.p)
        chisqs_diag = np.array([np.sqrt(np.random.chisquare(2*a_val - self.beta*df)) for df in dfs])
        dfs = np.flip(dfs)
        chisqs_offdiag = np.array([np.sqrt(np.random.chisquare(self.beta*df)) for df in dfs[:-1]])
        # calculating tridiagonal diagonals
        diag = np.array([chisqs_diag[0]**2]+[chisqs_diag[i+1]**2 + \
                         chisqs_offdiag[i]**2 for i in range(self.p-1)])
        offdiag = np.multiply(chisqs_offdiag, chisqs_diag[:-1])
        # inserting diagonals
        diagonals = [offdiag, diag, offdiag]
        mtx = sparse.diags(diagonals, [-1, 0, 1])
        # converting to numpy array
        self.matrix = mtx.toarray()
        return self.matrix


    def eigvals(self):
        """Calculates the random matrix eigenvalues.

        Calculates the random matrix eigenvalues using numpy standard procedure.
        If the matrix ensemble is symmetric, a faster algorithm is used.

        Returns:
            numpy array with the calculated eigenvalues.

        """
        return np.linalg.eigvalsh(self.matrix)

    def eigval_hist(self, bins, interval=None, density=False, norm_const=None):
        if self.use_tridiagonal:
            if norm_const:
                return tridiag_eigval_hist(self.matrix*norm_const, bins=bins,
                                           interval=interval, density=density)
            return tridiag_eigval_hist(self.matrix, bins=bins, interval=interval, density=density)

        return super().eigval_hist(bins, interval, density, norm_const)

    def plot_eigval_hist(self, bins, interval=None, density=False, norm_const=None, fig_path=None):
        """Calculates and plots the histogram of the matrix eigenvalues

        Calculates and plots the histogram of the current sampled matrix eigenvalues.
        Gaussian (Hermite) ensemble and Wishart (Laguerre) ensemble have improved
        routines to avoid calculating the eigenvalues, so the histogram
        is built using certain techniques to boost efficiency.

        Args:
            bins (int or sequence): If bins is an integer, it defines the number of
                equal-width bins in the range. If bins is a sequence, it defines the
                bin edges, including the left edge of the first bin and the right
                edge of the last bin; in this case, bins may be unequally spaced.
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram.
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
        if norm_const is None:
            norm_const = 1/self.n
        if interval is None:
            ratio = self.p/self.n
            lambda_plus = (1 + np.sqrt(ratio))**2
            lambda_minus = (1 - np.sqrt(ratio))**2
            xmin = lambda_minus
            xmax = lambda_plus
            interval = (xmin, xmax)

        if self.use_tridiagonal:
            observed, bins = tridiag_eigval_hist(self.matrix*norm_const, bins=bins,
                                                 interval=interval, density=density)
            width = bins[1]-bins[0]
            plt.bar(bins[:-1], observed, width=width, align='edge')
            plt.title("Eigenvalue density histogram (matrix size: "+\
                      str(len(self.matrix))+"x"+str(len(self.matrix))+")", fontweight="bold")
            plt.xlabel("x")
            plt.ylabel("density")
            # Saving plot or showing it
            if fig_path:
                plt.savefig(fig_path, dpi=1000)
            else:
                plt.show()

        else:
            super().plot_eigval_hist(bins, interval, density, norm_const, fig_path)

    def eigval_pdf(self):
        '''Calculates joint eigenvalue pdf.

        Calculates joint eigenvalue probability density function given the current
            random matrix (so its eigenvalues). This function depends on beta, i.e.,
            in the sub-Wishart ensemble.

        Returns:
            real number. Value of the joint pdf of the current eigenvalues.

        References:
            Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        '''
        a_val = self.beta*self.n/2
        p_aux = 1 + self.beta/2*(self.p - 1)
        # calculating Laguerre eigval pdf constant depeding on beta
        const_beta = 2**(-self.p*a_val)
        for j in range(self.p):
            const_beta *= special.gamma(1 + self.beta/2)/ \
                          (special.gamma(1 + self.beta*j/2)*\
                            special.gamma(a_val - self.beta/2*(self.p - j)))
        # calculating eigenvalues
        eigvals = np.linalg.eigvals(self.matrix)
        n_eigvals = len(eigvals)
        # calculating first prod
        prod1 = 1
        for j in range(n_eigvals):
            for i in range(j):
                prod1 *= np.abs(eigvals[i] - eigvals[j])**self.beta
        # calculating second prod
        prod2 = np.prod(eigvals**(a_val - p_aux))
        # calculating exponential term
        exp_val = np.exp(-np.sum((eigvals**2)/2))
        # calculating Laguerre eigval pdf
        return const_beta * prod1 * prod2 * exp_val
