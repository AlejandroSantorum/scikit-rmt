"""Wishart Ensemble Module

This module contains the implementation of the Wishart Ensemble, also
known as Laguerre Ensemble. This ensemble of random matrices contains
mainly three sub-ensembles: Wishart Real Ensemble, Wishart Complex Ensemble
and Wishart Quaternion Ensemble.

"""

from typing import Union, Sequence, Tuple
import numpy as np
from scipy import sparse, special

from .base_ensemble import _Ensemble
from .tridiagonal_utils import tridiag_eigval_hist
from .spectral_law import MarchenkoPasturDistribution


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
            size p times p. If it is a WishartQuaternion, the random matrix
            is of size 2p times 2p.
        beta (int): descriptive integer of the Wishart ensemble type.
            For Real beta=1, for Complex beta=2, for Quaternion beta=4.
        p (int): number of rows of the guassian matrix that generates
            the matrix of the corresponding ensemble.
        n (int): number of columns of the guassian matrix that generates
            the matrix of the corresponding ensemble.
        tridiagonal_form (bool): if set to True, Gaussian Ensemble
            matrices are sampled in its tridiagonal form, which has the same
            eigenvalues than its standard form. Otherwise, it is sampled using
            its standard form.
        sigma (float): scale (standard deviation) of the random entries of the
            sampled matrix.

    References:
        - Albrecht, J. and Chan, C.P. and Edelman, A.
            "Sturm sequences and random eigenvalue distributions".
            Foundations of Computational Mathematics. 9.4 (2008): 461-483.
        - Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
            Journal of Mathematical Physics. 43.11 (2002): 5830-5847.
        - Bar, Z.D. and Silverstain, J.W.
            Spectral Analysis of Large Dimensional Random Matrices.
            2nd edition. Springer. (2010).

    """

    def __init__(
        self,
        beta: int,
        p: int,
        n: int,
        tridiagonal_form: bool = False,
        sigma: float = 1.0,
        random_state: int = None
    ) -> None:
        """Constructor for WishartEnsemble class.

        Initializes an instance of this class with the given parameters.

        Args:
            beta (int): descriptive integer of the Wishart ensemble type.
                For Real beta=1, for Complex beta=2, for Quaternion beta=4
            p (int): number of rows of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            n (int): number of columns of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            tridiagonal_form (bool, default=False): if set to True, Wishart Ensemble
                matrices are sampled in its tridiagonal form, which has the same
                eigenvalues than its standard form.
            sigma (float, 1.0): scale (standard deviation) of the random entries of the
                sampled matrix.
            random_state (int, default=None): random seed to initialize the pseudo-random
                number generator of numpy before sampling the random matrix instance. This
                has to be any integer between 0 and 2**32 - 1 (inclusive), or None (default).
                If None, the seed is obtained from the clock.

        """
        if beta not in [1,2,4]:
            raise ValueError(f"Invalid beta: {beta}. Beta value has to be 1, 2 or 4.")

        super().__init__()
        # pylint: disable=invalid-name
        self.p = p
        self.n = n
        self.beta = beta
        self.tridiagonal_form = tridiagonal_form
        self.sigma = sigma
        self._eigvals = None
        self.matrix = self.sample(random_state=random_state)
        # default eigenvalue normalization constant
        self.eigval_norm_const = 1/self.n
        self._compute_parameters()
        # scikit-rmt class implementing the corresponding spectral law
        self._law_class = MarchenkoPasturDistribution(
            beta=self.beta, ratio=self.ratio, sigma=self.sigma
        )

    def _compute_parameters(self) -> None:
        # calculating constants depending on matrix sizes
        self.ratio = self.p/self.n
        self.lambda_plus = self.beta * self.sigma**2 * (1 + np.sqrt(self.ratio))**2
        self.lambda_minus = self.beta * self.sigma**2 * (1 - np.sqrt(self.ratio))**2

    def resample(self, tridiagonal_form: bool = None, random_state: int = None) -> np.ndarray:
        """Re-samples a random matrix from the Wishart ensemble with the specified form.

        It re-samples a random matrix from the Wishart ensemble with the specified form.
        If the specified form is different than the original form (tridiagonal vs standard)
        the property ``self.tridiagonal_form`` is updated and the random matrix is sampled
        with the updated form. If ``tridiagonal_form`` is not specified, this methods returns
        a re-sampled random matrix of the initialized form by calling the method ``sample``.

        Args:
            tridiagonal_form (bool, default=None): form to generate the new random matrix sample.
                If set to True, a random matrix in tridiagonal form is returned. Otherwise, the
                random matrix is sampled in standard form.
            random_state (int, default=None): random seed to initialize the pseudo-random
                    number generator of numpy. This has to be any integer between 0 and 2**32 - 1
                    (inclusive), or None (default). If None, the seed is obtained from the clock.

        Returns:
            (ndarray) numpy array containing new matrix sampled.

        References:
            - Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.
        """
        # pylint: disable=arguments-renamed
        if tridiagonal_form is not None:
            # The type of sampled matrix can be specified, changing the random matrix
            # form if the argument ``tridiagonal_form`` is provided.
            self.tridiagonal_form = tridiagonal_form
        return self.sample(random_state=random_state)

    # pylint: disable=inconsistent-return-statements
    def sample(self, random_state: int = None) -> np.ndarray:
        """Samples new Wishart Ensemble random matrix.

        The sampling algorithm depends on the specification of
        ``tridiagonal_form`` parameter. If ``tridiagonal_form`` is
        set to True, a Wishart Ensemble random matrix in its
        tridiagonal form is sampled. Otherwise, it is sampled
        using the standard form.

        Args:
            random_state (int, default=None): random seed to initialize the pseudo-random
                number generator of numpy. This has to be any integer between 0 and 2**32 - 1
                (inclusive), or None (default). If None, the seed is obtained from the clock.

        Returns:
            (ndarray) numpy array containing new matrix sampled.

        References:
            - Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.
        """
        if random_state is not None:
            np.random.seed(random_state)

        if self.tridiagonal_form:
            if self.p > self.n:
                # check reference ("Matrix Models for Beta Ensembles"): page 5, table 1.
                raise ValueError(
                    "Error: cannot use tridiagonal form if 'p' (degrees of freedom)"
                    " is greater than 'n' (sample size).\n"
                    f"\t Provided n={self.n} and p={self.p}."
                    " Set `tridiagonal_form=False` or increase sample size (`n`)."
                )

            if self.sigma != 1.0:
                raise ValueError(
                    "Error: cannot sample tridiagonal random matrix using non-unitary scale"
                    f" (sigma = {self.sigma}).\n"
                    "\t Set `sigma=1.0` (default) or deactivate tridiagonal sampling."
                )

            return self.sample_tridiagonal()

        if self.beta == 1:
            return self._sample_wre()
        if self.beta == 2:
            return self._sample_wce()
        if self.beta == 4:
            return self._sample_wqe()

    def _sample_wre(self) -> np.ndarray:
        p_size = self.p
        n_size = self.n
        # p by n matrix of random Gaussians
        mtx = np.random.randn(p_size,n_size) * self.sigma
        # symmetrize matrix
        self.matrix = np.matmul(mtx, mtx.transpose())
        # setting array of eigenvalues to None to force re-computing them
        self._eigvals = None
        return self.matrix

    def _sample_wce(self) -> np.ndarray:
        p_size = self.p
        n_size = self.n
        # p by n random complex matrix of random Gaussians
        mtx = np.random.randn(p_size,n_size) * self.sigma \
              + 1j*np.random.randn(p_size,n_size) * self.sigma
        # hermitian matrix
        self.matrix = np.matmul(mtx, mtx.transpose().conj())
        # setting array of eigenvalues to None to force re-computing them
        self._eigvals = None
        return self.matrix

    def _sample_wqe(self) -> np.ndarray:
        p_size = self.p
        n_size = self.n
        # p by n random complex matrix of random Gaussians
        x_mtx = np.random.randn(p_size,n_size) * self.sigma \
                + 1j*np.random.randn(p_size,n_size) * self.sigma
        # p by n random complex matrix of random Gaussians
        y_mtx = np.random.randn(p_size,n_size) * self.sigma \
                + 1j*np.random.randn(p_size,n_size) * self.sigma
        # [X Y; -conj(Y) conj(X)]
        mtx = np.block([
                        [x_mtx              , y_mtx],
                        [-np.conjugate(y_mtx), np.conjugate(x_mtx)]
                    ])
        # hermitian matrix
        self.matrix = np.matmul(mtx, mtx.transpose().conj())
        # setting array of eigenvalues to None to force re-computing them
        self._eigvals = None
        return self.matrix

    def sample_tridiagonal(self) -> np.ndarray:
        '''Samples a Wishart Ensemble random matrix in its tridiagonal form.

        Samples a random matrix of the specified Wishart Ensemble (remember,
        beta=1 is Real, beta=2 is Complex and beta=4 is Quaternion) in its
        tridiagonal form.

        Returns:
            numpy array containing new matrix sampled.

        References:
            - Albrecht, J. and Chan, C.P. and Edelman, A.
                "Sturm sequences and random eigenvalue distributions".
                Foundations of Computational Mathematics. 9.4 (2008): 461-483.
            - Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        '''
        a_val = self.n*self.beta/2
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
        # setting array of eigenvalues to None to force re-computing them
        self._eigvals = None
        return self.matrix

    def eigvals(self, normalize: bool = False) -> np.ndarray:
        """Computes the random matrix eigenvalues.

        Calculates the random matrix eigenvalues using numpy standard procedure.
        If the matrix ensemble is symmetric, a faster algorithm is used.

        Returns:
            numpy array with the calculated eigenvalues.

        """
        norm_const = self.eigval_norm_const if normalize else 1.0

        if self._eigvals is not None:
            return norm_const * self._eigvals

        # always storing non-normalized eigenvalues
        self._eigvals = np.linalg.eigvalsh(self.matrix)
        return norm_const * self._eigvals

    def eigval_hist(
        self,
        bins: Union[int, Sequence],
        interval: Tuple = None,
        density: bool = False,
        normalize: bool = False,
        avoid_img: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # pylint: disable=signature-differs
        if interval is None:
            if normalize:
                interval = (self.lambda_minus, self.lambda_plus)
            else:
                interval = (self.n*self.lambda_minus, self.n*self.lambda_plus)

        if self.tridiagonal_form:
            if normalize:
                return tridiag_eigval_hist(
                    self.eigval_norm_const * self.matrix,
                    bins=bins,
                    interval=interval,
                    density=density,
                )
            return tridiag_eigval_hist(self.matrix, bins=bins, interval=interval, density=density)

        return super().eigval_hist(
            bins, interval=interval, density=density, normalize=normalize, avoid_img=avoid_img
        )

    def plot_eigval_hist(
        self,
        bins: Union[int, Sequence] = 100,
        interval: Tuple = None,
        density: bool = False,
        normalize: bool = False,
        savefig_path: str = None,
    ) -> None:
        """Computes and plots the histogram of the matrix eigenvalues.

        Calculates and plots the histogram of the current sampled matrix eigenvalues.
        Gaussian (Hermite) ensemble and Wishart (Laguerre) ensemble have improved
        routines to avoid calculating the eigenvalues, so the histogram
        is built using certain techniques to boost efficiency.

        Args:
            bins (int or sequence, default=100): If bins is an integer, it defines the number of
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
            normalize (bool, default=False): Whether to normalize the computed eigenvalues
                by the default normalization constant (see references). Defaults to False, i.e.,
                the eigenvalues are not normalized. Normalization makes the eigenvalues to be
                in the same support independently of the sample size.
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
        # pylint: disable=arguments-differ
        # the default interval will be computed in the method `eigval_hist`
        # if the given interval is None
        super().plot_eigval_hist(
            bins=bins,
            interval=interval,
            density=density,
            normalize=normalize,
            savefig_path=savefig_path,
        )

    def joint_eigval_pdf(self, eigvals: np.ndarray = None) -> float:
        '''Computes joint eigenvalue pdf.

        Calculates joint eigenvalue probability density function given an array of
        eigenvalues. If the array of eigenvalues is not provided, the current random
        matrix sample (so its eigenvalues) is used. This function depends on beta,
        i.e., in the sub-Wishart ensemble.

        Args:
            eigvals (np.ndarray, default=None): numpy array with the values (eigenvalues)
                to evaluate the joint pdf in.

        Returns:
            real number. Value of the joint pdf of the eigenvalues.

        References:
            - Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        '''
        if eigvals is None:
            # calculating eigenvalues
            eigvals = self.eigvals()
        n_eigvals = len(eigvals)

        a_val = self.beta*self.n/2
        p_aux = 1 + self.beta/2*(self.p - 1)
        # calculating Laguerre eigval pdf constant depeding on beta
        const_beta = 2**(-self.p*a_val)
        for j in range(self.p):
            const_beta *= special.gamma(1 + self.beta/2)/ \
                          (special.gamma(1 + self.beta*j/2)*\
                            special.gamma(a_val - self.beta/2*(self.p - j)))
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
