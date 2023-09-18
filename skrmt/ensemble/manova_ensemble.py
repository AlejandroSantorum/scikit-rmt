"""Manova Ensemble Module

This module contains the implementation of the Manova Ensemble, also
known as Jacobi Ensemble. This ensemble of random matrices contains
mainly three sub-ensembles: Manova Real Ensemble, Manova Complex Ensemble
and Manova Quaternion Ensemble.

"""

from typing import Union, Sequence, Tuple
import numpy as np
from scipy import special

from .base_ensemble import _Ensemble
from .spectral_law import ManovaSpectrumDistribution


#########################################################################
### Manova Ensemble = Jacobi Ensemble

class ManovaEnsemble(_Ensemble):
    """General Manova Ensemble class.

    This class contains common attributes and methods for all the
    Manova ensembles. It also defines the basic interface to be
    supported by inherited classes. Manova Ensembles are divided in:
    - Manova Real Ensemble (MRE, beta=1): the random matrices of
    this ensemble are formed by sampling two random real standard
    guassian matrices (X and Y) of size m times n1 and m times n2
    respectively. Then, matrix A = (X * X') / (X * X' + Y * Y')
    generates a matrix of the Manova Real Ensemble.
    - Manova Complex Ensemble (MCE, beta=2): the random matrices
    of this ensemble are formed by sampling two random complex
    standard guassian matrices (X and Y) of size m times n1 and
    m times n2 respectively. Then, matrix A = (X * X') / (X * X' + Y * Y')
    generates a matrix of the Manova Complex Ensemble.
    - Manova Quaternion Ensemble (MQE, beta=4): the random matrices
    of this ensemble are formed by: sampling two random complex
    standard guassian matrices (X1 and X2), both of size m times n1.
    Another two random complex standard guassian matrices (Y1 and Y2),
    both of size m times n2, are sampled. They are stacked forming matrices
    X and Y:
    X = [X1  X2; -conj(X2)  conj(X1)]
    Y = [Y1  Y2; -conj(Y2)  conj(Y1)]
    Finally, matrix A = (X * X') / (X * X' + Y * Y') generates a matrix of
    the Manova Quaternion Ensemble of size m times m.

    Attributes:
        matrix (numpy array): instance of the ManovaReal, ManovaComplex
            or ManovaQuaternion random ensembles. If it is an instance
            of ManovaReal or ManovaComplex, the random matrix is of
            size m times m. If it is a ManovaQuaternion, the random matrix
            is of size 2m times 2m.
        beta (int): descriptive integer of the Manova ensemble type.
            For Real beta=1, for Complex beta=2, for Quaternion beta=4.
        m (int): number of rows of the random guassian matrices that
            generates the matrix of the corresponding ensemble.
        n1 (int): number of columns of the first random guassian matrix
            that generates the matrix of the corresponding ensemble.
        n2 (int): number of columns of the second random guassian matrix
            that generates the matrix of the corresponding ensemble.

    References:
        - Erdos, L. and Farrell, B.
            "Local Eigenvalue Density for General MANOVA Matrices".
            Journal of Statistical Physics. 152.6 (2013): 1003-1032.
        - Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
            Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

    """

    def __init__(self, beta: int, m: int, n1: int, n2: int, random_state: int = None) -> None:
        """Constructor for ManovaEnsemble class.

        Initializes an instance of this class with the given parameters.

        Args:
            beta (int): descriptive integer of the Manova ensemble type.
                For Real beta=1, for Complex beta=2, for Quaternion beta=4.
            m (int): number of rows of the random guassian matrices that
                generates the matrix of the corresponding ensemble.
            n1 (int): number of columns of the first random guassian matrix
                that generates the matrix of the corresponding ensemble.
            n2 (int): number of columns of the second random guassian matrix
                that generates the matrix of the corresponding ensemble.
            random_state (int, default=None): random seed to initialize the pseudo-random
                number generator of numpy before sampling the random matrix instance. This 
                has to be any integer between 0 and 2**32 - 1 (inclusive), or None (default).
                If None, the seed is obtained from the clock.

        """
        super().__init__()
        # pylint: disable=invalid-name
        self.m = m
        self.n1 = n1
        self.n2 = n2
        self.beta = beta
        self._eigvals = None
        self.matrix = self.sample(random_state=random_state)
        # scikit-rmt class implementing the corresponding spectral law
        self._law_class = ManovaSpectrumDistribution(
            beta=self.beta, ratio_a=self.n1/self.m, ratio_b=self.n2/self.m
        )

    def resample(self, random_state: int = None) -> np.ndarray:
        """Re-samples new Manova Ensemble random matrix.

        It re-samples a new random matrix from the Manova ensemble. This is an alias
        for method ``sample``.

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
        return self.sample(random_state=random_state)

    # pylint: disable=inconsistent-return-statements
    def sample(self, random_state: int = None) -> np.ndarray:
        """Samples new Manova Ensemble random matrix.

        The sampling algorithm depends on the specification of
        beta parameter. If beta=1, Manova Real is sampled; if
        beta=2 Manova Complex is sampled and if beta=4
        Manova Quaternion is sampled.

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

        if self.beta == 1:
            return self._sample_mre()
        if self.beta == 2:
            return self._sample_mce()
        if self.beta == 4:
            return self._sample_mqe()

    def _sample_mre(self) -> np.ndarray:
        m_size = self.m
        n1_size = self.n1
        n2_size = self.n2
        # m by n1 random real matrix of random Gaussians
        x_mtx = np.random.randn(m_size,n1_size)
        # m by n2 random real matrix of random Gaussians
        y_mtx = np.random.randn(m_size,n2_size)
        # A1 = X * X'
        a1_mtx = np.matmul(x_mtx, x_mtx.transpose())
        # A2 = X * X' + Y * Y'
        a2_mtx = a1_mtx + np.matmul(y_mtx, y_mtx.transpose())
        # A = (X * X') / (X * X' + Y * Y') = (X * X') * (X * X' + Y * Y')^(-1)
        self.matrix = np.matmul(a1_mtx, np.linalg.inv(a2_mtx))
        # setting array of eigenvalues to None to force re-computing them
        self._eigvals = None
        return self.matrix

    def _sample_mce(self) -> np.ndarray:
        m_size = self.m
        n1_size = self.n1
        n2_size = self.n2
        # m by n1 random complex matrix of random Gaussians
        x_mtx = np.random.randn(m_size,n1_size) + (0+1j)*np.random.randn(m_size,n1_size)
        # m by n2 random complex matrix of random Gaussians
        y_mtx = np.random.randn(m_size,n2_size) + (0+1j)*np.random.randn(m_size,n2_size)
        # A1 = X * X'
        a1_mtx = np.matmul(x_mtx, x_mtx.transpose().conj())
        # A2 = X * X' + Y * Y'
        a2_mtx = a1_mtx + np.matmul(y_mtx, y_mtx.transpose().conj())
        # A = (X * X') / (X * X' + Y * Y') = (X * X') * (X * X' + Y * Y')^(-1)
        self.matrix = np.matmul(a1_mtx, np.linalg.inv(a2_mtx))
        # setting array of eigenvalues to None to force re-computing them
        self._eigvals = None
        return self.matrix

    def _sample_mqe(self) -> np.ndarray:
        m_size = self.m
        n1_size = self.n1
        n2_size = self.n2
        # m by n1 random complex matrix of random Gaussians
        x1_mtx = np.random.randn(m_size,n1_size) + (0+1j)*np.random.randn(m_size,n1_size)
        # m by n1 random complex matrix of random Gaussians
        x2_mtx = np.random.randn(m_size,n1_size) + (0+1j)*np.random.randn(m_size,n1_size)
        # m by n2 random complex matrix of random Gaussians
        y1_mtx = np.random.randn(m_size,n2_size) + (0+1j)*np.random.randn(m_size,n2_size)
        # m by n2 random complex matrix of random Gaussians
        y2_mtx = np.random.randn(m_size,n2_size) + (0+1j)*np.random.randn(m_size,n2_size)
        # X = [X1 X2; -conj(X2) conj(X1)]
        x_mtx = np.block([
                        [x1_mtx               , x2_mtx],
                        [-np.conjugate(x2_mtx), np.conjugate(x1_mtx)]
                        ])
        # Y = [Y1 Y2; -conj(Y2) conj(Y1)]
        y_mtx = np.block([
                         [y1_mtx               , y2_mtx],
                         [-np.conjugate(y2_mtx), np.conjugate(y1_mtx)]
                        ])
        # A1 = X * X'
        a1_mtx = np.matmul(x_mtx, x_mtx.transpose().conj())
        # A2 = X * X' + Y * Y'
        a2_mtx = a1_mtx + np.matmul(y_mtx, y_mtx.transpose().conj())
        # A = (X * X') / (X * X' + Y * Y') = (X * X') * (X * X' + Y * Y')^(-1)
        self.matrix = np.matmul(a1_mtx, np.linalg.inv(a2_mtx))
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
        self._eigvals = np.linalg.eigvals(self.matrix)
        return norm_const * self._eigvals

    def _plot_eigval_hist(
        self,
        bins: Union[int, Sequence],
        interval: Tuple = (0,1),
        density: bool = False,
        normalize: bool = False,
        avoid_img: bool = True,
    ) -> None:
        return super()._plot_eigval_hist(
            bins=bins, interval=interval, density=density, normalize=normalize, avoid_img=avoid_img,
        )

    def plot_eigval_hist(
        self,
        bins: Union[int, Sequence] = 100,
        interval: Tuple = (0,1),
        density: bool = False,
        normalize: bool = False,
        savefig_path: str = None,
    ) -> None:
        """Computes and plots the histogram of the matrix eigenvalues

        Calculates and plots the histogram of the current sampled matrix eigenvalues.

        Args:
            bins (int or sequence, default=100): If bins is an integer, it defines the number of
                equal-width bins in the range. If bins is a sequence, it defines the
                bin edges, including the left edge of the first bin and the right
                edge of the last bin; in this case, bins may be unequally spaced.
            interval (tuple, default=(0,1)): Delimiters (xmin, xmax) of the histogram.
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
            - Erdos, L. and Farrell, B.
                "Local Eigenvalue Density for General MANOVA Matrices".
                Journal of Statistical Physics. 152.6 (2013): 1003-1032.
            - Dumitriu, I. and Edelman, A.
                "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        """
        # pylint: disable=too-many-arguments
        # pylint: disable=arguments-differ
        return super().plot_eigval_hist(
            bins=bins,
            interval=interval,
            density=density,
            normalize=normalize,
            savefig_path=savefig_path,
            avoid_img=True,
        )

    def joint_eigval_pdf(self, eigvals: np.ndarray = None) -> float:
        '''Computes joint eigenvalue pdf.

        Calculates joint eigenvalue probability density function given an array of
        eigenvalues. If the array of eigenvalues is not provided, the current random
        matrix sample (so its eigenvalues) is used. This function depends on beta,
        i.e., in the sub-Manova ensemble.

        Args:
            eigvals (np.ndarray, default=None): numpy array with the values (eigenvalues)
                to evaluate the joint pdf in.

        Returns:
            real number. Value of the joint pdf of the eigenvalues.

        References:
            - Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        '''
        # pylint: disable=invalid-name
        a1 = self.beta*self.n1/2
        a2 = self.beta*self.n2/2
        p = 1 + self.beta/2*(self.m - 1)

        if eigvals is None:
            # calculating eigenvalues
            eigvals = self.eigvals()
        n_eigvals = len(eigvals)

        # calculating Jacobi eigval pdf constant depeding on beta
        const_beta = 1
        for j in range(self.m):
            const_beta *= (special.gamma(1 + self.beta/2) * \
                            special.gamma(a1 + a2 -self.beta/2*(self.m - j)))/ \
                          (special.gamma(1 + self.beta*j/2) * \
                            special.gamma(a1 - self.beta/2*(self.m - j)) * \
                              special.gamma(a2 - self.beta/2*(self.m - j)))

        # calculating first prod
        prod1 = 1
        for j in range(n_eigvals):
            for i in range(j):
                prod1 *= np.abs(eigvals[i] - eigvals[j])**self.beta
        # calculating second prod
        prod2 = 1
        for j in range(n_eigvals):
            prod2 *= eigvals[j]**(a1-p) * (1 - eigvals[j])**(a2-p)
        # calculating Jacobi eigval pdf
        return const_beta * prod1 * prod2
