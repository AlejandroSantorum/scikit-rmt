"""Gaussian Ensemble Module

This module contains the implementation of the Gaussian Ensemble, also
known as Hermite Ensemble. This ensemble of random matrices contains
mainly three sub-ensembles: Gaussian Orthogonal Ensemble (GOE),
Gaussian Unitary Ensemble (GUE) and Gaussian Symplectic Ensemble (GSE).

"""

from typing import Union, Sequence, Tuple
import numpy as np
from scipy import sparse, special

from .base_ensemble import _Ensemble
from .tridiagonal_utils import tridiag_eigval_hist
from .spectral_law import WignerSemicircleDistribution

#########################################################################
### Gaussian Ensemble = Hermite Ensemble

class GaussianEnsemble(_Ensemble):
    """General Gaussian Ensemble class.

    This class contains common attributes and methods for all the
    gaussian ensembles. Gaussian Ensembles are divided in:
    - Gaussian Orthogonal Ensemble (GOE, beta=1): the distribution of the
    matrices of this ensemble are invariant under orthogonal conjugation,
    i.e., if X is in GOE(n) and O is an orthogonal matrix, then
    O*X*O^T is equally distributed as X.
    - Gaussian Unitary Ensemble (GUE, beta=2): the distribution of
    the matrices of this ensemble are invariant under unitary conjugation,
    i.e., if X is in GUE(n) and O is an unitary matrix, then O*X*O^T
    is equally distributed as X.
    - Gaussian Symplectic Ensemble (GSE, beta=4): the distribution of
    the matrices of this ensemble are invariant under conjugation
    by the symplectic group.

    Attributes:
        matrix (numpy array): instance of the GOE, GUE or GSE random
            matrix ensemble of size n times n if it is GOE or GUE, or
            of size 2n times 2n if it is GSE.
        beta (int): descriptive integer of the gaussian ensemble type.
            For GOE beta=1, for GUE beta=2, for GSE beta=4.
        n (int): random matrix size. Gaussian ensemble matrices are
            squared matrices. GOE and GUE are of size n times n,
            and GSE are of size 2n times 2n.
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

    """

    def __init__(
        self,
        beta: int,
        n: int,
        tridiagonal_form: bool = False,
        sigma: float = 1.0,
        random_state: int = None,
    ) -> None:
        """Constructor for GaussianEnsemble class.

        Initializes an instance of this class with the given parameters.

        Args:
            beta (int, default=1): descriptive integer of the gaussian ensemble type.
                For GOE beta=1, for GUE beta=2, for GSE beta=4.
            n (int): random matrix size. Gaussian ensemble matrices are
                squared matrices. GOE and GUE are of size n times n,
                and GSE are of size 2n times 2n.
            tridiagonal_form (bool, default=False): if set to True, Gaussian Ensemble
                matrices are sampled in its tridiagonal form, which has the same
                eigenvalues than its standard form. Otherwise, it is sampled using
                its standard form.
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
        self.n = n
        self.beta = beta
        self.tridiagonal_form = tridiagonal_form
        self.sigma = sigma
        self.radius = 2 * np.sqrt(self.beta) * self.sigma
        self._eigvals = None
        self.matrix = self.sample(random_state=random_state)

        # default eigenvalue normalization constant
        self.eigval_norm_const = 1/np.sqrt(2*self.n) if self.beta==4 else 1/np.sqrt(self.n)

        # non-normalized semicircle interval (keys are betas)
        self._non_normalized_interval = {
            1: (-2*np.sqrt(self.n), 2*np.sqrt(self.n)),
            2: (-2*np.sqrt(2*self.n), 2*np.sqrt(2*self.n)),
            4: (-4*np.sqrt(2*self.n), 4*np.sqrt(2*self.n)),
        }

        # scikit-rmt class implementing the corresponding spectral law
        self._law_class = WignerSemicircleDistribution(beta=self.beta, center=0.0, sigma=self.sigma)

    def resample(self, tridiagonal_form: bool = None, random_state: int = None) -> np.ndarray:
        """Re-samples a random matrix from the Gaussian ensemble with the specified form.

        It re-samples a random matrix from the Gaussian ensemble with the specified form.
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
        """Samples new Gaussian Ensemble random matrix.

        The sampling algorithm depends on the specification of
        ``tridiagonal_form`` parameter. If ``tridiagonal_form`` is 
        set to True, a Gaussian Ensemble random matrix in its
        tridiagonal form is sampled. Otherwise, it is sampled using 
        the standard form.

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
            return self.sample_tridiagonal()

        if self.beta == 1:
            return self._sample_goe()
        if self.beta == 2:
            return self._sample_gue()
        if self.beta == 4:
            return self._sample_gse()

    def _sample_goe(self) -> np.ndarray:
        # n by n matrix of random Gaussians
        mtx = np.random.randn(self.n,self.n) * self.sigma
        # symmetrize matrix
        self.matrix = (mtx + mtx.transpose())/np.sqrt(2)
        # setting array of eigenvalues to None to force re-computing them
        self._eigvals = None
        return self.matrix

    def _sample_gue(self) -> np.ndarray:
        size = self.n
        # n by n random complex matrix
        mtx = np.random.randn(size,size)*self.sigma + 1j*np.random.randn(size,size)*self.sigma
        # hermitian matrix
        self.matrix = (mtx + mtx.transpose().conj())/np.sqrt(2)
        # setting array of eigenvalues to None to force re-computing them
        self._eigvals = None
        return self.matrix

    def _sample_gse(self) -> np.ndarray:
        size = self.n
        # n by n random complex matrix
        x_mtx = np.random.randn(size,size)*self.sigma + 1j*np.random.randn(size,size)*self.sigma
        # another n by n random complex matrix
        y_mtx = np.random.randn(size,size)*self.sigma + 1j*np.random.randn(size,size)*self.sigma
        # [X Y; -conj(Y) conj(X)]
        mtx = np.block([
                       [x_mtx               , y_mtx],
                       [-np.conjugate(y_mtx), np.conjugate(x_mtx)]
                        ])
        # hermitian matrix
        # Before: self.matrix = (mtx + mtx.transpose().conj())/np.sqrt(2)
        self.matrix = (mtx + mtx.transpose().conj())
        # setting array of eigenvalues to None to force re-computing them
        self._eigvals = None
        return self.matrix

    def sample_tridiagonal(self) -> np.ndarray:
        '''Samples a Gaussian Ensemble random matrix in its tridiagonal form.

        Samples a random matrix of the specified Gaussian Ensemble (remember,
        beta=1 is GOE, beta=2 is GUE and beta=4 is GSE) in its tridiagonal
        form.

        Returns:
            numpy array containing new matrix sampled.

        References:
            - Albrecht, J. and Chan, C.P. and Edelman, A.
                "Sturm sequences and random eigenvalue distributions".
                Foundations of Computational Mathematics. 9.4 (2008): 461-483.
            - Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        '''
        if self.sigma != 1.0:
            raise ValueError(
                "Error: cannot sample tridiagonal random matrix using non-unitary scale"
                f" (sigma = {self.sigma}).\n"
                "\t Set `sigma=1.0` (default) or deactivate tridiagonal sampling."
            )

        size = 2*self.n if self.beta==4 else self.n
        # sampling diagonal normals
        # normals = (1/np.sqrt(2)) * np.random.normal(loc=0, scale=np.sqrt(2), size=size)
        normals = np.random.normal(loc=0, scale=1, size=size)
        # sampling chi-squares
        dfs = np.flip(np.arange(1, size))
        # chisqs = (1/np.sqrt(2)) * \
        #          np.array([np.sqrt(np.random.chisquare(df*self.beta)) for df in dfs])
        chisqs = np.array([np.sqrt(np.random.chisquare(df*self.beta)) for df in dfs])
        # inserting diagonals
        diagonals = [chisqs, normals, chisqs]
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

        # always storing non-normalized eigenvalues
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
                interval = (-self.radius, self.radius)
            else:
                interval = self._non_normalized_interval[self.beta]

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
        # pylint: disable=arguments-differ
        # the default interval will be computed in the method `eigval_hist` if interval is None
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
        i.e., in the sub-Gaussian ensemble.

        Args:
            eigvals (np.ndarray, default=None): numpy array with the values (eigenvalues)
                to evaluate the joint pdf in.

        Returns:
            real number. Value of the joint pdf of the eigenvalues.

        References:
            - Dumitriu, I. and Edelman, A.
                "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        '''
        if eigvals is None:
            # calculating eigenvalues
            eigvals = self.eigvals()
        n_eigvals = len(eigvals)

        # calculating Hermite eigval pdf constant depeding on beta
        const_beta = (2*np.pi)**(-self.n/2)
        for j in range(self.n):
            const_beta *= special.gamma(1 + self.beta/2)/special.gamma(1 + self.beta*j/2)
        # calculating prod
        pdf = 1
        for j in range(n_eigvals):
            for i in range(j):
                pdf *= np.abs(eigvals[i] - eigvals[j])**self.beta
        # calculating exponential term
        exp_val = np.exp(-np.sum((eigvals**2)/2))
        # calculating Hermite eigval pdf
        return const_beta * pdf * exp_val
