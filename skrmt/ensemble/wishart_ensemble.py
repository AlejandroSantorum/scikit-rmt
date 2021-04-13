"""Wishart Ensemble Module

This module contains the implementation of the Wishart Ensemble, also
known as Laguerre Ensemble. This ensemble of random matrices contains
mainly three sub-ensembles: Wishart Real Ensemble, Wishart Complex Ensemble
and Wishart Quaternion Ensemble.

"""

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import sparse

from ._base_ensemble import _Ensemble
from .tridiagonal_utils import tridiag_eigval_hist


#########################################################################
### Wishart Ensemble = Laguerre Ensemble

class WishartEnsemble(_Ensemble, metaclass=ABCMeta):
    """General Wishart Ensemble class.

    This class contains common attributes and methods for all the
    Wishart ensembles. It also defines the basic interface to be
    supported by inherited classes.

    Attributes:
        beta (int): descriptive integer of the Wishart ensemble type.
            For Real beta=1, for Complex beta=2, for Quaternion beta=4.
        p (int): number of rows of the guassian matrix that generates
            the matrix of the corresponding ensemble.
        n (int): number of columns of the guassian matrix that generates
            the matrix of the corresponding ensemble.

    """

    @abstractmethod
    def __init__(self, p, n, beta=1):
        """Constructor for WishartEnsemble class.

        Initializes an instance of this class with the given parameters.

        Args:
            p (int): number of rows of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            n (int): number of columns of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            beta (int, default=1): descriptive integer of the Wishart ensemble type.
                For Real beta=1, for Complex beta=2, for Quaternion beta=4.

        """
        self.p = p
        self.n = n
        self.beta = beta
    
    def set_size(self, p, n, resample_mtx=False):
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

    @abstractmethod
    def sample(self):
        pass

    def sample_tridiagonal(self):
        '''Samples a Wishart Ensemble random matrix in its tridiagonal form.

        Samples a random matrix of the specified Wishart Ensemble (remember,
        beta=1 is Real, beta=2 is Complex and beta=4 is Quaternion) in its
        tridiagonal form.

        Returns:
            numpy array containing new matrix sampled.

        References:
            Albrecht, J. and Chan, C.P. and Edelman, A. "Sturm sequences and random eigenvalue distributions".
                Foundations of Computational Mathematics. 9.4 (2008): 461-483.
            Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.
            
        '''
        a = self.n*self.beta/ 2
        # sampling chi-squares
        dfs = np.arange(self.p)
        chisqs_diag = np.array([np.sqrt(np.random.chisquare(2*a - self.beta*df)) for df in dfs])
        dfs = np.flip(dfs)
        chisqs_offdiag = np.array([np.sqrt(np.random.chisquare(self.beta*df)) for df in dfs[:-1]])
        # calculating tridiagonal diagonals
        diag = np.array([chisqs_diag[0]**2]+[chisqs_diag[i+1]**2 + chisqs_offdiag[i]**2 for i in range(self.p-1)])
        offdiag = np.multiply(chisqs_offdiag, chisqs_diag[:-1])
        # inserting diagonals
        diagonals = [offdiag, diag, offdiag]
        M = sparse.diags(diagonals, [-1, 0, 1])
        # converting to numpy array
        self.matrix = M.toarray()
        return self.matrix
        

    def eigvals(self):
        """Calculates the random matrix eigenvalues.

        Calculates the random matrix eigenvalues using numpy standard procedure.
        If the matrix ensemble is symmetric, a faster algorithm is used.

        Returns:
            numpy array with the calculated eigenvalues.

        """
        return np.linalg.eigvalsh(self.matrix)

    def eigval_hist(self, bins, interval=None, normed_hist=True):
        if self.use_tridiagonal:
            return tridiag_eigval_hist(self.matrix, bins=bins, interval=interval, norm=normed_hist)
        else:
            return super().eigval_hist(bins, interval, normed_hist)

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
        a = self.beta*self.n/2
        p = 1 + self.beta/2*(self.p - 1)
        # calculating Laguerre eigval pdf constant depeding on beta
        const_beta = 2**(-self.p*a)
        for j in range(self.p):
            const_beta *= sp.special.gamma(1 + self.beta/2)/ \
                          (sp.special.gamma(1 + self.beta*j/2)*sp.special.gamma(a - self.beta/2*(self.p - j)))
        # calculating eigenvalues
        eigvals = np.linalg.eigvals(self.matrix)
        n_eigvals = len(eigvals)
        # calculating first prod
        prod1 = 1
        for j in range(n_eigvals):
            for i in range(j):
                prod1 *= np.abs(eigvals[i] - eigvals[j])**self.beta
        # calculating second prod
        prod2 = np.prod(eigvals**(a - p))
        # calculating exponential term
        exp_val = np.exp(-np.sum((eigvals**2)/2))
        # calculating Laguerre eigval pdf
        return const_beta * prod1 * prod2 * exp_val
    

##########################################
### Wishart real

class WishartReal(WishartEnsemble):
    """Wishart Real Ensemble class.

    The random matrices of this ensemble are formed by multiplying
    a random real standard gaussian matrix of size p times n by its
    transpose.

    Attributes:
        matrix (numpy array): instance of the random matrix ensemble
            of size p times p.
        use_tridiagonal (bool): if set to True, WishartReal matrices
            are sampled in its tridiagonal form, which has the same
            eigenvalues than its standard form.

    """

    def __init__(self, p, n, use_tridiagonal=False):
        """Constructor for WishartReal class.

        Initializes an instance of this class with the given parameters,
        calling the parent class constructor and sampling a random instance.

        Args:
            p (int): number of rows of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            n (int): number of columns of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            use_tridiagonal (bool, default=False): if set to True, WishartReal
                matrices are sampled in its tridiagonal form, which has the
                same eigenvalues than its standard form.

        """
        super().__init__(p=p, n=n, beta=1)
        self.use_tridiagonal = use_tridiagonal
        self.matrix = self.sample()

    def sample(self):
        if self.use_tridiagonal:
            return self.sample_tridiagonal()
        else:
            return self.sample_WisReal_matrix()

    def sample_WisReal_matrix(self):
        p = self.p
        n = self.n
        # p by n matrix of random Gaussians
        A = np.random.randn(p,n)
        # symmetrize matrix
        self.matrix = np.matmul(A, A.transpose())
        return self.matrix


##########################################
### Wishart complex

class WishartComplex(WishartEnsemble):
    """Wishart Complex Ensemble class.

    The random matrices of this ensemble are formed by multiplying
    a random complex standard gaussian matrix of size p times n by its
    transpose.

    Attributes:
        matrix (numpy array): instance of the random matrix ensemble
            of size p times p.
        use_tridiagonal (bool): if set to True, WishartComplex matrices
            are sampled in its tridiagonal form, which has the same
            eigenvalues than its standard form.

    """

    def __init__(self, p, n, use_tridiagonal=False):
        """Constructor for WishartComplex class.

        Initializes an instance of this class with the given parameters,
        calling the parent class constructor and sampling a random instance.

        Args:
            p (int): number of rows of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            n (int): number of columns of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            use_tridiagonal (bool, default=False): if set to True, WishartComplex
                matrices are sampled in its tridiagonal form, which has the
                same eigenvalues than its standard form.

        """
        super().__init__(p=p, n=n, beta=2)
        self.use_tridiagonal = use_tridiagonal
        self.matrix = self.sample()
    
    def sample(self):
        if self.use_tridiagonal:
            return self.sample_tridiagonal()
        else:
            return self.sample_WisComplex_matrix()

    def sample_WisComplex_matrix(self):
        p = self.p
        n = self.n
        # p by n random complex matrix of random Gaussians
        A = np.random.randn(p,n) + (0+1j)*np.random.randn(p,n)
        # hermitian matrix
        self.matrix = np.matmul(A, A.transpose())
        return self.matrix


##########################################
### Wishart quaternion

class WishartQuaternion(WishartEnsemble):
    """Wishart Quaternion Ensemble class.

    The random matrices of this ensemble are formed by: sampling two
    random complex standard guassian matrices (X and Y), stacking them
    to create matrix A = [X  Y; -conj(Y)  conj(X)]. Finally matrix
    A is multiplied by its transpose in order to generate a matrix of
    the Wishart Quaternion Ensemble.

    Attributes:
        matrix (numpy array): instance of the random matrix ensemble
            of size 2p times 2p.
        use_tridiagonal (bool): if set to True, WishartQuaternion matrices
            are sampled in its tridiagonal form, which has the same
            eigenvalues than its standard form.

    """

    def __init__(self, p, n, use_tridiagonal=False):
        """Constructor for WishartQuaternion class.

        Initializes an instance of this class with the given parameters,
        calling the parent class constructor and sampling a random instance.

        Args:
            p (int): number of rows of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            n (int): number of columns of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            use_tridiagonal (bool, default=False): if set to True, WishartQuaternion
                matrices are sampled in its tridiagonal form, which has the
                same eigenvalues than its standard form.

        """
        super().__init__(p=p, n=n, beta=4)
        self.use_tridiagonal = use_tridiagonal
        self.matrix = self.sample()
    
    def sample(self):
        if self.use_tridiagonal:
            return self.sample_tridiagonal()
        else:
            return self.sample_WisQuatern_matrix()

    def sample_WisQuatern_matrix(self):
        p = self.p
        n = self.n
        # p by n random complex matrix of random Gaussians
        X = np.random.randn(p,n) + (0+1j)*np.random.randn(p,n)
        # p by n random complex matrix of random Gaussians
        Y = np.random.randn(p,n) + (0+1j)*np.random.randn(p,n)
        # [X Y; -conj(Y) conj(X)] 
        A = np.block([
                        [X               , Y],
                        [-np.conjugate(Y), np.conjugate(X)]
                    ])
        # hermitian matrix
        self.matrix = np.matmul(A, A.transpose())
        return self.matrix