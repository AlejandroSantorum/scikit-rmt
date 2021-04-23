"""Gaussian Ensemble Module

This module contains the implementation of the Gaussian Ensemble, also
known as Hermite Ensemble. This ensemble of random matrices contains
mainly three sub-ensembles: Gaussian Orthogonal Ensemble (GOE),
Gaussian Unitary Ensemble (GUE) and Gaussian Symplectic Ensemble (GSE).

"""

import numpy as np
from scipy import sparse
from scipy import special

from ._base_ensemble import _Ensemble
from .tridiagonal_utils import tridiag_eigval_hist

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
        use_tridiagonal (bool): if set to True, Gaussian Ensemble
            matrices are sampled in its tridiagonal form, which has the same
            eigenvalues than its standard form. Otherwise, it is sampled using
            its standard form.
        
    References:
        Albrecht, J. and Chan, C.P. and Edelman, A. "Sturm sequences and random eigenvalue distributions".
            Foundations of Computational Mathematics. 9.4 (2008): 461-483.
        Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
            Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

    """

    def __init__(self, beta, n, use_tridiagonal=False):
        """Constructor for GaussianEnsemble class.

        Initializes an instance of this class with the given parameters.

        Args:
            n (int): random matrix size. Gaussian ensemble matrices are
                squared matrices. GOE and GUE are of size n times n,
                and GSE are of size 2n times 2n.
            beta (int, default=1): descriptive integer of the gaussian ensemble type.
                For GOE beta=1, for GUE beta=2, for GSE beta=4.
            use_tridiagonal (bool, default=False): if set to True, Gaussian Ensemble
            matrices are sampled in its tridiagonal form, which has the same
            eigenvalues than its standard form. Otherwise, it is sampled using
            its standard form.

        """
        self.n = n
        self.beta = beta
        self.use_tridiagonal = use_tridiagonal
        self.matrix = self.sample()

    def set_size(self, n, resample_mtx=False):
        """Setter of matrix size.

        Sets the matrix size. Useful if it has been initialized with a different value.

        Args:
            n (int): new random matrix size. Gaussian ensemble matrices are
                squared matrices. GOE and GUE are of size n times n, and 
                GSE are of size 2n times 2n.    
            resample_mtx (bool, default=False): If set to True, the ensemble matrix is
                resampled with the new dimensions.

        """
        self.n = n
        if resample_mtx:
            self.matrix = self.sample()

    def sample(self):
        """Samples new Gaussian Ensemble random matrix.

        The sampling algorithm depends on the specification of 
        use_tridiagonal parameter. If use_tridiagonal is set to True,
        a Gaussian Ensemble random matrix in its tridiagonal form
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
        else:
            if self.beta == 1:
                return self._sample_goe()
            elif self.beta == 2:
                return self._sample_gue()
            elif self.beta == 4:
                return self._sample_gse()

    def _sample_goe(self):
        # n by n matrix of random Gaussians
        A = np.random.randn(self.n,self.n)
        # symmetrize matrix
        self.matrix = (A + A.transpose())/2
        return self.matrix

    def _sample_gue(self):
        n = self.n
        # n by n random complex matrix
        A = np.random.randn(n,n) + (0+1j)*np.random.randn(n,n)
        # hermitian matrix
        self.matrix = (A + A.transpose())/2
        return self.matrix

    def _sample_gse(self):
        n = self.n
        # n by n random complex matrix
        X = np.random.randn(n,n) + (0+1j)*np.random.randn(n,n)
        # another n by n random complex matrix
        Y = np.random.randn(n,n) + (0+1j)*np.random.randn(n,n)
        # [X Y; -conj(Y) conj(X)] 
        A = np.block([
                        [X               , Y],
                        [-np.conjugate(Y), np.conjugate(X)]
                        ])
        # hermitian matrix
        self.matrix = (A + A.transpose())/2
        return self.matrix

    def sample_tridiagonal(self):
        '''Samples a Gaussian Ensemble random matrix in its tridiagonal form.

        Samples a random matrix of the specified Gaussian Ensemble (remember,
        beta=1 is GOE, beta=2 is GUE and beta=4 is GSE) in its tridiagonal
        form.

        Returns:
            numpy array containing new matrix sampled.

        References:
            Albrecht, J. and Chan, C.P. and Edelman, A. "Sturm sequences and random eigenvalue distributions".
                Foundations of Computational Mathematics. 9.4 (2008): 461-483.
            Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        '''
        # sampling diagonal normals
        normals = (1/np.sqrt(2)) * np.random.normal(loc=0, scale=np.sqrt(2), size=self.n)
        # sampling chi-squares
        dfs = np.flip(np.arange(1, self.n))
        chisqs = (1/np.sqrt(2)) * np.array([np.sqrt(np.random.chisquare(df*self.beta)) for df in dfs])
        # inserting diagonals
        diagonals = [chisqs, normals, chisqs]
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
        in the sub-Gaussian ensemble.

        Returns:
            real number. Value of the joint pdf of the current eigenvalues.

        References:
            Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.
            
        '''
        # calculating Hermite eigval pdf constant depeding on beta        
        const_beta = (2*np.pi)**(-self.n/2)
        for j in range(self.n):
            const_beta *= special.gamma(1 + self.beta/2)/special.gamma(1 + self.beta*j/2)
        # calculating eigenvalues
        eigvals = self.eigvals()
        n_eigvals = len(eigvals)
        # calculating prod
        pdf = 1
        for j in range(n_eigvals):
            for i in range(j):
                pdf *= np.abs(eigvals[i] - eigvals[j])**self.beta
        # calculating exponential term
        exp_val = np.exp(-np.sum((eigvals**2)/2))
        # calculating Hermite eigval pdf
        return const_beta * pdf * exp_val
