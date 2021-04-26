"""Circular Ensemble Module

This module contains the implementation of the Circular Ensemble.
This ensemble of random matrices contains mainly three sub-ensembles:
Circular Orthogonal Ensemble (COE), Circular Unitary Ensemble (CUE)
and Circular Symplectic Ensemble (CSE).

"""

import cmath
import numpy as np

from ._base_ensemble import _Ensemble


#########################################################################
### Circular Ensemble

class CircularEnsemble(_Ensemble):
    """General Circular Ensemble class.

    This class contains common attributes and methods for all the
    Circular ensembles. Circular Ensembles are divided in:
    - Circular Orthogonal Ensemble (COE, beta=1): the distribution
    of the matrices of this ensemble are invariant under orthogonal
    conjugation, i.e., if X is in COE(n) and O is an orthogonal matrix,
    then O*X*O^T is equally distributed as X.
    - Circular Unitary Ensemble (CUE, beta=2): the distribution of
    the matrices of this ensemble are invariant under unitary
    conjugation, i.e., if X is in CUE(n) and O is an unitary matrix,
    then O*X*O^T is equally distributed as X.
    - Circular Symplectic Ensemble (CSE, beta=4): the distribution
    of the matrices of this ensemble are invariant under conjugation
    by the symplectic group.

    Attributes:
        matrix (numpy array): instance of the COE, CUE or CSE random
            matrix ensemble of size n times n if it is COE or CUE, or
            of size 2n times 2n if it is CSE.
        beta (int): descriptive integer of the gaussian ensemble type.
            For COE beta=1, for CUE beta=2, for CSE beta=4.
        n (int): random matrix size. Circular ensemble matrices are
            squared matrices. COE and CUE are of size n times n,
            and CSE are of size 2n times 2n.

    References:
        Killip, R. and Zozhan, R.
            Matrix Models AND Eigenvalue Statistics for Truncations of
            Classical Ensembles of Random Unitary Matrices.
            Communications in Mathematical Physics. 349 (2017): 991-1027.
        "Circular ensemble". Wikipedia.
            en.wikipedia.org/wiki/Circular_ensemble

    """

    def __init__(self, beta, n):
        """Constructor for CircularEnsemble class.

        Initializes an instance of this class with the given parameters.

        Args:
            beta (int): descriptive integer of the Circular ensemble type.
                For COE beta=1, for CUE beta=2, for CSE beta=4.
            n (int): random matrix size. Circular ensemble matrices are
                squared matrices. COE and CUE are of size n times n,
                and CSE are of size 2n times 2n.

        """
        self.n = n
        self.beta = beta
        self.matrix = self.sample()
    
    def set_size(self, n, resample_mtx=False):
        """Setter of matrix size.

        Sets the matrix size. Useful if it has been initialized with a different value.

        Args:
            n (int): random matrix size. Circular ensemble matrices are
                squared matrices. COE and CUE are of size n times n,
                and CSE are of size 2n times 2n.

        """
        self.n = n
        if resample_mtx:
            self.matrix = self.sample()

    def sample(self):
        """Samples new Circular Ensemble random matrix.

        The sampling algorithm depends on the specification of 
        beta parameter. If beta=1, COE matrix is sampled; if
        beta=2 CUE matrix is sampled and if beta=4 
        CSE matrix is sampled.

        Returns:
            numpy array containing new matrix sampled.

        References:
            Killip, R. and Zozhan, R.
                Matrix Models AND Eigenvalue Statistics for Truncations of
                Classical Ensembles of Random Unitary Matrices.
                Communications in Mathematical Physics. 349 (2017): 991-1027.
            "Circular ensemble". Wikipedia.
                en.wikipedia.org/wiki/Circular_ensemble 

        """
        if self.beta == 1:
            return self._sample_coe()
        elif self.beta == 2:
            return self._sample_cue()
        elif self.beta == 4:
            return self._sample_cse()
        
    def _sample_Haar_mtx(self, size):
        """Samples Haar-distributed matrices.

        Samples Haar-distributed matrices that are useful to generate
        random matrices for COE, CUE and CSE ensembles.

        Args:
            n (int): matrix size. 

        Returns:
            numpy array containing Haar-distributed random matrix.
        """
        # n by n random complex matrix
        X = np.random.randn(size,size) + (0+1j)*np.random.randn(size,size)
        # orthonormalizing matrix using QR algorithm
        Q,R = np.linalg.qr(X)
        # the resulting Q is Haar-distributed
        return Q

    def _sample_coe(self):
        n = self.n
        # sampling unitary Haar-distributed matrix
        U = self._sample_Haar_mtx(self.n)
        # mapping to Circular Orthogonal Ensemble
        self.matrix = np.matmul(U.transpose(), U)
        return self.matrix

    def _sample_cue(self):
        n = self.n
        # sampling unitary Haar-distributed matrix
        self.matrix = self._sample_Haar_mtx(self.n)
        return self.matrix

    def _sample_cse(self):
        n = self.n
        # sampling unitary Haar-distributed matrix of size 2n
        U = self._sample_Haar_mtx(2*self.n)
        # mapping to Circular Symplectic Ensemble
        J = self._build_J_mtx(2*self.n)
        # U_R = J * U^T * J^T
        U_R_1 = np.matmul(J, U.transpose())
        U_R = np.matmul(U_R_1, J.transpose())
        # A = U^R * U
        self.matrix = np.matmul(U_R, U)
        return self.matrix

    def _build_J_mtx(self, size):
        """Creates an useful matrix to sample CSE matrices.

        Creates matrix J of zeros but with the upper-diagonal
        set to -1 and the lower-diagonal set to 1. This matrix
        is useful in the sampling algorithm of CSE matrices.

        Args:
            n (int): matrix size. 

        Returns:
            numpy array containing J matrix.
        
        References:
            Killip, R. and Zozhan, R.
                Matrix Models AND Eigenvalue Statistics for Truncations of
                Classical Ensembles of Random Unitary Matrices.
                Communications in Mathematical Physics. 349 (2017): 991-1027.
            "Circular ensemble". Wikipedia.
                en.wikipedia.org/wiki/Circular_ensemble
        """
        J = np.zeros((size,size))
        # selecting indices
        inds = np.arange(size-1)
        # selecting upper-diagonal indices
        J[inds, inds+1] = -1
        # selecting lower-diagonal indices
        J[inds+1, inds] = 1
        return J
    
    def eigvals(self):
        """Calculates the random matrix eigenvalues.

        Calculates the random matrix eigenvalues using numpy standard procedure.
        If the matrix ensemble is symmetric, a faster algorithm is used.

        Returns:
            numpy array with the calculated eigenvalues.

        """
        return np.linalg.eigvals(self.matrix)

    def eigval_pdf(self):
        '''Calculates joint eigenvalue pdf.

        Calculates joint eigenvalue probability density function given the current 
            random matrix (so its eigenvalues). This function depends on beta, i.e.,
            in the sub-Circular ensemble.

        Returns:
            real number. Value of the joint pdf of the current eigenvalues.

        References:
            Killip, R. and Zozhan, R.
                Matrix Models AND Eigenvalue Statistics for Truncations of
                Classical Ensembles of Random Unitary Matrices.
                Communications in Mathematical Physics. 349 (2017): 991-1027.
            "Circular ensemble". Wikipedia.
                en.wikipedia.org/wiki/Circular_ensemble 
            
        '''
        # calculating Circular eigval pdf constant depeding on beta        
        const_beta = (2*np.pi)**self.n * sp.special.gamma(1 + self.n*self.beta/2)/(sp.special.gamma(1 + self.beta/2)**self.n)
        # calculating eigenvalues
        eigvals = np.linalg.eigvals(self.matrix)
        n_eigvals = len(eigvals)
        # calculating prod
        pdf = 1
        for k in range(n_eigvals):
            for i in range(k):
                pdf *= np.abs(complex(0, np.exp(eigvals[i])) - complex(0,np.exp(eigvals[k])))**self.beta
        # calculating Circular eigval pdf
        return (1/const_beta) * pdf
