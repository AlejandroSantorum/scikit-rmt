"""Circular Ensemble Module

This module contains the implementation of the Circular Ensemble.
This ensemble of random matrices contains mainly three sub-ensembles:
Circular Orthogonal Ensemble (COE), Circular Unitary Ensemble (CUE)
and Circular Symplectic Ensemble (CSE).

"""

from abc import abstractmethod
import numpy as np

from ._base_ensemble import _Ensemble


#########################################################################
### Circular Ensemble

class CircularEnsemble(_Ensemble):
    """General Circular Ensemble class.

    This class contains common attributes and methods for all the
    Circular ensembles. It also defines the basic interface to be
    supported by inherited classes.

    Attributes:
        beta (int): descriptive integer of the gaussian ensemble type.
            For COE beta=1, for CUE beta=2, for CSE beta=4.
        n (int): random matrix size. Circular ensemble matrices are
            squared matrices. COE and CUE are of size n times n,
            and CSE are of size 2n times 2n.

    """

    def __init__(self, n, beta=1):
        """Constructor for CircularEnsemble class.

        Initializes an instance of this class with the given parameters.

        Args:
            n (int): random matrix size. Circular ensemble matrices are
            squared matrices. COE and CUE are of size n times n,
            and CSE are of size 2n times 2n.
            beta (int, default=1): descriptive integer of the Circular ensemble type.
                For COE beta=1, for CUE beta=2, for CSE beta=4.

        """
        self.n = n
        self.beta = beta

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def eigval_pdf(self):
        pass


def sample_Haar_mtx(n):
    """Samples Haar-distributed matrices.

    Samples Haar-distributed matrices that are useful to generate
    random matrices for COE, CUE and CSE ensembles.

    Args:
        n (int): matrix size. 

    Returns:
        numpy array containing Haar-distributed random matrix.
    """
    # n by n random complex matrix
    X = np.random.randn(n,n) + (0+1j)*np.random.randn(n,n)
    # orthonormalizing matrix using QR algorithm
    Q,R = np.linalg.qr(X)
    # the resulting Q is Haar-distributed
    return Q


##########################################
### Circular Orthogonal Ensemble = COE

class COE(CircularEnsemble):
    """Circular Orthogonal Ensemble class.

    The distribution of the matrices of this ensemble are invariant
    under orthogonal conjugation, i.e., if X is in COE(n) and O
    is an orthogonal matrix, then O*X*O^T is equally distributed
    as X.

    Attributes:
        matrix (numpy array): instance of the random matrix ensemble
            of size n times n.

    """

    def __init__(self, n):
        """Constructor for COE class.

        Initializes an instance of this class with the given parameters,
        calling the parent class constructor and sampling a random instance.
        Matrices of this ensemble are formed using matrices of the CUE
        esemble. If U is in CUE(n), then U'*U is in COE(n).

        Args:
            n (int): random matrix size. COE matrices are
                squared matrices of size n times n.

        """
        super().__init__(n=n, beta=1)
        self.matrix = self.sample()

    def sample(self):
        n = self.n
        # sampling unitary Haar-distributed matrix
        U = sample_Haar_mtx(n)
        # mapping to Circular Orthogonal Ensemble
        self.matrix = np.matmul(U.transpose(), U)
        return self.matrix


##########################################
### Circular Unitary Ensemble = CUE

class CUE(CircularEnsemble):
    """Circular Unitary Ensemble class.

    The distribution of the matrices of this ensemble are invariant
    under unitary conjugation, i.e., if X is in CUE(n) and O
    is an unitary matrix, then O*X*O^T is equally distributed
    as X.

    Attributes:
        matrix (numpy array): instance of the random matrix ensemble
            of size n times n.

    """

    def __init__(self, n):
        """Constructor for CUE class.

        Initializes an instance of this class with the given parameters,
        calling the parent class constructor and sampling a random instance.
        Matrices of this ensemble are formed by sampling Haar-distributed
        matrices.

        Args:
            n (int): random matrix size. CUE matrices are
                squared matrices of size n times n.

        """
        super().__init__(n=n, beta=2)
        self.matrix = self.sample()

    def sample(self):
        n = self.n
        # sampling unitary Haar-distributed matrix
        self.matrix = sample_Haar_mtx(n)
        return self.matrix


##########################################
### Circular Symplectic Ensemble = CSE

def _build_J_mtx(n):
    """Creates an useful matrix to sample CSE matrices.

    Creates matrix J of zeros but with the upper-diagonal
    set to -1 and the lower-diagonal set to 1. This matrix
    is useful in the sampling algorithm of CSE matrices.

    Args:
        n (int): matrix size. 

    Returns:
        numpy array containing J matrix.
    """
    J = np.zeros((n,n))
    # selecting indices
    inds = np.arange(n-1)
    # selecting upper-diagonal indices
    J[inds, inds+1] = -1
    # selecting lower-diagonal indices
    J[inds+1, inds] = 1
    return J

class CSE(CircularEnsemble):
    """Circular Symplectic Ensemble class.

    The distribution of the matrices of this ensemble are invariant
    under conjugation by the symplectic group.

    Attributes:
        matrix (numpy array): instance of the CSE random matrix
            ensemble of size 2n times 2n.

    """

    def __init__(self, n):
        """Constructor for CSE class.

        Initializes an instance of this class with the given parameters,
        calling the parent class constructor and sampling a random instance.
        A matrix M of this ensemble is formed by: sampling squared Haar-distributed
        matrix U of size 2n, and a matrix J of zeros but with the
        upper-diagonal set to -1 and the lower-diagonal set to 1. Then,
        U^R = J*U'*J and, finally, M = U^R * U.

        Args:
            n (int): random matrix size. CSE matrices are
                squared matrices of size 2n times 2n.

        """
        super().__init__(n=n, beta=4)
        self.matrix = self.sample()

    def sample(self):
        n = self.n
        # sampling unitary Haar-distributed matrix of size 2n
        U = sample_Haar_mtx(2*n)
        # mapping to Circular Symplectic Ensemble
        J = _build_J_mtx(2*n)
        # U_R = J * U^T * J^T
        U_R_1 = np.matmul(J, U.transpose())
        U_R = np.matmul(U_R_1, J.transpose())
        # A = U^R * U
        self.matrix = np.matmul(U_R, U)
        return self.matrix