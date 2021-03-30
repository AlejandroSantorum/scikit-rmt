
from abc import abstractmethod
import numpy as np

from ._base_ensemble import _Ensemble


#########################################################################
### Circular Ensemble

class CircularEnsemble(_Ensemble):

    def __init__(self, beta=1):
        self.beta = beta

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def eigval_pdf(self):
        pass


def sample_Haar_mtx(n):
    # n by n random complex matrix
    X = np.random.randn(n,n) + (0+1j)*np.random.randn(n,n)
    # orthonormalizing matrix using QR algorithm
    Q,R = np.linalg.qr(X)
    # the resulting Q is Haar-distributed
    return Q


##########################################
### Circular Orthogonal Ensemble = COE

class COE(CircularEnsemble):

    def __init__(self, n):
        super().__init__(beta=1)
        self.matrix = self.sample(n)

    def sample(self, n):
        # sampling unitary Haar-distributed matrix
        U = sample_Haar_mtx(n)
        # mapping to Circular Orthogonal Ensemble
        self.matrix = np.matmul(U.transpose(), U)
        return self.matrix


##########################################
### Circular Unitary Ensemble = CUE

class CUE(CircularEnsemble):

    def __init__(self, n):
        super().__init__(beta=2)
        self.matrix = self.sample(n)

    def sample(self, n):
        # sampling unitary Haar-distributed matrix
        self.matrix = sample_Haar_mtx(n)
        return self.matrix


##########################################
### Circular Symplectic Ensemble = CSE

def _build_J_mtx(n):
    J = np.zeros((n,n))
    # selecting indices
    inds = np.arange(n-1)
    # selecting upper-diagonal indices
    J[inds, inds+1] = -1
    # selecting lower-diagonal indices
    J[inds+1, inds] = 1
    return J

class CSE(CircularEnsemble):

    def __init__(self, n):
        super().__init__(beta=4)
        self.matrix = self.sample(n)

    def sample(self, n):
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