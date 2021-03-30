
from abc import abstractmethod
import numpy as np

from ._base_ensemble import _Ensemble


#########################################################################
### Manova Ensemble = Jacobi Ensemble

class ManovaEnsemble(_Ensemble):

    def __init__(self, beta=1):
        self.beta = beta

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def eigval_pdf(self):
        pass
    

##########################################
### Manova real

class ManovaReal(ManovaEnsemble):

    def __init__(self, m, n1, n2):
        super().__init__(beta=1)
        self.matrix = self.sample(m, n1, n2)

    def sample(self, m, n1, n2):
        # m by n1 random real matrix of random Gaussians
        X = np.random.randn(m,n1)
        # m by n2 random real matrix of random Gaussians
        Y = np.random.randn(m,n2)
        # A1 = X * X'
        A1 = np.matmul(X, X.transpose())
        # A2 = X * X' + Y * Y'
        A2 = A1 + np.matmul(Y, Y.transpose())
        # A = (X * X') / (X * X' + Y * Y') = (X * X') * (X * X' + Y * Y')^(-1)
        self.matrix = np.matmul(A1, np.linalg.inv(A2))
        return self.matrix


##########################################
### Manova complex

class ManovaComplex(ManovaEnsemble):

    def __init__(self, m, n1, n2):
        super().__init__(beta=2)
        self.matrix = self.sample(m, n1, n2)

    def sample(self, m, n1, n2):
        # m by n1 random complex matrix of random Gaussians
        X = np.random.randn(m,n1) + (0+1j)*np.random.randn(m,n1)
        # m by n2 random complex matrix of random Gaussians
        Y = np.random.randn(m,n2) + (0+1j)*np.random.randn(m,n2)
        # A1 = X * X'
        A1 = np.matmul(X, X.transpose())
        # A2 = X * X' + Y * Y'
        A2 = A1 + np.matmul(Y, Y.transpose())
        # A = (X * X') / (X * X' + Y * Y') = (X * X') * (X * X' + Y * Y')^(-1)
        self.matrix = np.matmul(A1, np.linalg.inv(A2))
        return self.matrix


##########################################
### Manova quaternion

class ManovaQuaternion(ManovaEnsemble):

    def __init__(self, m, n1, n2):
        super().__init__(beta=4)
        self.matrix = self.sample(m, n1, n2)

    def sample(self, m, n1, n2):
        # m by n1 random complex matrix of random Gaussians
        X1 = np.random.randn(m,n1) + (0+1j)*np.random.randn(m,n1)
        # m by n1 random complex matrix of random Gaussians
        X2 = np.random.randn(m,n1) + (0+1j)*np.random.randn(m,n1)
        # m by n2 random complex matrix of random Gaussians
        Y1 = np.random.randn(m,n2) + (0+1j)*np.random.randn(m,n2)
        # m by n2 random complex matrix of random Gaussians
        Y2 = np.random.randn(m,n2) + (0+1j)*np.random.randn(m,n2)
        # X = [X1 X2; -conj(X2) conj(X1)] 
        X = np.block([
                        [X1               , X2],
                        [-np.conjugate(X2), np.conjugate(X1)]
                    ])
        # Y = [Y1 Y2; -conj(Y2) conj(Y1)] 
        Y = np.block([
                        [Y1               , Y2],
                        [-np.conjugate(Y2), np.conjugate(Y1)]
                    ])
        # A1 = X * X'
        A1 = np.matmul(X, X.transpose())
        # A2 = X * X' + Y * Y'
        A2 = A1 + np.matmul(Y, Y.transpose())
        # A = (X * X') / (X * X' + Y * Y') = (X * X') * (X * X' + Y * Y')^(-1)
        self.matrix = np.matmul(A1, np.linalg.inv(A2))
        return self.matrix