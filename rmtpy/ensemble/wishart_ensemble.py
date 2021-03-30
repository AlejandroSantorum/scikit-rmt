
from abc import abstractmethod
import numpy as np

from ._base_ensemble import _Ensemble


#########################################################################
### Wishart Ensemble = Laguerre Ensemble

class WishartEnsemble(_Ensemble):

    def __init__(self, beta=1):
        self.beta = beta

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def eigval_pdf(self):
        pass
    

##########################################
### Wishart real

class WishartReal(WishartEnsemble):

    def __init__(self, p, n):
        super().__init__(beta=1)
        self.matrix = self.sample(p, n)

    def sample(self, p, n):
        # p by n matrix of random Gaussians
        A = np.random.randn(p,n)
        # symmetrize matrix
        self.matrix = np.matmul(A, A.transpose())
        return self.matrix


##########################################
### Wishart complex

class WishartComplex(WishartEnsemble):

    def __init__(self, p, n):
        super().__init__(beta=2)
        self.matrix = self.sample(p, n)

    def sample(self, p, n):
        # p by n random complex matrix of random Gaussians
        A = np.random.randn(p,n) + (0+1j)*np.random.randn(p,n)
        # hermitian matrix
        self.matrix = np.matmul(A, A.transpose())
        return self.matrix


##########################################
### Wishart quaternion

class WishartQuaternion(WishartEnsemble):

    def __init__(self, p, n):
        super().__init__(beta=4)
        self.matrix = self.sample(p, n)

    def sample(self, p, n):
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