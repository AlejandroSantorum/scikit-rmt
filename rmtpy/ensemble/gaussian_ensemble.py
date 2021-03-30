
from abc import abstractmethod
import numpy as np

from ._base_ensemble import _Ensemble


#########################################################################
### Gaussian Ensemble = Hermite Ensemble

class GaussianEnsemble(_Ensemble):

    def __init__(self, beta=1):
        self.beta = beta

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def eigval_pdf(self):
        pass
    

##########################################
### Gaussian Orthogonal Ensemble = GOE

def _sample_goe(n):
        # n by n matrix of random Gaussians
        A = np.random.randn(n,n)
        # symmetrize matrix
        return (A + A.transpose())/2

class GOE(GaussianEnsemble):

    def __init__(self, n):
        super().__init__(beta=1)
        self.matrix = _sample_goe(n)

    def sample(self, n):
        self.matrix = self._sample_goe(n)
        return self.matrix


##########################################
### Gaussian UNITARY Ensemble = GUE

def _sample_gue(n):
        # n by n random complex matrix
        A = np.random.randn(n,n) + (0+1j)*np.random.randn(n,n)
        # hermitian matrix
        return (A + A.transpose())/2

class GUE(GaussianEnsemble):

    def __init__(self, n):
        super().__init__(beta=2)
        self.matrix = _sample_gue(n)

    def sample(self, n):
        self.matrix = _sample_gue(n)
        return self.matrix


##########################################
### Gaussian Orthogonal Ensemble = GOE

def _sample_gse(n):
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
    return (A + A.transpose())/2

class GSE(GaussianEnsemble):

    def __init__(self, n):
        super().__init__(beta=4)
        self.matrix = _sample_gse(n)

    def sample(self, n):
        self.matrix = _sample_gse(n)
        return self.matrix