"""Gaussian Ensemble Module

This module contains the implementation of the Gaussian Ensemble, also
known as Hermite Ensemble. This ensemble of random matrices contains
mainly three sub-ensembles: Gaussian Orthogonal Ensemble (GOE),
Gaussian Unitary Ensemble (GUE) and Gaussian Symplectic Ensemble (GSE).

"""

from abc import abstractmethod
import numpy as np

from ._base_ensemble import _Ensemble


#########################################################################
### Gaussian Ensemble = Hermite Ensemble

class GaussianEnsemble(_Ensemble):
    """General Gaussian Ensemble class.

    This class contains common attributes and methods for all the
    gaussian ensembles. It also defines the basic interface to be
    supported by inherited classes.

    Attributes:
        beta (int): descriptive integer of the gaussian ensemble type.
            For GOE beta=1, for GUE beta=2, for GSE beta=4.
        n (int): random matrix size. Gaussian ensemble matrices are
            squared matrices. GOE and GUE are of size n times n,
            and GSE are of size 2n times 2n.

    """

    def __init__(self, n, beta=1):
        """Constructor for GaussianEnsemble class.

        Initializes an instance of this class with the given parameters.

        Args:
            n (int): random matrix size. Gaussian ensemble matrices are
            squared matrices. GOE and GUE are of size n times n,
            and GSE are of size 2n times 2n.
            beta (int, default=1): descriptive integer of the gaussian ensemble type.
                For GOE beta=1, for GUE beta=2, for GSE beta=4.

        """
        self.n = n
        self.beta = beta

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def eigval_pdf(self):
        pass
    

##########################################
### Gaussian Orthogonal Ensemble = GOE

class GOE(GaussianEnsemble):
    """Gaussian Orthogonal Ensemble class.

    The distribution of the matrices of this ensemble are invariant
    under orthogonal conjugation, i.e., if X is in GOE(n) and O
    is an orthogonal matrix, then O*X*O^T is equally distributed
    as X.

    Attributes:
        matrix (numpy array): instance of the random matrix ensemble
            of size n times n.

    """

    def __init__(self, n):
        """Constructor for GOE class.

        Initializes an instance of this class with the given parameters,
        calling the parent class constructor and sampling a random instance.

        Args:
            n (int): random matrix size. GOE matrices are squared matrices
                of size n times n.

        """
        super().__init__(n=n, beta=1)
        self.matrix = self.sample()

    def sample(self):
        n = self.n
        # n by n matrix of random Gaussians
        A = np.random.randn(n,n)
        # symmetrize matrix
        self.matrix = (A + A.transpose())/2
        return self.matrix


##########################################
### Gaussian Unitary Ensemble = GUE

class GUE(GaussianEnsemble):
    """Gaussian Unitary Ensemble class.

    The distribution of the matrices of this ensemble are invariant
    under unitary conjugation, i.e., if X is in GUE(n) and O
    is an unitary matrix, then O*X*O^T is equally distributed
    as X.

    Attributes:
        matrix (numpy array): instance of the random matrix ensemble
            of size n times n.

    """

    def __init__(self, n):
        """Constructor for GUE class.

        Initializes an instance of this class with the given parameters,
        calling the parent class constructor and sampling a random instance.

        Args:
            n (int): random matrix size. GUE matrices are squared matrices
                of size n times n.

        """
        super().__init__(n=n, beta=2)
        self.matrix = self.sample()

    def sample(self):
        n = self.n
        # n by n random complex matrix
        A = np.random.randn(n,n) + (0+1j)*np.random.randn(n,n)
        # hermitian matrix
        self.matrix = (A + A.transpose())/2
        return self.matrix


##########################################
### Gaussian Symplectic Ensemble = GSE

class GSE(GaussianEnsemble):
    """Gaussian Symplectic Ensemble class.

    The distribution of the matrices of this ensemble are invariant
    under conjugation by the symplectic group.

    Attributes:
        matrix (numpy array): instance of the GSE random matrix
            ensemble of size 2n times 2n.

    """

    def __init__(self, n):
        """Constructor for GSE class.

        Initializes an instance of this class with the given parameters,
        calling the parent class constructor and sampling a random instance.

        Args:
            n (int): random matrix size. GSE matrices are squared matrices
                of size 2n times 2n.

        """
        super().__init__(n=n, beta=4)
        self.matrix = self.sample()

    def sample(self):
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