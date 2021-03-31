"""Wishart Ensemble Module

This module contains the implementation of the Wishart Ensemble, also
known as Laguerre Ensemble. This ensemble of random matrices contains
mainly three sub-ensembles: Wishart Real Ensemble, Wishart Complex Ensemble
and Wishart Quaternion Ensemble.

"""

from abc import abstractmethod
import numpy as np

from ._base_ensemble import _Ensemble


#########################################################################
### Wishart Ensemble = Laguerre Ensemble

class WishartEnsemble(_Ensemble):
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

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def eigval_pdf(self):
        pass
    

##########################################
### Wishart real

class WishartReal(WishartEnsemble):
    """Wishart Real Ensemble class.

    The random matrices of this ensemble are formed by multiplying
    a random real standard gaussian matrix of size p times n by its
    transpose.

    Attributes:
        matrix (numpy array): instance of the random matrix ensemble
            of size p times n.

    """

    def __init__(self, p, n):
        """Constructor for WishartReal class.

        Initializes an instance of this class with the given parameters,
        calling the parent class constructor and sampling a random instance.

        Args:
            p (int): number of rows of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            n (int): number of columns of the guassian matrix that generates
                the matrix of the corresponding ensemble.

        """
        super().__init__(p=p, n=n, beta=1)
        self.matrix = self.sample()

    def sample(self):
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
            of size p times n.

    """

    def __init__(self, p, n):
        """Constructor for WishartComplex class.

        Initializes an instance of this class with the given parameters,
        calling the parent class constructor and sampling a random instance.

        Args:
            p (int): number of rows of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            n (int): number of columns of the guassian matrix that generates
                the matrix of the corresponding ensemble.

        """
        super().__init__(p=p, n=n, beta=2)
        self.matrix = self.sample()

    def sample(self):
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
            of size p times n.

    """

    def __init__(self, p, n):
        """Constructor for WishartQuaternion class.

        Initializes an instance of this class with the given parameters,
        calling the parent class constructor and sampling a random instance.

        Args:
            p (int): number of rows of the guassian matrix that generates
                the matrix of the corresponding ensemble.
            n (int): number of columns of the guassian matrix that generates
                the matrix of the corresponding ensemble.

        """
        super().__init__(p=p, n=n, beta=4)
        self.matrix = self.sample()

    def sample(self):
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