"""Gaussian Ensemble Module

This module contains the implementation of the Gaussian Ensemble, also
known as Hermite Ensemble. This ensemble of random matrices contains
mainly three sub-ensembles: Gaussian Orthogonal Ensemble (GOE),
Gaussian Unitary Ensemble (GUE) and Gaussian Symplectic Ensemble (GSE).

"""

from abc import abstractmethod
import numpy as np
import scipy as sp

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

    def set_size(self, n):
        """Setter of matrix size.

        Sets the matrix size. Useful if it has been initialized with a different value.

        Args:
            n (int): new random matrix size. Gaussian ensemble matrices are
            squared matrices. GOE and GUE are of size n times n, and 
            GSE are of size 2n times 2n.

        """
        self.n = n

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

    def __init__(self, n, use_tridiagonal=False):
        """Constructor for GOE class.

        Initializes an instance of this class with the given parameters,
        calling the parent class constructor and sampling a random instance.

        Args:
            n (int): random matrix size. GOE matrices are squared matrices
                of size n times n.
            
            use_tridiagonal (bool, default=False): if set to True, GOE matrices
                are sampled in its tridiagonal form, which has the same
                eigenvalues than its standard form.

        """
        super().__init__(n=n, beta=1)
        self.use_tridiagonal = use_tridiagonal
        self.matrix = self.sample()

    def sample(self):
        if self.use_tridiagonal:
            return self._sample_goe_tridiagonal()
        else:
            return self._sample_goe_matrix()
    
    def _sample_goe_matrix(self):
        # n by n matrix of random Gaussians
        A = np.random.randn(self.n,self.n)
        # symmetrize matrix
        self.matrix = (A + A.transpose())/2
        return self.matrix
    
    def _sample_goe_tridiagonal(self):
        # sampling diagonal normals
        normals = (1/np.sqrt(2)) * np.random.normal(loc=0, scale=np.sqrt(2), size=self.n)
        # sampling chi-squares
        dfs = np.arange(1, self.n)
        chisqs = (1/np.sqrt(2)) * [np.sqrt(np.random.chisquare(df)) for df in dfs]
        # inserting diagonals
        diags = [chisqs, normals, chisqs]
        M = sp.sparse.diags(diagonals, [-1, 0, 1])
        # converting to numpy array
        self.matrix = M.toarray()
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