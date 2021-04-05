"""Gaussian Ensemble Module

This module contains the implementation of the Gaussian Ensemble, also
known as Hermite Ensemble. This ensemble of random matrices contains
mainly three sub-ensembles: Gaussian Orthogonal Ensemble (GOE),
Gaussian Unitary Ensemble (GUE) and Gaussian Symplectic Ensemble (GSE).

"""

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import sparse
from scipy import special

from ._base_ensemble import _Ensemble


#########################################################################
### Gaussian Ensemble = Hermite Ensemble

class GaussianEnsemble(_Ensemble, metaclass=ABCMeta):
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

    @abstractmethod
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

    @abstractmethod
    def sample(self):
        pass

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
        use_tridiagonal (bool): if set to True, GOE matrices
            are sampled in its tridiagonal form, which has the same
            eigenvalues than its standard form.

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
            return self.sample_tridiagonal()
        else:
            return self.sample_goe_matrix()

    def sample_goe_matrix(self):
        # n by n matrix of random Gaussians
        A = np.random.randn(self.n,self.n)
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
        use_tridiagonal (bool): if set to True, GUE matrices
            are sampled in its tridiagonal form, which has the same
            eigenvalues than its standard form.

    """

    def __init__(self, n, use_tridiagonal=False):
        """Constructor for GUE class.

        Initializes an instance of this class with the given parameters,
        calling the parent class constructor and sampling a random instance.

        Args:
            n (int): random matrix size. GUE matrices are squared matrices
                of size n times n.
            use_tridiagonal (bool, default=False): if set to True, GUE matrices
                are sampled in its tridiagonal form, which has the same
                eigenvalues than its standard form.

        """
        super().__init__(n=n, beta=2)
        self.use_tridiagonal = use_tridiagonal
        self.matrix = self.sample()

    def sample(self):
        if self.use_tridiagonal:
            return self.sample_tridiagonal()
        else:
            return self.sample_gue_matrix()

    def sample_gue_matrix(self):
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
        use_tridiagonal (bool): if set to True, GSE matrices
            are sampled in its tridiagonal form, which has the same
            eigenvalues than its standard form.

    """

    def __init__(self, n, use_tridiagonal=False):
        """Constructor for GSE class.

        Initializes an instance of this class with the given parameters,
        calling the parent class constructor and sampling a random instance.

        Args:
            n (int): random matrix size. GSE matrices are squared matrices
                of size 2n times 2n.
            use_tridiagonal (bool, default=False): if set to True, GSE matrices
                are sampled in its tridiagonal form, which has the same
                eigenvalues than its standard form.

        """
        super().__init__(n=n, beta=4)
        self.use_tridiagonal = use_tridiagonal
        self.matrix = self.sample()
    
    def sample(self):
        if self.use_tridiagonal:
            return self.sample_tridiagonal()
        else:
            return self.sample_gse_matrix()

    def sample_gse_matrix(self):
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