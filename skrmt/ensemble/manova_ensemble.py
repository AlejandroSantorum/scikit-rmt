"""Manova Ensemble Module

This module contains the implementation of the Manova Ensemble, also
known as Jacobi Ensemble. This ensemble of random matrices contains
mainly three sub-ensembles: Manova Real Ensemble, Manova Complex Ensemble
and Manova Quaternion Ensemble.

"""

import numpy as np

from ._base_ensemble import _Ensemble


#########################################################################
### Manova Ensemble = Jacobi Ensemble

class ManovaEnsemble(_Ensemble):
    """General Manova Ensemble class.

    This class contains common attributes and methods for all the
    Manova ensembles. It also defines the basic interface to be
    supported by inherited classes. Manova Ensembles are divided in:
    - Manova Real Ensemble (MRE, beta=1): the random matrices of
    this ensemble are formed by sampling two random real standard
    guassian matrices (X and Y) of size m times n1 and m times n2
    respectively. Then, matrix A = (X * X') / (X * X' + Y * Y')
    generates a matrix of the Manova Real Ensemble.
    - Manova Complex Ensemble (MCE, beta=2): the random matrices
    of this ensemble are formed by sampling two random complex
    standard guassian matrices (X and Y) of size m times n1 and
    m times n2 respectively. Then, matrix A = (X * X') / (X * X' + Y * Y')
    generates a matrix of the Manova Complex Ensemble.
    - Manova Quaternion Ensemble (MQE, beta=4): the random matrices
    of this ensemble are formed by: sampling two random complex
    standard guassian matrices (X1 and X2), both of size m times n1.
    Another two random complex standard guassian matrices (Y1 and Y2),
    both of size m times n2, are sampled. They are stacked forming matrices
    X and Y:
    X = [X1  X2; -conj(X2)  conj(X1)]
    Y = [Y1  Y2; -conj(Y2)  conj(Y1)]
    Finally, matrix A = (X * X') / (X * X' + Y * Y') generates a matrix of
    the Manova Quaternion Ensemble.

    Attributes:
        beta (int): descriptive integer of the Manova ensemble type.
            For Real beta=1, for Complex beta=2, for Quaternion beta=4.
        m (int): number of rows of the random guassian matrices that
            generates the matrix of the corresponding ensemble.
        n1 (int): number of columns of the first random guassian matrix
            that generates the matrix of the corresponding ensemble.
        n2 (int): number of columns of the second random guassian matrix
            that generates the matrix of the corresponding ensemble.

    """

    def __init__(self, beta, m, n1, n2):
        """Constructor for ManovaEnsemble class.

        Initializes an instance of this class with the given parameters.

        Args:
            beta (int): descriptive integer of the Manova ensemble type.
                For Real beta=1, for Complex beta=2, for Quaternion beta=4.
            m (int): number of rows of the random guassian matrices that
                generates the matrix of the corresponding ensemble.
            n1 (int): number of columns of the first random guassian matrix
                that generates the matrix of the corresponding ensemble.
            n2 (int): number of columns of the second random guassian matrix
                that generates the matrix of the corresponding ensemble.

        """
        self.m = m
        self.n1 = n1
        self.n2 = n2
        self.beta = beta
        self.matrix = self.sample()
    
    def set_size(self, m, n1, n2, resample_mtx=False):
        """Setter of matrix size.

        Sets the matrix size. Useful if it has been initialized with a different value.

        Args:
            m (int): number of rows of the random guassian matrices that
                generates the matrix of the corresponding ensemble.
            n1 (int): number of columns of the first random guassian matrix
                that generates the matrix of the corresponding ensemble.
            n2 (int): number of columns of the second random guassian matrix
                that generates the matrix of the corresponding ensemble.
            resample_mtx (bool, default=False): If set to True, the ensemble matrix is
                resampled with the new dimensions.

        """
        self.m = m
        self.n1 = n1
        self.n2 = n2
        if resample_mtx:
            self.matrix = self.sample()

    def sample(self):
        """Samples new Manova Ensemble random matrix.

        The sampling algorithm depends on the specification of 
        beta parameter. If beta=1, Manova Real is sampled; if
        beta=2 Manova Complex is sampled and if beta=4 
        Manova Quaternion is sampled.

        Returns:
            numpy array containing new matrix sampled.

        References:
            Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        """
        if self.beta == 1:
            return self._sample_mre()
        elif self.beta == 2:
            return self._sample_mce()
        elif self.beta == 4:
            return self._sample_mqe()

    def _sample_mre(self):
        m = self.m
        n1 = self.n1
        n2 = self.n2
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

    def _sample_mce(self):
        m = self.m
        n1 = self.n1
        n2 = self.n2
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

    def _sample_mqe(self):
        m = self.m
        n1 = self.n1
        n2 = self.n2
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
            in the sub-Manova ensemble.

        Returns:
            real number. Value of the joint pdf of the current eigenvalues.

        References:
            Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.
            
        '''
        a1 = self.beta*self.n1/2
        a2 = self.beta*self.n2/2
        p = 1 + self.beta/2*(self.m - 1)

        # calculating Jacobi eigval pdf constant depeding on beta
        const_beta = 1
        for j in range(self.m):
            const_beta *= (sp.special.gamma(1 + self.beta/2) * sp.special.gamma(a1 + a2 -self.beta/2*(self.m - j)))/ \
                          (sp.special.gamma(1 + self.beta*j/2)*sp.special.gamma(a1 - self.beta/2*(self.m - j))*sp.special.gamma(a2 - self.beta/2*(self.m - j)))
        # calculating eigenvalues
        eigvals = np.linalg.eigvals(self.matrix)
        n_eigvals = len(eigvals)
        # calculating first prod
        prod1 = 1
        for j in range(n_eigvals):
            for i in range(j):
                prod1 *= np.abs(eigvals[i] - eigvals[j])**self.beta
        # calculating second prod
        prod2 = 1
        for j in range(n_eigvals):
            prod2 *= eigvals[j]**(a1-p) * (1 - eigvals[j])**(a2-p)
        # calculating Jacobi eigval pdf
        return const_beta * prod1 * prod2
