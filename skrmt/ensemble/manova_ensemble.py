"""Manova Ensemble Module

This module contains the implementation of the Manova Ensemble, also
known as Jacobi Ensemble. This ensemble of random matrices contains
mainly three sub-ensembles: Manova Real Ensemble, Manova Complex Ensemble
and Manova Quaternion Ensemble.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

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
        matrix (numpy array): instance of the ManovaReal, ManovaComplex
            or ManovaQuaternion random ensembles. If it is an instance
            of ManovaReal or ManovaComplex, the random matrix is of
            size n times n. If it is a ManovaQuaternion, the random matrix
            is of size 2n times 2n.
        beta (int): descriptive integer of the Manova ensemble type.
            For Real beta=1, for Complex beta=2, for Quaternion beta=4.
        m (int): number of rows of the random guassian matrices that
            generates the matrix of the corresponding ensemble.
        n1 (int): number of columns of the first random guassian matrix
            that generates the matrix of the corresponding ensemble.
        n2 (int): number of columns of the second random guassian matrix
            that generates the matrix of the corresponding ensemble.

    References:
        Dumitriu, I. and Edelman, A. "Matrix Models for Beta Ensembles".
            Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

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
        super().__init__()
        # pylint: disable=invalid-name
        self.m = m
        self.n1 = n1
        self.n2 = n2
        self.beta = beta
        self.matrix = self.sample()

    def set_size(self, m, n1, n2, resample_mtx=False):
        # pylint: disable=arguments-differ
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

    # pylint: disable=inconsistent-return-statements
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
        if self.beta == 2:
            return self._sample_mce()
        if self.beta == 4:
            return self._sample_mqe()

    def _sample_mre(self):
        m_size = self.m
        n1_size = self.n1
        n2_size = self.n2
        # m by n1 random real matrix of random Gaussians
        x_mtx = np.random.randn(m_size,n1_size)
        # m by n2 random real matrix of random Gaussians
        y_mtx = np.random.randn(m_size,n2_size)
        # A1 = X * X'
        a1_mtx = np.matmul(x_mtx, x_mtx.transpose())
        # A2 = X * X' + Y * Y'
        a2_mtx = a1_mtx + np.matmul(y_mtx, y_mtx.transpose())
        # A = (X * X') / (X * X' + Y * Y') = (X * X') * (X * X' + Y * Y')^(-1)
        self.matrix = np.matmul(a1_mtx, np.linalg.inv(a2_mtx))
        return self.matrix

    def _sample_mce(self):
        m_size = self.m
        n1_size = self.n1
        n2_size = self.n2
        # m by n1 random complex matrix of random Gaussians
        x_mtx = np.random.randn(m_size,n1_size) + (0+1j)*np.random.randn(m_size,n1_size)
        # m by n2 random complex matrix of random Gaussians
        y_mtx = np.random.randn(m_size,n2_size) + (0+1j)*np.random.randn(m_size,n2_size)
        # A1 = X * X'
        a1_mtx = np.matmul(x_mtx, x_mtx.transpose())
        # A2 = X * X' + Y * Y'
        a2_mtx = a1_mtx + np.matmul(y_mtx, y_mtx.transpose())
        # A = (X * X') / (X * X' + Y * Y') = (X * X') * (X * X' + Y * Y')^(-1)
        self.matrix = np.matmul(a1_mtx, np.linalg.inv(a2_mtx))
        return self.matrix

    def _sample_mqe(self):
        m_size = self.m
        n1_size = self.n1
        n2_size = self.n2
        # m by n1 random complex matrix of random Gaussians
        x1_mtx = np.random.randn(m_size,n1_size) + (0+1j)*np.random.randn(m_size,n1_size)
        # m by n1 random complex matrix of random Gaussians
        x2_mtx = np.random.randn(m_size,n1_size) + (0+1j)*np.random.randn(m_size,n1_size)
        # m by n2 random complex matrix of random Gaussians
        y1_mtx = np.random.randn(m_size,n2_size) + (0+1j)*np.random.randn(m_size,n2_size)
        # m by n2 random complex matrix of random Gaussians
        y2_mtx = np.random.randn(m_size,n2_size) + (0+1j)*np.random.randn(m_size,n2_size)
        # X = [X1 X2; -conj(X2) conj(X1)]
        x_mtx = np.block([
                        [x1_mtx               , x2_mtx],
                        [-np.conjugate(x2_mtx), np.conjugate(x1_mtx)]
                        ])
        # Y = [Y1 Y2; -conj(Y2) conj(Y1)]
        y_mtx = np.block([
                         [y1_mtx               , y2_mtx],
                         [-np.conjugate(y2_mtx), np.conjugate(y1_mtx)]
                        ])
        # A1 = X * X'
        a1_mtx = np.matmul(x_mtx, x_mtx.transpose())
        # A2 = X * X' + Y * Y'
        a2_mtx = a1_mtx + np.matmul(y_mtx, y_mtx.transpose())
        # A = (X * X') / (X * X' + Y * Y') = (X * X') * (X * X' + Y * Y')^(-1)
        self.matrix = np.matmul(a1_mtx, np.linalg.inv(a2_mtx))
        return self.matrix

    def eigvals(self):
        """Calculates the random matrix eigenvalues.

        Calculates the random matrix eigenvalues using numpy standard procedure.
        If the matrix ensemble is symmetric, a faster algorithm is used.

        Returns:
            numpy array with the calculated eigenvalues.

        """
        return np.linalg.eigvals(self.matrix)

    def plot_eigval_hist(self, bins, interval=None, density=False, norm_const=None, fig_path=None):
        """Calculates and plots the histogram of the matrix eigenvalues

        Calculates and plots the histogram of the current sampled matrix eigenvalues.
        It is important to underline that this function works with real and complex
        eigenvalues: if the matrix eigenvalues are complex, they are plotted in the
        complex plane next to a heap map to study eigenvalue density.

        Args:
            bins (int or sequence): If bins is an integer, it defines the number of
                equal-width bins in the range. If bins is a sequence, it defines the
                bin edges, including the left edge of the first bin and the right
                edge of the last bin; in this case, bins may be unequally spaced.
            interval (tuple, default=None): Delimiters (xmin, xmax) of the histogram.
                The lower and upper range of the bins. Lower and upper outliers are ignored.
            density (bool, default=False): If True, draw and return a probability
                density: each bin will display the bin's raw count divided by the total
                number of counts and the bin width, so that the area under the histogram
                integrates to 1. If set to False, the absolute frequencies of the eigenvalues
                are returned.
            norm_const (float, default=None): Eigenvalue normalization constant. By default,
                it is set to None, so eigenvalues are not normalized. However, it is advisable
                to specify a normalization constant to observe eigenvalue spectrum, e.g.
                1/sqrt(n/2) if you want to analyze Wigner's Semicircular Law.
            fig_path (string, default=None): path to save the created figure. If it is not
                provided, the plot is shown are the end of the routine.

        References:
            Dumitriu, I. and Edelman, A.
                "Matrix Models for Beta Ensembles".
                Journal of Mathematical Physics. 43.11 (2002): 5830-5847.

        """
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        if self.beta == 1:
            return super().plot_eigval_hist(bins, interval, density, norm_const, fig_path)

        if (interval is not None) and not isinstance(interval, tuple):
            raise ValueError("interval argument must be a tuple (or None)")

        eigvals = self.eigvals()
        xvals = eigvals.real
        yvals = eigvals.imag

        if interval is None:
            rang = ((xvals.min(), xvals.max()), (yvals.min(), yvals.max()))
            extent = [xvals.min(), xvals.max(), yvals.min(), yvals.max()]
        else:
            rang = (interval, interval)
            extent = [interval[0], interval[1], interval[0], interval[1]]

        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.set_figheight(5)
        fig.set_figwidth(13)
        fig.subplots_adjust(hspace=.5)

        axes[0].set_xlim(rang[0][0], rang[0][1])
        axes[0].set_ylim(rang[1][0], rang[1][1])
        axes[0].plot(xvals, yvals, 'o')
        axes[0].set_title('Eigenvalue plot')
        axes[0].set_xlabel('real')
        axes[0].set_ylabel('imaginary')

        h2d,_,_,img = axes[1].hist2d(xvals, yvals, range=rang,
                                   cmap=plt.cm.get_cmap('nipy_spectral'))
        fig.colorbar(img, ax=axes[1])
        axes[1].cla()
        axes[1].imshow(h2d.transpose(), origin='lower', interpolation="bilinear", extent=extent)
        axes[1].set_title('Heatmap eigenvalue plot')
        axes[1].set_xlabel('real')
        axes[1].set_ylabel('imaginary')

        # Saving plot or showing it
        if fig_path:
            plt.savefig(fig_path, dpi=1200)
        else:
            plt.show()

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
        # pylint: disable=invalid-name
        a1 = self.beta*self.n1/2
        a2 = self.beta*self.n2/2
        p = 1 + self.beta/2*(self.m - 1)

        # calculating Jacobi eigval pdf constant depeding on beta
        const_beta = 1
        for j in range(self.m):
            const_beta *= (special.gamma(1 + self.beta/2) * \
                            special.gamma(a1 + a2 -self.beta/2*(self.m - j)))/ \
                          (special.gamma(1 + self.beta*j/2) * \
                            special.gamma(a1 - self.beta/2*(self.m - j)) * \
                              special.gamma(a2 - self.beta/2*(self.m - j)))
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
