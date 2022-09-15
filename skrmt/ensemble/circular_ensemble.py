"""Circular Ensemble Module

This module contains the implementation of the Circular Ensemble.
This ensemble of random matrices contains mainly three sub-ensembles:
Circular Orthogonal Ensemble (COE), Circular Unitary Ensemble (CUE)
and Circular Symplectic Ensemble (CSE).

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

from ._base_ensemble import _Ensemble



def _sample_haar_mtx(size):
    """Samples Haar-distributed matrices.

    Samples Haar-distributed matrices that are useful to generate
    random matrices for COE, CUE and CSE ensembles.

    Args:
        n (int): matrix size.

    Returns:
        numpy array containing Haar-distributed random matrix.
    """
    # n by n random complex matrix
    x_mtx = np.random.randn(size,size) + (0+1j)*np.random.randn(size,size)
    # orthonormalizing matrix using QR algorithm
    q_mtx, _ = np.linalg.qr(x_mtx)
    # the resulting Q is Haar-distributed
    return q_mtx


#########################################################################
### Circular Ensemble

class CircularEnsemble(_Ensemble):
    """General Circular Ensemble class.

    This class contains common attributes and methods for all the
    Circular ensembles. Circular Ensembles are divided in:
    - Circular Orthogonal Ensemble (COE, beta=1): the distribution
    of the matrices of this ensemble are invariant under orthogonal
    conjugation, i.e., if X is in COE(n) and O is an orthogonal matrix,
    then O*X*O^T is equally distributed as X.
    - Circular Unitary Ensemble (CUE, beta=2): the distribution of
    the matrices of this ensemble are invariant under unitary
    conjugation, i.e., if X is in CUE(n) and O is an unitary matrix,
    then O*X*O^T is equally distributed as X.
    - Circular Symplectic Ensemble (CSE, beta=4): the distribution
    of the matrices of this ensemble are invariant under conjugation
    by the symplectic group.

    Attributes:
        matrix (numpy array): instance of the COE, CUE or CSE random
            matrix ensemble of size n times n if it is COE or CUE, or
            of size 2n times 2n if it is CSE.
        beta (int): descriptive integer of the gaussian ensemble type.
            For COE beta=1, for CUE beta=2, for CSE beta=4.
        n (int): random matrix size. Circular ensemble matrices are
            squared matrices. COE and CUE are of size n times n,
            and CSE are of size 2n times 2n.

    References:
        Killip, R. and Zozhan, R.
            Matrix Models AND Eigenvalue Statistics for Truncations of
            Classical Ensembles of Random Unitary Matrices.
            Communications in Mathematical Physics. 349 (2017): 991-1027.
        "Circular ensemble". Wikipedia.
            en.wikipedia.org/wiki/Circular_ensemble

    """

    def __init__(self, beta, n):
        """Constructor for CircularEnsemble class.

        Initializes an instance of this class with the given parameters.

        Args:
            beta (int): descriptive integer of the Circular ensemble type.
                For COE beta=1, for CUE beta=2, for CSE beta=4.
            n (int): random matrix size. Circular ensemble matrices are
                squared matrices. COE and CUE are of size n times n,
                and CSE are of size 2n times 2n.

        """
        super().__init__()
        # pylint: disable=invalid-name
        self.n = n
        self.beta = beta
        self.matrix = self.sample()

    def set_size(self, n, resample_mtx=False):
        # pylint: disable=arguments-differ
        """Setter of matrix size.

        Sets the matrix size. Useful if it has been initialized with a different value.

        Args:
            n (int): random matrix size. Circular ensemble matrices are
                squared matrices. COE and CUE are of size n times n,
                and CSE are of size 2n times 2n.

        """
        self.n = n
        if resample_mtx:
            self.matrix = self.sample()

    # pylint: disable=inconsistent-return-statements
    def sample(self):
        """Samples new Circular Ensemble random matrix.

        The sampling algorithm depends on the specification of
        beta parameter. If beta=1, COE matrix is sampled; if
        beta=2 CUE matrix is sampled and if beta=4
        CSE matrix is sampled.

        Returns:
            numpy array containing new matrix sampled.

        References:
            Killip, R. and Zozhan, R.
                Matrix Models AND Eigenvalue Statistics for Truncations of
                Classical Ensembles of Random Unitary Matrices.
                Communications in Mathematical Physics. 349 (2017): 991-1027.
            "Circular ensemble". Wikipedia.
                en.wikipedia.org/wiki/Circular_ensemble

        """
        if self.beta == 1:
            return self._sample_coe()
        if self.beta == 2:
            return self._sample_cue()
        if self.beta == 4:
            return self._sample_cse()

    def _sample_coe(self):
        # sampling unitary Haar-distributed matrix
        u_mtx = _sample_haar_mtx(self.n)
        # mapping to Circular Orthogonal Ensemble
        self.matrix = np.matmul(u_mtx.transpose(), u_mtx)
        return self.matrix

    def _sample_cue(self):
        # sampling unitary Haar-distributed matrix
        self.matrix = _sample_haar_mtx(self.n)
        return self.matrix

    def _sample_cse(self):
        # sampling unitary Haar-distributed matrix of size 2n
        u_mtx = _sample_haar_mtx(2*self.n)
        # mapping to Circular Symplectic Ensemble
        j_mtx = self._build_j_mtx()
        # U_R = J * U^T * J^T
        u_r_aux = np.matmul(j_mtx, u_mtx.transpose())
        u_r_mtx = np.matmul(u_r_aux, j_mtx.transpose())
        # A = U^R * U
        self.matrix = np.matmul(u_r_mtx, u_mtx)
        return self.matrix

    def _build_j_mtx(self):
        """Creates an useful matrix to sample CSE matrices.

        Creates matrix J of zeros but with the upper-diagonal
        set to -1 and the lower-diagonal set to 1. This matrix
        is useful in the sampling algorithm of CSE matrices.

        Args:
            n (int): matrix size.

        Returns:
            numpy array containing J matrix.

        References:
            Killip, R. and Zozhan, R.
                Matrix Models AND Eigenvalue Statistics for Truncations of
                Classical Ensembles of Random Unitary Matrices.
                Communications in Mathematical Physics. 349 (2017): 991-1027.
            "Circular ensemble". Wikipedia.
                en.wikipedia.org/wiki/Circular_ensemble
        """
        size = 2*self.n
        j_mtx = np.zeros((size,size))
        # selecting indices
        inds = np.arange(size-1)
        # selecting upper-diagonal indices
        j_mtx[inds, inds+1] = -1
        # selecting lower-diagonal indices
        j_mtx[inds+1, inds] = 1
        return j_mtx

    def eigvals(self):
        """Calculates the random matrix eigenvalues.

        Calculates the random matrix eigenvalues using numpy standard procedure.
        If the matrix ensemble is symmetric, a faster algorithm is used.

        Returns:
            numpy array with the calculated eigenvalues.

        """
        if self.beta == 1:
            return np.linalg.eigvalsh(self.matrix)
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
            Killip, R. and Zozhan, R.
                Matrix Models AND Eigenvalue Statistics for Truncations of
                Classical Ensembles of Random Unitary Matrices.
                Communications in Mathematical Physics. 349 (2017): 991-1027.

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
            rang_val = self.beta/2
            rang = ((-rang_val, rang_val), (-rang_val, rang_val))
            extent = [-rang_val, rang_val, -rang_val, rang_val]
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

        plt.suptitle("matrix size: "+\
                      str(len(self.matrix))+"x"+str(len(self.matrix)), fontweight="bold")

        # Saving plot or showing it
        if fig_path:
            plt.savefig(fig_path, dpi=600)
        else:
            plt.show()


    def eigval_pdf(self):
        '''Calculates joint eigenvalue pdf.

        Calculates joint eigenvalue probability density function given the
            current random matrix (so its eigenvalues). This function depends
            on beta, i.e., in the sub-Circular ensemble.

        Returns:
            real number. Value of the joint pdf of the current eigenvalues.

        References:
            Killip, R. and Zozhan, R.
                Matrix Models AND Eigenvalue Statistics for Truncations of
                Classical Ensembles of Random Unitary Matrices.
                Communications in Mathematical Physics. 349 (2017): 991-1027.
            "Circular ensemble". Wikipedia.
                en.wikipedia.org/wiki/Circular_ensemble

        '''
        # calculating Circular eigval pdf constant depeding on beta
        const_beta = (2*np.pi)**self.n * \
                     special.gamma(1 + self.n*self.beta/2)/(special.gamma(1 + self.beta/2)**self.n)
        # calculating eigenvalues
        eigvals = np.linalg.eigvals(self.matrix)
        n_eigvals = len(eigvals)
        # calculating prod
        pdf = 1
        for k in range(n_eigvals):
            for i in range(k):
                complex_num = complex(0, np.exp(eigvals[i])) - complex(0,np.exp(eigvals[k]))
                pdf *= np.abs(complex_num)**self.beta
        # calculating Circular eigval pdf
        return (1/const_beta) * pdf
