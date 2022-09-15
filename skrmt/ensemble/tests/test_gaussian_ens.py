'''Gaussian Ensemble Test module

Testing GaussianEnsemble module
'''

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
)

from skrmt.ensemble import GaussianEnsemble

##########################################
### Gaussian Orthogonal Ensemble = GOE

def test_goe_init():
    '''Testing GOE init
    '''
    n_size = 3

    np.random.seed(1)
    goe = GaussianEnsemble(beta=1, n=n_size)

    assert goe.matrix.shape == (n_size,n_size)

    assert_almost_equal(goe.matrix, np.array([[ 1.62434536, -0.84236252, 0.60832001],
                                              [-0.84236252, 0.86540763, -1.5313728 ],
                                              [ 0.60832001, -1.5313728, 0.3190391 ]]),
                        decimal=4)


def test_goe_symmetric():
    '''Testing that GOE matrix is symmetric
    '''
    n_size = 5
    goe = GaussianEnsemble(beta=1, n=n_size)

    m_mtx = goe.matrix
    assert (m_mtx.transpose() == m_mtx).all()



def test_goe_set_size():
    '''Testing setter to change matrix sizes of GOE
    '''
    n1_size = 3
    n2_size = 5

    goe = GaussianEnsemble(beta=1, n=n1_size)
    assert goe.n == n1_size
    assert goe.matrix.shape == (n1_size,n1_size)

    goe.set_size(n2_size, resample_mtx=False)
    assert goe.n == n2_size
    assert goe.matrix.shape == (n1_size,n1_size)

    goe.set_size(n2_size, resample_mtx=True)
    assert goe.n == n2_size
    assert goe.matrix.shape == (n2_size,n2_size)


def test_goe_build_tridiagonal():
    '''Testing tridiagonal form of GSE
    '''
    n_size = 5
    beta = 1

    np.random.seed(1)
    goe = GaussianEnsemble(beta=1, n=n_size, use_tridiagonal=True)

    np.random.seed(1)
    normals = (1/np.sqrt(2)) * np.random.normal(loc=0, scale=np.sqrt(2), size=n_size)
    dfs = np.flip(np.arange(1, n_size))
    chisqs = (1/np.sqrt(2)) * np.array([np.sqrt(np.random.chisquare(df*beta)) for df in dfs])

    for i in range(n_size):
        assert normals[i] == goe.matrix[i][i]
    for i in range(n_size-1):
        assert chisqs[i] == goe.matrix[i][i+1]
        assert chisqs[i] == goe.matrix[i+1][i]


def test_beta1_eigval_pdf():
    '''Testing joint eigenvalue pdf
    '''
    n_size = 3
    goe = GaussianEnsemble(beta=1, n=n_size)

    goe.matrix = np.zeros((n_size,n_size))
    assert goe.eigval_pdf() == 0.0

    goe.matrix = np.eye(n_size)
    assert goe.eigval_pdf() == 0.0

    goe.matrix = 10*np.eye(n_size)
    assert goe.eigval_pdf() == 0.0


def test_goe_tridiag_hist():
    '''Testing tridiagonal histogram of GOE
    '''
    n_size = 50
    goe1 = GaussianEnsemble(beta=1, n=n_size, use_tridiagonal=False)
    goe2 = GaussianEnsemble(beta=1, n=n_size, use_tridiagonal=True)

    goe1.matrix = goe2.matrix

    nbins = 10
    interval = (-2,2)
    to_norm = False # without normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = goe1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = goe2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm)
    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_array_equal(hist_nottridiag, hist_tridiag)

    to_norm = True # normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = goe1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = goe2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm)

    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_array_equal(hist_nottridiag, hist_tridiag)

    const = 1/np.sqrt(n_size/2)
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = goe1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm, norm_const=const)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = goe2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm, norm_const=const)

    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_array_equal(hist_nottridiag, hist_tridiag)



##########################################
### Gaussian Unitary Ensemble = GUE

def test_gue_init():
    '''Testing GUE init
    '''
    n_size = 3

    np.random.seed(1)
    gue = GaussianEnsemble(beta=2, n=n_size)

    assert gue.matrix.shape == (n_size,n_size)

    mtx_sol = [[1.62434536+0.j, -0.84236252+0.89226257j,\
                0.60832001-0.48012472j],
               [-0.84236252-0.89226257j, 0.86540763+0.j,\
                -1.5313728 +0.65309882j],
               [0.60832001+0.48012472j, -1.5313728 -0.65309882j,\
                0.3190391 +0.j]]

    assert_almost_equal(gue.matrix, np.array(mtx_sol),
                        decimal=4)


def test_gue_hermitian():
    '''Testing that GUE matrix is hermitian
    '''
    n_size = 5
    gue = GaussianEnsemble(beta=2, n=n_size)

    m_mtx = gue.matrix
    assert (m_mtx.transpose().conj() == m_mtx).all()


def test_gue_set_size():
    '''Testing setter to change matrix sizes of GUE
    '''
    n1_size = 5
    n2_size = 8

    gue = GaussianEnsemble(beta=2, n=n1_size)
    assert gue.n == n1_size
    assert gue.matrix.shape == (n1_size,n1_size)

    gue.set_size(n2_size, resample_mtx=False)
    assert gue.n == n2_size
    assert gue.matrix.shape == (n1_size,n1_size)

    gue.set_size(n2_size, resample_mtx=True)
    assert gue.n == n2_size
    assert gue.matrix.shape == (n2_size,n2_size)


def test_gue_build_tridiagonal():
    '''Testing tridiagonal form of GUE
    '''
    n_size = 5
    beta = 2

    np.random.seed(1)
    gue = GaussianEnsemble(beta=2, n=n_size, use_tridiagonal=True)

    np.random.seed(1)
    normals = (1/np.sqrt(2)) * np.random.normal(loc=0, scale=np.sqrt(2), size=n_size)
    dfs = np.flip(np.arange(1, n_size))
    chisqs = (1/np.sqrt(2)) * np.array([np.sqrt(np.random.chisquare(df*beta)) for df in dfs])

    for i in range(n_size):
        assert normals[i] == gue.matrix[i][i]
    for i in range(n_size-1):
        assert chisqs[i] == gue.matrix[i][i+1]
        assert chisqs[i] == gue.matrix[i+1][i]


def test_gue_tridiag_hist():
    '''Testing tridiagonal histogram of GUE
    '''
    n_size = 50
    gue1 = GaussianEnsemble(beta=2, n=n_size, use_tridiagonal=False)
    gue2 = GaussianEnsemble(beta=2, n=n_size, use_tridiagonal=True)

    gue1.matrix = gue2.matrix

    nbins = 10
    interval = (-2,2)
    to_norm = False # without normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = gue1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = gue2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm)
    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_array_equal(hist_nottridiag, hist_tridiag)

    to_norm = True # normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = gue1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = gue2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm)
    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_array_equal(hist_nottridiag, hist_tridiag)


##########################################
### Gaussian Symplectic Ensemble = GSE

def test_gse_init():
    '''Testing GSE init
    '''
    n_size = 2
    np.random.seed(1)
    gse = GaussianEnsemble(beta=4, n=n_size)

    assert gse.matrix.shape == (2*n_size,2*n_size)

    mtx_sol = [[1.62434536+0.86540763j, -0.56996408-0.27836347j,\
                0.-0.3224172j, -0.85573916+0.37485754j],
               [-0.56996408-0.27836347j, -1.07296862-0.7612069j,\
                0.85573916+0.37485754j, 0.-1.09989127j],
               [0.-0.3224172j, 0.85573916+0.37485754j,\
                1.62434536-0.86540763j, -0.56996408+0.27836347j],
               [-0.85573916+0.37485754j, 0.-1.09989127j, \
                -0.56996408+0.27836347j, -1.07296862+0.7612069j]]

    assert_almost_equal(gse.matrix, np.array(mtx_sol),
                        decimal=4)


def test_gse_symmetric():
    '''Testing that GSE matrix is symmetric
    '''
    n_size = 5
    gse = GaussianEnsemble(beta=4, n=n_size)

    m_tx = gse.matrix
    assert (m_tx.transpose() == m_tx).all()


def test_gse_set_size():
    '''Testing setter to change matrix sizes of GSE
    '''
    n1_size = 4
    n2_size = 9

    gse = GaussianEnsemble(beta=4, n=n1_size)
    assert gse.n == n1_size
    assert gse.matrix.shape == (2*n1_size,2*n1_size)

    gse.set_size(n2_size, resample_mtx=False)
    assert gse.n == n2_size
    assert gse.matrix.shape == (2*n1_size,2*n1_size)

    gse.set_size(n2_size, resample_mtx=True)
    assert gse.n == n2_size
    assert gse.matrix.shape == (2*n2_size,2*n2_size)


def test_gse_build_tridiagonal():
    '''Testing tridiagonal form of GSE
    '''
    n_size = 5
    beta = 4

    np.random.seed(1)
    gse = GaussianEnsemble(beta=4, n=n_size, use_tridiagonal=True)

    np.random.seed(1)
    normals = (1/np.sqrt(2)) * np.random.normal(loc=0, scale=np.sqrt(2), size=n_size)
    dfs = np.flip(np.arange(1, n_size))
    chisqs = (1/np.sqrt(2)) * np.array([np.sqrt(np.random.chisquare(df*beta)) for df in dfs])

    for i in range(n_size):
        assert normals[i] == gse.matrix[i][i]
    for i in range(n_size-1):
        assert chisqs[i] == gse.matrix[i][i+1]
        assert chisqs[i] == gse.matrix[i+1][i]


def test_gse_tridiag_hist():
    '''Testing tridiagonal histogram of GSE
    '''
    n_size = 50
    gse1 = GaussianEnsemble(beta=4, n=n_size, use_tridiagonal=False)
    gse2 = GaussianEnsemble(beta=4, n=n_size, use_tridiagonal=True)

    gse1.matrix = gse2.matrix

    nbins = 10
    interval = (-2,2)
    to_norm = False # without normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = gse1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = gse2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm)
    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_array_equal(hist_nottridiag, hist_tridiag)

    to_norm = True # normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = gse1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = gse2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm)
    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_array_equal(hist_nottridiag, hist_tridiag)
