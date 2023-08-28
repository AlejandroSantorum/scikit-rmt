'''Gaussian Ensemble Test module

Testing GaussianEnsemble module
'''

import pytest
import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
)

from skrmt.ensemble import GaussianEnsemble



def test_init_exception():
    with pytest.raises(ValueError):
        _ = GaussianEnsemble(beta=3, n=100)

def test_build_tridiagonal_exception():
    with pytest.raises(ValueError):
        _ = GaussianEnsemble(beta=1, n=100, sigma=2.0, tridiagonal_form=True)


##########################################
### Gaussian Orthogonal Ensemble = GOE

def test_goe_init():
    '''Testing GOE init
    '''
    n_size = 3

    np.random.seed(1)
    goe = GaussianEnsemble(beta=1, n=n_size)

    assert goe.matrix.shape == (n_size,n_size)

    assert_almost_equal(goe.matrix, np.array([[ 2.29717124, -1.1912805 ,  0.8602944 ],
                                              [-1.1912805 ,  1.22387121, -2.16568818],
                                              [0.8602944 , -2.16568818,  0.45118942]]),
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

    goe.set_size(n2_size, resample_mtx=True, random_state=1)
    assert goe.n == n2_size
    assert goe.matrix.shape == (n2_size,n2_size)


def test_goe_resample():
    '''Testing resample of GOE
    '''
    n_size = 5
    goe = GaussianEnsemble(beta=1, n=n_size, random_state=1)
    assert goe.tridiagonal_form == False
    
    prev_mtx = np.copy(goe.matrix)
    goe.resample(random_state=1)
    assert_array_equal(prev_mtx, goe.matrix)

    goe.resample(tridiagonal_form=True, random_state=1)
    assert goe.tridiagonal_form == True


def test_goe_build_tridiagonal():
    '''Testing tridiagonal form of GSE
    '''
    n_size = 5
    beta = 1

    np.random.seed(1)
    goe = GaussianEnsemble(beta=1, n=n_size, tridiagonal_form=True)

    np.random.seed(1)
    normals = np.random.normal(loc=0, scale=1, size=n_size)
    dfs = np.flip(np.arange(1, n_size))
    chisqs = np.array([np.sqrt(np.random.chisquare(df*beta)) for df in dfs])

    for i in range(n_size):
        assert normals[i] == goe.matrix[i][i]
    for i in range(n_size-1):
        assert chisqs[i] == goe.matrix[i][i+1]
        assert chisqs[i] == goe.matrix[i+1][i]


def test_beta1_joint_eigval_pdf():
    '''Testing joint eigenvalue pdf
    '''
    n_size = 3
    goe = GaussianEnsemble(beta=1, n=n_size)

    goe.matrix = np.zeros((n_size,n_size))
    assert goe.joint_eigval_pdf() == 0.0

    goe.matrix = np.eye(n_size)
    assert goe.joint_eigval_pdf() == 0.0

    goe.matrix = 10*np.eye(n_size)
    assert goe.joint_eigval_pdf() == 0.0


def test_goe_tridiag_hist():
    '''Testing tridiagonal histogram of GOE
    '''
    n_size = 50
    goe1 = GaussianEnsemble(beta=1, n=n_size, tridiagonal_form=False)
    goe2 = GaussianEnsemble(beta=1, n=n_size, tridiagonal_form=True)

    goe1.matrix = goe2.matrix

    nbins = 10
    interval = (-2,2)
    to_norm = False # without density normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = goe1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm, normalize=False)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = goe2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm, normalize=False)
    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_array_equal(hist_nottridiag, hist_tridiag)

    to_norm = True # density normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = goe1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm, normalize=False)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = goe2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm, normalize=False)

    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_array_equal(hist_nottridiag, hist_tridiag)

    to_norm = False # no density normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = goe1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm, normalize=False)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = goe2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm, normalize=False)

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

    mtx_sol = [[2.29717124+0.j, -1.1912805+1.26184983j,\
                0.8602944-0.67899889j],
               [-1.1912805-1.26184983j, 1.22387121+0.j, \
                -2.16568818+0.92362122j],
               [0.8602944+0.67899889j, -2.16568818-0.92362122j,\
                0.45118942+0.j]]

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

    gue.set_size(n2_size, resample_mtx=True, random_state=1)
    assert gue.n == n2_size
    assert gue.matrix.shape == (n2_size,n2_size)


def test_gue_resample():
    '''Testing resample of GUE
    '''
    n_size = 5
    gue = GaussianEnsemble(beta=2, n=n_size, random_state=1)
    assert gue.tridiagonal_form == False
    
    prev_mtx = np.copy(gue.matrix)
    gue.resample(random_state=1)
    assert_array_equal(prev_mtx, gue.matrix)

    gue.resample(tridiagonal_form=True, random_state=1)
    assert gue.tridiagonal_form == True


def test_gue_build_tridiagonal():
    '''Testing tridiagonal form of GUE
    '''
    n_size = 5
    beta = 2

    np.random.seed(1)
    gue = GaussianEnsemble(beta=2, n=n_size, tridiagonal_form=True)

    np.random.seed(1)
    normals = np.random.normal(loc=0, scale=1, size=n_size)
    dfs = np.flip(np.arange(1, n_size))
    chisqs = np.array([np.sqrt(np.random.chisquare(df*beta)) for df in dfs])

    for i in range(n_size):
        assert normals[i] == gue.matrix[i][i]
    for i in range(n_size-1):
        assert chisqs[i] == gue.matrix[i][i+1]
        assert chisqs[i] == gue.matrix[i+1][i]


def test_gue_tridiag_hist():
    '''Testing tridiagonal histogram of GUE
    '''
    n_size = 50
    gue1 = GaussianEnsemble(beta=2, n=n_size, tridiagonal_form=False)
    gue2 = GaussianEnsemble(beta=2, n=n_size, tridiagonal_form=True)

    gue1.matrix = gue2.matrix

    nbins = 10
    interval = (-2,2)
    to_norm = False # without density normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = gue1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm, normalize=False)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = gue2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm, normalize=False)
    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_array_equal(hist_nottridiag, hist_tridiag)

    to_norm = True # density normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = gue1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm, normalize=False)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = gue2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm, normalize=False)
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

    mtx_sol = [
        [3.24869073+0.j, -1.13992817-4.04635046j, 0.+0.j, -1.71147831-1.5178238j],
        [-1.13992817+4.04635046j, -2.14593724+0.j, 1.71147831+1.5178238j, 0.+0.j],
        [0.+0.j, 1.71147831-1.5178238j, 3.24869073+0.j, -1.13992817+4.04635046j],
        [-1.71147831+1.5178238j, 0.+0.j, -1.13992817-4.04635046j, -2.14593724+0.j],
    ]

    assert_almost_equal(gse.matrix, np.asarray(mtx_sol),
                        decimal=4)


def test_gse_hermitian():
    '''Testing that GSE matrix is hermitian
    '''
    n_size = 5
    gse = GaussianEnsemble(beta=4, n=n_size)

    m_tx = gse.matrix
    assert (m_tx.transpose().conj() == m_tx).all()


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

    gse.set_size(n2_size, resample_mtx=True, random_state=1)
    assert gse.n == n2_size
    assert gse.matrix.shape == (2*n2_size,2*n2_size)


def test_gse_resample():
    '''Testing resample of GSE
    '''
    n_size = 5
    gse = GaussianEnsemble(beta=4, n=n_size, random_state=1)
    assert gse.tridiagonal_form == False
    
    prev_mtx = np.copy(gse.matrix)
    gse.resample(random_state=1)
    assert_array_equal(prev_mtx, gse.matrix)

    gse.resample(tridiagonal_form=True, random_state=1)
    assert gse.tridiagonal_form == True


def test_gse_build_tridiagonal():
    '''Testing tridiagonal form of GSE
    '''
    n_size = 5
    beta = 4

    np.random.seed(1)
    gse = GaussianEnsemble(beta=4, n=n_size, tridiagonal_form=True)

    np.random.seed(1)
    n_size *= 2  # WQE matrices are 2p times 2p
    normals = np.random.normal(loc=0, scale=1, size=n_size)
    dfs = np.flip(np.arange(1, n_size))
    chisqs = np.array([np.sqrt(np.random.chisquare(df*beta)) for df in dfs])

    for i in range(n_size):
        assert normals[i] == gse.matrix[i][i]
    for i in range(n_size-1):
        assert chisqs[i] == gse.matrix[i][i+1]
        assert chisqs[i] == gse.matrix[i+1][i]


def test_gse_tridiag_hist():
    '''Testing tridiagonal histogram of GSE
    '''
    n_size = 50
    gse1 = GaussianEnsemble(beta=4, n=n_size, tridiagonal_form=False)
    gse2 = GaussianEnsemble(beta=4, n=n_size, tridiagonal_form=True)

    gse1.matrix = gse2.matrix

    nbins = 10
    interval = (-2,2)
    to_norm = False # without density normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = gse1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm, normalize=False)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = gse2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm, normalize=False)
    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_array_equal(hist_nottridiag, hist_tridiag)

    to_norm = True # density normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = gse1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm, normalize=False)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = gse2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm, normalize=False)
    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_array_equal(hist_nottridiag, hist_tridiag)
