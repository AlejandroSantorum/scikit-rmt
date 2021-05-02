'''Wishart Ensemble Test module

Testing WishartEnsemble module
'''

import numpy as np
from scipy import sparse
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
)

from skrmt.ensemble import WishartEnsemble


##########################################
### Wishart Real Ensemble = WRE

def test_wishart_real_init():
    '''Testing WRE init
    '''
    p_size = 3
    n_size = 5

    np.random.seed(1)
    wre = WishartEnsemble(beta=1, p=p_size, n=n_size)

    assert wre.matrix.shape == (p_size,p_size)

    assert_almost_equal(wre.matrix, np.array([[5.1919012, -4.96197148, 5.1988152],
                                            [-4.96197148, 9.08485594, -7.11948641],
                                            [5.1988152, -7.11948641, 7.91882311]]),
                        decimal=7)


def test_wre_symmetric():
    '''Testing that WRE matrix is symmetric
    '''
    p_size = 3
    n_size = 5
    wre = WishartEnsemble(beta=1, p=p_size, n=n_size)

    mtx = wre.matrix
    assert (mtx.transpose() == mtx).all()


def test_wre_set_size():
    '''Testing setter to change matrix sizes of WRE
    '''
    p1_size, n1_size = 3, 5
    p2_size, n2_size = 4, 6

    ens = WishartEnsemble(beta=1, p=p1_size, n=n1_size)
    assert ens.p == p1_size
    assert ens.n == n1_size
    assert ens.matrix.shape == (p1_size,p1_size)

    ens.set_size(p=p2_size, n=n2_size, resample_mtx=False)
    assert ens.p == p2_size
    assert ens.n == n2_size
    assert ens.matrix.shape == (p1_size,p1_size)

    ens.set_size(p=p2_size, n=n2_size, resample_mtx=True)
    assert ens.p == p2_size
    assert ens.n == n2_size
    assert ens.matrix.shape == (p2_size,p2_size)


def test_wre_build_tridiagonal():
    '''Testing tridiagonal form of WRE
    '''
    p_size, n_size = 3, 5
    beta = 1

    # sampling WishartReal tridiagonal
    np.random.seed(1)
    wre = WishartEnsemble(beta=1, p=p_size, n=n_size, use_tridiagonal=True)

    # sampling chi-squares and finding tridiagonal matrix in two ways
    np.random.seed(1)
    a_val = n_size*beta/ 2
    dfs = np.arange(p_size)
    chisqs_diag = np.array([np.sqrt(np.random.chisquare(2*a_val - beta*df)) for df in dfs])
    dfs = np.flip(dfs)
    chisqs_offdiag = np.array([np.sqrt(np.random.chisquare(beta*df)) for df in dfs[:-1]])
    diagonals = [chisqs_offdiag, chisqs_diag]
    mtx = sparse.diags(diagonals, [-1, 0])
    mtx = mtx.toarray()

    diag = np.array([chisqs_diag[0]**2]+[chisqs_diag[i+1]**2 + chisqs_offdiag[i]**2 \
                     for i in range(p_size-1)])
    offdiag = np.multiply(chisqs_offdiag, chisqs_diag[:-1])

    for i in range(p_size):
        assert diag[i] == wre.matrix[i][i]
    for i in range(p_size-1):
        assert offdiag[i] == wre.matrix[i][i+1]
        assert offdiag[i] == wre.matrix[i+1][i]

    assert_almost_equal(wre.matrix, np.dot(mtx, mtx.transpose()), decimal=7)


def test_beta1_eigval_pdf():
    '''Testing joint eigenvalue pdf
    '''
    p_size, n_size = 3, 5
    wre = WishartEnsemble(beta=1, p=p_size, n=n_size)

    wre.matrix = np.zeros((p_size,p_size))
    assert wre.eigval_pdf() == 0.0

    wre.matrix = np.eye(p_size)
    assert wre.eigval_pdf() == 0.0

    wre.matrix = 10*np.eye(p_size)
    assert wre.eigval_pdf() == 0.0


def test_wre_tridiag_hist():
    '''Testing tridiagonal histogram of WRE
    '''
    p_size, n_size = 50, 100
    wre1 = WishartEnsemble(beta=1, p=p_size, n=n_size, use_tridiagonal=False)
    wre2 = WishartEnsemble(beta=1, p=p_size, n=n_size, use_tridiagonal=True)

    wre1.matrix = wre2.matrix

    nbins = 20
    interval = (0,400)
    to_norm = False # without normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = wre1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = wre2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm)

    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_array_equal(hist_nottridiag, hist_tridiag)

    to_norm = True # normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = wre1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = wre2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm)

    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_almost_equal(hist_nottridiag, hist_tridiag, decimal=7)

    const = 1/n_size
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = wre1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm, norm_const=const)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = wre2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm, norm_const=const)

    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_array_equal(hist_nottridiag, hist_tridiag)


##########################################
### Wishart Complex Ensemble = WCE

def test_wishart_complex_init():
    '''Testing WCE init
    '''
    p_size = 3
    n_size = 8

    np.random.seed(1)
    wce = WishartEnsemble(beta=2, p=p_size, n=n_size)

    assert wce.matrix.shape == (p_size,p_size)

    mtx_sol = [[10.95414504+1.18652432j, 6.49496169+2.33465966j, \
                -0.41256652-0.97218795j],
               [6.49496169+2.33465966j, 3.04633885+0.74413531j, \
                -3.62175065-2.09216231j],
               [-0.41256652-0.97218795j, -3.62175065-2.09216231j, \
                -4.3742556 +4.41905659j]]

    assert_almost_equal(wce.matrix, np.array(mtx_sol), decimal=7)


def test_wce_symmetric():
    '''Testing that WCE matrix is symmetric
    '''
    p_size = 3
    n_size = 5
    wce = WishartEnsemble(beta=2, p=p_size, n=n_size)

    mtx= wce.matrix
    assert (mtx.transpose() == mtx).all()


def test_wce_set_size():
    '''Testing setter to change matrix sizes of WCE
    '''
    p1_size, n1_size = 5, 10
    p2_size, n2_size = 7, 14

    ens = WishartEnsemble(beta=2, p=p1_size, n=n1_size)
    assert ens.p == p1_size
    assert ens.n == n1_size
    assert ens.matrix.shape == (p1_size,p1_size)

    ens.set_size(p=p2_size, n=n2_size, resample_mtx=False)
    assert ens.p == p2_size
    assert ens.n == n2_size
    assert ens.matrix.shape == (p1_size,p1_size)

    ens.set_size(p=p2_size, n=n2_size, resample_mtx=True)
    assert ens.p == p2_size
    assert ens.n == n2_size
    assert ens.matrix.shape == (p2_size,p2_size)


def test_wce_build_tridiagonal():
    '''Testing tridiagonal form of WCE
    '''
    p_size, n_size = 3, 5
    beta = 2

    # sampling WishartComplex tridiagonal
    np.random.seed(1)
    wce = WishartEnsemble(beta=2, p=p_size, n=n_size, use_tridiagonal=True)

    # sampling chi-squares and finding tridiagonal matrix in two ways
    np.random.seed(1)
    a_val = n_size*beta/ 2
    dfs = np.arange(p_size)
    chisqs_diag = np.array([np.sqrt(np.random.chisquare(2*a_val - beta*df)) for df in dfs])
    dfs = np.flip(dfs)
    chisqs_offdiag = np.array([np.sqrt(np.random.chisquare(beta*df)) for df in dfs[:-1]])
    diagonals = [chisqs_offdiag, chisqs_diag]
    mtx = sparse.diags(diagonals, [-1, 0])
    mtx = mtx.toarray()

    diag = np.array([chisqs_diag[0]**2]+[chisqs_diag[i+1]**2 + chisqs_offdiag[i]**2 \
                     for i in range(p_size-1)])
    offdiag = np.multiply(chisqs_offdiag, chisqs_diag[:-1])

    for i in range(p_size):
        assert diag[i] == wce.matrix[i][i]
    for i in range(p_size-1):
        assert offdiag[i] == wce.matrix[i][i+1]
        assert offdiag[i] == wce.matrix[i+1][i]

    assert_almost_equal(wce.matrix, np.dot(mtx, mtx.transpose()), decimal=7)


def test_wce_tridiag_hist():
    '''Testing tridiagonal histogram of WCE
    '''
    p_size, n_size = 50, 100
    wce1 = WishartEnsemble(beta=2, p=p_size, n=n_size, use_tridiagonal=False)
    wce2 = WishartEnsemble(beta=2, p=p_size, n=n_size, use_tridiagonal=True)

    wce1.matrix = wce2.matrix

    nbins = 20
    interval = (0,400)
    to_norm = False # without normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = wce1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = wce2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm)

    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_array_equal(hist_nottridiag, hist_tridiag)

    to_norm = True # normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = wce1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = wce2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm)

    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_almost_equal(hist_nottridiag, hist_tridiag, decimal=7)



##########################################
### Wishart Quaternion Ensemble = WQE

def test_wishart_quatern_init():
    '''Testing WQE init
    '''
    p_size = 2
    n_size = 5

    np.random.seed(1)
    wqe = WishartEnsemble(beta=4, p=p_size, n=n_size)

    assert wqe.matrix.shape == (2*p_size,2*p_size)

    mtx_sol = [[-6.01217359e-01+7.71388293j, -2.91531515e+00-6.39395501j,\
                8.10811919e-17+6.4759398j, 2.16853073e+00-0.3065013j],
               [-2.91531515e+00-6.39395501j, 3.83549752e+00+5.28492599j,\
                -2.16853073e+00-0.3065013j, -1.30942157e-16-7.29353772j],
               [8.10811919e-17+6.4759398j, -2.16853073e+00-0.3065013j,\
                -6.01217359e-01-7.71388293j, -2.91531515e+00+6.39395501j],
               [2.16853073e+00-0.3065013j, -1.30942157e-16-7.29353772j,\
                -2.91531515e+00+6.39395501j, 3.83549752e+00-5.28492599j]]

    assert_almost_equal(wqe.matrix, np.array(mtx_sol), decimal=7)


def test_wqe_symmetric():
    '''Testing that WQE matrix is symmetric
    '''
    p_size = 3
    n_size = 5
    wqe = WishartEnsemble(beta=4, p=p_size, n=n_size)

    mtx = wqe.matrix
    assert (mtx.transpose() == mtx).all()


def test_wqe_set_size():
    '''Testing setter to change matrix sizes of WQE
    '''
    p_size1, n_size1 = 2, 3
    p_size2, n_size2 = 4, 5

    ens = WishartEnsemble(beta=4, p=p_size1, n=n_size1)
    assert ens.p == p_size1
    assert ens.n == n_size1
    assert ens.matrix.shape == (2*p_size1,2*p_size1)

    ens.set_size(p=p_size2, n=n_size2, resample_mtx=False)
    assert ens.p == p_size2
    assert ens.n == n_size2
    assert ens.matrix.shape == (2*p_size1,2*p_size1)

    ens.set_size(p=p_size2, n=n_size2, resample_mtx=True)
    assert ens.p == p_size2
    assert ens.n == n_size2
    assert ens.matrix.shape == (2*p_size2,2*p_size2)


def test_wqe_build_tridiagonal():
    '''Testing tridiagonal form of WQE
    '''
    p_size, n_size = 3, 5
    beta = 4

    # sampling WishartQuaternion tridiagonal
    np.random.seed(1)
    wqe = WishartEnsemble(beta=4, p=p_size, n=n_size, use_tridiagonal=True)

    # sampling chi-squares and finding tridiagonal matrix in two ways
    np.random.seed(1)
    a_val = n_size*beta/ 2
    dfs = np.arange(p_size)
    chisqs_diag = np.array([np.sqrt(np.random.chisquare(2*a_val - beta*df)) for df in dfs])
    dfs = np.flip(dfs)
    chisqs_offdiag = np.array([np.sqrt(np.random.chisquare(beta*df)) for df in dfs[:-1]])
    diagonals = [chisqs_offdiag, chisqs_diag]
    mtx = sparse.diags(diagonals, [-1, 0])
    mtx = mtx.toarray()

    diag = np.array([chisqs_diag[0]**2]+[chisqs_diag[i+1]**2 + chisqs_offdiag[i]**2 \
                     for i in range(p_size-1)])
    offdiag = np.multiply(chisqs_offdiag, chisqs_diag[:-1])

    for i in range(p_size):
        assert diag[i] == wqe.matrix[i][i]
    for i in range(p_size-1):
        assert offdiag[i] == wqe.matrix[i][i+1]
        assert offdiag[i] == wqe.matrix[i+1][i]

    assert_almost_equal(wqe.matrix, np.dot(mtx, mtx.transpose()), decimal=7)


def test_wqe_tridiag_hist():
    '''Testing tridiagonal histogram of WQE
    '''
    p_size, n_size = 50, 100
    wqe1 = WishartEnsemble(beta=4, p=p_size, n=n_size, use_tridiagonal=False)
    wqe2 = WishartEnsemble(beta=4, p=p_size, n=n_size, use_tridiagonal=True)

    wqe1.matrix = wqe2.matrix

    nbins = 20
    interval = (0,400)
    to_norm = False # without normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = wqe1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = wqe2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm)

    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_array_equal(hist_nottridiag, hist_tridiag)

    to_norm = True # normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = wqe1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = wqe2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm)

    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_almost_equal(hist_nottridiag, hist_tridiag, decimal=7)
