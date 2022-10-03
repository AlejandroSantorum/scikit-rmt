'''Wishart Ensemble Test module

Testing WishartEnsemble module
'''

from unicodedata import decimal
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

    mtx_sol = [[17.27142618+0.j, 4.56375544+1.14721231j, \
                -5.16181787+0.9188017j],
               [4.56375544-1.14721231j, 15.53877203+0.j, \
                2.44108976+10.47646539j],
               [-5.16181787-0.9188017j, 2.44108976-10.47646539j, \
                13.83214319+0.j]]

    assert_almost_equal(wce.matrix, np.array(mtx_sol), decimal=7)


def test_wce_hermitian():
    '''Testing that WCE matrix is hermitian
    '''
    p_size = 3
    n_size = 5
    wce = WishartEnsemble(beta=2, p=p_size, n=n_size)

    mtx = wce.matrix
    assert_almost_equal(mtx.transpose().conj(), mtx, decimal=7)


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

    mtx_sol = [[1.97823453e+01+0.00000000e+00j, -6.78596934e+00-5.77477475e+00j,\
                5.04164630e-17+4.44089210e-16j, 4.98065254e+00+2.42862094e+00j],
               [-6.78596934e+00+5.77477475e+00j, 1.77567967e+01+0.00000000e+00j,\
                -4.98065254e+00-2.42862094e+00j, 4.13628735e-17+4.44089210e-16j],
               [5.04164630e-17-4.44089210e-16j, -4.98065254e+00+2.42862094e+00j,\
                1.97823453e+01+0.00000000e+00j, -6.78596934e+00+5.77477475e+00j],
               [4.98065254e+00-2.42862094e+00j, 4.13628735e-17-4.44089210e-16j,\
                -6.78596934e+00-5.77477475e+00j, 1.77567967e+01+0.00000000e+00j]]

    assert_almost_equal(wqe.matrix, np.array(mtx_sol), decimal=7)


def test_wqe_hermitian():
    '''Testing that WQE matrix is hermitian
    '''
    p_size = 3
    n_size = 5
    wqe = WishartEnsemble(beta=4, p=p_size, n=n_size)

    mtx = wqe.matrix
    assert_almost_equal(mtx, mtx.transpose().conj(), decimal=7)


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
