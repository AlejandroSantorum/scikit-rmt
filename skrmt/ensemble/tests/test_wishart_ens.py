"""Wishart Ensemble Test module

Testing WishartEnsemble module
"""

import pytest
import numpy as np
from scipy import sparse
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
)

from skrmt.ensemble import WishartEnsemble



def test_init_exception():
    """Test init exception - invalid beta
    """
    with pytest.raises(ValueError):
        _ = WishartEnsemble(beta=3, p=100, n=300)

def test_tridiagonal_ratio_exception():
    """Test tridiagonal ratio exception (ratio has to be < 1 if tridiagonal_form=True)
    """
    with pytest.raises(ValueError):
        _ = WishartEnsemble(beta=1, p=300, n=100, tridiagonal_form=True)

def test_tridiagonal_sigma_exception():
    """Test tridiagonal sigma exception
    """
    with pytest.raises(ValueError):
        _ = WishartEnsemble(beta=1, p=100, n=300, sigma=2.0, tridiagonal_form=True)


##########################################
### Wishart Real Ensemble = WRE

def test_wishart_real_init():
    """Testing WRE init
    """
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
    """Testing that WRE matrix is symmetric
    """
    p_size = 3
    n_size = 5
    wre = WishartEnsemble(beta=1, p=p_size, n=n_size)

    mtx = wre.matrix
    assert (mtx.transpose() == mtx).all()


def test_wre_resample():
    """Testing resample of WRE
    """
    p_size = 3
    n_size = 5
    wre = WishartEnsemble(beta=1, p=p_size, n=n_size, random_state=1)
    assert not wre.tridiagonal_form

    prev_mtx = np.copy(wre.matrix)
    wre.resample(random_state=1)
    assert_array_equal(prev_mtx, wre.matrix)

    wre.resample(tridiagonal_form=True, random_state=1)
    assert wre.tridiagonal_form


def test_wre_build_tridiagonal():
    """Testing tridiagonal form of WRE
    """
    p_size, n_size = 3, 5
    beta = 1

    # sampling WishartReal tridiagonal
    np.random.seed(1)
    wre = WishartEnsemble(beta=1, p=p_size, n=n_size, tridiagonal_form=True)

    # sampling chi-squares and finding tridiagonal matrix in two ways
    np.random.seed(1)
    a_val = n_size*beta/2
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


def test_beta1_joint_eigval_pdf():
    """Testing joint eigenvalue pdf
    """
    p_size, n_size = 3, 5
    wre = WishartEnsemble(beta=1, p=p_size, n=n_size)

    wre.matrix = np.zeros((p_size,p_size))
    assert wre.joint_eigval_pdf() == 0.0

    wre.matrix = np.eye(p_size)
    assert wre.joint_eigval_pdf() == 0.0

    wre.matrix = 10*np.eye(p_size)
    assert wre.joint_eigval_pdf() == 0.0


def test_wre_tridiag_hist():
    """Testing tridiagonal histogram of WRE
    """
    p_size, n_size = 50, 100
    wre1 = WishartEnsemble(beta=1, p=p_size, n=n_size, tridiagonal_form=False)
    wre2 = WishartEnsemble(beta=1, p=p_size, n=n_size, tridiagonal_form=True)

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

    to_norm = True # density normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = wre1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm, normalize=True)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = wre2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm, normalize=True)

    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_almost_equal(hist_nottridiag, hist_tridiag, decimal=7)

    to_norm = False # no density normalization
    # calculating histogram using standard naive procedure
    hist_nottridiag, bins_nottridiag = wre1.eigval_hist(bins=nbins, interval=interval,
                                                        density=to_norm, normalize=True)
    # calculating histogram using tridiagonal procedure
    hist_tridiag, bins_tridiag = wre2.eigval_hist(bins=nbins, interval=interval,
                                                  density=to_norm, normalize=True)

    assert_array_equal(bins_nottridiag, bins_tridiag)
    assert_array_equal(hist_nottridiag, hist_tridiag)


##########################################
### Wishart Complex Ensemble = WCE

def test_wishart_complex_init():
    """Testing WCE init
    """
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
    """Testing that WCE matrix is hermitian
    """
    p_size = 3
    n_size = 5
    wce = WishartEnsemble(beta=2, p=p_size, n=n_size)

    mtx = wce.matrix
    assert_almost_equal(mtx.transpose().conj(), mtx, decimal=7)


def test_wce_resample():
    """Testing resample of WCE
    """
    p_size = 3
    n_size = 5
    wce = WishartEnsemble(beta=2, p=p_size, n=n_size, random_state=1)
    assert not wce.tridiagonal_form

    prev_mtx = np.copy(wce.matrix)
    wce.resample(random_state=1)
    assert_array_equal(prev_mtx, wce.matrix)

    wce.resample(tridiagonal_form=True, random_state=1)
    assert wce.tridiagonal_form


def test_wce_build_tridiagonal():
    """Testing tridiagonal form of WCE
    """
    p_size, n_size = 3, 5
    beta = 2

    # sampling WishartComplex tridiagonal
    np.random.seed(1)
    wce = WishartEnsemble(beta=2, p=p_size, n=n_size, tridiagonal_form=True)

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
    """Testing tridiagonal histogram of WCE
    """
    p_size, n_size = 50, 100
    wce1 = WishartEnsemble(beta=2, p=p_size, n=n_size, tridiagonal_form=False)
    wce2 = WishartEnsemble(beta=2, p=p_size, n=n_size, tridiagonal_form=True)

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
    """Testing WQE init
    """
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
    """Testing that WQE matrix is hermitian
    """
    p_size = 3
    n_size = 5
    wqe = WishartEnsemble(beta=4, p=p_size, n=n_size)

    mtx = wqe.matrix
    assert_almost_equal(mtx, mtx.transpose().conj(), decimal=7)


def test_wqe_resample():
    """Testing resample of WQE
    """
    p_size = 3
    n_size = 5
    wqe = WishartEnsemble(beta=4, p=p_size, n=n_size, random_state=1)
    assert not wqe.tridiagonal_form

    prev_mtx = np.copy(wqe.matrix)
    wqe.resample(random_state=1)
    assert_array_equal(prev_mtx, wqe.matrix)

    wqe.resample(tridiagonal_form=True, random_state=1)
    assert wqe.tridiagonal_form


def test_wqe_build_tridiagonal():
    """Testing tridiagonal form of WQE
    """
    p_size, n_size = 2, 5
    beta = 4

    # sampling WishartQuaternion tridiagonal
    np.random.seed(1)
    wqe = WishartEnsemble(beta=4, p=p_size, n=n_size, tridiagonal_form=True)

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
    """Testing tridiagonal histogram of WQE
    """
    p_size, n_size = 50, 100
    wqe1 = WishartEnsemble(beta=4, p=p_size, n=n_size, tridiagonal_form=False)
    wqe2 = WishartEnsemble(beta=4, p=p_size, n=n_size, tridiagonal_form=True)

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
