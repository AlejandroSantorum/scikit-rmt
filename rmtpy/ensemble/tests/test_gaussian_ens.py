import pytest

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_allclose,
)

from rmtpy.ensemble import GOE, GUE, GSE


##########################################
### Gaussian Orthogonal Ensemble = GOE

def test_goe_init():
    N = 3

    np.random.seed(1)
    goe = GOE(n=N)

    assert(goe.matrix.shape == (N,N))

    assert_almost_equal(goe.matrix, np.array([[ 1.62434536, -0.84236252, 0.60832001],
                                              [-0.84236252, 0.86540763, -1.5313728 ],
                                              [ 0.60832001, -1.5313728, 0.3190391 ]]),
                        decimal=4)


def test_goe_set_size():
    N1 = 3
    N2 = 5

    goe = GOE(n=N1)
    assert(goe.n == N1)
    assert(goe.matrix.shape == (N1,N1))

    goe.set_size(N2, resample_mtx=False)
    assert(goe.n == N2)
    assert(goe.matrix.shape == (N1,N1))

    goe.set_size(N2, resample_mtx=True)
    assert(goe.n == N2)
    assert(goe.matrix.shape == (N2,N2))


##########################################
### Gaussian Unitary Ensemble = GUE

def test_gue_init():
    N = 3

    np.random.seed(1)
    gue = GUE(n=N)

    assert(gue.matrix.shape == (N,N))

    assert_almost_equal(gue.matrix, np.array([[1.62434536-0.24937038j, -0.84236252+0.56984537j, 0.60832001-1.58001599j],
                                             [-0.84236252+0.56984537j, 0.86540763-0.38405435j, -1.5313728 +0.48067062j],
                                             [0.60832001-1.58001599j, -1.5313728 +0.48067062j, 0.3190391 -0.87785842j]]),
                        decimal=4)


def test_gue_set_size():
    N1 = 5
    N2 = 8

    gue = GUE(n=N1)
    assert(gue.n == N1)
    assert(gue.matrix.shape == (N1,N1))

    gue.set_size(N2, resample_mtx=False)
    assert(gue.n == N2)
    assert(gue.matrix.shape == (N1,N1))

    gue.set_size(N2, resample_mtx=True)
    assert(gue.n == N2)
    assert(gue.matrix.shape == (N2,N2))


##########################################
### Gaussian Symplectic Ensemble = GSE

def test_gse_init():
    N = 2 
    np.random.seed(1)
    gse = GSE(n=N)

    assert(gse.matrix.shape == (2*N,2*N))

    assert_almost_equal(gse.matrix, np.array([[1.62434536+0.86540763j, -0.56996408-0.27836347j, 0.-0.3224172j, -0.85573916+0.37485754j],
                                             [-0.56996408-0.27836347j, -1.07296862-0.7612069j, 0.85573916+0.37485754j, 0.-1.09989127j],
                                             [0.-0.3224172j, 0.85573916+0.37485754j, 1.62434536-0.86540763j, -0.56996408+0.27836347j],
                                             [-0.85573916+0.37485754j, 0.-1.09989127j, -0.56996408+0.27836347j, -1.07296862+0.7612069j ]]),
                        decimal=4)


def test_gse_set_size():
    N1 = 4
    N2 = 9

    gse = GSE(n=N1)
    assert(gse.n == N1)
    assert(gse.matrix.shape == (2*N1,2*N1))

    gse.set_size(N2, resample_mtx=False)
    assert(gse.n == N2)
    assert(gse.matrix.shape == (2*N1,2*N1))

    gse.set_size(N2, resample_mtx=True)
    assert(gse.n == N2)
    assert(gse.matrix.shape == (2*N2,2*N2))