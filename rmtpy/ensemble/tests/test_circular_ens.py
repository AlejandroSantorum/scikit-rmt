import pytest

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_allclose,
)

from rmtpy.ensemble import COE, CUE, CSE


##########################################
### Circular Orthogonal Ensemble = COE

def test_coe_init():
    N = 3

    np.random.seed(1)
    coe = COE(n=N)

    assert(coe.matrix.shape == (N,N))

    assert_almost_equal(coe.matrix, np.array([[0.66482467-0.48190422j, -0.4274577+0.09781321j, -0.05592612-0.36105573j],
                                             [-0.4274577+0.09781321j, -0.80810768+0.1559807j, 0.36012685+0.02555646j],
                                             [-0.05592612-0.36105573j, 0.36012685+0.02555646j, 0.8408391-0.17075173j]]),
                        decimal=4)


def test_coe_symmetric():
    N = 5
    coe = COE(n=N)

    M = coe.matrix
    assert((M.transpose() == M).all() == True)
                

def test_coe_set_size():
    N1 = 3
    N2 = 5

    coe = COE(n=N1)
    assert(coe.n == N1)
    assert(coe.matrix.shape == (N1,N1))

    coe.set_size(N2, resample_mtx=False)
    assert(coe.n == N2)
    assert(coe.matrix.shape == (N1,N1))

    coe.set_size(N2, resample_mtx=True)
    assert(coe.n == N2)
    assert(coe.matrix.shape == (N2,N2))


##########################################
### Circular Unitary Ensemble = CUE

def test_cue_init():
    N = 3

    np.random.seed(1)
    cue = CUE(n=N)

    assert(cue.matrix.shape == (N,N))

    assert_almost_equal(cue.matrix, np.array([[-0.56689951+0.08703072j, 0.00821467-0.66547876j, -0.38691909+0.28002633j],
                                             [0.37446802+0.1125242j, -0.23983754+0.18549333j, -0.86852536-0.02908408j],
                                             [-0.60894251+0.38386407j, 0.1958485 +0.65328713j, -0.12797213+0.01788349j]]),
                        decimal=4)
                    

def test_cue_set_size():
    N1 = 5
    N2 = 10

    cue = CUE(n=N1)
    assert(cue.n == N1)
    assert(cue.matrix.shape == (N1,N1))

    cue.set_size(N2, resample_mtx=False)
    assert(cue.n == N2)
    assert(cue.matrix.shape == (N1,N1))

    cue.set_size(N2, resample_mtx=True)
    assert(cue.n == N2)
    assert(cue.matrix.shape == (N2,N2))


##########################################
### Circular Symplectic Ensemble = CSE

def test_cse_init():
    N = 2 
    np.random.seed(1)
    cse = CSE(n=N)

    assert(cse.matrix.shape == (2*N,2*N))

    assert_almost_equal(cse.matrix, np.array([[4.43078888e-01-8.29771592e-01j, -6.00429653e-17+1.43650461e-17j, 9.66935997e-02+7.40605433e-01j, 3.43966100e-01-3.01198149e-02j],
                                             [4.56225923e-01+3.23872465e-02j, 3.46385288e-01-1.57037703e+00j, 4.56225923e-01+3.23872465e-02j, -1.22164739e+00-7.84199458e-01j],
                                             [-9.56297276e-01+7.17094554e-01j, -3.43966100e-01+3.01198149e-02j, 6.11735400e-01-6.90830132e-02j, -3.43966100e-01+3.01198149e-02j],
                                             [-4.56225923e-01-3.23872465e-02j, 9.66935997e-02+7.40605433e-01j, 1.94983679e-17+3.89651615e-18j, 7.08429000e-01+6.71522420e-01j]]),
                        decimal=4)


def test_cse_set_size():
    N1 = 4
    N2 = 9

    cse = CSE(n=N1)
    assert(cse.n == N1)
    assert(cse.matrix.shape == (2*N1,2*N1))

    cse.set_size(N2, resample_mtx=False)
    assert(cse.n == N2)
    assert(cse.matrix.shape == (2*N1,2*N1))

    cse.set_size(N2, resample_mtx=True)
    assert(cse.n == N2)
    assert(cse.matrix.shape == (2*N2,2*N2))