'''Circular Ensemble Test module

Testing CircularEnsemble module
'''

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
)

from skrmt.ensemble import CircularEnsemble


##########################################
### Circular Orthogonal Ensemble = COE

def test_coe_init():
    '''Testing COE init
    '''
    n_size = 3

    np.random.seed(1)
    coe = CircularEnsemble(beta=1, n=n_size)

    assert coe.matrix.shape == (n_size,n_size)

    mtx_sol = [[0.66482467-0.48190422j, -0.4274577+0.09781321j, -0.05592612-0.36105573j],
               [-0.4274577+0.09781321j, -0.80810768+0.1559807j, 0.36012685+0.02555646j],
               [-0.05592612-0.36105573j, 0.36012685+0.02555646j, 0.8408391-0.17075173j]]

    assert_almost_equal(coe.matrix, np.array(mtx_sol), decimal=7)


def test_coe_symmetric():
    '''Testing that COE matrix is symmetric
    '''
    n_size = 5
    coe = CircularEnsemble(beta=1, n=n_size)

    mtx = coe.matrix
    assert (mtx.transpose() == mtx).all()


def test_coe_set_size():
    '''Testing setter to change matrix sizes of COE
    '''
    n1_size = 3
    n2_size = 5

    coe = CircularEnsemble(beta=1, n=n1_size)
    assert coe.n == n1_size
    assert coe.matrix.shape == (n1_size,n1_size)

    coe.set_size(n2_size, resample_mtx=False)
    assert coe.n == n2_size
    assert coe.matrix.shape == (n1_size,n1_size)

    coe.set_size(n2_size, resample_mtx=True)
    assert coe.n == n2_size
    assert coe.matrix.shape == (n2_size,n2_size)


def test_coe_eigvals():
    '''Testing all eigenvalues of a COE matrix are real
    '''
    n_size = 5
    coe = CircularEnsemble(beta=1, n=n_size)

    vals = coe.eigvals()

    assert_array_equal(vals.imag, 0.0)



def test_beta1_eigval_pdf():
    '''Testing joint eigenvalue pdf
    '''
    n_size = 3
    coe = CircularEnsemble(beta=1, n=n_size)

    coe.matrix = np.zeros((n_size,n_size))
    assert coe.eigval_pdf() == 0.0

    coe.matrix = np.eye(n_size)
    assert coe.eigval_pdf() == 0.0

    coe.matrix = 10*np.eye(n_size)
    assert coe.eigval_pdf() == 0.0


##########################################
### Circular Unitary Ensemble = CUE

def test_cue_init():
    '''Testing CUE init
    '''
    n_size = 3

    np.random.seed(1)
    cue = CircularEnsemble(beta=2, n=n_size)

    assert cue.matrix.shape == (n_size,n_size)

    mtx_sol = [[-0.56689951+0.08703072j, 0.00821467-0.66547876j, -0.38691909+0.28002633j],
               [0.37446802+0.1125242j, -0.23983754+0.18549333j, -0.86852536-0.02908408j],
               [-0.60894251+0.38386407j, 0.1958485 +0.65328713j, -0.12797213+0.01788349j]]

    assert_almost_equal(cue.matrix, np.array(mtx_sol), decimal=7)


def test_cue_set_size():
    '''Testing setter to change matrix sizes of CUE
    '''
    n1_size = 5
    n2_size = 10

    cue = CircularEnsemble(beta=2, n=n1_size)
    assert cue.n == n1_size
    assert cue.matrix.shape == (n1_size,n1_size)

    cue.set_size(n2_size, resample_mtx=False)
    assert cue.n == n2_size
    assert cue.matrix.shape == (n1_size,n1_size)

    cue.set_size(n2_size, resample_mtx=True)
    assert cue.n == n2_size
    assert cue.matrix.shape == (n2_size,n2_size)


def test_cue_eigvals():
    '''Testing all eigenvalues of a CUE matrix have module 1
    '''
    n_size = 5
    cue = CircularEnsemble(beta=2, n=n_size)

    vals = cue.eigvals()

    mods = np.absolute(vals)
    assert_almost_equal(mods, 1.0, decimal=12)


##########################################
### Circular Symplectic Ensemble = CSE

def test_cse_init():
    '''Testing CSE init
    '''
    n_size = 2
    np.random.seed(1)
    cse = CircularEnsemble(beta=4, n=n_size)

    assert cse.matrix.shape == (2*n_size,2*n_size)

    mtx_sol = [[4.43078888e-01-8.29771592e-01j, -6.00429653e-17+1.43650461e-17j,\
                9.66935997e-02+7.40605433e-01j, 3.43966100e-01-3.01198149e-02j],
               [4.56225923e-01+3.23872465e-02j, 3.46385288e-01-1.57037703e+00j, \
                4.56225923e-01+3.23872465e-02j, -1.22164739e+00-7.84199458e-01j],
               [-9.56297276e-01+7.17094554e-01j, -3.43966100e-01+3.01198149e-02j, \
                6.11735400e-01-6.90830132e-02j, -3.43966100e-01+3.01198149e-02j],
               [-4.56225923e-01-3.23872465e-02j, 9.66935997e-02+7.40605433e-01j, \
                1.94983679e-17+3.89651615e-18j, 7.08429000e-01+6.71522420e-01j]]

    assert_almost_equal(cse.matrix, np.array(mtx_sol), decimal=7)


def test_cse_set_size():
    '''Testing setter to change matrix sizes of CSE
    '''
    n1_size = 4
    n2_size = 9

    cse = CircularEnsemble(beta=4, n=n1_size)
    assert cse.n == n1_size
    assert cse.matrix.shape == (2*n1_size,2*n1_size)

    cse.set_size(n2_size, resample_mtx=False)
    assert cse.n == n2_size
    assert cse.matrix.shape == (2*n1_size,2*n1_size)

    cse.set_size(n2_size, resample_mtx=True)
    assert cse.n == n2_size
    assert cse.matrix.shape == (2*n2_size,2*n2_size)
