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


def test_coe_eigvals():
    '''Testing all eigenvalues of a COE matrix have module 1
    '''
    n_size = 5
    coe = CircularEnsemble(beta=1, n=n_size)

    vals = coe.eigvals()

    mods = np.absolute(vals)
    assert_almost_equal(mods, 1.0, decimal=12)


def test_beta1_joint_eigval_pdf():
    '''Testing joint eigenvalue pdf
    '''
    n_size = 3
    coe = CircularEnsemble(beta=1, n=n_size)

    coe.matrix = np.zeros((n_size,n_size))
    assert coe.joint_eigval_pdf() == 0.0

    coe.matrix = np.eye(n_size)
    assert coe.joint_eigval_pdf() == 0.0

    coe.matrix = 10*np.eye(n_size)
    assert coe.joint_eigval_pdf() == 0.0


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


def test_cue_eigvals():
    '''Testing all eigenvalues of a CUE matrix have module 1
    '''
    n_size = 5
    cue = CircularEnsemble(beta=2, n=n_size)

    vals = cue.eigvals()

    mods = np.absolute(vals)
    assert_almost_equal(mods, 1.0, decimal=12)


def test_beta2_joint_eigval_pdf():
    '''Testing joint eigenvalue pdf
    '''
    n_size = 3
    cue = CircularEnsemble(beta=2, n=n_size)

    cue.matrix = np.zeros((n_size,n_size))
    assert cue.joint_eigval_pdf() == 0.0

    cue.matrix = np.eye(n_size)
    assert cue.joint_eigval_pdf() == 0.0

    cue.matrix = 10*np.eye(n_size)
    assert cue.joint_eigval_pdf() == 0.0


##########################################
### Circular Symplectic Ensemble = CSE

def test_cse_init():
    '''Testing CSE init
    '''
    n_size = 2
    np.random.seed(1)
    cse = CircularEnsemble(beta=4, n=n_size)

    assert cse.matrix.shape == (2*n_size,2*n_size)

    mtx_sol = np.array(
        [
            [ 6.45393580e-01-4.76777901e-01j, -7.32722934e-18-2.77555756e-17j,
              3.28888045e-01+2.22882248e-01j,  2.17002314e-01-3.88865161e-01j],
            [ 1.20088950e-17-5.55111512e-17j,  6.45393580e-01-4.76777901e-01j,
              1.77812336e-01+4.08275508e-01j, -3.49164384e-01+1.89547027e-01j],
            [-3.49164384e-01+1.89547027e-01j, -2.17002314e-01+3.88865161e-01j,
             5.95514436e-01+5.37784898e-01j,  6.02588126e-18+0.00000000e+00j],
            [-1.77812336e-01-4.08275508e-01j,  3.28888045e-01+2.22882248e-01j,
             1.08100510e-17+0.00000000e+00j,  5.95514436e-01+5.37784898e-01j]
        ]
    )

    assert_almost_equal(cse.matrix, np.array(mtx_sol), decimal=7)


def test_cse_eigvals():
    '''Testing all eigenvalues of a CSE matrix have module 1
    '''
    n_size = 5
    cse = CircularEnsemble(beta=4, n=n_size)

    vals = cse.eigvals()

    mods = np.absolute(vals)
    assert_almost_equal(mods, 1.0, decimal=12)


def test_beta4_joint_eigval_pdf():
    '''Testing joint eigenvalue pdf
    '''
    n_size = 3
    cse = CircularEnsemble(beta=4, n=n_size)

    cse.matrix = np.zeros((n_size,n_size))
    assert cse.joint_eigval_pdf() == 0.0

    cse.matrix = np.eye(n_size)
    assert cse.joint_eigval_pdf() == 0.0

    cse.matrix = 10*np.eye(n_size)
    assert cse.joint_eigval_pdf() == 0.0
