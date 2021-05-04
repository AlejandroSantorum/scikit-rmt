'''Tridiagonalization Test module

Testing tridiagonalization using Householder reduction
'''

import numpy as np
from numpy.testing import (
    assert_almost_equal,
)

from skrmt.ensemble import householder_reduction


def test_householder_reduction():
    '''Testing householder reduction method for tridiagonalization
    '''
    mtx = np.asarray([[4,2,-2,1], [2,3,2,1], [-2,2,1,0], [1,1,0,2]])
    sol = np.asarray([[4, -3, 0, 0], [-3, 2/3, 5/3, 0], [0, 5/3, 3, 4/3], [0,0,4/3, 7/3]])

    reduction = householder_reduction(mtx)

    assert_almost_equal(reduction, sol, decimal=7)


def test_householder_reduction_eigvals():
    '''Testing householder reduction method keeps eigenvalues after tridiagonalization
    '''
    mtx = np.asarray([[4,2,-2,1], [2,3,2,1], [-2,2,1,0], [1,1,0,2]])
    sol = np.asarray([[4, -3, 0, 0], [-3, 2/3, 5/3, 0], [0, 5/3, 3, 4/3], [0,0,4/3, 7/3]])

    mtx_eigvals = np.linalg.eigvals(mtx)
    sol_eigvals = np.linalg.eigvals(sol)

    reduction = householder_reduction(mtx)
    reduction_eigvals = np.linalg.eigvals(reduction)

    assert_almost_equal(sol_eigvals, reduction_eigvals, decimal=7)
    assert_almost_equal(mtx_eigvals, reduction_eigvals, decimal=7)
