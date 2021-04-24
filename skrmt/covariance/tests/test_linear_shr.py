import pytest

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
)

from skrmt.covariance import linear_shrinkage_estimator


def test_estimate_1():
    # input data matrix
    X = np.array([[2, 0, 0], [2, 0, 0], [2, 0, 0], [1, 0, 0], [1, 0, 0],
                  [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
                  [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 2]])

    # solution using official matlab code (Ledoit and Wolf)
    sol = np.array([[0.571596585433017, -0.119074762494837, -0.142889714993804],
                    [-0.119074762494837, 0.285817155445408, -0.089306071871128],
                    [-0.142889714993804, -0.089306071871128, 0.387030703566020]])

    # linsh = LinearShrinkage()
    # sigma_tilde = linsh.estimate(X)
    sigma_tilde = linear_shrinkage_estimator(X)

    assert_almost_equal(sigma_tilde, sol, decimal=10)


def test_estimate_2():
    # input data matrix
    X = np.array([[5, 0, 0], [4, 0, 0], [3, 0, 0], [2, 0, 0], [1, 0, 0],
                  [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5],
                  [0, 3, 0], [0, 4, 0], [0, 5, 0], [0, 1, 0], [0, 10, 0]])

    # solution using official matlab code (Ledoit and Wolf)
    sol = np.array([[4.157119427352015, -0.175394418448355, -0.114387664205449],
                    [-0.175394418448355, 4.734650034184860, -0.175394418448355],
                    [-0.114387664205449, -0.175394418448355, 4.157119427352015]])

    # linsh = LinearShrinkage()
    # sigma_tilde = linsh.estimate(X)
    sigma_tilde = linear_shrinkage_estimator(X)

    assert_almost_equal(sigma_tilde, sol, decimal=10)