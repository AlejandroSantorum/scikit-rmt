import pytest

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
)

from skrmt.covariance import analytical_shrinkage_estimator


def test_estimate_1():
    # input data matrix
    X = np.array([[2, 0, 0], [2, 0, 0], [2, 0, 0], [1, 0, 0], [1, 0, 0],
              [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
              [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 2]])

    # solution using official matlab code (Ledoit and Wolf)
    sol = np.array([[0.551752555904758, -0.176171945854672, -0.127673277267705],
                    [-0.176171945854672, 0.317339265808824, -0.236795281588382],
                    [-0.127673277267705, -0.236795281588382, 0.474063174598454]])

    # analsh = AnalyticalShrinkage()
    # sigma_tilde = analsh.estimate(X)
    sigma_tilde = analytical_shrinkage_estimator(X)

    assert_almost_equal(sigma_tilde, sol, decimal=7)


def test_estimate_2():
    # input data matrix
    X = np.array([[5, 0, 0], [4, 0, 0], [3, 0, 0], [2, 0, 0], [1, 0, 0],
                  [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5],
                  [0, 3, 0], [0, 4, 0], [0, 5, 0], [0, 1, 0], [0, 10, 0]])

    # solution using official matlab code (Ledoit and Wolf)
    sol = np.array([[3.84630121492766, -1.3927903545004, -1.42093779975966],
                    [-1.3927903545004, 7.91982043683479, -1.3927903545004],
                    [-1.42093779975966, -1.3927903545004, 3.84630121492766]])

    # analsh = AnalyticalShrinkage()
    # sigma_tilde = analsh.estimate(X)
    sigma_tilde = analytical_shrinkage_estimator(X)

    assert_almost_equal(sigma_tilde, sol, decimal=7)