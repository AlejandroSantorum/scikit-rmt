import pytest

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_allclose,
)

from skrmt.covariance import sample_estimator
from skrmt.covariance import linear_shrinkage_estimator
from skrmt.covariance import analytical_shrinkage_estimator


################################################################
# SAMPLE ESTIMATOR 

def test_sampleEstimator():
    # input data matrix
    X = np.array([[2, 0, 0], [2, 0, 0], [2, 0, 0], [1, 0, 0], [1, 0, 0],
                  [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
                  [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 2]])

    # sample_est = SampleEstimator()
    # sigma_sample = sample_est.estimate(X)
    sigma_sample = sample_estimator(X)

    X = X - X.mean(axis=0)
    numpy_sol = np.cov(X, rowvar=False)

    assert_almost_equal(sigma_sample, numpy_sol, decimal=10)



################################################################
# LINEAR ESTIMATOR 

def test_linearEstimator1():
    # input data matrix
    X = np.array([[2, 0, 0], [2, 0, 0], [2, 0, 0], [1, 0, 0], [1, 0, 0],
                  [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
                  [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 2]])

    # solution using official matlab code (Ledoit and Wolf)
    sol = np.array([[0.571596585433017, -0.119074762494837, -0.142889714993804],
                    [-0.119074762494837, 0.285817155445408, -0.089306071871128],
                    [-0.142889714993804, -0.089306071871128, 0.387030703566020]])

    sigma_tilde = linear_shrinkage_estimator(X)

    assert_almost_equal(sigma_tilde, sol, decimal=10)


def test_linearEstimator2():
    # input data matrix
    X = np.array([[5, 0, 0], [4, 0, 0], [3, 0, 0], [2, 0, 0], [1, 0, 0],
                  [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5],
                  [0, 3, 0], [0, 4, 0], [0, 5, 0], [0, 1, 0], [0, 10, 0]])

    # solution using official matlab code (Ledoit and Wolf)
    sol = np.array([[4.157119427352015, -0.175394418448355, -0.114387664205449],
                    [-0.175394418448355, 4.734650034184860, -0.175394418448355],
                    [-0.114387664205449, -0.175394418448355, 4.157119427352015]])

    sigma_tilde = linear_shrinkage_estimator(X)

    assert_almost_equal(sigma_tilde, sol, decimal=10)


################################################################
# ANALYTICAL ESTIMATOR 

def test_analyticalEstimator1():
    # input data matrix
    X = np.array([[2, 0, 0], [2, 0, 0], [2, 0, 0], [1, 0, 0], [1, 0, 0],
              [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
              [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 2]])

    # solution using official matlab code (Ledoit and Wolf)
    sol = np.array([[0.551752555904758, -0.176171945854672, -0.127673277267705],
                    [-0.176171945854672, 0.317339265808824, -0.236795281588382],
                    [-0.127673277267705, -0.236795281588382, 0.474063174598454]])

    sigma_tilde = analytical_shrinkage_estimator(X)

    assert_almost_equal(sigma_tilde, sol, decimal=7)


def test_analyticalEstimator2():
    # input data matrix
    X = np.array([[5, 0, 0], [4, 0, 0], [3, 0, 0], [2, 0, 0], [1, 0, 0],
                  [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5],
                  [0, 3, 0], [0, 4, 0], [0, 5, 0], [0, 1, 0], [0, 10, 0]])

    # solution using official matlab code (Ledoit and Wolf)
    sol = np.array([[3.84630121492766, -1.3927903545004, -1.42093779975966],
                    [-1.3927903545004, 7.91982043683479, -1.3927903545004],
                    [-1.42093779975966, -1.3927903545004, 3.84630121492766]])

    sigma_tilde = analytical_shrinkage_estimator(X)

    assert_almost_equal(sigma_tilde, sol, decimal=7)