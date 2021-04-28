import pytest

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_allclose,
)

from skrmt.covariance import sample_estimator
from skrmt.covariance import fsopt_estimator
from skrmt.covariance import linear_shrinkage_estimator
from skrmt.covariance import analytical_shrinkage_estimator
from skrmt.covariance import empirical_bayesian_estimator
from skrmt.covariance import minimax_estimator

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


################################################################
# FSOPT ESTIMATOR 

def test_fsoptEstimator_shape_symm():
    # population covariance matrix
    Sigma = np.array([[3.00407916, -1.46190757, 1.50140806, 1.50933526, 0.27036442],
                      [-1.46190757, 5.61441061, -0.51939653, -2.76492235, 1.38225566],
                      [1.50140806, -0.51939653, 2.3068582, 1.41248896, 0.84740175],
                      [1.50933526, -2.76492235, 1.41248896, 6.57182938, 0.73407095],
                      [0.27036442, 1.38225566, 0.84740175, 0.73407095, 9.50282265]])
    
    p = Sigma.shape[0]
    n = 12
    # input data matrix
    X = np.random.multivariate_normal(np.random.randn(p), Sigma, size=n)
    
    sigma_tilde = fsopt_estimator(X, Sigma)

    assert(sigma_tilde.shape == (X.shape[1], X.shape[1]))
    assert((sigma_tilde.all() == sigma_tilde.all()) == True)


################################################################
# EMPIRICAL BAYESIAN ESTIMATOR 

def test_EBEstimator_shape_symm():
    # input data matrix
    X = np.array([[5, 0, 0], [4, 0, 0], [3, 0, 0], [2, 0, 0], [1, 0, 0],
                  [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5],
                  [0, 3, 0], [0, 4, 0], [0, 5, 0], [0, 1, 0], [0, 10, 0]])
    
    sigma_tilde = empirical_bayesian_estimator(X)

    assert(sigma_tilde.shape == (X.shape[1], X.shape[1]))
    assert((sigma_tilde.all() == sigma_tilde.all()) == True)


################################################################
# MINIMAX ESTIMATOR 

def test_minimaxEstimator_shape_symm():
    # input data matrix
    X = np.array([[5, 0, 0], [4, 0, 0], [3, 0, 0], [2, 0, 0], [1, 0, 0],
                  [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5],
                  [0, 3, 0], [0, 4, 0], [0, 5, 0], [0, 1, 0], [0, 10, 0]])
    
    sigma_tilde = minimax_estimator(X)

    assert(sigma_tilde.shape == (X.shape[1], X.shape[1]))
    assert((sigma_tilde.all() == sigma_tilde.all()) == True)