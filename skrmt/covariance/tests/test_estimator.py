import pytest

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_allclose,
)

from skrmt.covariance import sample_estimator


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