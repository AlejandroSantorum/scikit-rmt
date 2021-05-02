'''Metrics Test module

Testing Metrics module
'''

import numpy as np
import pytest

from skrmt.covariance import sample_estimator
from skrmt.covariance import fsopt_estimator
from skrmt.covariance import loss_frobenius, loss_mv, prial_mv



def test_prial_sample():
    '''Testing prial evaluated in sample covariance matrix
    '''
    # population covariance matrix
    sigma = np.array([[3.00407916, -1.46190757, 1.50140806, 1.50933526, 0.27036442],
                      [-1.46190757, 5.61441061, -0.51939653, -2.76492235, 1.38225566],
                      [1.50140806, -0.51939653, 2.3068582, 1.41248896, 0.84740175],
                      [1.50933526, -2.76492235, 1.41248896, 6.57182938, 0.73407095],
                      [0.27036442, 1.38225566, 0.84740175, 0.73407095, 9.50282265]])

    p_size, n_size = sigma.shape[0], 3*sigma.shape[0]
    # input data matrix
    mtx = np.random.multivariate_normal(np.random.randn(p_size), sigma, size=n_size)

    sigma_sample = sample_estimator(mtx)
    sigma_fsopt = fsopt_estimator(mtx, sigma)

    exp_sample = loss_mv(sigma_tilde=sigma_sample, sigma=sigma)
    exp_sigma_tilde = loss_mv(sigma_tilde=sigma_sample, sigma=sigma)
    exp_fsopt = loss_mv(sigma_tilde=sigma_fsopt, sigma=sigma)

    prial = prial_mv(exp_sample=exp_sample, exp_sigma_tilde=exp_sigma_tilde, exp_fsopt=exp_fsopt)

    assert prial == 0.0


def test_prial_fsopt():
    '''Testing prial evaluated in finite-sample optimal covariance matrix
    '''
    # population covariance matrix
    sigma = np.array([[3.00407916, -1.46190757, 1.50140806, 1.50933526, 0.27036442],
                      [-1.46190757, 5.61441061, -0.51939653, -2.76492235, 1.38225566],
                      [1.50140806, -0.51939653, 2.3068582, 1.41248896, 0.84740175],
                      [1.50933526, -2.76492235, 1.41248896, 6.57182938, 0.73407095],
                      [0.27036442, 1.38225566, 0.84740175, 0.73407095, 9.50282265]])

    p_size, n_size = sigma.shape[0], 3*sigma.shape[0]
    # input data matrix
    mtx = np.random.multivariate_normal(np.random.randn(p_size), sigma, size=n_size)

    sigma_sample = sample_estimator(mtx)
    sigma_fsopt = fsopt_estimator(mtx, sigma)

    exp_sample = loss_mv(sigma_tilde=sigma_sample, sigma=sigma)
    exp_sigma_tilde = loss_mv(sigma_tilde=sigma_fsopt, sigma=sigma)
    exp_fsopt = loss_mv(sigma_tilde=sigma_fsopt, sigma=sigma)

    prial = prial_mv(exp_sample=exp_sample, exp_sigma_tilde=exp_sigma_tilde, exp_fsopt=exp_fsopt)

    assert prial == 1.0


def test_loss_frobenius():
    '''Testing Frobenius loss is zero when applied at the same matrix
    '''
    # population covariance matrix
    sigma = np.array([[3.00407916, -1.46190757, 1.50140806, 1.50933526, 0.27036442],
                      [-1.46190757, 5.61441061, -0.51939653, -2.76492235, 1.38225566],
                      [1.50140806, -0.51939653, 2.3068582, 1.41248896, 0.84740175],
                      [1.50933526, -2.76492235, 1.41248896, 6.57182938, 0.73407095],
                      [0.27036442, 1.38225566, 0.84740175, 0.73407095, 9.50282265]])

    assert loss_frobenius(sigma, sigma) == 0


def test_exceptions():
    '''Testing loss function raises exception when a non-invertible matrix is introduced
    '''
    mtx1 = np.zeros((5, 5))
    mtx2 = np.array([[3.00407916, -1.46190757, 1.50140806, 1.50933526, 0.27036442],
                     [-1.46190757, 5.61441061, -0.51939653, -2.76492235, 1.38225566],
                     [1.50140806, -0.51939653, 2.3068582, 1.41248896, 0.84740175],
                     [1.50933526, -2.76492235, 1.41248896, 6.57182938, 0.73407095],
                     [0.27036442, 1.38225566, 0.84740175, 0.73407095, 9.50282265]])

    with pytest.raises(ValueError):
        loss_mv(mtx1, mtx2)

    with pytest.raises(ValueError):
        loss_mv(mtx2, mtx1)
