import numpy as np

from .estimator import SampleEstimator, FSOptEstimator


def loss_mv(Sigma_tilde, Sigma):
    p = len(Sigma)

    try:
        Sigma_tilde_inv = np.linalg.inv(Sigma_tilde)
    except np.linalg.LinAlgError:
        raise ValueError("Unable to invert estimated covariance matrix")
    
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        raise ValueError("Unable to invert population covariance matrix")
    

    M = np.matmul(np.matmul(Sigma_tilde_inv, Sigma), Sigma_tilde_inv)
    loss = np.trace(M)/p/(np.trace(Sigma_tilde_inv)/p)**2 - p/np.trace(Sigma_inv)

    return loss


def loss_frobenius(Sigma_tilde, Sigma):
    p = len(Sigma)
    loss = np.trace(Sigma_tilde - Sigma)**2/p

    return loss


# Percentage relative improvement in average loss
def PRIAL_mv(E_Sn, E_Sigma_tilde, E_Sstar):
    return (E_Sn - E_Sigma_tilde)/(E_Sn - E_Sstar)

# Percentage relative improvement in average loss
'''
def PRIAL_mv(Sigma_tilde, Sigma, X):
    sample_est = SampleEstimator()
    S = sample_est.estimate(X)

    fsopt = FSOptEstimator(Sigma)
    S_star = fsopt.estimate(X)

    loss_S_Sigma = loss_mv(S, Sigma)
    prial = (loss_S_Sigma - loss_mv(Sigma_tilde, Sigma))/(loss_S_Sigma - loss_mv(S_star, Sigma))

    return prial
'''


