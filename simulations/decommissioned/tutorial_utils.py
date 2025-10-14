
# general libraries
import time
import math
import numpy as np
import matplotlib.pyplot as plt
# my own modules
import sys
sys.path.append('../skrmt')

from covariance import (
    sample_estimator,
    fsopt_estimator,
    loss_mv,
    prial_mv,
)


####################################################################################
# COVARIANCE SIMS

def sample_rand_orthogonal_mtx(n):
    # n by n random complex matrix
    X = np.random.randn(n,n)
    # orthonormalizing matrix using QR algorithm
    Q,_ = np.linalg.qr(X)
    return Q


def sample_diagEig_mtx(p, values, prop):
    n_per_val = []

    for i in range(len(values[:-1])):
        n_per_val.append(math.floor(p * prop[i]))
    n_per_val.append(p - np.sum(n_per_val))

    eigvals = []
    for (i,nval) in enumerate(n_per_val):
        eigvals += [values[i]]*nval

    # shuffling eigenvalues
    np.random.shuffle(eigvals)
    # building diagonal matrix
    M = np.diag(eigvals)
    return M


def sample_pop_cov(p, values, prop, diag=False):
    if diag:
        return sample_diagEig_mtx(p, values, prop)
    else:
        O = sample_rand_orthogonal_mtx(p)
        M = sample_diagEig_mtx(p, values, prop)
        # O M O.T preserves original eigenvalues (O is an orthogonal rotation)
        return np.matmul(np.matmul(O, M), O.T) # sampling \Sigma


def sample_dataset(p, n, Sigma):
    X = np.random.multivariate_normal(np.random.randn(p), Sigma, size=n)
    return X


def cov_estim_simulation(p, n, estimators, eigvals, props, nreps=100):
    # adviced to check prial_mv formula to understand the code below
    Sn_idx = 0
    Sstar_idx = 1
    Sigma_tilde_idx = 2
    # generating population covariance matrix
    Sigma = sample_pop_cov(p, eigvals, props)

    # matrices/arrays of results
    # +2 because sample and FSOptimal estimators are always considered
    LOSSES = np.zeros((len(estimators)+2, 3))
    PRIALS = np.zeros(len(estimators)+2)
    TIMES = np.zeros((len(estimators)+2))

    for (idx, estimator) in enumerate(estimators):
        t1 = time.time()
        for i in range(nreps):
            # sampling random dataset from fixed population covariance matrix
            X = sample_dataset(p=p, n=n, Sigma=Sigma)
            # estimating sample cov
            Sample = sample_estimator(X)
            # estimating S_star
            S_star = fsopt_estimator(X, Sigma)
            # estimating population covariance matrix using current estimator
            Sigma_tilde = estimator(X)
            # calculating losses
            loss_Sn = loss_mv(sigma_tilde=Sample, sigma=Sigma)
            loss_Sstar = loss_mv(sigma_tilde=S_star, sigma=Sigma)
            loss_Sigma_tilde = loss_mv(sigma_tilde=Sigma_tilde, sigma=Sigma)
            LOSSES[idx][Sn_idx] += loss_Sn
            LOSSES[idx][Sstar_idx] += loss_Sstar
            LOSSES[idx][Sigma_tilde_idx] += loss_Sigma_tilde
        t2 = time.time()
        TIMES[idx] = (t2-t1)*1000/nreps # time needed in ms (meaned by number of repetitions)
        LOSSES[idx] /= p
        PRIALS[idx] = prial_mv(exp_sample=LOSSES[idx][Sn_idx],
                               exp_sigma_tilde=LOSSES[idx][Sigma_tilde_idx],
                               exp_fsopt=LOSSES[idx][Sstar_idx])
        
    # Sample estimator
    t1 = time.time()
    for i in range(nreps):
        # sampling random dataset from fixed population covariance matrix
        X = sample_dataset(p=p, n=n, Sigma=Sigma)
        # estimating sample cov
        Sample = sample_estimator(X)
        # estimating S_star
        S_star = fsopt_estimator(X, Sigma)
        # estimating population covariance matrix using sample estimator
        Sigma_tilde = sample_estimator(X)
        # calculating losses
        loss_Sn = loss_mv(sigma_tilde=Sample, sigma=Sigma)
        loss_Sstar = loss_mv(sigma_tilde=S_star, sigma=Sigma)
        loss_Sigma_tilde = loss_mv(sigma_tilde=Sigma_tilde, sigma=Sigma)
        LOSSES[-2][Sn_idx] += loss_Sn
        LOSSES[-2][Sstar_idx] += loss_Sstar
        LOSSES[-2][Sigma_tilde_idx] += loss_Sigma_tilde
    t2 = time.time()
    TIMES[-2] = (t2-t1)*1000/nreps # time needed in ms (meaned by number of repetitions)
    LOSSES[-2] /= p
    PRIALS[-2] = prial_mv(exp_sample=LOSSES[-2][Sn_idx],
                          exp_sigma_tilde=LOSSES[-2][Sigma_tilde_idx],
                          exp_fsopt=LOSSES[-2][Sstar_idx])
    
    # FSOpt estimator
    t1 = time.time()
    for i in range(nreps):
        # sampling random dataset from fixed population covariance matrix
        X = sample_dataset(p=p, n=n, Sigma=Sigma)
        # estimating sample cov
        Sample = sample_estimator(X)
        # estimating S_star
        S_star = fsopt_estimator(X, Sigma)
        # estimating population covariance matrix using current estimator
        Sigma_tilde = fsopt_estimator(X, Sigma)
        # calculating losses
        loss_Sn = loss_mv(sigma_tilde=Sample, sigma=Sigma)
        loss_Sstar = loss_mv(sigma_tilde=S_star, sigma=Sigma)
        loss_Sigma_tilde = loss_mv(sigma_tilde=Sigma_tilde, sigma=Sigma)
        LOSSES[-1][Sn_idx] += loss_Sn
        LOSSES[-1][Sstar_idx] += loss_Sstar
        LOSSES[-1][Sigma_tilde_idx] += loss_Sigma_tilde
    t2 = time.time()
    TIMES[-1] = (t2-t1)*1000/nreps # time needed in ms (meaned by number of repetitions)
    LOSSES[-1] /= p
    PRIALS[-1] = prial_mv(exp_sample=LOSSES[-1][Sn_idx],
                          exp_sigma_tilde=LOSSES[-1][Sigma_tilde_idx],
                          exp_fsopt=LOSSES[-1][Sstar_idx])
        
    return LOSSES, PRIALS, TIMES



def plot_cov_estimator_sim(estimators, labels, eigvals, props, P_list,
                           N=None, ratio=3, nreps=None, metric='prial'):

    # +2 because Sample and FSOptimal estimators are always considered
    MEASURES = np.zeros((len(P_list), len(estimators)+2))
    labels += ['Sample', 'FSOpt']

    ratios = []

    for (idx, p) in enumerate(P_list):
        if N is None:
            n = ratio*p
        else:
            n = N
            ratios.append(p/n)
        if nreps is None:
            nreps = int(max(100, min(1000, 10000/p)))

        losses, prials, times = cov_estim_simulation(p, n, estimators, eigvals, props, nreps=nreps)
        if metric == 'prial':
            MEASURES[idx] = prials
        elif metric == 'loss':
            MEASURES[idx] = losses
        elif metric == 'time':
            MEASURES[idx] = times

    if N is None:
        lines = plt.plot(P_list, MEASURES, '-D')
        plt.xlabel('Matrix dimension p')
    else:
        lines = plt.plot(ratios, MEASURES, '-D')
        plt.xlabel('Ratio p/n')
    plt.legend(lines, labels)

    if metric == 'prial':
        plt.title('Evolution of PRIAL (reps='+str(nreps)+')')
        plt.ylabel('PRIAL')
    elif metric == 'loss':
        plt.title('Evolution of Loss (reps='+str(nreps)+')')
        plt.ylabel('Loss')
    elif metric == 'time':
        plt.title('Duration study on average (reps='+str(nreps)+')')
        plt.ylabel('time (ms)')
