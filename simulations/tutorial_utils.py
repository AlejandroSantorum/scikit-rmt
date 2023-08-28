
# general libraries
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
# my own modules
import sys
sys.path.append('../skrmt')

from ensemble import GaussianEnsemble, WishartEnsemble
from covariance import sample_estimator, fsopt_estimator
from covariance import loss_mv, prial_mv


####################################################################################
# TRIDIAGONAL SIMS

def _build_m_labels(M):
    labels = []
    for m in M:
        labels.append("m = "+str(m))
    return labels



def _plot_times(N_list, bins_list, times_naive, times_tridiag):
    # creating subplots
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(5)
    fig.set_figwidth(13)
    fig.subplots_adjust(hspace=.5)

    # labels for plots and nodes for interpolation
    labels = _build_m_labels(bins_list)
    nodes = np.linspace(N_list[0], N_list[-1], 1000)

    # Fitting line naive computation
    y_naive = times_naive[:,0]
    a, m, b = np.polyfit(N_list, y_naive, 2)
    y_smooth_naive = a*nodes**2 + m*nodes + b
    #spl = make_interp_spline(N_list, y_naive, k=2)
    #y_smooth_naive = spl(nodes)

    # Fitting line tridiagonal computational (smallest bin)
    y_tridiag_low = times_tridiag[:,0]
    m, b = np.polyfit(N_list, y_tridiag_low, 1)
    y_smooth_tridiag_low = m*nodes + b
    #spl = make_interp_spline(N_list, y_tridiag_low, k=1)
    #y_smooth_tridiag_low = spl(nodes)

    # Fitting line tridiagonal computational (largest bin)
    y_tridiag_up = times_tridiag[:,-1]
    m, b = np.polyfit(N_list, y_tridiag_up, 1)
    y_smooth_tridiag_up = m*nodes + b
    #spl = make_interp_spline(N_list, y_tridiag_up, k=1)
    #y_smooth_tridiag_up = spl(nodes)

    # Naive plot
    lines = axes[0].plot(N_list, times_naive)
    fit_line = axes[0].plot(nodes, y_smooth_naive, '--')
    legend1 = axes[0].legend(lines, labels, loc=0)
    legend2 = axes[0].legend(fit_line, ['fit'], loc=9)
    axes[0].add_artist(legend1)
    axes[0].add_artist(legend2)
    axes[0].set_title('Naive computation')
    axes[0].set_xlabel('n (matrix size)')
    axes[0].set_ylabel('Time (ms.)')

    # Tridiagonal plot
    lines = axes[1].plot(N_list, times_tridiag)
    fit_line1 = axes[1].plot(nodes, y_smooth_tridiag_low, '--')
    fit_line2 = axes[1].plot(nodes, y_smooth_tridiag_up, '--')
    legend1 = axes[1].legend(lines, labels, loc=0)
    legend2 = axes[1].legend(fit_line1, ['lower fit'], loc=4)
    legend3 = axes[1].legend(fit_line2, ['upper fit'], loc=9)
    axes[1].add_artist(legend1)
    axes[1].add_artist(legend2)
    axes[1].add_artist(legend3)
    axes[1].set_title('Tridiagonal Sturm computation')
    axes[1].set_xlabel('n (matrix size)')
    _ = axes[1].set_ylabel('Time (ms.)')

    plt.show()



def gaussian_tridiagonal_sim(N_list, bins_list, nreps=10):
    # time lists
    times_naive = np.zeros((len(N_list), len(bins_list)))
    times_tridiag = np.zeros((len(N_list), len(bins_list)))

    # default interval and norm const
    interval = (-2, 2)
    to_norm = False

    # simulating times
    for (i, n) in enumerate(N_list):
        for (j, m) in enumerate(bins_list):
            for _ in range(nreps):
                goe1 = GaussianEnsemble(beta=1, n=n, tridiagonal_form=False)
                t1 = time.time()
                eig_hist_nt, bins_nt = goe1.eigval_hist(bins=m, interval=interval, density=to_norm)
                t2 = time.time()
                times_naive[i][j] += (t2 - t1)*1000 # ms

                goe2 = GaussianEnsemble(beta=1, n=n, tridiagonal_form=True)
                t1 = time.time()
                eig_hist_nt, bins_nt = goe2.eigval_hist(bins=m, interval=interval, density=to_norm)
                t2 = time.time()
                times_tridiag[i][j] += (t2 - t1)*1000 # ms
        
            times_naive[i][j] /= nreps
            times_tridiag[i][j] /= nreps
    
    _plot_times(N_list, bins_list, times_naive, times_tridiag)



def wishart_tridiagonal_sim(N_list, bins_list, nreps=10):
    # time lists
    times_naive = np.zeros((len(N_list), len(bins_list)))
    times_tridiag = np.zeros((len(N_list), len(bins_list)))

    # default interval and norm const
    interval = (0, 4)
    to_norm = False

    # simulating times
    for (i, n) in enumerate(N_list):
        for (j, m) in enumerate(bins_list):
            for _ in range(nreps):
                wre1 = WishartEnsemble(beta=1, p=n, n=3*n, tridiagonal_form=False)
                t1 = time.time()
                eig_hist_nt, bins_nt = wre1.eigval_hist(bins=m, interval=interval, density=to_norm)
                t2 = time.time()
                times_naive[i][j] += (t2 - t1)*1000 # ms

                wre2 = WishartEnsemble(beta=1, p=n, n=3*n, tridiagonal_form=True)
                t1 = time.time()
                eig_hist_nt, bins_nt = wre2.eigval_hist(bins=m, interval=interval, density=to_norm)
                t2 = time.time()
                times_tridiag[i][j] += (t2 - t1)*1000 # ms
        
            times_naive[i][j] /= nreps
            times_tridiag[i][j] /= nreps
    
    _plot_times(N_list, bins_list, times_naive, times_tridiag)


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
