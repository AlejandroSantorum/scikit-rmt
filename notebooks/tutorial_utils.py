
# general libraries
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
# my own modules
import sys
sys.path.append('../skrmt')

from ensemble import GaussianEnsemble, WishartEnsemble


def _build_m_labels(M):
    labels = []
    for m in M:
        labels.append("m = "+str(m))
    return labels



def plot_times(N_list, bins_list, times_naive, times_tridiag):
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
    legend2 = axes[1].legend(fit_line1, ['lower fit'], loc=8)
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
                goe1 = GaussianEnsemble(beta=1, n=n, use_tridiagonal=False)
                t1 = time.time()
                eig_hist_nt, bins_nt = goe1.eigval_hist(bins=m, interval=interval, density=to_norm)
                t2 = time.time()
                times_naive[i][j] += (t2 - t1)*1000 # ms

                goe2 = GaussianEnsemble(beta=1, n=n, use_tridiagonal=True)
                t1 = time.time()
                eig_hist_nt, bins_nt = goe2.eigval_hist(bins=m, interval=interval, density=to_norm)
                t2 = time.time()
                times_tridiag[i][j] += (t2 - t1)*1000 # ms
        
            times_naive[i][j] /= nreps
            times_tridiag[i][j] /= nreps
    
    plot_times(N_list, bins_list, times_naive, times_tridiag)



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
                wre1 = WishartEnsemble(beta=1, p=n, n=3*n, use_tridiagonal=False)
                t1 = time.time()
                eig_hist_nt, bins_nt = wre1.eigval_hist(bins=m, interval=interval, density=to_norm)
                t2 = time.time()
                times_naive[i][j] += (t2 - t1)*1000 # ms

                wre2 = WishartEnsemble(beta=1, p=n, n=3*n, use_tridiagonal=True)
                t1 = time.time()
                eig_hist_nt, bins_nt = wre2.eigval_hist(bins=m, interval=interval, density=to_norm)
                t2 = time.time()
                times_tridiag[i][j] += (t2 - t1)*1000 # ms
        
            times_naive[i][j] /= nreps
            times_tridiag[i][j] /= nreps
    
    plot_times(N_list, bins_list, times_naive, times_tridiag)

