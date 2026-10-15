
# general libraries
import time
import numpy as np
import matplotlib.pyplot as plt
# my own modules
import sys
sys.path.append('../skrmt')

from ensemble import GaussianEnsemble, WishartEnsemble


####################################################################################
# TRIDIAGONAL SIMS

def _build_m_labels(M):
    labels = []
    for m in M:
        labels.append("m = "+str(m))
    return labels



def _plot_times(N_list, bins_list, times):
    # times has shape (3, len(N_list), len(bins_list)): dense, tridiagonal, Sturm
    titles = [
        'Dense eigenvalues: O(n^3)',
        'Tridiagonal eigenvalues: O(n^2)',
        'Sturm sequences: O(mn)',
    ]
    degrees = [3, 2, 1]
    labels = _build_m_labels(bins_list)
    nodes = np.linspace(N_list[0], N_list[-1], 1000)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(19, 5))
    for ax, method_times, title, deg in zip(axes, times, titles, degrees):
        lines = ax.plot(N_list, method_times)
        # polyfit of the smallest and largest bin size, dashed
        for j in (0, -1):
            coef = np.polyfit(N_list, method_times[:, j], deg)
            ax.plot(nodes, np.polyval(coef, nodes), '--', color=lines[j].get_color())
        ax.legend(lines, labels)
        ax.set_title(f'{title} (dashed: degree {deg} fit)')
        ax.set_xlabel('n (matrix size)')
        ax.set_ylabel('Time (ms.)')
    fig.tight_layout()
    plt.show()


def _time_ms(fn, *args, **kwargs):
    t1 = time.time()
    fn(*args, **kwargs)
    return (time.time() - t1)*1000


def _simulate_times(make_ensemble, N_list, bins_list, interval, nreps, seed):
    # axis 0: dense eigenvalues, tridiagonal eigenvalues, Sturm sequences
    times = np.zeros((3, len(N_list), len(bins_list)))
    to_norm = False
    np.random.seed(seed)

    for (i, n) in enumerate(N_list):
        for (j, m) in enumerate(bins_list):
            for _ in range(nreps):
                dense = make_ensemble(n, tridiagonal_form=False)
                times[0][i][j] += _time_ms(dense.eigval_hist, bins=m, interval=interval,
                                           density=to_norm)

                tridiag = make_ensemble(n, tridiagonal_form=True)
                times[1][i][j] += _time_ms(
                    lambda: np.histogram(tridiag.eigvals(), bins=m, range=interval,
                                         density=to_norm))

                sturm = make_ensemble(n, tridiagonal_form=True)
                times[2][i][j] += _time_ms(sturm.eigval_hist, bins=m, interval=interval,
                                           density=to_norm)

            times[:, i, j] /= nreps

    return times


def _gaussian_ensemble(n, tridiagonal_form):
    return GaussianEnsemble(beta=1, n=n, tridiagonal_form=tridiagonal_form)


def _wishart_ensemble(n, tridiagonal_form):
    return WishartEnsemble(beta=1, p=n, n=3*n, tridiagonal_form=tridiagonal_form)



def gaussian_tridiagonal_sim(N_list, bins_list, nreps=10, seed=0):
    times = _simulate_times(_gaussian_ensemble, N_list, bins_list, (-2, 2), nreps, seed)
    _plot_times(N_list, bins_list, times)



def wishart_tridiagonal_sim(N_list, bins_list, nreps=10, seed=0):
    times = _simulate_times(_wishart_ensemble, N_list, bins_list, (0, 4), nreps, seed)
    _plot_times(N_list, bins_list, times)
