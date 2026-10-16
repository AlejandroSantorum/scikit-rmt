
# general libraries
import time
import numpy as np
import tqdm
from datetime import timedelta
import matplotlib.pyplot as plt
# my own modules
import sys
sys.path.append("../skrmt")

from ensemble import GaussianEnsemble, WishartEnsemble


####################################################################################
# TRIDIAGONAL SIMS
def _build_m_labels(bins_list):
    return [f"m = {m}" for m in bins_list]


def _plot_times(N_list, bins_list, times):
    """Plot benchmark times in ascending order, regardless of input order.

    times has shape (3, len(N_list), len(bins_list)):
    dense, tridiagonal, Sturm.
    """
    titles = [
        r"Dense eigenvalues: $O(n^3)$",
        r"Tridiagonal eigenvalues: $O(n^2)$",
        r"Sturm sequences: $O(mn)$",
    ]
    degrees = [3, 2, 1]

    # sort matrix sizes and bin counts independently
    n_order = np.argsort(N_list)
    m_order = np.argsort(bins_list)

    N_sorted = np.asarray(N_list)[n_order]
    bins_sorted = np.asarray(bins_list)[m_order]

    # reorder timings to match the sorted matrix sizes and bin counts
    times_sorted = times[:, n_order, :][:, :, m_order]

    labels = _build_m_labels(bins_sorted)
    nodes = np.linspace(N_sorted[0], N_sorted[-1], 1000)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(19, 5))

    for ax, method_times, title, degree in zip(
        axes, times_sorted, titles, degrees
    ):
        lines = ax.plot(N_sorted, method_times)

        # fit the smallest and largest bin counts
        for j in (0, -1):
            coef = np.polyfit(N_sorted, method_times[:, j], degree)
            ax.plot(
                nodes,
                np.polyval(coef, nodes),
                "--",
                color=lines[j].get_color(),
            )

        ax.legend(lines, labels)
        ax.set_title(
            f"{title}\n(Dashed: degree-{degree} polynomial fits)"
        )
        ax.set_xlabel("n (matrix size)")
        ax.set_ylabel("Time (ms)")

    fig.tight_layout()
    plt.show()


def _time_ms(fn, *args, **kwargs):
    """Return execution time in milliseconds."""
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return (time.perf_counter() - t0) * 1000


def _get_interval(ensemble, interval):
    """Resolve the histogram interval for a given ensemble.

    If interval is explicitly supplied, use it.
    Otherwise, use the ensemble non-normalized interval.
    """
    if interval is not None:
        return interval

    # the library defines the correct non-normalized support by beta
    if hasattr(ensemble, "_non_normalized_interval"):
        return ensemble._non_normalized_interval[ensemble.beta]

    raise ValueError(
        "No default histogram interval is available for this ensemble. "
        "Please provide interval explicitly."
    )


def _simulate_times(make_ensemble, N_list, bins_list, interval, nreps, seed):
    """Benchmark dense, tridiagonal, and Sturm histogram methods.

    Displays one progress bar covering all matrix sizes, bin counts,
    and repetitions. The bar estimates the total simulation duration.
    """
    times = np.zeros((3, len(N_list), len(bins_list)))
    density = False
    np.random.seed(seed)

    # Total number of (n, m, repetition) combinations.
    total_steps = len(N_list) * len(bins_list) * nreps
    start_time = time.perf_counter()

    with tqdm.tqdm(
        total=total_steps,
        desc="Simulating",
        unit="rep",
        dynamic_ncols=True,
    ) as pbar:

        for i, n in enumerate(N_list):
            # Resolve the appropriate interval for this matrix size.
            reference = make_ensemble(n, tridiagonal_form=True)
            interval_n = _get_interval(reference, interval)

            for j, m in enumerate(bins_list):
                for rep in range(nreps):

                    # Dense eigenvalue computation + histogramming.
                    dense = make_ensemble(n, tridiagonal_form=False)
                    times[0, i, j] += _time_ms(
                        dense.eigval_hist,
                        bins=m,
                        interval=interval_n,
                        density=density,
                    )

                    # Tridiagonal eigenvalue computation + histogramming.
                    tridiag = make_ensemble(n, tridiagonal_form=True)
                    times[1, i, j] += _time_ms(
                        lambda: np.histogram(
                            tridiag.eigvals(),
                            bins=m,
                            range=interval_n,
                            density=density,
                        )
                    )

                    # Sturm-sequence histogram.
                    sturm = make_ensemble(n, tridiagonal_form=True)
                    times[2, i, j] += _time_ms(
                        sturm.eigval_hist,
                        bins=m,
                        interval=interval_n,
                        density=density,
                    )

                    # Update the average timings for this combination.
                    pbar.update(1)
                    pbar.set_postfix(n=n, m=m, rep=f"{rep + 1}/{nreps}")

                    # Estimate total runtime from completed repetitions.
                    elapsed = time.perf_counter() - start_time
                    estimated_total = elapsed * total_steps / pbar.n

                    pbar.set_postfix(
                        n=n,
                        m=m,
                        rep=f"{rep + 1}/{nreps}",
                        total_est= str(
                            timedelta(seconds=int(estimated_total))
                        ),
                    )

                times[:, i, j] /= nreps

    return times


def _gaussian_ensemble(n, tridiagonal_form):
    return GaussianEnsemble(
        beta=1,
        n=n,
        tridiagonal_form=tridiagonal_form,
    )


def _wishart_ensemble(n, tridiagonal_form):
    return WishartEnsemble(
        beta=1,
        p=n,
        n=3*n,
        tridiagonal_form=tridiagonal_form,
    )


def gaussian_tridiagonal_sim(N_list, bins_list, nreps=10, seed=0):
    """Benchmark Gaussian ensemble histogram methods."""
    times = _simulate_times(
        _gaussian_ensemble,
        N_list,
        bins_list,
        interval=None,  # use the scikit-rmt size-dependent interval
        nreps=nreps,
        seed=seed,
    )
    _plot_times(N_list, bins_list, times)


def wishart_tridiagonal_sim(N_list, bins_list, nreps=10, seed=0):
    """Benchmark Wishart ensemble histogram methods."""
    times = _simulate_times(
        _wishart_ensemble,
        N_list,
        bins_list,
        interval=None,  # use the scikit-rmt size-dependent interval
        nreps=nreps,
        seed=seed,
    )
    _plot_times(N_list, bins_list, times)
