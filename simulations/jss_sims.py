import os
import sys
import time
from importlib import reload
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# This will add the directory of the script to the Python path
sys.path.append(os.path.dirname(__file__))

try:
    sys.path.append("./scikit-rmt")
    sys.path.append("../")
    sys.path.append("../scikit-rmt")
except BaseException:
    pass

from skrmt.ensemble.gaussian_ensemble import GaussianEnsemble
from skrmt.ensemble.wishart_ensemble import WishartEnsemble
from skrmt.ensemble.manova_ensemble import ManovaEnsemble
from skrmt.ensemble.circular_ensemble import CircularEnsemble
from skrmt.ensemble.spectral_law import (
    WignerSemicircleDistribution,
    MarchenkoPasturDistribution,
    TracyWidomDistribution,
    ManovaSpectrumDistribution,
)
from skrmt.ensemble.utils import (
    standard_vs_tridiag_hist, plot_spectral_hist_and_law,
)

IMGS_DIRNAME = "skrmt_sim_imgs"
SCRIPT_PATH = os.path.dirname(__file__)

BOLD_CHAR = '\033[1m'
END_CHAR = '\033[0m'


def _setup_img_dir():
    """Creates a directory (if ti does not exist) in the same locations
    as this script for the simulation images of scikit-rmt.
    """
    # creating a directory in the same place as this script for the images
    imgs_dir_path = os.path.join(SCRIPT_PATH, IMGS_DIRNAME)
    if not os.path.exists(imgs_dir_path):
        print(f"Creating directory {IMGS_DIRNAME} at {SCRIPT_PATH} for the images.")
        os.makedirs(imgs_dir_path)

    print(f"The images of the Figures will be stored at {imgs_dir_path}")


def __restore_plt():
    reload(matplotlib)
    reload(plt)
    plt.clf()
    plt.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcdefaults()


def plot_figure_1():
    print(f"Generating images of Figure 1. This may take {BOLD_CHAR}some seconds{END_CHAR}...")

    ens = GaussianEnsemble(beta=1, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig1_goe_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    __restore_plt()

    ens = GaussianEnsemble(beta=2, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig1_gue_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    __restore_plt()

    ens = GaussianEnsemble(beta=4, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig1_gse_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 1")


def plot_figure_2():
    print(f"Generating images of Figure 2. This may take {BOLD_CHAR}some seconds{END_CHAR}...")

    ens = WishartEnsemble(beta=1, p=1000, n=5000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig2_wre_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    __restore_plt()

    ens = WishartEnsemble(beta=2, p=1000, n=5000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig2_wce_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    __restore_plt()

    ens = WishartEnsemble(beta=4, p=1000, n=5000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig2_wqe_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 2")


def plot_figure_3():
    print(f"Generating images of Figure 3. This may take {BOLD_CHAR}some seconds{END_CHAR}...")

    ens = ManovaEnsemble(beta=1, m=1000, n1=2000, n2=2000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig3_mre_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    __restore_plt()

    ens = ManovaEnsemble(beta=2, m=1000, n1=2000, n2=2000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig3_mce_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    __restore_plt()

    ens = ManovaEnsemble(beta=4, m=1000, n1=2000, n2=2000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig3_mqe_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 3")


def plot_figure_4():
    print(f"Generating images of Figure 4. This may take {BOLD_CHAR}some seconds{END_CHAR}...")

    ens = CircularEnsemble(beta=1, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig4_coe_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    __restore_plt()

    ens = CircularEnsemble(beta=2, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig4_cue_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    __restore_plt()

    ens = CircularEnsemble(beta=4, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig4_cse_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 4")


def plot_figure_5():
    print(f"Generating images of Figure 5. This may take {BOLD_CHAR}some seconds{END_CHAR}...")

    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig5_goe_1000.png")
    ens = GaussianEnsemble(beta=1, n=1000)
    standard_vs_tridiag_hist(ensemble=ens, bins=60, random_state=10, savefig_path=ens_figpath)
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 5")


def plot_figure_6():
    print(f"Generating images of Figure 6. This may take {BOLD_CHAR}several minutes (~45 mins.){END_CHAR} ...")

    # matrix sizes
    n_list = [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    # number of bins
    m_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    # Init random seed, even though it is not crucial since this is a time-based simulation
    np.random.seed(1)

    figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig6_goe_tridiagonal_vs_standard.png")
    _gaussian_tridiagonal_sim(
        N_list=n_list, bins_list=m_list, nreps=20, savefig_path=figpath,
    )

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 6")


def plot_figure_7():
    print(f"Generating images of Figure 7. This may take {BOLD_CHAR}several minutes (~50 mins.){END_CHAR} ...")

    # matrix sizes
    n_list = [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    # number of bins
    m_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    # Init random seed, even though it is not crucial since this is a time-based simulation
    np.random.seed(1)

    figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig7_wre_tridiagonal_vs_standard.png")
    _wishart_tridiagonal_sim(
        N_list=n_list, bins_list=m_list, nreps=20, savefig_path=figpath,
    )

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 7")


def plot_figure_8():
    print(f"Generating images of Figure 8. This may take {BOLD_CHAR}few minutes{END_CHAR}...")

    ### Subplot 1
    np.random.seed(1)

    t = 3_000_000

    lambdas_1 = []
    lambdas_2 = []
    for i in range(t):
        goe = GaussianEnsemble(n=2, beta=1, tridiagonal_form=False)

        eigvals = 1/np.sqrt(2) * goe.eigvals()
        lambdas_1.append(eigvals[0])
        lambdas_2.append(eigvals[1])


    ### Subplot 2
    np.random.seed(1)

    goe = GaussianEnsemble(n=2, beta=1)

    t = 100
    radius = 3

    # remember, lambda_1 <= lambda_2
    x_vals = np.linspace(-radius, radius, t)
    y_vals = np.linspace(-radius, radius, t)

    joint_pdf = []
    for i in range(t):
        aux_jpdf = []
        for j in range(t):
            # picking eigenvalues
            v1 = x_vals[j]
            v2 = y_vals[i]

            if v2 < v1:
                # joint eigenvalue pdf is zero since we do not consider lambda_2 < lambda_1
                aux_jpdf.append(0.0)
            
            else:
                # computing eigenvalue joint pdf
                jepdf = goe.joint_eigval_pdf(np.asarray([v1, v2]))
                aux_jpdf.append(jepdf)

        joint_pdf.append(aux_jpdf)

    joint_pdf = np.asarray(joint_pdf)

    ### Plotting
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(5)
    fig.set_figwidth(13)
    fig.subplots_adjust(hspace=.1)

    _ = axes[0].hist2d(lambdas_1, lambdas_2, bins=30, range=[[-3,3], [-3,3]])
    _ = axes[0].set_aspect('equal', adjustable='box')
    _ = axes[0].set_xlabel(r"$\lambda_1$", fontsize=12)
    _ = axes[0].set_ylabel(r"$\lambda_2$", fontsize=12)
    _ = axes[0].set_title("Histogram of eigenvalues of GOE matrices 2x2")

    img = axes[1].contourf(x_vals, y_vals, joint_pdf, extend="both")
    _ = axes[1].set_xlabel(r"$\lambda_1$", fontsize=12)
    _ = axes[1].set_ylabel(r"$\lambda_2$", fontsize=12)
    _ = fig.colorbar(img, label="joint eigenvalue PDF", ax=axes)
    _ = axes[1].set_title("Joint Eigenvalue PDF for GOE matrices 2x2")

    figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig8_goe_2x2.png")
    plt.savefig(figpath)
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 8")


def plot_figure_9():
    print(f"Generating images of Figure 9. This may take {BOLD_CHAR}some seconds{END_CHAR}...")

    plt.rcParams['figure.dpi'] = 100

    xx = np.linspace(-10, 10, num=4000)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

    for sigma in [0.5, 1.0, 2.0, 4.0]:
        wsd = WignerSemicircleDistribution(beta=1, center=0.0, sigma=sigma)

        # computing pdf
        y1 = wsd.pdf(xx)
        # computing cdf    
        y2 = wsd.cdf(xx)

        ax1.plot(xx, y1, label=f"$\sigma$ = {sigma} (R = ${wsd.radius}$)")
        ax2.plot(xx, y2, label=f"$\sigma$ = {sigma} (R = ${wsd.radius}$)")

    ax1.legend()
    ax1.set_xlabel("x", fontweight="bold")
    ax1.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    ax1.set_ylabel("density", fontweight="bold")
    ax1.set_title("Probability density function (PDF)")

    ax2.legend()
    ax2.set_xlabel("x", fontweight="bold")
    ax2.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    ax2.set_ylabel("probability", fontweight="bold")
    ax2.set_title("Cumulative distribution function (CDF)")

    fig.suptitle("Wigner Semicircle law", fontweight="bold")

    figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig9_wsl_pdf_cdf.png")
    plt.savefig(figpath)
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 9")


def plot_figure_10():
    print(f"Generating images of Figure 10. This may take {BOLD_CHAR}some seconds{END_CHAR}...")

    wsd1 = WignerSemicircleDistribution(beta=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig10_wsl_beta1.png")
    wsd1.plot_empirical_pdf(
        sample_size=10000,
        bins=60,
        density=True,
        plot_law_pdf=True,
        random_state=1,
        savefig_path=ens_figpath,
    )
    __restore_plt()

    wsd2 = WignerSemicircleDistribution(beta=2)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig10_wsl_beta2.png")
    wsd2.plot_empirical_pdf(
        sample_size=10000,
        bins=60,
        density=True,
        plot_law_pdf=True,
        random_state=2,
        savefig_path=ens_figpath,
    )
    __restore_plt()

    wsd4 = WignerSemicircleDistribution(beta=4)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig10_wsl_beta4.png")
    wsd4.plot_empirical_pdf(
        sample_size=10000,
        bins=60,
        density=True,
        plot_law_pdf=True,
        random_state=4,
        savefig_path=ens_figpath,
    )
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 10")


def plot_figure_11():
    print(f"Generating images of Figure 11. This may take {BOLD_CHAR}some seconds{END_CHAR}...")

    goe = GaussianEnsemble(beta=1, n=100, tridiagonal_form=True, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig11_goe_100.png")
    plot_spectral_hist_and_law(ensemble=goe, bins=60, savefig_path=ens_figpath)
    __restore_plt()

    goe = GaussianEnsemble(beta=1, n=1000, tridiagonal_form=True, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig11_goe_1000.png")
    plot_spectral_hist_and_law(ensemble=goe, bins=60, savefig_path=ens_figpath)
    __restore_plt()

    goe = GaussianEnsemble(beta=1, n=10000, tridiagonal_form=True, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig11_goe_10000.png")
    plot_spectral_hist_and_law(ensemble=goe, bins=60, savefig_path=ens_figpath)
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 11")


def plot_figure_12():
    print(f"Generating images of Figure 12. This may take {BOLD_CHAR}some seconds{END_CHAR}...")

    plt.rcParams['figure.dpi'] = 100

    x = np.linspace(-5, 2, num=1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

    for beta in [1,2,4]:
        twd = TracyWidomDistribution(beta=beta)

        y_pdf = twd.pdf(x)
        y_cdf = twd.cdf(x)

        ax1.plot(x, y_pdf, label=f"$\\beta$ = {beta}")
        ax2.plot(x, y_cdf, label=f"$\\beta$ = {beta}")

    ax1.legend()
    ax1.set_xlabel("x", fontweight="bold")
    ax1.set_ylabel("density", fontweight="bold")
    ax1.set_title("Probability density function (PDF)")

    ax2.legend()
    ax2.set_xlabel("x", fontweight="bold")
    ax2.set_ylabel("probability", fontweight="bold")
    ax2.set_title("Cumulative distribution function (CDF)")

    fig.suptitle("Tracy-Widom law", fontweight="bold")

    figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig12_twl_pdf_cdf.png")
    plt.savefig(figpath)
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 12")


def plot_figure_13():
    print(f"Generating images of Figure 13. This may take {BOLD_CHAR}several minutes (~45 mins.){END_CHAR}...")
    
    twd1 = TracyWidomDistribution(beta=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig13_twl_beta1.png")
    twd1.plot_empirical_pdf(
        sample_size=10000,
        bins=60,
        density=True,
        plot_law_pdf=True,
        random_state=1,
        savefig_path=ens_figpath,
    )
    __restore_plt()

    twd2 = TracyWidomDistribution(beta=2)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig13_twl_beta2.png")
    twd2.plot_empirical_pdf(
        sample_size=10000,
        bins=60,
        density=True,
        plot_law_pdf=True,
        random_state=2,
        savefig_path=ens_figpath,
    )
    __restore_plt()

    twd4 = TracyWidomDistribution(beta=4)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig13_twl_beta4.png")
    twd4.plot_empirical_pdf(
        sample_size=10000,
        bins=60,
        density=True,
        plot_law_pdf=True,
        random_state=4,
        savefig_path=ens_figpath,
    )
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 13")


def plot_figure_14():
    print(f"Generating images of Figure 14. This may take {BOLD_CHAR}some minutes{END_CHAR}...")
    
    twd = TracyWidomDistribution(beta=1)

    ens = GaussianEnsemble(beta=1, n=10)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig14_twl_n10.png")
    twd.plot_ensemble_max_eigvals(
        ensemble=ens, n_eigvals=10000, bins=50, random_state=2, savefig_path=ens_figpath,
    )
    __restore_plt()

    ens = GaussianEnsemble(beta=1, n=100)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig14_twl_n100.png")
    twd.plot_ensemble_max_eigvals(
        ensemble=ens, n_eigvals=10000, bins=50, random_state=2, savefig_path=ens_figpath,
    )
    __restore_plt()

    ens = GaussianEnsemble(beta=1, n=1000)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig14_twl_n1000.png")
    twd.plot_ensemble_max_eigvals(
        ensemble=ens, n_eigvals=10000, bins=50, random_state=2, savefig_path=ens_figpath,
    )
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 14")


def plot_figure_15():
    print(f"Generating images of Figure 15. This may take {BOLD_CHAR}some seconds{END_CHAR}...")

    plt.rcParams['figure.dpi'] = 100

    x1 = np.linspace(0, 4, num=1000)
    x2 = np.linspace(0, 5, num=2000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

    for ratio in [0.2, 0.4, 0.6, 1.0, 1.4]:
        mpl = MarchenkoPasturDistribution(beta=1, ratio=ratio, sigma=1.0)

        y1 = mpl.pdf(x1)
        y2 = mpl.pdf(x2)

        ax1.plot(x1, y1, label=f"$\lambda$ = {ratio} ")
        ax2.plot(x2, y2, label=f"$\lambda$ = {ratio} ")

    ax1.legend()
    ax1.set_ylim(0, 1.4)
    ax1.set_xlabel("x", fontweight="bold")
    ax1.set_ylabel("density", fontweight="bold")

    ax2.legend()
    ax2.set_ylim(0, 1.4)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("x", fontweight="bold")
    ax2.set_ylabel("density", fontweight="bold")

    fig.suptitle("Marchenko-Pastur probability density function (PDF)", fontweight="bold")

    figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig15_mpl_pdf.png")
    plt.savefig(figpath)
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 15")


def plot_figure_16():
    print(f"Generating images of Figure 16. This may take {BOLD_CHAR}some seconds{END_CHAR}...")
    
    mpd1 = MarchenkoPasturDistribution(beta=1, ratio=1/5)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig16_mpl_beta1.png")
    mpd1.plot_empirical_pdf(
        sample_size=10000,
        bins=60,
        density=True,
        plot_law_pdf=True,
        random_state=1,
        savefig_path=ens_figpath,
    )
    __restore_plt()

    mpd2 = MarchenkoPasturDistribution(beta=2, ratio=1/5)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig16_mpl_beta2.png")
    mpd2.plot_empirical_pdf(
        sample_size=10000,
        bins=60,
        density=True,
        plot_law_pdf=True,
        random_state=2,
        savefig_path=ens_figpath,
    )
    __restore_plt()

    mpd4 = MarchenkoPasturDistribution(beta=4, ratio=1/5)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig16_mpl_beta4.png")
    mpd4.plot_empirical_pdf(
        sample_size=10000,
        bins=60,
        density=True,
        plot_law_pdf=True,
        random_state=4,
        savefig_path=ens_figpath,
    )
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 16")


def plot_figure_17():
    print(f"Generating images of Figure 17. This may take {BOLD_CHAR}some seconds{END_CHAR}...")

    wre = WishartEnsemble(beta=1, p=100, n=500, tridiagonal_form=True, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig17_wre_100.png")
    plot_spectral_hist_and_law(ensemble=wre, bins=60, savefig_path=ens_figpath)
    __restore_plt()

    wre = WishartEnsemble(beta=1, p=1000, n=5000, tridiagonal_form=True, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig17_wre_1000.png")
    plot_spectral_hist_and_law(ensemble=wre, bins=60, savefig_path=ens_figpath)
    __restore_plt()

    wre = WishartEnsemble(beta=1, p=10000, n=50000, tridiagonal_form=True, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig17_wre_10000.png")
    plot_spectral_hist_and_law(ensemble=wre, bins=60, savefig_path=ens_figpath)
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 17")


def plot_figure_18():
    print(f"Generating images of Figure 18. This may take {BOLD_CHAR}some seconds{END_CHAR}...")

    np.random.seed(1)

    plt.rcParams['figure.dpi'] = 100

    x1 = np.linspace(0, 1, num=1000)
    x2 = np.linspace(0, 1, num=1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

    for a in [1.0, 1.2, 1.4, 1.6]:
        for b in [2.0]:
            msd = ManovaSpectrumDistribution(beta=1, ratio_a=a, ratio_b=b)

            y1 = msd.pdf(x1)
            y2 = msd.pdf(x2)

            ax1.plot(x1, y1, label=f"$a$ = {a}, $b$ = {b}")
            ax2.plot(x2, y2, label=f"$a$ = {a}, $b$ = {b}")

    ax1.legend()
    ax1.set_xlabel("x", fontweight="bold")
    ax1.set_ylabel("density", fontweight="bold")

    ax2.legend()
    ax2.set_ylim(0, 4)
    ax2.set_xlabel("x", fontweight="bold")
    ax2.set_ylabel("density", fontweight="bold")

    fig.suptitle("Manova spectrum probability density function (PDF)", fontweight="bold")

    figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig18_msd_pdf.png")
    plt.savefig(figpath)
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 18")


def plot_figure_19():
    print(f"Generating images of Figure 19. This may take {BOLD_CHAR}some seconds{END_CHAR}...")
    
    mpd1 = ManovaSpectrumDistribution(beta=1, ratio_a=2, ratio_b=2)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig19_msd_beta1.png")
    mpd1.plot_empirical_pdf(
        sample_size=10000,
        bins=60,
        density=True,
        plot_law_pdf=True,
        random_state=1,
        savefig_path=ens_figpath,
    )
    __restore_plt()

    mpd2 = ManovaSpectrumDistribution(beta=2, ratio_a=2, ratio_b=2)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig19_msd_beta2.png")
    mpd2.plot_empirical_pdf(
        sample_size=10000,
        bins=60,
        density=True,
        plot_law_pdf=True,
        random_state=2,
        savefig_path=ens_figpath,
    )
    __restore_plt()

    mpd4 = ManovaSpectrumDistribution(beta=4, ratio_a=2, ratio_b=2)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig19_msd_beta4.png")
    mpd4.plot_empirical_pdf(
        sample_size=10000,
        bins=60,
        density=True,
        plot_law_pdf=True,
        random_state=4,
        savefig_path=ens_figpath,
    )
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 19")


def plot_figure_20():
    print(f"Generating images of Figure 20. This may take {BOLD_CHAR}some minutes (~5 mins.){END_CHAR} ...")

    mre = ManovaEnsemble(beta=1, m=100, n1=200, n2=200, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig20_mre_100.png")
    plot_spectral_hist_and_law(ensemble=mre, bins=60, savefig_path=ens_figpath)
    __restore_plt()

    mre = ManovaEnsemble(beta=1, m=1000, n1=2000, n2=2000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig20_mre_1000.png")
    plot_spectral_hist_and_law(ensemble=mre, bins=60, savefig_path=ens_figpath)
    __restore_plt()

    mre = ManovaEnsemble(beta=1, m=10000, n1=20000, n2=20000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig20_mre_10000.png")
    plot_spectral_hist_and_law(ensemble=mre, bins=60, savefig_path=ens_figpath)
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 20")



def __build_m_labels(M):
    """Auxiliary function to build plot labels
    """
    labels = [("m = "+str(m)) for m in M]
    return labels


def __plot_times(N_list, bins_list, times_naive, times_tridiag, savefig_path):
    """Useful function to plot the tridiagonal optimization simulations
    """
    # creating subplots
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(5)
    fig.set_figwidth(13)
    fig.subplots_adjust(hspace=.5)

    # labels for plots and nodes for interpolation
    labels = __build_m_labels(bins_list)
    nodes = np.linspace(N_list[0], N_list[-1], 1000)

    # Fitting line naive computation
    y_naive = times_naive[:,0]
    a, m, b = np.polyfit(N_list, y_naive, 2)
    y_smooth_naive = a*nodes**2 + m*nodes + b

    # Naive plot
    lines = axes[0].plot(N_list, times_naive)
    axes[0].plot(nodes, y_smooth_naive, '--')
    legend1 = axes[0].legend(lines, labels, loc=0)
    axes[0].add_artist(legend1)
    axes[0].set_title('Naive computation')
    axes[0].set_xlabel('n (matrix size)')
    axes[0].set_ylabel('Time (ms.)')

    lines_tridiag = axes[1].plot(N_list, times_tridiag)
    # Fitting lines in tridiagonal computation
    for idx in range(times_tridiag.shape[1]):
        y_tridiag = times_tridiag[:,idx]
        m, b = np.polyfit(N_list, y_tridiag, 1)
        y_smooth_tridiag = m*nodes + b
        axes[1].plot(nodes, y_smooth_tridiag, '--')

    # Tridiagonal legends
    legend1 = axes[1].legend(lines_tridiag, labels, loc=0)
    axes[1].add_artist(legend1)
    axes[1].set_title('Tridiagonal Sturm computation')
    axes[1].set_xlabel('n (matrix size)')
    _ = axes[1].set_ylabel('Time (ms.)')

    plt.savefig(savefig_path)


def _gaussian_tridiagonal_sim(N_list, bins_list, savefig_path, nreps=10):
    """Creating tridiagonal simulation graphic using GOE (tridiagonal vs standard)
    """
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
    
    __plot_times(N_list, bins_list, times_naive, times_tridiag, savefig_path)
    

def _wishart_tridiagonal_sim(N_list, bins_list, savefig_path, nreps=10):
    """Creating tridiagonal simulation graphic using WRE (tridiagonal vs standard)
    """
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
    
    __plot_times(N_list, bins_list, times_naive, times_tridiag, savefig_path)



def main():
    """MAIN FUNCTION"""
    _setup_img_dir()

    # Gaussian ensemble
    plot_figure_1()
    # Wishart ensemble
    plot_figure_2()
    # Manova ensemble
    plot_figure_3()
    # Circular ensemble
    plot_figure_4()

    # Standard vs tridiagonal histograms 
    plot_figure_5()

    # Tridiagonal optimization
    plot_figure_6()
    plot_figure_7()

    # Joint eigenvalue PDF
    plot_figure_8()

    # Wigner Semicircle law
    plot_figure_9()
    plot_figure_10()
    plot_figure_11()

    # Tracy-Widom law
    plot_figure_12()
    plot_figure_13()
    plot_figure_14()

    # Marchenko-Pastur law
    plot_figure_15()
    plot_figure_16()
    plot_figure_17()

    # Manova Spectrum distr.
    plot_figure_18()
    plot_figure_19()
    plot_figure_20()


if __name__ == "__main__":
    main()
