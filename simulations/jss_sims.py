import os
import sys
import time
from importlib import reload
import numpy as np
from scipy.stats import rice
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


def plot_figure_22():
    print(f"Generating images of Figure 22. This may take {BOLD_CHAR}some minutes (~10 mins.){END_CHAR} ...")

    # Get original simulation MRI image
    current_filepath = os.path.dirname(__file__)
    tumor_slice = np.load(f"{current_filepath}/sim_data/sample_tumor_img.npy")
    tumor_slice = np.pad(tumor_slice, pad_width=5, mode="constant", constant_values=0)

    # Number of measurements (snapshots)
    n_snapshots = 100
    # RNG seed
    seed = 1
    # Noise parameters
    sigma = 35  # Marchenko-Pastur sigma ~ noise standard deviation
    rice_b = 2  # Rice distribution parameter
    # Denoising parameters
    window_size = 16  # 16x16 MPPCA window size

    noise_corruptor = ImgNoiseCorruptor(original_img=tumor_slice)
    snapshots, fg_masks = noise_corruptor.generate_rician_noisy_displaced_imgs(
        n_snapshots=n_snapshots,
        rice_b=rice_b,
        sigma=sigma,
        seed=seed,
    )

    # Sample of corrupted snapshot
    corrupted_slice = snapshots[0]

    # Apply MP-PCA
    denoised_snapshots = denoise_mppca(snapshots, sigma=sigma, window_size=window_size)
    # Adjust background and normalize denoised images in [0, 255]
    denoised_snapshots = apply_foreground_masks(snapshots=denoised_snapshots, fg_masks=fg_masks)
    denoised_snapshots = normalize_imgs_0_255(snapshots=denoised_snapshots)

    # Sample of denoised snapshot
    denoised_slice = denoised_snapshots[0]

    # Denoising by averaging
    denoised_by_avg_slice = (tumor_slice > 0.0) * np.mean(snapshots, axis=0)

    # Average SNR and PSNR of simulated corrupted snapshots
    ss_avg_snr, ss_std_snr = average_snr(ref_img=tumor_slice, test_imgs=snapshots)
    ss_avg_psnr, ss_std_psnr = average_psnr(ref_img=tumor_slice, test_imgs=snapshots)

    # SNR and PSNR of denoised images by averaging
    den_means_snr = snr(ref_img=tumor_slice, test_img=denoised_by_avg_slice)
    den_means_psnr = psnr(ref_img=tumor_slice, test_img=denoised_by_avg_slice)

    # Average SNR and PSNR of denoised images using MP-PCA
    den_ss_avg_snr, den_ss_std_snr = average_snr(ref_img=tumor_slice, test_imgs=denoised_snapshots)
    den_ss_avg_psnr, den_ss_std_psnr = average_psnr(ref_img=tumor_slice, test_imgs=denoised_snapshots)

    print(f"\tAverage SNR of simulated snapshots: {ss_avg_snr} ± {ss_std_snr}.")
    print(f"\tAverage PSNR of simulated snapshots: {ss_avg_psnr} ± {ss_std_psnr}.")

    print(f"\tSNR of image denoised by averaging: {den_means_snr}.")
    print(f"\tPSNR of image denoised by averaging: {den_means_psnr}.")

    print(f"\tAverage SNR of snapshots using MP-PCA: {den_ss_avg_snr} ± {den_ss_std_snr}.")
    print(f"\tAverage PSNR of snapshots using MP-PCA: {den_ss_avg_psnr} ± {den_ss_std_psnr}.")

    img_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig22_a_noisy.png")
    plt.imshow(corrupted_slice, cmap="gray", origin="lower")
    plt.savefig(img_figpath)
    __restore_plt()

    img_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig22_b_denoised_avg.png")
    plt.imshow(denoised_by_avg_slice, cmap="gray", origin="lower")
    plt.savefig(img_figpath)
    __restore_plt()

    img_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig22_c_denoised_mppca.png")
    plt.imshow(denoised_slice, cmap="gray", origin="lower")
    plt.savefig(img_figpath)
    __restore_plt()

    img_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig22_d_denoised_original.png")
    plt.imshow(tumor_slice, cmap="gray", origin="lower")
    plt.savefig(img_figpath)
    __restore_plt()

    print(f"{BOLD_CHAR}[DONE]{END_CHAR} - Images Figure 22")


####################################################################################
# FUNCTIONS FOR TRIDIAGONAL MATRICES SIMULATIONS
###

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


####################################################################################
# FUNCTIONS FOR MRI IMAGE DENOISING PRACTICAL ILLUSTRATION
###

def norm_img_0_255(img: np.ndarray):
    min_val = np.min(img)
    max_val = np.max(img)
    return 255 * (img - min_val) / (max_val - min_val)


def normalize_imgs_0_255(snapshots: np.ndarray) -> np.ndarray:
    norm_snapshots = []
    for ss in snapshots:
        norm_snapshots.append(norm_img_0_255(img=ss))
    return np.asarray(norm_snapshots)


def apply_foreground_masks(snapshots: np.ndarray, fg_masks: np.ndarray) -> np.ndarray:
    assert snapshots.shape == fg_masks.shape

    mask_snapshots = []
    for idx, den_ss in enumerate(snapshots):
        fg_m = fg_masks[idx]
        mask_snapshots.append(fg_m * den_ss)
    mask_snapshots = np.asarray(mask_snapshots)

    assert mask_snapshots.shape == snapshots.shape
    return mask_snapshots


def denoise_local_mppca(X: np.ndarray, sigma: float) -> np.ndarray:
    """Denoises local region represented by X using Marchenko-Pastur PCA
    denoising algorithm.

    Args:
        X (np.dnarray): `p` times `n` matrix representing a local region to denoise.
            `p` is the number of different measurements, and `n` the number of pixels
            in the region to denoise.
        sigma (float): Marchenko-Pastur parameter, that approximates the level of noise.

    Returns:
        np.ndarray: denoised matrix X.
    """
    # p := number of measurements
    # n := number of pixels
    (p, n) = X.shape
    wre = WishartEnsemble(beta=1, p=p, n=n, sigma=sigma)
    wre.matrix = X

    # Principal Component Analysis (PCA) via SVD
    U, S, Vh = np.linalg.svd((1/np.sqrt(n)) * wre.matrix, full_matrices=False)

    # nullifying noisy eigenvalues
    denoised_S = np.where(S <= np.sqrt(wre.lambda_plus), 0, S)

    # reconstructing denoised X
    denoised_X = np.sqrt(n) * np.dot(U * denoised_S, Vh)

    return denoised_X


def denoise_mppca(snapshots: np.ndarray, sigma: float, window_size: int = 16) -> np.ndarray:
    """Performns PCA-based image denoising using the Marchenko-Pastur law.
    Denoises a set of images (snapshots) which are all noisy measurements
    of the same region of interest. For example, a set of MRI images
    focused on the same brain region.
    This function iterates over patches or windows of the image of size
    `window_size x window_size`. The denoised pixels contained in several
    windows are averaged.

    Args:
        snapshots (np.ndarray): set of images which are all noisy measurements
            of the same region of interest. 3D dimensional numpy array of size
            (N images, height, width).
        sigma (float): Marchenko-Pastur parameter, that approximates the
            level of noise.

    Returns:
        np.ndarray: denoised snapshots.
    """
    n_snapshots, img_height, img_width = snapshots.shape
    print(
        f"Denoising {n_snapshots} snapshots of size {img_height}x{img_width} (sigma = {sigma})."
    )

    denoised_snapshots = np.zeros_like(snapshots)

    for i in range(img_height - window_size + 1):
        for j in range(img_width - window_size + 1):
            locally_denoised_sss = np.zeros_like(snapshots)
            local_window = snapshots[:,i:i+window_size,j:j+window_size]
            local_X = np.reshape(local_window, (n_snapshots, window_size**2))

            denoised_local_X = denoise_local_mppca(X=local_X, sigma=sigma)

            denoised_local_window = np.reshape(denoised_local_X, (n_snapshots, window_size, window_size))
            locally_denoised_sss[:,i:i+window_size,j:j+window_size] = denoised_local_window

            denoised_snapshots += locally_denoised_sss

    for i in range(img_height):
        for j in range(img_width):
            # average by the number of times a pixel has been denoised
            average_by = min(i+1, img_height-i, window_size) * min(j+1, img_width-j, window_size)
            denoised_snapshots[:,i,j] /= average_by

    return denoised_snapshots


def snr(ref_img: np.ndarray, test_img: np.ndarray) -> float:
    """
    Computes Signal-to-Noise ratio measured in dB

    Input:
        - ref_img (np.ndarray): 2D numpy array.
        - test_img (np.ndarray): 2D numpy array.
    
    Returns:
        float: signal-to-noise ratio.
    
    Reference:
        - "SNR, PSNR, RMSE, MAE". Daniel Sage at the Biomedical Image Group, EPFL, Switzerland.
            http://bigwww.epfl.ch/sage/soft/snr/
        - D. Sage, M. Unser, "Teaching Image-Processing Programming in Java".
            IEEE Signal Processing Magazine, vol. 20, no. 6, pp. 43-52, November 2003.
            http://bigwww.epfl.ch/publications/sage0303.html
    """
    # checking they are 2D images
    assert len(ref_img.shape) == 2
    # checking both images have the same size
    assert ref_img.shape == test_img.shape

    numerator = np.sum(ref_img**2)
    denominator = np.sum((ref_img - test_img)**2)
    return 10 * np.log10(numerator/denominator)


def psnr(ref_img: np.ndarray, test_img: np.ndarray) -> float:
    """
    Computes Peak Signal-to-Noise ratio measured in dB

    Input:
        - ref_img (np.ndarray): 2D numpy array.
        - test_img (np.ndarray): 2D numpy array.
    
    Returns:
        float: peak signal-to-noise ratio.
    
    Reference:
        - https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    # checking they are 2D images
    assert len(ref_img.shape) == 2
    # checking both images have the same size
    assert ref_img.shape == test_img.shape

    mse = 1/(np.prod(ref_img.shape)) * np.sum((ref_img - test_img)**2)
    return 20 * np.log10(255.0 / np.sqrt(mse))


def average_snr(ref_img, test_imgs):
    snrs = [snr(ref_img=ref_img, test_img=t_img) for t_img in test_imgs]
    return np.mean(snrs), np.std(snrs)

def average_psnr(ref_img, test_imgs):
    psnrs = [psnr(ref_img=ref_img, test_img=t_img) for t_img in test_imgs]
    return np.mean(psnrs), np.std(psnrs)



class ImgNoiseCorruptor:
    """Generates corrupted images from a given original image by
    injecting different types of noise.
    
    Attributes:
        original_img (np.ndarray): 2d numpy array representing the original image
                of size (n_rows, n_cols) = (height, width).
    """
    DEFAULT_MAX_PIXEL_DISPLACEMENT = 4
    
    def __init__(
        self,
        original_img: np.ndarray,
        max_pixel_displacement: int = DEFAULT_MAX_PIXEL_DISPLACEMENT
    ):
        self.original_img = original_img
        self.max_pixel_displacement = max_pixel_displacement

    def _set_seed(self, seed: int = None) -> None:
        if seed is not None:
            np.random.seed(seed)

    def displace_img_horizontally(
        self, img: np.ndarray = None, max_displacement: int = None, seed: int = None
    ) -> np.ndarray:
        self._set_seed(seed)
        if img is None:
            img = self.original_img
        if max_displacement is None:
            max_displacement = self.max_pixel_displacement

        n_pixels_to_move = np.random.choice(np.arange(max_displacement+1), size=1)[0]
        if n_pixels_to_move == 0:
            return img

        zero_img = np.zeros(img.shape)
        move_to_right = np.random.choice([True, False], size=1)[0]
        if move_to_right:
            return np.hstack((zero_img[:,:(2*n_pixels_to_move)], img[:,n_pixels_to_move:-n_pixels_to_move]))
        else:
            return np.hstack((img[:,n_pixels_to_move:-n_pixels_to_move], zero_img[:,-(2*n_pixels_to_move):]))

    def displace_img_vertically(
        self, img: np.ndarray = None, max_displacement: int = None, seed: int = None
    ) -> np.ndarray:
        self._set_seed(seed)
        if img is None:
            img = self.original_img
        if max_displacement is None:
            max_displacement = self.max_pixel_displacement

        n_pixels_to_move = np.random.choice(np.arange(max_displacement+1), size=1)[0]
        if n_pixels_to_move == 0:
            return img

        zero_img = np.zeros(img.shape)
        move_up = np.random.choice([True, False], size=1)[0]
        if move_up:
            return np.vstack((zero_img[:(2*n_pixels_to_move),:], img[n_pixels_to_move:-n_pixels_to_move,:]))
        else:
            return np.vstack((img[n_pixels_to_move:-n_pixels_to_move,:], zero_img[:(2*n_pixels_to_move),:]))
        
    def generate_displaced_imgs(
        self, n_snapshots: int,  max_displacement: int = None, seed: int = None
    ) -> np.ndarray:
        self._set_seed(seed)
        if max_displacement is None:
            max_displacement = self.max_pixel_displacement

        snapshots = []
        for _ in range(n_snapshots):
            snapsht = np.copy(self.original_img)
            vdispl_snapsht = self.displace_img_vertically(img=snapsht)
            displ_snapsht = self.displace_img_horizontally(img=vdispl_snapsht)
            snapshots.append(displ_snapsht)

        return snapshots

    def add_rician_noise_fg(self, rice_b, sigma, img = None, center_dist = False, seed = None):
        """Adds Rician noise to the foreground part of the image.
        
        Returns: tuple with the corrupted img and the used foreground mask.
        """
        self._set_seed(seed)
        if img is None:
            img = self.original_img

        fg_mask = img > 0.0    
        n_rician_samples = np.prod(img.shape)
        rician_samples = rice.rvs(b=rice_b, scale=sigma, size=n_rician_samples)
        if center_dist:
            rice_dist_mean = rice.mean(b=rice_b, scale=sigma)
            rician_samples -= rice_dist_mean
        rician_noise_mtx = np.reshape(rician_samples, img.shape)
        rician_noisy_img = img + rician_noise_mtx
        # Returning the image corrupted just in the foreground (ignoring the background)
        return fg_mask * rician_noisy_img, fg_mask

    def generate_rician_noisy_displaced_imgs(
        self,
        n_snapshots: int,
        sigma: float,
        rice_b: float,
        max_displacement: int = None,
        center_noise_dist: bool = False,
        seed: int = None,
    ) -> np.ndarray:
        if seed is not None: np.random.seed(seed)

        displaced_snapshots = self.generate_displaced_imgs(
            n_snapshots=n_snapshots, max_displacement=max_displacement
        )

        fg_masks = []
        corrupted_snapshots = []
        for displ_snapsh in displaced_snapshots:
            rician_noised_ss, fg_m = self.add_rician_noise_fg(
                img=displ_snapsh, rice_b=rice_b, sigma=sigma, center_dist=center_noise_dist
            )
            corrupted_snapshots.append(rician_noised_ss)
            fg_masks.append(fg_m)

        return np.stack(corrupted_snapshots, axis=0), np.stack(fg_masks, axis=0)


def main():
    """MAIN FUNCTION"""
    _setup_img_dir()

    # # Gaussian ensemble
    # plot_figure_1()
    # # Wishart ensemble
    # plot_figure_2()
    # # Manova ensemble
    # plot_figure_3()
    # # Circular ensemble
    # plot_figure_4()

    # # Standard vs tridiagonal histograms 
    # plot_figure_5()

    # # Tridiagonal optimization
    # plot_figure_6()
    # plot_figure_7()

    # # Joint eigenvalue PDF
    # plot_figure_8()

    # # Wigner Semicircle law
    # plot_figure_9()
    # plot_figure_10()
    # plot_figure_11()

    # # Tracy-Widom law
    # plot_figure_12()
    # plot_figure_13()
    # plot_figure_14()

    # # Marchenko-Pastur law
    # plot_figure_15()
    # plot_figure_16()
    # plot_figure_17()

    # # Manova Spectrum distr.
    # plot_figure_18()
    # plot_figure_19()
    # plot_figure_20()

    # MRI image denoising
    plot_figure_22()


if __name__ == "__main__":
    main()
