import os
import sys
import numpy as np
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)
sys.path.append(dirname[:dirname.rfind("/")])  # TODO: Remove

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
from skrmt.ensemble.utils import standard_vs_tridiag_hist

IMGS_DIRNAME = "skrmt_sim_imgs"
SCRIPT_PATH = os.path.dirname(__file__)


def setup_img_dir():
    """Creates a directory (if ti does not exist) in the same locations
    as this script for the simulation images of scikit-rmt.
    """
    # creating a directory in the same place as this script for the images
    imgs_dir_path = os.path.join(SCRIPT_PATH, IMGS_DIRNAME)
    if not os.path.exists(imgs_dir_path):
        print(f"Creating directory {IMGS_DIRNAME} at {SCRIPT_PATH} for the images.")
        os.makedirs(imgs_dir_path)

    print(f"The images of the Figures will be stored at {imgs_dir_path}")


def plot_figure_1():
    print("Generating images of Figure 1. This may take some seconds...")

    ens = GaussianEnsemble(beta=1, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig1_goe_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    ens = GaussianEnsemble(beta=2, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig1_gue_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    ens = GaussianEnsemble(beta=4, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig1_gse_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    print("[DONE] - Images Figure 1")


def plot_figure_2():
    print("Generating images of Figure 2. This may take some seconds...")

    ens = WishartEnsemble(beta=1, p=1000, n=5000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig2_wre_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    ens = WishartEnsemble(beta=2, p=1000, n=5000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig2_wce_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    ens = WishartEnsemble(beta=4, p=1000, n=5000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig2_wqe_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    print("[DONE] - Images Figure 2")


def plot_figure_3():
    print("Generating images of Figure 3. This may take some seconds...")

    ens = ManovaEnsemble(beta=1, m=1000, n1=2000, n2=2000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig3_mre_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    ens = ManovaEnsemble(beta=2, m=1000, n1=2000, n2=2000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig3_mce_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    ens = ManovaEnsemble(beta=4, m=1000, n1=2000, n2=2000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig3_mqe_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    print("[DONE] - Images Figure 3")


def plot_figure_4():
    print("Generating images of Figure 4. This may take some seconds...")

    ens = CircularEnsemble(beta=1, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig4_coe_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    ens = CircularEnsemble(beta=2, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig4_cue_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    ens = CircularEnsemble(beta=4, n=1000, random_state=1)
    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig4_cse_1000.png")
    ens.plot_eigval_hist(bins=80, density=True, savefig_path=ens_figpath)
    plt.clf()

    print("[DONE] - Images Figure 4")


def plot_figure_5():
    print("Generating images of Figure 5. This may take some seconds...")

    ens_figpath = os.path.join(SCRIPT_PATH, IMGS_DIRNAME, "fig5_goe_1000.png")
    ens = GaussianEnsemble(beta=1, n=1000)
    standard_vs_tridiag_hist(ensemble=ens, bins=60, random_state=10, savefig_path=ens_figpath)

    print("[DONE] - Images Figure 5")


# TODO: Function for Figure 6 !!!
# TODO: Function for Figure 7 !!!


def plot_figure_8():
    print("Generating images of Figure 8. This may take some few minutes...")

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

    print("[DONE] - Images Figure 8")



def main():
    setup_img_dir()
    # plot_figure_1()
    # plot_figure_2()
    # plot_figure_3()
    # plot_figure_4()

    # plot_figure_5()

    # TODO: Figure 6 !!!!
    # TODO: Figure 7 !!!!

    plot_figure_8()



if __name__ == "__main__":
    main()